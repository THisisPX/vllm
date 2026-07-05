#!/usr/bin/env python3
"""EAP Feasibility Analysis — distribution-level metrics.

Key question: Does the Prefill expert frequency distribution predict
the Decode expert frequency distribution?

If yes → EAP (probabilistic profile) has predictive value.
If no  → even a 6.5 KB frequency summary can't help cold-start.

Metrics (all per MoE layer, then aggregated):

1. **KL Divergence**  D_KL(P_decode || P_prefill)
   Small → prefill distribution is a good prior for decode.
   Large → decode demands very different experts than prefill.

2. **Frequency-Weighted Overlap** (a.k.a. soft Recall)
   For each expert e:
     overlap = Σ min(w_prefill[e], w_decode[e])
   where w_prefill[e] = count[e] / Σ counts.
   Ranges [0, 1]; 1 = identical distributions.

3. **EAP Cache Hit Rate Simulation**
   Simulate a weighted cache: prefill top-K experts are "pre-loaded"
   (hit=1), remaining experts have hit probability proportional to
   their prefill frequency.  Compare against:
     (a) random cache (no prefill info)
     (b) oracle cache (perfect decode prediction)

4. **Rank Correlation**  Spearman ρ between prefill and decode
   expert rankings.  ρ close to 1 = strong rank alignment.

Usage:
    python analysis/eap_feasibility.py \\
        --input traces/phase2_sharegpt/traces.jsonl \\
        --output analysis/results/eap/
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

matplotlib.use("Agg")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_grouped(path: str) -> dict[str, list[dict]]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec["request_id"].startswith("_warmup"):
                continue
            records.append(rec)

    groups = defaultdict(list)
    for r in records:
        groups[r["request_id"]].append(r)
    return dict(groups)


def _counts_to_probs(counter: Counter, n_experts: int) -> np.ndarray:
    """Convert counter to probability distribution (smoothed)."""
    probs = np.zeros(n_experts, dtype=np.float64)
    total = sum(counter.values())
    if total == 0:
        probs.fill(1.0 / n_experts)
        return probs
    # Add Laplace smoothing (pseudocount = 1) to avoid log(0)
    for e, c in counter.items():
        probs[e] = c + 1.0
    probs /= probs.sum()
    return probs


# ---------------------------------------------------------------------------
# Per-layer distribution metrics
# ---------------------------------------------------------------------------


def compute_distribution_metrics(
    groups: dict[str, list[dict]],
    num_layers: int,
    num_experts: int = 64,
) -> dict:
    """Compute KL, soft-overlap, Spearman ρ per layer across all requests."""
    all_kl: dict[int, list[float]] = defaultdict(list)
    all_overlap: dict[int, list[float]] = defaultdict(list)
    all_rho: dict[int, list[float]] = defaultdict(list)

    for records in groups.values():
        prefill = defaultdict(Counter)
        decode = defaultdict(Counter)
        for r in records:
            li = r["layer_idx"]
            if r["phase"] == "prefill":
                for e in r["experts"]:
                    prefill[li][e] += 1
            elif r["phase"] == "decode":
                for e in r["experts"]:
                    decode[li][e] += 1

        for li in range(num_layers):
            pf_probs = _counts_to_probs(prefill.get(li, Counter()), num_experts)
            dc_probs = _counts_to_probs(decode.get(li, Counter()), num_experts)

            # KL divergence: D_KL(P_decode || P_prefill)
            kl = np.sum(dc_probs * np.log(dc_probs / pf_probs))
            all_kl[li].append(float(kl))

            # Frequency-weighted overlap: Σ min(pf[e], dc[e])
            overlap = np.sum(np.minimum(pf_probs, dc_probs))
            all_overlap[li].append(float(overlap))

            # Spearman rank correlation
            if pf_probs.sum() > 0 and dc_probs.sum() > 0:
                rho, _ = stats.spearmanr(pf_probs, dc_probs)
                all_rho[li].append(float(rho) if not np.isnan(rho) else 0.0)
            else:
                all_rho[li].append(0.0)

    means = {}
    for li in range(num_layers):
        means[str(li)] = {
            "kl_divergence": float(np.mean(all_kl[li])),
            "kl_std": float(np.std(all_kl[li])),
            "soft_overlap": float(np.mean(all_overlap[li])),
            "overlap_std": float(np.std(all_overlap[li])),
            "spearman_rho": float(np.mean(all_rho[li])),
            "rho_std": float(np.std(all_rho[li])),
        }
    return means


# ---------------------------------------------------------------------------
# EAP Cache Hit Rate Simulation
# ---------------------------------------------------------------------------


def simulate_eap_cache(
    groups: dict[str, list[dict]],
    num_layers: int,
    num_experts: int = 64,
    cache_size: int = 16,
    decode_window: int = 64,
) -> dict:
    """Simulate EAP-guided cache vs random vs oracle.

    For each request:
    1. Build EAP from prefill (frequency histogram).
    2. Initialize weighted cache: top cache_size experts by prefill
       frequency are "pre-loaded", remaining weighted by frequency.
    3. For first decode_window tokens, measure hit rate.
    4. Compare against:
       - random: cache_size random experts
       - oracle: cache_size most frequent decode experts (upper bound)
    """
    eap_hits: dict[int, list[float]] = defaultdict(list)
    random_hits: dict[int, list[float]] = defaultdict(list)
    oracle_hits: dict[int, list[float]] = defaultdict(list)

    rng = np.random.default_rng(42)

    for records in groups.values():
        prefill = defaultdict(Counter)
        decode_list = defaultdict(list)
        for r in records:
            li = r["layer_idx"]
            if r["phase"] == "prefill":
                for e in r["experts"]:
                    prefill[li][e] += 1
            else:
                dti = r.get("decode_token_idx", -1)
                if dti >= 0:
                    decode_list[li].append((dti, tuple(r["experts"])))

        for li in range(num_layers):
            pf = prefill.get(li, Counter())
            dl = decode_list.get(li, [])
            if not pf or not dl:
                continue

            # Build EAP cache: top-K by prefill frequency.
            eap_ranked = [e for e, _ in pf.most_common()]
            eap_cache = set(eap_ranked[:cache_size])

            # Random cache.
            random_cache = set(rng.choice(num_experts, size=cache_size, replace=False))

            # Oracle cache: most frequent in first decode_window.
            dc_counts = Counter()
            dl_for_window = [d for d in dl if d[0] < decode_window]
            if not dl_for_window:
                continue
            for _, experts in dl_for_window:
                for e in experts:
                    dc_counts[e] += 1
            oracle_ranked = [e for e, _ in dc_counts.most_common()]
            oracle_cache = set(oracle_ranked[:cache_size])

            # Measure hits across all decode tokens.
            eap_token_hits = []
            random_token_hits = []
            oracle_token_hits = []
            for dti, experts in dl:
                actual = set(experts)
                eap_token_hits.append(len(actual & eap_cache) / len(actual))
                random_token_hits.append(len(actual & random_cache) / len(actual))
                oracle_token_hits.append(len(actual & oracle_cache) / len(actual))

            eap_hits[li].append(float(np.mean(eap_token_hits)))
            random_hits[li].append(float(np.mean(random_token_hits)))
            oracle_hits[li].append(float(np.mean(oracle_token_hits)))

    result = {}
    for li in range(num_layers):
        result[str(li)] = {
            "eap_hit_rate": float(np.mean(eap_hits.get(li, [0]))),
            "random_hit_rate": float(np.mean(random_hits.get(li, [0]))),
            "oracle_hit_rate": float(np.mean(oracle_hits.get(li, [0]))),
            "eap_vs_random_delta": float(
                np.mean(eap_hits.get(li, [0])) - np.mean(random_hits.get(li, [0]))
            ),
        }
    return result


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_distribution_metrics(
    metrics: dict, num_layers: int, num_experts: int, output_dir: Path
) -> None:
    layers = [str(i) for i in range(num_layers)]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. KL Divergence
    ax = axes[0, 0]
    kl_vals = [metrics[li]["kl_divergence"] for li in layers]
    kl_err = [metrics[li]["kl_std"] for li in layers]
    ax.bar(range(num_layers), kl_vals, yerr=kl_err, color="#FF9800",
           edgecolor="white", capsize=2)
    ax.axhline(y=np.mean(kl_vals), color="red", linestyle="--",
               label=f"Mean: {np.mean(kl_vals):.3f}")
    ax.set_xlabel("MoE Layer Index")
    ax.set_ylabel("D_KL(Decode || Prefill)")
    ax.set_title("KL Divergence per Layer\n(smaller = better prediction)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # 2. Frequency-weighted overlap
    ax = axes[0, 1]
    ov_vals = [metrics[li]["soft_overlap"] for li in layers]
    ax.bar(range(num_layers), ov_vals, color="#4CAF50", edgecolor="white")
    ax.axhline(y=np.mean(ov_vals), color="red", linestyle="--",
               label=f"Mean: {np.mean(ov_vals):.3f}")
    ax.set_xlabel("MoE Layer Index")
    ax.set_ylabel("Frequency-Weighted Overlap")
    ax.set_title("Soft Expert Overlap per Layer\n(1.0 = identical distributions)")
    ax.set_ylim(0, 1.02)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # 3. Spearman rank correlation
    ax = axes[1, 0]
    rho_vals = [metrics[li]["spearman_rho"] for li in layers]
    rho_err = [metrics[li]["rho_std"] for li in layers]
    colors = ["#2196F3" if r > 0.3 else "#F44336" if r < 0.1 else "#FF9800"
              for r in rho_vals]
    ax.bar(range(num_layers), rho_vals, yerr=rho_err, color=colors,
           edgecolor="white", capsize=2)
    ax.axhline(y=np.mean(rho_vals), color="red", linestyle="--",
               label=f"Mean ρ: {np.mean(rho_vals):.3f}")
    ax.set_xlabel("MoE Layer Index")
    ax.set_ylabel("Spearman ρ")
    ax.set_title("Rank Correlation per Layer\n(ρ > 0.5 = strong alignment)")
    ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.5, label="ρ=0.5")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # 4. Summary text
    ax = axes[1, 1]
    ax.axis("off")
    mean_kl = np.mean(kl_vals)
    mean_ov = np.mean(ov_vals)
    mean_rho = np.mean(rho_vals)

    # Interpret
    if mean_kl < 0.5 and mean_rho > 0.3 and mean_ov > 0.3:
        verdict = "✓ EAP LIKELY VIABLE"
        color = "green"
    elif mean_kl < 1.0 and mean_rho > 0.15:
        verdict = "~ EAP MARGINALLY VIABLE"
        color = "orange"
    else:
        verdict = "✗ EAP UNLIKELY TO HELP"
        color = "red"

    # Random baseline
    random_overlap = float(cache_size := 16) / num_experts
    random_rho = 0.0
    random_kl = np.log(num_experts / (cache_size / 2 + 1)) if cache_size > 0 else 3.0

    summary_lines = [
        f"=== EAP FEASIBILITY VERDICT ===",
        f"",
        f"Verdict: {verdict}",
        f"",
        f"--- Measured ---",
        f"KL Divergence:    {mean_kl:.4f}",
        f"Soft Overlap:     {mean_ov:.4f}",
        f"Spearman ρ:       {mean_rho:.4f}",
        f"",
        f"--- Random Baseline ---",
        f"KL (random):      {random_kl:.2f}",
        f"Soft Overlap:     {random_overlap:.4f}",
        f"Spearman ρ:       0.000",
        f"",
        f"--- Interpretation ---",
        f"KL < 1.0 + ρ > 0.2 → EAP provides signal",
        f"above random baseline",
    ]
    ax.text(0.05, 0.95, "\n".join(summary_lines),
            transform=ax.transAxes, fontsize=11, fontfamily="monospace",
            va="top", linespacing=1.4,
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    fig.suptitle(f"EAP Feasibility Analysis ({num_layers} layers × {num_experts} experts)",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_dir / "eap_distribution_metrics.png", dpi=120)
    plt.close(fig)
    print(f"  ✓ eap_distribution_metrics.png saved")


def plot_cache_simulation(cache_result: dict, num_layers: int, output_dir: Path) -> None:
    layers = [str(i) for i in range(num_layers)]
    eap_vals = [cache_result[li]["eap_hit_rate"] for li in layers]
    random_vals = [cache_result[li]["random_hit_rate"] for li in layers]
    oracle_vals = [cache_result[li]["oracle_hit_rate"] for li in layers]

    x = np.arange(num_layers)
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(x - width, eap_vals, width, color="#2196F3", label="EAP-guided cache")
    ax.bar(x, random_vals, width, color="#9E9E9E", label="Random cache")
    ax.bar(x + width, oracle_vals, width, color="#4CAF50", label="Oracle (upper bound)")

    ax.set_xlabel("MoE Layer Index")
    ax.set_ylabel("Cache Hit Rate")
    ax.set_title("Cache Hit Rate: EAP vs Random vs Oracle\n(prefill-guided cache init, cache_size=16)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 1.02)

    # Annotation
    mean_eap = np.mean(eap_vals)
    mean_rand = np.mean(random_vals)
    delta_pct = (mean_eap - mean_rand) / mean_rand * 100 if mean_rand > 0 else 0
    ax.text(0.98, 0.95,
            f"EAP mean:   {mean_eap:.3f}\nRandom:     {mean_rand:.3f}\nDelta:      {delta_pct:+.1f}%",
            transform=ax.transAxes, fontsize=10, fontfamily="monospace",
            va="top", ha="right",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    fig.tight_layout()
    fig.savefig(output_dir / "eap_cache_simulation.png", dpi=120)
    plt.close(fig)
    print(f"  ✓ eap_cache_simulation.png saved (EAP Δ vs random: {delta_pct:+.1f}%)")


def plot_kl_vs_rho_scatter(metrics: dict, num_layers: int, output_dir: Path) -> None:
    """Scatter plot: each layer = one point (KL, ρ). Which layers are predictable?"""
    layers = [str(i) for i in range(num_layers)]
    kl_vals = [metrics[li]["kl_divergence"] for li in layers]
    rho_vals = [metrics[li]["spearman_rho"] for li in layers]

    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(kl_vals, rho_vals, c=range(num_layers),
                         cmap="viridis", s=80, edgecolors="white", linewidths=0.5)
    for i, li in enumerate(layers):
        ax.annotate(li, (kl_vals[i], rho_vals[i]),
                    fontsize=7, xytext=(5, 3), textcoords="offset points")

    # Quadrant lines
    ax.axhline(y=0.2, color="gray", linestyle=":", alpha=0.5)
    ax.axvline(x=1.0, color="gray", linestyle=":", alpha=0.5)

    # Label quadrants
    ax.text(0.1, 0.46, "GOOD\n(low KL + high ρ)\nEAP works best here",
            ha="left", fontsize=8, color="green", fontweight="bold")
    ax.text(2.5, 0.46, "WEAK SIGNAL\n(high KL + high ρ)\nEAP still useful?",
            ha="left", fontsize=8, color="orange", fontweight="bold")
    ax.text(0.1, 0.05, "NOISY\n(low KL + low ρ)",
            ha="left", fontsize=8, color="gray")

    ax.set_xlabel("KL Divergence (Decode || Prefill)")
    ax.set_ylabel("Spearman ρ")
    ax.set_title("Per-Layer EAP Predictability\n(quadrant: low KL + high ρ = EAP-friendly)")

    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Layer Index")

    fig.tight_layout()
    fig.savefig(output_dir / "eap_kl_vs_rho.png", dpi=120)
    plt.close(fig)
    print(f"  ✓ eap_kl_vs_rho.png saved")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="EAP Feasibility Analysis")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--num-experts", type=int, default=64)
    p.add_argument("--cache-size", type=int, default=16)
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    groups = load_grouped(str(input_path))
    num_reqs = len(groups)
    print(f"Loaded {num_reqs} real requests")

    # Infer num_layers from data.
    all_records = [r for recs in groups.values() for r in recs]
    num_layers = max(r["layer_idx"] for r in all_records) + 1
    print(f"  {num_layers} MoE layers, {args.num_experts} experts")

    # --- Distribution metrics ---
    print("\nComputing distribution-level metrics...")
    metrics = compute_distribution_metrics(groups, num_layers, args.num_experts)

    # Print summary
    kl_vals = [metrics[str(li)]["kl_divergence"] for li in range(num_layers)]
    ov_vals = [metrics[str(li)]["soft_overlap"] for li in range(num_layers)]
    rho_vals = [metrics[str(li)]["spearman_rho"] for li in range(num_layers)]

    print()
    print("=" * 70)
    print("EAP FEASIBILITY SUMMARY")
    print("=" * 70)
    print(f"  KL Divergence (mean ± std):  {np.mean(kl_vals):.4f} ± {np.std(kl_vals):.4f}")
    print(f"    (< 1.0 → good, 1.0-3.0 → moderate, > 3.0 → poor)")
    print(f"  Soft Overlap  (mean ± std):  {np.mean(ov_vals):.4f} ± {np.std(ov_vals):.4f}")
    print(f"    (> 0.25 → above random)")
    print(f"  Spearman ρ    (mean ± std):  {np.mean(rho_vals):.4f} ± {np.std(rho_vals):.4f}")
    print(f"    (> 0.3 → meaningful rank alignment)")
    print()

    # Layer breakdown: best and worst.
    layers_ranked = sorted(range(num_layers), key=lambda li: rho_vals[li], reverse=True)
    print("  Best  layers (highest ρ):", layers_ranked[:5],
          "ρ =", [round(rho_vals[li], 3) for li in layers_ranked[:5]])
    print("  Worst layers (lowest ρ): ", layers_ranked[-5:],
          "ρ =", [round(rho_vals[li], 3) for li in layers_ranked[-5:]])
    print()

    # --- Cache simulation ---
    print("Simulating EAP cache...")
    cache_result = simulate_eap_cache(
        groups, num_layers, args.num_experts, args.cache_size
    )
    eap_hr = np.mean([cache_result[str(li)]["eap_hit_rate"] for li in range(num_layers)])
    rand_hr = np.mean([cache_result[str(li)]["random_hit_rate"] for li in range(num_layers)])
    oracle_hr = np.mean([cache_result[str(li)]["oracle_hit_rate"] for li in range(num_layers)])

    print(f"  EAP cache hit rate:    {eap_hr:.4f}")
    print(f"  Random cache hit rate: {rand_hr:.4f}")
    print(f"  Oracle cache hit rate: {oracle_hr:.4f} (upper bound)")
    print(f"  Δ (EAP − Random):      {eap_hr - rand_hr:+.4f}")
    print(f"  EAP / Oracle ratio:    {eap_hr / oracle_hr:.1%} of optimal" if oracle_hr > 0 else "")

    # --- Verdict ---
    print()
    print("=" * 70)
    if np.mean(rho_vals) > 0.2 and eap_hr > rand_hr * 1.1:
        print("VERDICT: EAP provides statistically meaningful signal.")
        print("  Prefill → Decode expert frequency distribution has predictive value.")
        print("  The 6.5 KB EAP metadata is worth transmitting alongside KV Cache.")
    elif np.mean(rho_vals) > 0.1:
        print("VERDICT: EAP provides marginal signal — borderline viability.")
        print("  KL/overlap metrics suggest some structure but limited predictability.")
        print("  Consider model-specific or layer-specific EAP (only deep layers).")
    else:
        print("VERDICT: EAP signal is weak — unlikely to overcome cold-start.")
        print("  Prefill and Decode expert distributions are substantially different.")
        print("  EAP may not provide enough benefit to justify the 6.5 KB metadata.")
    print("=" * 70)

    # Save.
    summary = {
        "num_requests": num_reqs,
        "num_layers": num_layers,
        "num_experts": args.num_experts,
        "cache_size": args.cache_size,
        "mean_kl": float(np.mean(kl_vals)),
        "mean_soft_overlap": float(np.mean(ov_vals)),
        "mean_spearman_rho": float(np.mean(rho_vals)),
        "eap_hit_rate": float(eap_hr),
        "random_hit_rate": float(rand_hr),
        "oracle_hit_rate": float(oracle_hr),
        "per_layer_metrics": metrics,
        "per_layer_cache": cache_result,
    }
    (output_dir / "eap_feasibility.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )

    # Plots.
    plot_distribution_metrics(metrics, num_layers, args.num_experts, output_dir)
    plot_cache_simulation(cache_result, num_layers, output_dir)
    plot_kl_vs_rho_scatter(metrics, num_layers, output_dir)

    print(f"\nAll outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
