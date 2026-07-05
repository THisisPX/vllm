#!/usr/bin/env python3
"""Recall analysis v2 — multi-angle Prefill↔Decode expert overlap.

Adds analyses motivated by BEAM and ReMoE:

1. **Per-layer Recall@N(K)** — deep vs shallow layers may differ
2. **Step-to-step EOR** (ReMoE-style) — adjacent decode token overlap
3. **Jaccard similarity** — union-aware overlap, not just Top-N intersection
4. **Expert probability mass overlap** — weighted by frequency, not binary
5. **Layer-wise expert concentration** — entropy/HHI per layer

Usage:
    python analysis/recall_analysis_v2.py \
        --input traces/phase2_sharegpt/traces.jsonl \
        --output analysis/results/recall_v2/
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

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
            rid = rec["request_id"]
            if rid.startswith("_warmup"):
                continue
            records.append(rec)

    groups = defaultdict(list)
    for r in records:
        groups[r["request_id"]].append(r)
    return dict(groups)


# ---------------------------------------------------------------------------
# Metric 1: Per-layer Recall@N(K)
# ---------------------------------------------------------------------------

def compute_per_layer_recall(
    groups: dict[str, list[dict]],
    num_layers: int,
    N_values=(4, 8, 16),
    K_values=(8, 16, 32, 64, 128),
) -> dict:
    """Return means[layer][N][K], stds[layer][N][K]."""
    all_req = defaultdict(list)  # layer -> list of {N: {K: recall}}

    for rid, records in groups.items():
        prefill = defaultdict(Counter)
        decode = defaultdict(list)
        for r in records:
            li = r["layer_idx"]
            if r["phase"] == "prefill":
                for e in r["experts"]:
                    prefill[li][e] += 1
            else:
                dti = r.get("decode_token_idx", -1)
                if dti >= 0:
                    decode[li].append((dti, tuple(r["experts"])))

        for li in range(num_layers):
            pc = prefill.get(li)
            dl = decode.get(li)
            if not pc or not dl:
                continue
            dl.sort(key=lambda x: x[0])

            lay_res = {}
            for N in N_values:
                topN_pf = {e for e, _ in pc.most_common(N)}
                lay_res[N] = {}
                for K in K_values:
                    dc_counts = Counter()
                    for _, experts in dl[:K]:
                        for e in experts:
                            dc_counts[e] += 1
                    topN_dc = {e for e, _ in dc_counts.most_common(N)}
                    if not topN_dc:
                        lay_res[N][K] = 1.0
                    else:
                        lay_res[N][K] = len(topN_pf & topN_dc) / len(topN_dc)
            all_req[str(li)].append(lay_res)

    means, stds = {}, {}
    for li_str, results in all_req.items():
        means[li_str] = {}
        stds[li_str] = {}
        for N in N_values:
            means[li_str][str(N)] = {}
            stds[li_str][str(N)] = {}
            for K in K_values:
                vals = [r[N][K] for r in results]
                means[li_str][str(N)][K] = float(np.mean(vals))
                stds[li_str][str(N)][K] = float(np.std(vals))
    return means, stds


# ---------------------------------------------------------------------------
# Metric 2: Step-to-step EOR (ReMoE-style)
# ---------------------------------------------------------------------------

def compute_step_to_step_eor(
    groups: dict[str, list[dict]],
    num_layers: int,
) -> dict[str, list[float]]:
    """Compute per-layer expert overlap between adjacent decode tokens.

    EOR(d) = |experts(d) ∩ experts(d-1)| / |experts(d-1) ∪ experts(d)|
    """
    all_layer_eor = defaultdict(list)

    for records in groups.values():
        decode_by_layer = defaultdict(list)
        for r in records:
            if r["phase"] == "decode":
                dti = r.get("decode_token_idx", -1)
                if dti >= 0:
                    decode_by_layer[r["layer_idx"]].append((dti, set(r["experts"])))

        for li, tokens in decode_by_layer.items():
            tokens.sort(key=lambda x: x[0])
            for i in range(1, len(tokens)):
                prev = tokens[i - 1][1]
                curr = tokens[i][1]
                union = prev | curr
                if union:
                    intersection = prev & curr
                    all_layer_eor[str(li)].append(len(intersection) / len(union))

    return dict(all_layer_eor)


# ---------------------------------------------------------------------------
# Metric 3: Jaccard similarity (Prefill↔Decode window)
# ---------------------------------------------------------------------------

def compute_jaccard_prefill_decode(
    groups: dict[str, list[dict]],
    num_layers: int,
    K_values=(8, 16, 32, 64, 128),
) -> dict[str, list[float]]:
    """Per-layer Jaccard between Prefill expert set and first-K Decode set."""
    all_layer = defaultdict(list)

    for records in groups.values():
        prefill = defaultdict(set)
        decode = defaultdict(list)
        for r in records:
            li = r["layer_idx"]
            if r["phase"] == "prefill":
                prefill[li] |= set(r["experts"])
            else:
                dti = r.get("decode_token_idx", -1)
                if dti >= 0:
                    decode[li].append((dti, set(r["experts"])))

        for li in range(num_layers):
            pf = prefill.get(li, set())
            dl = decode.get(li, [])
            if not pf or not dl:
                continue
            dl.sort(key=lambda x: x[0])
            for K in K_values:
                dc_set = set().union(*(s for _, s in dl[:K]))
                union = pf | dc_set
                jaccard = len(pf & dc_set) / len(union) if union else 1.0
                all_layer[str(li)].append(jaccard)

    means = {}
    for li_str, vals in all_layer.items():
        means[li_str] = {str(K): float(np.mean(vals)) for K, vals in
                         zip(K_values, zip(*[vals[i::len(K_values)]
                                              for i in range(len(K_values))]))}
    return means


# ---------------------------------------------------------------------------
# Metric 4: Expert concentration (HHI) per layer
# ---------------------------------------------------------------------------

def compute_expert_concentration(
    groups: dict[str, list[dict]],
    num_layers: int,
    num_experts: int = 64,
) -> dict[str, dict[str, float]]:
    """Per-layer Herfindahl-Hirschman Index (0=uniform, 1=monopoly)."""
    prefill_counts = defaultdict(Counter)
    decode_counts = defaultdict(Counter)

    for records in groups.values():
        for r in records:
            li = r["layer_idx"]
            if r["phase"] == "prefill":
                for e in r["experts"]:
                    prefill_counts[li][e] += 1
            else:
                for e in r["experts"]:
                    decode_counts[li][e] += 1

    result = {}
    for li in range(num_layers):
        pf_hhi = _compute_hhi(prefill_counts.get(li, Counter()), num_experts)
        dc_hhi = _compute_hhi(decode_counts.get(li, Counter()), num_experts)
        pf_entropy = _compute_entropy(prefill_counts.get(li, Counter()), num_experts)
        dc_entropy = _compute_entropy(decode_counts.get(li, Counter()), num_experts)
        result[str(li)] = {
            "prefill_hhi": pf_hhi,
            "decode_hhi": dc_hhi,
            "prefill_entropy": pf_entropy,
            "decode_entropy": dc_entropy,
            "hhi_delta": dc_hhi - pf_hhi,
        }
    return result


def _compute_hhi(counter: Counter, n_total: int) -> float:
    """Normalized HHI: 1/n → 1, where n = n_total."""
    if not counter:
        return 0.0
    total = sum(counter.values())
    if total == 0:
        return 0.0
    raw = sum((v / total) ** 2 for v in counter.values())
    # Normalize: unity-n = 1 - 1/n, so HHI_norm = (HHI_raw - 1/n) / (1 - 1/n)
    h_min = 1.0 / n_total
    return (raw - h_min) / (1 - h_min) if n_total > 1 else raw


def _compute_entropy(counter: Counter, n_total: int) -> float:
    """Normalized entropy: 0 → 1 where 1 = uniform across all experts."""
    if not counter or n_total <= 1:
        return 1.0
    total = sum(counter.values())
    if total == 0:
        return 1.0
    probs = np.array([v / total for v in counter.values()])
    raw = -np.sum(probs * np.log(probs + 1e-12))
    max_entropy = np.log(n_total)
    return float(raw / max_entropy) if max_entropy > 0 else 1.0


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_per_layer_recall(means, K_values, output_dir):
    """Heatmap: layer × K for N=8."""
    N_str = "8"
    layers = sorted(means.keys(), key=int)
    matrix = np.zeros((len(layers), len(K_values)))

    for i, li in enumerate(layers):
        for j, K in enumerate(K_values):
            matrix[i, j] = means[li][N_str].get(K, 0.0)

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=0.6, origin="lower")
    ax.set_xticks(range(len(K_values)))
    ax.set_xticklabels([str(k) for k in K_values])
    ax.set_xlabel("K (decode tokens)")
    ax.set_ylabel("MoE Layer Index")
    ax.set_title("Recall@8 per Layer (Prefill → Decode)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Recall@8")
    fig.tight_layout()
    fig.savefig(output_dir / "per_layer_recall.png", dpi=120)
    plt.close(fig)
    print(f"  ✓ per_layer_recall.png")


def plot_step_to_step_eor(eor_by_layer, output_dir):
    """Bar chart: mean step-to-step EOR per layer."""
    layers = sorted(eor_by_layer.keys(), key=int)
    means = [np.mean(eor_by_layer[li]) for li in layers]
    stds = [np.std(eor_by_layer[li]) for li in layers]
    x = np.arange(len(layers))

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(x, means, yerr=stds, color="#4CAF50", edgecolor="white", capsize=2)
    ax.axhline(y=np.mean(means), color="red", linestyle="--",
               label=f"Global mean: {np.mean(means):.3f}")
    ax.set_xlabel("MoE Layer Index")
    ax.set_ylabel("Jaccard EOR (step-to-step)")
    ax.set_title("ReMoE-style Step-to-Step Expert Overlap per Layer")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(output_dir / "step_to_step_eor.png", dpi=120)
    plt.close(fig)
    print(f"  ✓ step_to_step_eor.png (global mean: {np.mean(means):.3f})")


def plot_concentration(conc, output_dir):
    """Scatter: prefill HHI vs decode HHI per layer."""
    layers = sorted(conc.keys(), key=int)
    pf_hhi = [conc[li]["prefill_hhi"] for li in layers]
    dc_hhi = [conc[li]["decode_hhi"] for li in layers]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.scatter(pf_hhi, dc_hhi, c=range(len(layers)), cmap="viridis", s=60)
    for i, li in enumerate(layers):
        ax1.annotate(li, (pf_hhi[i], dc_hhi[i]), fontsize=7, alpha=0.7,
                     xytext=(3, 3), textcoords="offset points")
    ax1.plot([0, 1], [0, 1], "k--", alpha=0.3, label="y=x (no shift)")
    ax1.set_xlabel("Prefill HHI")
    ax1.set_ylabel("Decode HHI")
    ax1.set_title("Expert Concentration: Prefill vs Decode")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    hhi_delta = [conc[li]["hhi_delta"] for li in layers]
    colors = ["#F44336" if d > 0 else "#2196F3" for d in hhi_delta]
    ax2.bar(np.arange(len(layers)), hhi_delta, color=colors, edgecolor="white")
    ax2.set_xlabel("MoE Layer Index")
    ax2.set_ylabel("Δ HHI (Decode − Prefill)")
    ax2.set_title("Expert Concentration Shift per Layer")
    ax2.axhline(y=0, color="black", linewidth=0.5)
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.text(0.02, 0.95, "Red = Decode more concentrated\nBlue = Prefill more concentrated",
             transform=ax2.transAxes, fontsize=8, va="top")

    fig.tight_layout()
    fig.savefig(output_dir / "expert_concentration.png", dpi=120)
    plt.close(fig)
    print(f"  ✓ expert_concentration.png")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_parser():
    p = argparse.ArgumentParser(description="Recall analysis v2 — multi-angle")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Not found: {input_path}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    groups = load_grouped(str(input_path))
    num_layers = 26
    N_values = (4, 8, 16)
    K_values = (8, 16, 32, 64, 128)

    real_reqs = {k: v for k, v in groups.items() if v}
    print(f"Loaded {len(real_reqs)} real requests")

    # --- Metric 1: Per-layer Recall ---
    means, stds = compute_per_layer_recall(groups, num_layers, N_values, K_values)

    # Print per-layer summary for N=8.
    print()
    print("=" * 70)
    print("RECALL@8 PER-LAYER (Deep layers → more important for reasoning)")
    print("=" * 70)
    layers = sorted(means.keys(), key=int)
    for li in layers[:5] + ["..."] + layers[-5:]:
        if li == "...":
            print("  ...")
            continue
        row = f"  L{li:>4s}:"
        for K in K_values:
            row += f"  K={K:>4}: {means[li]['8'][K]:.3f}"
        print(row)
    print("-" * 70)

    # Mean for deep layers (last 8) vs shallow layers (first 8)
    shallow = sorted(means.keys(), key=int)[:8]
    deep = sorted(means.keys(), key=int)[-8:]
    for label, subset in [("Shallow (L0-L7)", shallow), ("Deep (L18-L25)", deep)]:
        for N in N_values:
            n_str = str(N)
            avg = np.mean([means[li][n_str][8] for li in subset])
            print(f"  {label} Recall@{N}(K=8): {avg:.3f}")
    print()

    # --- Metric 2: Step-to-step EOR ---
    eor = compute_step_to_step_eor(groups, num_layers)
    global_eor = np.mean([np.mean(v) for v in eor.values()])
    print(f"Step-to-step EOR (ReMoE-style, single request):")
    print(f"  Global mean: {global_eor:.3f}")
    print(f"  This is the decode→decode locality — baseline for EAP comparison")
    print()

    # --- Metric 3: Jaccard Prefill↔Decode ---
    jaccard = compute_jaccard_prefill_decode(groups, num_layers, K_values)
    jacc_means = {}
    for K in K_values:
        vals = [jaccard[li][str(K)] for li in sorted(jaccard.keys(), key=int)]
        jacc_means[K] = float(np.mean(vals))
    print("Jaccard similarity (Prefill↔Decode):")
    for K, v in jacc_means.items():
        print(f"  K={K:>4}: {v:.3f}")
    print()

    # --- Metric 4: Expert Concentration ---
    conc = compute_expert_concentration(groups, num_layers)
    pf_hhi_avg = np.mean([conc[li]["prefill_hhi"] for li in sorted(conc.keys(), key=int)])
    dc_hhi_avg = np.mean([conc[li]["decode_hhi"] for li in sorted(conc.keys(), key=int)])
    print(f"Expert concentration (normalized HHI, 0=uniform 1=monopoly):")
    print(f"  Prefill mean HHI: {pf_hhi_avg:.3f}")
    print(f"  Decode  mean HHI: {dc_hhi_avg:.3f}")
    print(f"  Δ (Decode − Prefill): {dc_hhi_avg - pf_hhi_avg:+.3f}")
    print(f"  {'Decode more concentrated → fewer experts dominate' if dc_hhi_avg > pf_hhi_avg else 'Prefill more concentrated'}")
    print()

    # Save results
    summary = {
        "num_requests": len(real_reqs),
        "num_layers": num_layers,
        "global_step_to_step_eor": global_eor,
        "jaccard_means": {str(k): v for k, v in jacc_means.items()},
        "concentration": {
            "prefill_mean_hhi": pf_hhi_avg,
            "decode_mean_hhi": dc_hhi_avg,
        },
        "per_layer_recall_N8": {
            li: {str(K): means[li]["8"][K] for K in K_values}
            for li in sorted(means.keys(), key=int)
        },
        "per_layer_eor": {
            li: float(np.mean(v)) for li, v in eor.items()
        },
        "per_layer_concentration": conc,
    }
    (output_dir / "recall_v2_results.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )

    # Plots
    plot_per_layer_recall(means, K_values, output_dir)
    plot_step_to_step_eor(eor, output_dir)
    plot_concentration(conc, output_dir)
    print(f"\nAll outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
