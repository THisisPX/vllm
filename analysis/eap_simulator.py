#!/usr/bin/env python3
"""Phase 3: Offline EAP Simulation against real trace data.

Simulates the full Prefill→Decode pipeline with EAP-guided expert
cache and measures cold-start reduction compared to baseline PD.

Three modes per request:
  - **baseline_pd**  — decode starts with empty cache (pure PD)
  - **eap**          — decode cache pre-loaded from EAP (our method)
  - **oracle**       — cache pre-loaded from decode ground truth (upper bound)

Metrics:
  - Cache hit rate per decode token
  - T90 (tokens to reach 90% steady-state)
  - Improvement during first 8 / 16 / 32 decode tokens

Usage:
    python analysis/eap_simulator.py \\
        --input traces/phase2_qwen3/traces.jsonl \\
        --output analysis/results/eap_sim/ \\
        --num-experts 128 \\
        --cache-size 32
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

# Import EAP from vllm._eap
import sys

_HERE = Path(__file__).resolve().parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from vllm._eap.profile import EapProfile
from vllm._eap.cache_manager import EapCacheManager


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


# ---------------------------------------------------------------------------
# Simulation
# ---------------------------------------------------------------------------


def simulate_one_request(
    records: list[dict],
    num_layers: int,
    num_experts: int,
    cache_size: int,
    eap_weight: float,
    decay_steps: int,
    k_preload: int,
    random_seed: int | None = None,
) -> dict:
    """Run all three modes on one request and return hit-rate traces."""

    # --- Build EAP from prefill ---
    eap = EapProfile(num_layers, num_experts)
    decode_tokens: dict[int, list[tuple[int, tuple[int, ...]]]] = defaultdict(list)
    for r in records:
        li = r["layer_idx"]
        if r["phase"] == "prefill":
            eap.record(li, r["experts"])
        else:
            dti = r.get("decode_token_idx", -1)
            if dti >= 0:
                decode_tokens[dti].append((li, tuple(r["experts"])))

    if not decode_tokens:
        return {}

    max_dti = max(decode_tokens.keys()) + 1

    # --- Baseline PD: empty cache ---
    cache_bl = EapCacheManager(
        num_layers, num_experts, cache_size,
        eap_weight=0.0, decay_steps=0,
    )
    bl_hits = np.zeros(max_dti)
    bl_accesses = np.zeros(max_dti)

    # --- EAP: pre-load from prefill ---
    cache_eap = EapCacheManager(
        num_layers, num_experts, cache_size,
        eap_weight=eap_weight, decay_steps=decay_steps,
    )
    cache_eap.init_from_eap(eap, k_preload=k_preload)
    eap_hits = np.zeros(max_dti)
    eap_accesses = np.zeros(max_dti)

    # --- Oracle: pre-load from decode ground truth ---
    from collections import Counter as _Counter
    oracle_eap = EapProfile(num_layers, num_experts)
    for dti, layer_recs in decode_tokens.items():
        for li, experts in layer_recs:
            oracle_eap.record(li, experts)

    cache_oracle = EapCacheManager(
        num_layers, num_experts, cache_size,
        eap_weight=1.0, decay_steps=99999,
    )
    cache_oracle.init_from_eap(oracle_eap, k_preload=min(cache_size, num_experts))
    oracle_hits = np.zeros(max_dti)
    oracle_accesses = np.zeros(max_dti)

    # --- Step through decode ---
    for dti in sorted(decode_tokens.keys()):
        layer_recs = decode_tokens[dti]
        cache_eap.step_decay()
        cache_bl.step_decay()

        for li, experts in layer_recs:
            bl_hits[dti] += cache_bl.access(li, list(experts))
            bl_accesses[dti] += len(experts)
            eap_hits[dti] += cache_eap.access(li, list(experts))
            eap_accesses[dti] += len(experts)
            oracle_hits[dti] += cache_oracle.access(li, list(experts))
            oracle_accesses[dti] += len(experts)

    return {
        "max_dti": max_dti,
        "bl_hit_rate": (bl_hits / bl_accesses.clip(min=1)).tolist(),
        "eap_hit_rate": (eap_hits / eap_accesses.clip(min=1)).tolist(),
        "oracle_hit_rate": (oracle_hits / oracle_accesses.clip(min=1)).tolist(),
    }


def simulate_all(
    groups: dict[str, list[dict]],
    num_layers: int,
    num_experts: int,
    cache_size: int,
    eap_weight: float,
    decay_steps: int,
    k_preload: int,
) -> dict:
    """Run simulation on all requests, aggregate."""
    all_bl = []
    all_eap = []
    all_oracle = []
    max_len = 0

    for rid, records in groups.items():
        result = simulate_one_request(
            records, num_layers, num_experts, cache_size,
            eap_weight, decay_steps, k_preload,
        )
        if not result:
            continue
        all_bl.append(result["bl_hit_rate"])
        all_eap.append(result["eap_hit_rate"])
        all_oracle.append(result["oracle_hit_rate"])
        max_len = max(max_len, result["max_dti"])

    # Pad to max_len and average.
    def _pad_mean(vecs: list[list[float]]) -> np.ndarray:
        mat = np.full((len(vecs), max_len), np.nan)
        for i, v in enumerate(vecs):
            mat[i, : len(v)] = v
        return np.nanmean(mat, axis=0)

    return {
        "num_requests": len(all_bl),
        "baseline_pd": _pad_mean(all_bl).tolist(),
        "eap": _pad_mean(all_eap).tolist(),
        "oracle": _pad_mean(all_oracle).tolist(),
    }


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def compute_t90(hit_rates: np.ndarray) -> float | None:
    """T90: first decode token at which rolling mean ≥ 0.9 × steady-state."""
    if len(hit_rates) < 5:
        return None
    steady = float(np.mean(hit_rates[-max(1, len(hit_rates) // 4):]))
    if steady <= 0:
        return None
    threshold = 0.9 * steady
    window = 5
    for i in range(len(hit_rates) - window + 1):
        if np.mean(hit_rates[i : i + window]) >= threshold:
            return float(i)
    return None


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_simulation(agg: dict, output_dir: Path) -> None:
    bl = np.array(agg["baseline_pd"])
    eap = np.array(agg["eap"])
    oracle = np.array(agg["oracle"])
    x = np.arange(len(bl))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # --- Full curves ---
    ax1.plot(x, bl, color="#F44336", linewidth=1.5, label="Baseline PD (cold)")
    ax1.plot(x, eap, color="#4CAF50", linewidth=1.5, label="EAP (ours)")
    ax1.plot(x, oracle, color="#2196F3", linewidth=1.2, linestyle="--",
             label="Oracle (upper bound)")
    ax1.set_xlabel("Decode Token Index")
    ax1.set_ylabel("Cache Hit Rate")
    ax1.set_title("Expert Cache Hit Rate: PD vs EAP vs Oracle")
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.02)

    # --- Zoom: first 32 tokens ---
    zoom = min(32, len(bl))
    ax2.plot(x[:zoom], bl[:zoom], color="#F44336", linewidth=2, marker="o", markersize=4,
             label="Baseline PD")
    ax2.plot(x[:zoom], eap[:zoom], color="#4CAF50", linewidth=2, marker="o", markersize=4,
             label="EAP")
    ax2.plot(x[:zoom], oracle[:zoom], color="#2196F3", linewidth=1.5, linestyle="--",
             label="Oracle")
    ax2.set_xlabel("Decode Token Index")
    ax2.set_ylabel("Cache Hit Rate")
    ax2.set_title("Cold-Start Window (first 32 tokens)")
    ax2.legend(loc="lower right")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 1.02)

    # --- Delta annotation ---
    delta_at_1 = (eap[0] - bl[0]) * 100
    delta_at_4 = np.mean(eap[:4] - bl[:4]) * 100
    delta_at_8 = np.mean(eap[:8] - bl[:8]) * 100
    ax2.text(0.98, 0.95,
             f"Δ at token 0:     {delta_at_1:+.1f}pp\n"
             f"Δ avg (0-3):     {delta_at_4:+.1f}pp\n"
             f"Δ avg (0-7):     {delta_at_8:+.1f}pp",
             transform=ax2.transAxes, fontsize=10, fontfamily="monospace",
             va="top", ha="right",
             bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.85))

    fig.tight_layout()
    fig.savefig(output_dir / "eap_cache_evolution.png", dpi=120)
    plt.close(fig)
    print(f"  ✓ eap_cache_evolution.png saved")


def plot_improvement_bars(agg: dict, output_dir: Path) -> None:
    bl = np.array(agg["baseline_pd"])
    eap = np.array(agg["eap"])

    windows = {"0-7": (0, 8), "0-15": (0, 16), "0-31": (0, 32), "All": (0, len(bl))}
    labels = list(windows.keys())
    bl_means = [float(np.mean(bl[s:e])) for s, e in windows.values()]
    eap_means = [float(np.mean(eap[s:e])) for s, e in windows.values()]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width/2, bl_means, width, color="#F44336", label="Baseline PD")
    bars2 = ax.bar(x + width/2, eap_means, width, color="#4CAF50", label="EAP (ours)")

    for bar, val in zip(bars1, bl_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", fontsize=9)
    for bar, val in zip(bars2, eap_means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean Cache Hit Rate")
    ax.set_title("EAP Improvement over Decode Windows")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    ax.set_ylim(0, 1.02)

    fig.tight_layout()
    fig.savefig(output_dir / "eap_improvement_bars.png", dpi=120)
    plt.close(fig)
    print(f"  ✓ eap_improvement_bars.png saved")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="EAP offline simulation against real trace")
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--num-experts", type=int, default=64)
    p.add_argument("--cache-size", type=int, default=16)
    p.add_argument("--eap-weight", type=float, default=0.5)
    p.add_argument("--decay-steps", type=int, default=32)
    p.add_argument("--k-preload", type=int, default=8)
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    groups = load_grouped(str(input_path))
    print(f"Loaded {len(groups)} requests from {input_path}")

    all_records = [r for recs in groups.values() for r in recs]
    num_layers = max(r["layer_idx"] for r in all_records) + 1
    print(f"  {num_layers} layers, {args.num_experts} experts")

    agg = simulate_all(
        groups, num_layers, args.num_experts,
        args.cache_size, args.eap_weight, args.decay_steps, args.k_preload,
    )

    bl = np.array(agg["baseline_pd"])
    eap = np.array(agg["eap"])
    oracle = np.array(agg["oracle"])

    t90_bl = compute_t90(bl)
    t90_eap = compute_t90(eap)
    t90_oracle = compute_t90(oracle)

    print()
    print("=" * 60)
    print("EAP SIMULATION RESULTS")
    print("=" * 60)
    print(f"  Requests:              {agg['num_requests']}")
    print(f"  Cache size:            {args.cache_size}")
    print(f"  EAP weight:            {args.eap_weight}")
    print(f"  Decay steps:           {args.decay_steps}")
    print(f"  K preload:             {args.k_preload}")
    print()
    print(f"  --- Hit Rates (mean) ---")
    print(f"  Baseline PD (all):     {float(np.mean(bl)):.4f}")
    print(f"  EAP (all):             {float(np.mean(eap)):.4f}")
    print(f"  Oracle (all):          {float(np.mean(oracle)):.4f}")
    print()
    print(f"  --- First-8 improvement ---")
    delta_8 = float(np.mean(eap[:8] - bl[:8]))
    print(f"  Δ EAP − Baseline:      {delta_8:+.4f} ({delta_8*100:+.1f}pp)")
    print()
    print(f"  --- T90 ---")
    print(f"  Baseline PD:  {t90_bl}")
    print(f"  EAP:          {t90_eap}")
    print(f"  Oracle:       {t90_oracle}")
    if t90_bl is not None and t90_eap is not None:
        print(f"  T90 reduction: {t90_bl - t90_eap:.1f} tokens")

    # EAP/Oracle ratio at key points
    for label, idx in [("token 0", 0), ("avg 0-7", slice(0, 8)),
                         ("avg 0-31", slice(0, 32))]:
        bl_v = float(np.mean(bl[idx])) if isinstance(idx, slice) else float(bl[idx])
        eap_v = float(np.mean(eap[idx])) if isinstance(idx, slice) else float(eap[idx])
        oracle_v = float(np.mean(oracle[idx])) if isinstance(idx, slice) else float(oracle[idx])
        eap_vs_oracle = (eap_v - bl_v) / (oracle_v - bl_v) * 100 if (oracle_v - bl_v) > 0 else 0
        print(f"  EAP effectiveness @ {label}: {eap_vs_oracle:.1f}% of optimal")

    # Save.
    results = {
        "config": {
            "cache_size": args.cache_size,
            "eap_weight": args.eap_weight,
            "decay_steps": args.decay_steps,
            "k_preload": args.k_preload,
            "num_experts": args.num_experts,
        },
        "num_requests": agg["num_requests"],
        "t90": {"baseline_pd": t90_bl, "eap": t90_eap, "oracle": t90_oracle},
        "mean_hit_rates": {
            "baseline_pd": float(np.mean(bl)),
            "eap": float(np.mean(eap)),
            "oracle": float(np.mean(oracle)),
        },
        "first_8_delta": float(delta_8),
        "hit_rate_curves": agg,
    }
    (output_dir / "eap_simulation.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False)
    )

    plot_simulation(agg, output_dir)
    plot_improvement_bars(agg, output_dir)
    print(f"\nAll outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
