#!/usr/bin/env python3
"""Recall analysis: can Prefill experts predict Early Decode experts?

Core metric:
    Ep = aggregate expert activations across all prefill tokens
    Ed(K) = expert activations over first K decode tokens
    Recall@N = |TopN(Ep) ∩ TopN(Ed(K))| / |TopN(Ed(K))|

If Recall@N is high, Prefill-derived EAP can meaningfully warm
the decode-side expert cache, reducing cold-start.

Usage:
    python analysis/recall_analysis.py \
        --input traces/phase1_random/traces.jsonl \
        --output analysis/results/recall/
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from itertools import islice
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")


# ---------------------------------------------------------------------------
# Data loading (same as Phase 1 data_loader)
# ---------------------------------------------------------------------------


def load_traces(path: str) -> list[dict]:
    """Load raw JSONL lines."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def group_by_request(records: list[dict]) -> dict[str, list[dict]]:
    """Group records by request_id, skip warmup."""
    groups = defaultdict(list)
    for r in records:
        rid = r["request_id"]
        if rid.startswith("_warmup"):
            continue
        groups[rid].append(r)
    return dict(groups)


# ---------------------------------------------------------------------------
# Core Recall computation
# ---------------------------------------------------------------------------


def compute_recall_for_request(
    records: list[dict],
    num_layers: int,
    N_values: tuple[int, ...] = (4, 8, 16),
    K_values: tuple[int, ...] = (8, 16, 32, 64, 128),
) -> dict[str, dict[int, dict[int, float]]]:
    """Compute Recall@N(K) for one request.

    Returns:
        {layer_idx: {N: {K: recall}}}
    """
    # Separate prefill and decode.
    prefill_by_layer: dict[int, Counter] = defaultdict(Counter)
    decode_by_layer: dict[int, list[tuple[int, tuple[int, ...]]]] = defaultdict(list)
    # decode entries: list of (decode_token_idx, experts_tuple)

    for r in records:
        li = r["layer_idx"]
        experts = tuple(r["experts"])
        if r["phase"] == "prefill":
            for e in experts:
                prefill_by_layer[li][e] += 1
        else:
            dti = r.get("decode_token_idx", -1)
            if dti >= 0:
                decode_by_layer[li].append((dti, experts))

    if not prefill_by_layer or not decode_by_layer:
        return {}

    results: dict[str, dict[int, dict[int, float]]] = {}

    for li in range(num_layers):
        prefill_counts = prefill_by_layer.get(li)
        decode_list = decode_by_layer.get(li)
        if not prefill_counts or not decode_list:
            continue

        # Sort decode by decode_token_idx.
        decode_list.sort(key=lambda x: x[0])

        layer_results: dict[int, dict[int, float]] = {}

        for N in N_values:
            # Top-N experts by prefill frequency.
            topN_prefill = {e for e, _ in prefill_counts.most_common(N)}

            layer_results[N] = {}
            for K in K_values:
                # Top-N experts in first K decode tokens.
                first_k = islice(decode_list, K)
                decode_counts: Counter = Counter()
                for _, experts in first_k:
                    for e in experts:
                        decode_counts[e] += 1
                topN_decode = {e for e, _ in decode_counts.most_common(N)}

                if not topN_decode:
                    layer_results[N][K] = 1.0  # edge case: empty decode
                else:
                    intersection = topN_prefill & topN_decode
                    layer_results[N][K] = len(intersection) / len(topN_decode)

        results[str(li)] = layer_results

    return results


def compute_aggregate_recall(
    all_requests: list[dict],
    num_layers: int,
    N_values: tuple[int, ...] = (4, 8, 16),
    K_values: tuple[int, ...] = (8, 16, 32, 64, 128),
) -> tuple[dict, dict, list]:
    """Run Recall@N across all requests, return aggregate stats.

    Returns:
        (means, stds, per_request_results)
        means[layer][N][K] = mean recall
        stds[layer][N][K] = std recall
    """
    all_results = []

    groups = group_by_request(all_requests)
    for rid, records in groups.items():
        result = compute_recall_for_request(records, num_layers, N_values, K_values)
        if result:
            all_results.append((rid, result))

    # Aggregate: mean ± std per (layer, N, K).
    means: dict = {}
    stds: dict = {}
    for layer_key in all_results[0][1] if all_results else {}:
        means[layer_key] = {}
        stds[layer_key] = {}
        for N in N_values:
            N_str = str(N)
            means[layer_key][N_str] = {}
            stds[layer_key][N_str] = {}
            for K in K_values:
                K_str = str(K)
                values = []
                for _, res in all_results:
                    v = res.get(layer_key, {}).get(N, {}).get(K)
                    if v is not None:
                        values.append(v)
                means[layer_key][N_str][K_str] = float(np.mean(values)) if values else 0.0
                stds[layer_key][N_str][K_str] = float(np.std(values)) if values else 0.0

    return means, stds, all_results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_recall_curves(
    means: dict,
    stds: dict,
    N_values: tuple[int, ...],
    K_values: tuple[int, ...],
    output_dir: Path,
) -> None:
    """Plot Recall@N vs K curves, one line per N, averaged across layers."""
    K_list = list(K_values)

    # Average across layers for each (N, K).
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # All-layers average
    for n_val in N_values:
        n_str = str(n_val)
        y = []
        y_err = []
        for k_val in K_values:
            k_str = str(k_val)
            vals = [means[li][n_str][k_str] for li in means]
            y.append(float(np.mean(vals)))
            y_err.append(float(np.std(vals)))
        y, y_err = np.array(y), np.array(y_err)
        ax1.errorbar(K_list, y, yerr=y_err, marker="o", markersize=5,
                      capsize=3, label=f"N={n_val}", linewidth=1.5)

    ax1.set_xlabel("K (first K decode tokens)")
    ax1.set_ylabel("Recall@N")
    ax1.set_title("Prefill→Decode Expert Recall (layer-average)")
    ax1.set_ylim(0, 1.02)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Per-layer heatmap (N=8 fixed, best tradeoff)
    fixed_n = 8
    layers = sorted(means.keys(), key=int)
    recall_matrix = np.zeros((len(layers), len(K_list)))
    for i, li in enumerate(layers):
        for j, k_val in enumerate(K_list):
            recall_matrix[i, j] = means[li].get(str(fixed_n), {}).get(str(k_val), 0.0)

    im = ax2.imshow(recall_matrix, aspect="auto", cmap="RdYlGn",
                     vmin=0, vmax=1, origin="lower")
    ax2.set_xticks(range(len(K_list)))
    ax2.set_xticklabels([str(k) for k in K_list])
    ax2.set_xlabel("K (decode tokens)")
    ax2.set_ylabel("MoE Layer Index")
    ax2.set_title(f"Recall@{fixed_n} per Layer")
    cbar = fig.colorbar(im, ax=ax2)
    cbar.set_label("Recall")

    fig.tight_layout()
    fig.savefig(output_dir / "recall_curves.png", dpi=120)
    plt.close(fig)
    print(f"  ✓ recall_curves.png saved")


def plot_recall_summary(
    means: dict,
    N_values: tuple[int, ...],
    K_values: tuple[int, ...],
    output_dir: Path,
) -> None:
    """Summary: Recall(N=8) at K=8, 16, 32, 64, 128 as bar chart."""
    N = 8
    layers = sorted(means.keys(), key=int)
    n_str = str(N)

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(layers))
    width = 0.16

    for i, K in enumerate(K_values):
        k_str = str(K)
        y = [means[li][n_str][k_str] for li in layers]
        ax.bar(x + i * width, y, width, label=f"K={K}")

    ax.set_xlabel("MoE Layer Index")
    ax.set_ylabel(f"Recall@{N}")
    ax.set_title(f"Per-Layer Recall@{N} at different decode windows K")
    ax.set_xticks(x + width * 2)
    ax.set_xticklabels([str(int(li)) for li in layers], rotation=90, fontsize=8)
    ax.legend()
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(output_dir / "recall_per_layer.png", dpi=120)
    plt.close(fig)
    print(f"  ✓ recall_per_layer.png saved")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


DEFAULT_N = (4, 8, 16)
DEFAULT_K = (8, 16, 32, 64, 128)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Recall analysis: Prefill→Decode expert overlap"
    )
    p.add_argument("--input", required=True, help="Path to traces.jsonl")
    p.add_argument("--output", required=True, help="Output directory")
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Trace file not found: {input_path}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load and find num_layers.
    records = load_traces(str(input_path))
    print(f"Loaded {len(records)} records from {input_path}")

    num_layers = max(r["layer_idx"] for r in records) + 1
    print(f"  {num_layers} MoE layers detected")

    groups = group_by_request(records)
    real_requests = {k: v for k, v in groups.items() if len(v) > 0}
    print(f"  {len(real_requests)} real requests")

    # Compute Recall.
    means, stds, per_request = compute_aggregate_recall(
        records, num_layers, DEFAULT_N, DEFAULT_K
    )

    if not means:
        print("No recall data computed — not enough prefill+decode pairs.")
        return

    # Print summary.
    print()
    print("=" * 70)
    print("RECALL@N SUMMARY (layer-average)")
    print("=" * 70)
    header = f"{'N\\K':>6}"
    for K in DEFAULT_K:
        header += f"  K={K:>4}"
    print(header)
    print("-" * 70)
    for N in DEFAULT_N:
        n_str = str(N)
        row = f"{'N=' + n_str:>6}"
        for K in DEFAULT_K:
            k_str = str(K)
            vals = [means[li][n_str][k_str] for li in means]
            row += f"  {np.mean(vals):.3f}"
        print(row)
    print("-" * 70)
    print()

    # Highlight the key finding.
    n8_k8 = np.mean([means[li]["8"]["8"] for li in means])
    n8_k32 = np.mean([means[li]["8"]["32"] for li in means])
    n8_k128 = np.mean([means[li]["8"]["128"] for li in means])
    print(f"Key metrics (N=8):")
    print(f"  Recall@8 (K=8):   {n8_k8:.3f}  ← early decode")
    print(f"  Recall@8 (K=32):  {n8_k32:.3f}  ← mid decode")
    print(f"  Recall@8 (K=128): {n8_k128:.3f}  ← long decode")
    print()

    # Save results.
    summary = {
        "num_requests": len(real_requests),
        "num_layers": num_layers,
        "N_values": list(DEFAULT_N),
        "K_values": list(DEFAULT_K),
        "means": means,
        "stds": stds,
        "key_metrics": {
            "recall_N8_K8": n8_k8,
            "recall_N8_K32": n8_k32,
            "recall_N8_K128": n8_k128,
        },
    }
    (output_dir / "recall_results.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False)
    )
    print(f"Results saved to {output_dir / 'recall_results.json'}")

    # Plot.
    plot_recall_curves(means, stds, DEFAULT_N, DEFAULT_K, output_dir)
    plot_recall_summary(means, DEFAULT_N, DEFAULT_K, output_dir)
    print(f"\nAll plots saved to {output_dir}")


if __name__ == "__main__":
    main()
