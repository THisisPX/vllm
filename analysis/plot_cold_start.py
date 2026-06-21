# SPDX-License-Identifier: Apache-2.0
"""Plotting script for cold-start simulation results.

Generates three figures from a ``results.json`` file produced by
``cold_start_simulator.py``:

1. **Cache hit-rate evolution** — x = decode token index,
   y = mean hit rate (Coupled vs PD) with ±1σ bands.
2. **T90 bar chart** — side-by-side bars for Coupled and PD
   T90 values.
3. **Per-layer heatmap** — x = decode token index,
   y = layer index, color = hit-rate delta (PD − Coupled).

Usage::

    python analysis/plot_cold_start.py \\
        --input results/run_001/results.json \\
        --output plots/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")  # non-interactive backend


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_results(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _extract_hit_rate_matrix(
    per_request: list[dict], key: str
) -> list[list[float]]:
    """Extract per-request hit-rate vectors and pad to max length."""
    vectors = [r[key] for r in per_request]
    return vectors


# ---------------------------------------------------------------------------
# Plot 1: Hit-rate evolution with ±1σ bands
# ---------------------------------------------------------------------------


def plot_hit_rate_evolution(
    results: dict[str, Any], output_dir: Path
) -> None:
    per_req = results["per_request"]
    coupled_vecs = _extract_hit_rate_matrix(per_req["coupled"], "cache_hit_rates")
    pd_vecs = _extract_hit_rate_matrix(per_req["pd"], "cache_hit_rates")

    if not coupled_vecs and not pd_vecs:
        print("  No hit-rate data to plot — skipping hit-rate evolution")
        return

    # Pad to max length.
    max_c = max((len(v) for v in coupled_vecs)) if coupled_vecs else 0
    max_pd = max((len(v) for v in pd_vecs)) if pd_vecs else 0
    max_len = max(max_c, max_pd)

    def _pad_mean_std(vecs: list[list[float]]) -> tuple[np.ndarray, np.ndarray]:
        mat = np.full((len(vecs), max_len), np.nan)
        for i, v in enumerate(vecs):
            mat[i, : len(v)] = v
        mean = np.nanmean(mat, axis=0)
        std = np.nanstd(mat, axis=0)
        return mean, std

    c_mean, c_std = _pad_mean_std(coupled_vecs)
    pd_mean, pd_std = _pad_mean_std(pd_vecs)
    x = np.arange(max_len)

    fig, ax = plt.subplots(figsize=(12, 5))

    ax.plot(x, c_mean, color="#2196F3", linewidth=1.5, label="Coupled")
    ax.fill_between(x, c_mean - c_std, c_mean + c_std,
                    color="#2196F3", alpha=0.15)
    ax.plot(x, pd_mean, color="#F44336", linewidth=1.5, label="PD")
    ax.fill_between(x, pd_mean - pd_std, pd_mean + pd_std,
                    color="#F44336", alpha=0.15)

    ax.set_xlabel("Decode Token Index")
    ax.set_ylabel("Cache Hit Rate")
    ax.set_title("Expert Cache Hit-Rate Evolution (mean ± 1σ)")
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "hit_rate_evolution.png", dpi=120)
    plt.close(fig)
    print(f"  ✓ hit_rate_evolution.png ({max_len} tokens)")


# ---------------------------------------------------------------------------
# Plot 2: T90 bar chart
# ---------------------------------------------------------------------------


def plot_t90(results: dict[str, Any], output_dir: Path) -> None:
    """Side-by-side bar chart of T90 values."""
    summary = results["summary"]
    t90_c = summary.get("mean_t90_coupled")
    t90_pd = summary.get("mean_t90_pd")

    if t90_c is None and t90_pd is None:
        print("  No T90 values — skipping T90 chart")
        return

    fig, ax = plt.subplots(figsize=(6, 5))

    labels = ["Coupled", "PD"]
    values = [t90_c or 0, t90_pd or 0]
    colors = ["#2196F3", "#F44336"]

    bars = ax.bar(labels, values, color=colors, width=0.45, edgecolor="white")

    for bar, val in zip(bars, [t90_c, t90_pd]):
        if val is not None:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                    f"{val:.1f}", ha="center", va="bottom", fontweight="bold")
        else:
            ax.text(bar.get_x() + bar.get_width() / 2, 5,
                    "N/A", ha="center", va="bottom", fontweight="bold",
                    color="gray")

    ax.set_ylabel("T90 (decode tokens)")
    ax.set_title("Time-to-90% Cache Hit Rate")
    ax.grid(True, alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(output_dir / "t90_comparison.png", dpi=120)
    plt.close(fig)
    print(f"  ✓ t90_comparison.png (Coupled={t90_c}, PD={t90_pd})")


# ---------------------------------------------------------------------------
# Plot 3: Per-layer heatmap (PD − Coupled delta)
# ---------------------------------------------------------------------------


def plot_layer_heatmap(results: dict[str, Any], output_dir: Path) -> None:
    """Heatmap showing which layers suffer most from cold-start."""
    per_req = results["per_request"]
    config = results["config"]
    num_layers = config.get("num_layers", 1)

    # Aggregate per-layer hit rates across requests.
    # per_layer[layer_idx][decode_token_idx]
    coupled_layers: dict[int, list[float]] = {}
    pd_layers: dict[int, list[float]] = {}

    for r in per_req["coupled"]:
        for li, vec in enumerate(r.get("per_layer_hit_rates", [])):
            coupled_layers.setdefault(li, []).extend(vec)
    for r in per_req["pd"]:
        for li, vec in enumerate(r.get("per_layer_hit_rates", [])):
            pd_layers.setdefault(li, []).extend(vec)

    if not coupled_layers and not pd_layers:
        print("  No per-layer data — skipping heatmap")
        return

    # Build a delta matrix: layers × min_decode_tokens.
    all_layers = sorted(set(coupled_layers.keys()) | set(pd_layers.keys()))
    coupled_lens = [len(coupled_layers.get(l, [])) for l in all_layers]
    pd_lens = [len(pd_layers.get(l, [])) for l in all_layers]
    min_len = min(
        min(coupled_lens) if coupled_lens else 0,
        min(pd_lens) if pd_lens else 0,
    )
    if min_len > 256:
        min_len = 256  # cap for readability

    if min_len == 0:
        print("  Insufficient per-layer data — skipping heatmap")
        return

    delta = np.zeros((len(all_layers), min_len))
    for row, layer in enumerate(all_layers):
        for col in range(min_len):
            c_val = coupled_layers.get(layer, [0])[col] if col < len(coupled_layers.get(layer, [])) else 0
            pd_val = pd_layers.get(layer, [0])[col] if col < len(pd_layers.get(layer, [])) else 0
            delta[row, col] = pd_val - c_val

    vmax = max(abs(delta.min()), abs(delta.max()), 1e-6)

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(
        delta, aspect="auto", cmap="RdBu_r",
        vmin=-vmax, vmax=vmax, origin="lower",
    )

    ax.set_xlabel("Decode Token Index")
    ax.set_ylabel("Layer Index")
    ax.set_title("Hit-Rate Delta (PD − Coupled) per Layer")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Δ Hit Rate")

    fig.tight_layout()
    fig.savefig(output_dir / "layer_heatmap.png", dpi=120)
    plt.close(fig)
    print(f"  ✓ layer_heatmap.png ({len(all_layers)} layers × {min_len} tokens)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Plot cold-start simulation results"
    )
    p.add_argument("--input", required=True, help="Path to results.json")
    p.add_argument("--output", required=True, help="Output directory for plots")
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Results file not found: {input_path}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = _load_results(input_path)

    config = results.get("config", {})
    summary = results.get("summary", {})
    print(f"Source: {config.get('source', '?')}")
    print(f"Requests: {summary.get('num_requests', 0)}")
    print(f"Cache capacity: {config.get('cache_capacity', '?')}")
    print()

    plot_hit_rate_evolution(results, output_dir)
    plot_t90(results, output_dir)
    plot_layer_heatmap(results, output_dir)

    print(f"\nAll plots saved to {output_dir}")


if __name__ == "__main__":
    main()
