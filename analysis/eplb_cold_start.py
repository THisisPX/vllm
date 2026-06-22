#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Analyze EPLB cold-start from ``physical_to_logical_map`` snapshots.

Computes arrangement distance across steps to measure how quickly
the expert-to-GPU mapping converges from the default (startup)
arrangement.

Metrics
-------
- **Per-step distance**: Fraction of physical slots whose mapping
  changed since the *previous* step.
- **Cumulative distance**: Fraction of slots that have changed at
  least once since step 0.
- **Convergence step**: First step at which per-step distance drops
  below *epsilon* (default 5%).

Usage
-----
    python analysis/eplb_cold_start.py \\
        --input traces/eplb_run/snapshots.npz \\
        --output analysis/results/eplb/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")


# ---------------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------------


def load_snapshots(npz_path: Path) -> dict[int, np.ndarray]:
    """Load EPLB snapshot archive.

    Returns:
        dict mapping step number → ``[n_layers, n_physicals]`` int32 array.
    """
    data = np.load(npz_path)
    snapshots: dict[int, np.ndarray] = {}
    for key in sorted(data.files, key=lambda k: int(k.split("_")[1])):
        step = int(key.split("_")[1])
        snapshots[step] = data[key].astype(np.int32)
    return snapshots


def compute_distances(
    snapshots: dict[int, np.ndarray],
) -> tuple[list[int], list[float], list[float]]:
    """Compute per-step and cumulative arrangement distances.

    Returns:
        (steps, per_step_distance, cumulative_distance)
    """
    sorted_steps = sorted(snapshots.keys())
    maps = [snapshots[s] for s in sorted_steps]

    per_step: list[float] = []
    cumulative: list[float] = []

    # Reference: the initial map at step 0.
    initial = maps[0]
    changed_ever = np.zeros(initial.shape, dtype=bool)

    for i, current in enumerate(maps):
        if i == 0:
            per_step.append(0.0)
            cumulative.append(0.0)
            continue

        prev = maps[i - 1]
        changed_now = (current != prev)
        per_step.append(float(changed_now.mean()))

        changed_ever = changed_ever | changed_now
        cumulative.append(float(changed_ever.mean()))

    return sorted_steps, per_step, cumulative


def find_convergence_step(
    steps: list[int],
    distances: list[float],
    epsilon: float = 0.05,
) -> int | None:
    """First step where per-step distance < epsilon."""
    for s, d in zip(steps, distances):
        if d < epsilon:
            return int(s)
    return None


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_distances(
    steps: list[int],
    per_step: list[float],
    cumulative: list[float],
    convergence_step: int | None,
    output_dir: Path,
) -> None:
    """Generate arrangement distance plots."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # --- Per-step distance (churn) ---
    ax1.plot(steps, [d * 100 for d in per_step],
             color="#F44336", linewidth=1.2, marker=".", markersize=4)
    if convergence_step is not None:
        ax1.axvline(convergence_step, color="gray", linestyle="--", alpha=0.7,
                    label=f"Convergence (step {convergence_step})")
    ax1.set_xlabel("EPLB Step")
    ax1.set_ylabel("Arrangement Churn (%)")
    ax1.set_title("Per-Step Arrangement Distance\n(Δ from previous step)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # --- Cumulative distance ---
    ax2.plot(steps, [d * 100 for d in cumulative],
             color="#2196F3", linewidth=1.5)
    ax2.set_xlabel("EPLB Step")
    ax2.set_ylabel("Cumulative Change (%)")
    ax2.set_title("Cumulative Arrangement Distance\n(slots changed at least once)")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_dir / "arrangement_distance.png", dpi=120)
    plt.close(fig)
    print(f"  ✓ arrangement_distance.png saved")


def plot_layer_heatmap_eplb(
    snapshots: dict[int, np.ndarray],
    steps: list[int],
    output_dir: Path,
) -> None:
    """Heatmap: layer × step showing fraction of slots changed per layer."""
    sorted_steps = sorted(snapshots.keys())
    maps = [snapshots[s] for s in sorted_steps]
    n_layers = maps[0].shape[0]

    # Build matrix: [n_layers, n_steps-1]
    layer_churn = np.zeros((n_layers, len(maps) - 1))
    for t in range(1, len(maps)):
        churn = (maps[t] != maps[t - 1]).astype(np.float32)
        layer_churn[:, t - 1] = churn.mean(axis=1)

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(
        layer_churn * 100, aspect="auto", cmap="YlOrRd",
        origin="lower", vmin=0,
    )
    ax.set_xlabel("EPLB Step Index")
    ax.set_ylabel("MoE Layer Index")
    ax.set_title("Per-Layer Arrangement Churn (%)")
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Churn (%)")

    fig.tight_layout()
    fig.savefig(output_dir / "layer_churn_heatmap.png", dpi=120)
    plt.close(fig)
    print(f"  ✓ layer_churn_heatmap.png saved ({n_layers} layers)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Analyze EPLB cold-start from PTL map snapshots"
    )
    p.add_argument("--input", required=True, help="Path to snapshots.npz")
    p.add_argument("--output", required=True, help="Output directory")
    p.add_argument("--epsilon", type=float, default=0.05,
                   help="Convergence threshold (default: 0.05 = 5%%)")
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Snapshots file not found: {input_path}")

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load.
    snapshots = load_snapshots(input_path)
    print(f"Loaded {len(snapshots)} snapshots from {input_path}")
    first = snapshots[min(snapshots.keys())]
    print(f"  Shape: {first.shape[0]} layers × {first.shape[1]} physical experts")
    print(f"  Steps: {sorted(snapshots.keys())[:10]}...")

    # Analyze.
    steps, per_step, cumulative = compute_distances(snapshots)
    conv = find_convergence_step(steps, per_step, args.epsilon)

    print()
    print("=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"Total steps:             {len(steps)}")
    if conv is not None:
        print(f"Convergence step (ε={args.epsilon}): {conv}  ✓")
    else:
        print(f"Convergence step (ε={args.epsilon}): not reached  ✗")

    first_step_map = snapshots[steps[0]]
    if len(steps) > 1:
        total_slots = first_step_map.size
        changed_between_first_two = (snapshots[steps[1]] != first_step_map).sum()
        print(f"Slots changed in first rebalance: {changed_between_first_two} "
              f"({changed_between_first_two / total_slots * 100:.1f}%)")
        print(f"Note: this is the TRUE cold-start cost — the fraction of "
              f"expert slots that must be physically migrated on first rebalance.")

    # Save results JSON.
    result = {
        "num_snapshots": len(snapshots),
        "num_layers": int(first.shape[0]),
        "num_physicals": int(first.shape[1]),
        "convergence_step": conv,
        "epsilon": args.epsilon,
        "per_step_distance": [round(d, 6) for d in per_step],
        "cumulative_distance": [round(d, 6) for d in cumulative],
        "steps": steps,
    }
    result_path = output_dir / "eplb_cold_start.json"
    result_path.write_text(json.dumps(result, indent=2))
    print(f"\nResults written to {result_path}")

    # Plot.
    plot_distances(steps, per_step, cumulative, conv, output_dir)
    plot_layer_heatmap_eplb(snapshots, steps, output_dir)

    print(f"\nAll plots saved to {output_dir}")


if __name__ == "__main__":
    main()
