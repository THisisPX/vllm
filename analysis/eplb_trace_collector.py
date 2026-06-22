# SPDX-License-Identifier: Apache-2.0
"""EPLB trace collector — records physical_to_logical_map snapshots.

Hooks into EplbState to capture expert arrangement evolution during
real EP+EPLB inference.  Each snapshot is a (step, layers × physicals)
numpy array saved as a compressed .npz file.

Usage:
    export VLLM_EPLB_TRACE_DIR="./traces/eplb_run_001"
    # ... start vLLM with EP > 1 and --enable-eplb ...
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class EplbTraceCollector:
    """Collects ``physical_to_logical_map`` snapshots during EPLB inference.

    One snapshot is recorded per engine step (whether or not a rebalance
    occurred — the map may be unchanged).

    Output files (written at ``close()``):
        ``snapshots.npz``
            Keys: ``step_N`` → ``int32[n_layers, n_physicals]`` array
        ``metadata.json``
            Config and step metadata.
    """

    def __init__(self, output_dir: str) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self._snapshots: dict[int, np.ndarray] = {}
        self._step_timestamps: list[tuple[int, float]] = []
        self._num_layers: int | None = None
        self._num_physicals: int | None = None

    # ------------------------------------------------------------------
    # Hook called from EplbState
    # ------------------------------------------------------------------

    def record_map(
        self,
        step: int,
        physical_to_logical_map: "torch.Tensor",  # noqa: F821
    ) -> None:
        """Record one PTL map snapshot.

        Called from ``EplbState`` after every rearrange (or at every
        step boundary).  The tensor is copied to CPU immediately.

        Args:
            step: Global EPLB step counter.
            physical_to_logical_map: ``[num_layers, num_physicals]``
                int tensor on GPU.
        """
        import torch

        # Copy to CPU once.
        cpu_map = physical_to_logical_map.cpu().to(torch.int32).numpy()

        if self._num_layers is None:
            self._num_layers, self._num_physicals = cpu_map.shape

        self._snapshots[step] = cpu_map.copy()
        self._step_timestamps.append((step, time.time()))

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Write all snapshots and metadata to disk."""
        if not self._snapshots:
            logger.warning("EPLB trace: no snapshots recorded")
            return

        # Save as compressed numpy archive.
        npz_path = self.output_dir / "snapshots.npz"
        np.savez_compressed(
            npz_path,
            **{f"step_{s}": m for s, m in self._snapshots.items()},
        )
        logger.info(
            "EPLB snapshots written: %d steps → %s (%.1f KB)",
            len(self._snapshots),
            npz_path,
            npz_path.stat().st_size / 1024,
        )

        # Write metadata.
        import json

        meta = {
            "num_layers": self._num_layers,
            "num_physicals": self._num_physicals,
            "num_snapshots": len(self._snapshots),
            "steps": sorted(self._snapshots.keys()),
        }
        meta_path = self.output_dir / "metadata.json"
        meta_path.write_text(json.dumps(meta, indent=2))
        logger.info("EPLB metadata written to %s", meta_path)

    @property
    def num_snapshots(self) -> int:
        return len(self._snapshots)
