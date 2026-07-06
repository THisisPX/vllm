# SPDX-License-Identifier: Apache-2.0
"""EapProfile — lightweight L×E expert frequency matrix.

A serializable per-request histogram of expert activations across
MoE layers.  Size: ``num_layers × num_experts × 4`` bytes.

For DeepSeek-V2-Lite (26 layers × 64 experts): ~6.5 KB.
For Qwen3-30B-A3B (48 layers × 128 experts): ~24 KB.

Serialisation uses raw bytes (little-endian uint32) for minimal
overhead during KV Cache transfer.
"""

from __future__ import annotations

import struct
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

# Magic header for format validation.
_EAP_MAGIC = b"EAP\x01"  # version 1


class EapProfile:
    """Per-request expert activation frequency histogram.

    Parameters
    ----------
    num_layers:
        Number of MoE layers.
    num_experts:
        Number of routed experts per layer.
    """

    __slots__ = ("_counts", "num_layers", "num_experts")

    def __init__(self, num_layers: int, num_experts: int) -> None:
        self.num_layers = num_layers
        self.num_experts = num_experts
        self._counts = np.zeros((num_layers, num_experts), dtype=np.uint32)

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(self, layer_idx: int, expert_ids: Sequence[int]) -> None:
        """Increment counters for one (token, layer) expert selection.

        Called during prefill for every token × layer.
        """
        if layer_idx < 0 or layer_idx >= self.num_layers:
            return
        arr = self._counts[layer_idx]
        for eid in expert_ids:
            if 0 <= eid < self.num_experts:
                arr[eid] += 1

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_counts(self, layer_idx: int) -> np.ndarray:
        """Return raw uint32 counts for one layer."""
        return self._counts[layer_idx]

    def get_probabilities(self, layer_idx: int) -> np.ndarray:
        """Return normalised probability distribution for one layer."""
        counts = self._counts[layer_idx].astype(np.float64)
        total = counts.sum()
        if total > 0:
            return counts / total
        return np.full(self.num_experts, 1.0 / self.num_experts)

    def top_k(self, layer_idx: int, k: int) -> list[int]:
        """Return top-K expert indices by count (descending)."""
        counts = self._counts[layer_idx]
        order = np.argsort(-counts)[:k]  # type: ignore[call-overload]
        return [int(e) for e in order if counts[e] > 0]

    @property
    def total_activations(self) -> int:
        """Total number of expert activations recorded."""
        return int(self._counts.sum())

    # ------------------------------------------------------------------
    # Serialisation (binary, ~6.5 KB for DSv2-Lite)
    # ------------------------------------------------------------------

    def to_bytes(self) -> bytes:
        """Encode to compact binary format.

        Layout: magic(4) + num_layers(2) + num_experts(2) + data(N×4)
        """
        header = _EAP_MAGIC
        header += struct.pack("<HH", self.num_layers, self.num_experts)
        return header + self._counts.tobytes()

    @classmethod
    def from_bytes(cls, data: bytes) -> "EapProfile":
        """Decode from binary format produced by :meth:`to_bytes`."""
        if not data.startswith(_EAP_MAGIC):
            raise ValueError("Invalid EAP magic bytes")
        num_layers, num_experts = struct.unpack_from("<HH", data, 4)
        expected_size = 8 + num_layers * num_experts * 4
        if len(data) < expected_size:
            raise ValueError(
                f"Truncated EAP data: expected {expected_size} bytes, "
                f"got {len(data)}"
            )
        profile = cls(num_layers, num_experts)
        profile._counts = np.frombuffer(
            data, dtype=np.uint32, count=num_layers * num_experts, offset=8
        ).reshape(num_layers, num_experts).copy()
        return profile

    def to_dict(self) -> dict:
        """Convert to JSON-serialisable dict (for debugging only)."""
        return {
            "num_layers": self.num_layers,
            "num_experts": self.num_experts,
            "total_activations": self.total_activations,
            "top_k_per_layer": {
                str(li): {
                    "top8": self.top_k(li, 8),
                    "top16": self.top_k(li, 16),
                }
                for li in range(self.num_layers)
            },
        }

    @property
    def size_bytes(self) -> int:
        """Serialised size in bytes."""
        return 8 + self.num_layers * self.num_experts * 4

    def __repr__(self) -> str:
        return (
            f"EapProfile(layers={self.num_layers}, experts={self.num_experts}, "
            f"size={self.size_bytes}B, activations={self.total_activations})"
        )
