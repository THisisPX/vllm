# SPDX-License-Identifier: Apache-2.0
"""Per-layer LRU expert cache for cold-start simulation.

Models the expert scheduler state as a fixed-capacity LRU cache
per MoE layer.  Each ``access()`` returns the number of cache hits
(experts already resident) and updates recency.

Assumptions (documented in output metadata):
- Cache entries are expert IDs (integers), no weight/bias.
- LRU eviction models a simplified EPLB without load-balancing
  pressure.  Real EPLB would also consider expert popularity
  across the batch — this simulation is strictly sequential
  (token by token, layer by layer).
- Capacity is uniform across layers.  In practice EPLB may
  allocate different capacities per layer, but for the kill-or-
  prove experiment a uniform capacity is a reasonable first
  approximation.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass


@dataclass
class CacheStats:
    """Aggregated statistics for one layer after a simulation run."""

    layer_idx: int
    total_accesses: int = 0
    total_hits: int = 0
    total_misses: int = 0

    @property
    def hit_rate(self) -> float:
        if self.total_accesses == 0:
            return 0.0
        return self.total_hits / self.total_accesses


class ExpertCache:
    """Per-layer LRU cache of expert IDs.

    Parameters
    ----------
    capacity:
        Maximum number of experts cached per layer.
    num_layers:
        Number of MoE layers.
    """

    def __init__(self, capacity: int, num_layers: int) -> None:
        if capacity <= 0:
            raise ValueError(f"capacity must be > 0, got {capacity}")
        if num_layers <= 0:
            raise ValueError(f"num_layers must be > 0, got {num_layers}")

        self.capacity = capacity
        self.num_layers = num_layers

        # Each layer maintains an OrderedDict where keys are expert IDs
        # and order reflects recency (most-recently-used at the end).
        self._layers: list[OrderedDict[int, None]] = [
            OrderedDict() for _ in range(num_layers)
        ]

        # Per-layer statistics.
        self._stats: list[CacheStats] = [
            CacheStats(layer_idx=i) for i in range(num_layers)
        ]

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def access(self, layer_idx: int, expert_ids: list[int]) -> int:
        """Access a set of experts for one (token, layer).

        Returns the number of cache hits (experts already resident).
        LRU order is updated for all accessed experts — hits move
        to MRU position; misses are inserted and may trigger eviction.
        """
        cache = self._layers[layer_idx]
        stats = self._stats[layer_idx]
        hits = 0

        for eid in expert_ids:
            stats.total_accesses += 1
            if eid in cache:
                hits += 1
                stats.total_hits += 1
                # Move to MRU position.
                cache.move_to_end(eid)
            else:
                stats.total_misses += 1
                # Insert at MRU position.
                cache[eid] = None
                # Evict LRU if over capacity.
                if len(cache) > self.capacity:
                    cache.popitem(last=False)

        return hits

    def hit_rate(self, layer_idx: int | None = None) -> float:
        """Aggregate hit rate across layers (or for a single layer)."""
        if layer_idx is not None:
            return self._stats[layer_idx].hit_rate
        total_accesses = sum(s.total_accesses for s in self._stats)
        if total_accesses == 0:
            return 0.0
        return sum(s.total_hits for s in self._stats) / total_accesses

    def per_layer_stats(self) -> list[CacheStats]:
        """Return a copy of per-layer statistics."""
        return list(self._stats)

    def resident_experts(self, layer_idx: int) -> set[int]:
        """Return the set of expert IDs currently cached in a layer."""
        return set(self._layers[layer_idx].keys())

    def reset(self) -> None:
        """Clear all caches and statistics (used for PD cold-start)."""
        for cache in self._layers:
            cache.clear()
        for stats in self._stats:
            stats.total_accesses = 0
            stats.total_hits = 0
            stats.total_misses = 0

    # ------------------------------------------------------------------
    # Serialisation helpers
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialise current state (for checkpointing / debugging)."""
        return {
            "capacity": self.capacity,
            "num_layers": self.num_layers,
            "layers": [
                {
                    "layer_idx": i,
                    "resident_experts": sorted(self._layers[i].keys()),
                    "stats": {
                        "total_accesses": s.total_accesses,
                        "total_hits": s.total_hits,
                        "total_misses": s.total_misses,
                        "hit_rate": round(s.hit_rate, 6),
                    },
                }
                for i, s in enumerate(self._stats)
            ],
        }
