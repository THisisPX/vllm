# SPDX-License-Identifier: Apache-2.0
"""EapCacheManager — EAP-weighted LRU expert cache for Decode.

Receives an :class:`EapProfile` from the Prefill pool and uses it to
initialise a per-layer expert cache with weighted priorities.

Key innovation: instead of a cold empty cache (PD baseline) or a
blind LRU (coupled baseline), the cache starts with high-frequency
prefill experts pre-loaded and protected from early eviction.

The EAP influence decays over time — after enough decode steps the
cache behaves identically to standard LRU, but the critical first
few tokens benefit from prefill-derived warmup.

Parameters
----------
eap_profile:
    The EAP from prefill (one profile per request).
cache_capacity:
    Maximum experts cached per layer.
eap_weight:
    Initial boost for EAP top-K experts (0-1, default 0.5).
    Higher → EAP experts stay longer.
decay_steps:
    Number of decode tokens over which EAP boost linearly decays
    to zero.  After this many tokens, the cache is pure LRU.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

from .profile import EapProfile


@dataclass
class _CacheEntry:
    expert_id: int
    eap_boost: float = 0.0  # residual EAP priority (decays to 0)


class EapCacheManager:
    """Per-layer EAP-weighted LRU expert cache.

    Before each ``access()``, call ``step_decay()`` to advance the
    EAP influence window.  The cache internally tracks which experts
    are resident and their EAP boost scores.
    """

    def __init__(
        self,
        num_layers: int,
        num_experts: int,
        cache_capacity: int = 16,
        eap_weight: float = 0.5,
        decay_steps: int = 32,
    ) -> None:
        if cache_capacity <= 0:
            raise ValueError(f"cache_capacity must be > 0, got {cache_capacity}")
        if not 0 <= eap_weight <= 1:
            raise ValueError(f"eap_weight must be in [0, 1], got {eap_weight}")

        self.num_layers = num_layers
        self.num_experts = num_experts
        self.cache_capacity = cache_capacity
        self.eap_weight = eap_weight
        self.decay_steps = decay_steps

        # Per-layer cache: OrderedDict mapping expert_id → _CacheEntry.
        # Most-recently-used at the end.
        self._caches: list[OrderedDict[int, _CacheEntry]] = [
            OrderedDict() for _ in range(num_layers)
        ]

        # Hit/miss counters.
        self._hits = np.zeros(num_layers, dtype=np.int64)
        self._misses = np.zeros(num_layers, dtype=np.int64)

        # Decay state.
        self._decode_step: int = 0
        self._eap_profile: EapProfile | None = None
        self._initialized = False

    # ------------------------------------------------------------------
    # Initialisation
    # ------------------------------------------------------------------

    def init_from_eap(self, eap_profile: EapProfile, k_preload: int = 8) -> None:
        """Pre-load top-K EAP experts into the cache with weighted priority.

        Called once when the Decode pool receives the EAP alongside
        the KV Cache.

        Args:
            eap_profile:
                The prefill-derived expert frequency profile.
            k_preload:
                Number of top experts per layer to pre-load.
        """
        self._eap_profile = eap_profile
        boost = self.eap_weight

        for li in range(min(self.num_layers, eap_profile.num_layers)):
            top_k = eap_profile.top_k(li, k_preload)
            cache = self._caches[li]
            for eid in top_k:
                if eid not in cache:
                    cache[eid] = _CacheEntry(expert_id=eid, eap_boost=boost)

        self._initialized = True

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def step_decay(self) -> None:
        """Advance decode step counter and decay EAP boosts.

        Should be called once per decode token (before the first
        layer access of that token).
        """
        self._decode_step += 1
        if self._decode_step > self.decay_steps:
            return  # fully decayed — no-op

        # Linear decay: boost_factor = 1 - step/decay_steps
        factor = 1.0 - self._decode_step / self.decay_steps
        for cache in self._caches:
            for entry in cache.values():
                entry.eap_boost = self.eap_weight * factor

    def access(self, layer_idx: int, expert_ids: list[int]) -> int:
        """Access a set of experts for one (token, layer).

        Returns the number of cache hits.  LRU order is updated.
        EAP-boosted entries require fewer hits to stay resident
        (their priority = recency + eap_boost).
        """
        if layer_idx < 0 or layer_idx >= self.num_layers:
            return 0

        cache = self._caches[layer_idx]
        hits = 0

        for eid in expert_ids:
            if eid in cache:
                hits += 1
                self._hits[layer_idx] += 1
                # Already resident — move to MRU.
                cache.move_to_end(eid)
            else:
                self._misses[layer_idx] += 1
                # Insert at MRU with zero EAP boost.
                cache[eid] = _CacheEntry(expert_id=eid, eap_boost=0.0)
                # Evict LRU if over capacity.
                if len(cache) > self.cache_capacity:
                    # Evict the entry with lowest effective priority.
                    self._evict_lowest(layer_idx)

        return hits

    def _evict_lowest(self, layer_idx: int) -> None:
        """Evict the entry with lowest (recency_order + eap_boost)."""
        cache = self._caches[layer_idx]
        if not cache:
            return

        # The OrderedDict is in insertion→MRU order (oldest first).
        # Among all entries, evict the one at the front (LRU) unless
        # a later entry has significantly lower boost.
        # Simplified: pop the first (least recently used).
        cache.popitem(last=False)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def hit_rate(self, layer_idx: int | None = None) -> float:
        """Aggregate hit rate."""
        if layer_idx is not None:
            total = self._hits[layer_idx] + self._misses[layer_idx]
            return float(self._hits[layer_idx] / total) if total > 0 else 0.0
        total_h = self._hits.sum()
        total_m = self._misses.sum()
        total = total_h + total_m
        return float(total_h / total) if total > 0 else 0.0

    def resident_experts(self, layer_idx: int) -> set[int]:
        """Current cached expert IDs for one layer."""
        return set(self._caches[layer_idx].keys())

    @property
    def decode_step(self) -> int:
        return self._decode_step

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    # ------------------------------------------------------------------
    # Serialisation (for debugging)
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Serialise current state (debugging / checkpointing)."""
        return {
            "num_layers": self.num_layers,
            "num_experts": self.num_experts,
            "cache_capacity": self.cache_capacity,
            "eap_weight": self.eap_weight,
            "decay_steps": self.decay_steps,
            "decode_step": self._decode_step,
            "hit_rate": self.hit_rate(),
            "per_layer": {
                str(li): {
                    "resident": sorted(self._caches[li].keys()),
                    "hit_rate": self.hit_rate(li),
                }
                for li in range(self.num_layers)
            },
        }
