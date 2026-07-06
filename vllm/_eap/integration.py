# SPDX-License-Identifier: Apache-2.0
"""Runtime integration — hooks EAP into vLLM's request lifecycle.

Activated via environment variable ``VLLM_EAP_ENABLE=1``.  On
Prefill completion the accumulator is finalised and the resulting
:class:`EapProfile` is stored in the request's trace headers for
transfer to the Decode pool alongside KV Cache.

Integration points
------------------
1. **GPU Model Runner init** : Create per-request ``EapAccumulator`` map.
2. **Expert trace callback** : During forward, route prefill tokens
   to the accumulator.
3. **Request completion callback** : When prefill finishes, finalise
   and serialise ``EapProfile``.

Assumptions (documented)
------------------------
- Works with the existing ``expert_trace_callback`` in ``ForwardContext``.
- EAP serialisation + transfer overhead is negligible (~6.5-24 KB).
- Decode pool is responsible for deserialising and applying EAP
  (handled by :class:`EapCacheManager`).
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING

import numpy as np
import torch

from .accumulator import EapAccumulator
from .profile import EapProfile

if TYPE_CHECKING:
    from vllm.forward_context import TraceContext

logger = logging.getLogger(__name__)

# Sentinel for "not a prefill token".
_PREFILL_PHASE = "prefill"


class EapIntegration:
    """Orchestrates EAP accumulation across all in-flight requests.

    This is a singleton attached to the GPU model runner.  It provides
    the callback that hooks into ``ForwardContext.expert_trace_callback``
    and manages per-request accumulator lifecycle.

    Parameters
    ----------
    num_layers, num_experts:
        Model dimensions (determined at model load time).
    """

    def __init__(self, num_layers: int, num_experts: int) -> None:
        self.num_layers = num_layers
        self.num_experts = num_experts
        self._accumulators: dict[str, EapAccumulator] = {}
        self._completed: dict[str, EapProfile] = {}

    # ------------------------------------------------------------------
    # Callback for ForwardContext
    # ------------------------------------------------------------------

    def make_trace_callback(
        self, trace_context: "TraceContext"
    ) -> Callable[[str, torch.Tensor], None]:
        """Return a callback suitable for ``ForwardContext.expert_trace_callback``.

        The callback uses ``TraceContext`` to determine which tokens
        belong to which request and whether they are prefill or decode.
        """
        eap = self

        def _callback(layer_name: str, topk_ids: torch.Tensor) -> None:
            """Record expert selections for all prefill tokens in this layer."""
            if not eap._accumulators:
                return

            # Move to CPU.
            if topk_ids.is_cuda:
                ids_np = topk_ids.cpu().numpy()
            else:
                ids_np = topk_ids.numpy()

            # Derive layer index from layer_name.
            # The layer_name format is "model.layers.N.mlp.experts".
            layer_idx = _parse_layer_idx(layer_name, eap.num_layers)
            if layer_idx is None:
                return

            num_tokens = ids_np.shape[0]
            for i in range(num_tokens):
                if i >= len(trace_context.req_indices):
                    break
                req_idx = trace_context.req_indices[i]
                if req_idx >= len(trace_context.request_ids):
                    break
                req_id = trace_context.request_ids[req_idx]

                if not trace_context.is_prefill[i]:
                    continue

                acc = eap._accumulators.get(req_id)
                if acc is None:
                    # Lazy creation on first prefill token.
                    acc = EapAccumulator(req_id, eap.num_layers, eap.num_experts)
                    eap._accumulators[req_id] = acc

                acc.record(layer_idx, list(int(e) for e in ids_np[i]),
                           is_prefill=True)

        return _callback

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def on_prefill_complete(self, request_id: str) -> EapProfile | None:
        """Finalise and retrieve EAP for a completed prefill.

        Returns ``None`` if the request was never seen (e.g. decode-only).
        """
        acc = self._accumulators.pop(request_id, None)
        if acc is None:
            return None
        profile = acc.finalize()
        self._completed[request_id] = profile
        return profile

    def get_eap_bytes(self, request_id: str) -> bytes | None:
        """Get serialised EAP for KV Cache transfer."""
        profile = self._completed.get(request_id)
        if profile is None:
            return None
        return profile.to_bytes()

    def get_eap(self, request_id: str) -> EapProfile | None:
        """Get EAP object (for in-process use)."""
        return self._completed.get(request_id)

    def remove(self, request_id: str) -> None:
        """Clean up request state."""
        self._accumulators.pop(request_id, None)
        self._completed.pop(request_id, None)

    @property
    def active_requests(self) -> int:
        return len(self._accumulators)

    @property
    def completed_requests(self) -> int:
        return len(self._completed)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_layer_idx(layer_name: str, num_layers: int) -> int | None:
    """Extract layer index from vLLM MoE layer name.

    Format: "model.layers.N.mlp.experts" → N
    """
    # Common vLLM naming patterns.
    for sep in (".mlp.experts", ".mlp", ".moe"):
        if sep in layer_name:
            prefix = layer_name.split(sep)[0]
            parts = prefix.rsplit(".", 1)
            if len(parts) == 2:
                try:
                    idx = int(parts[1])
                    if 0 <= idx < num_layers:
                        return idx
                except ValueError:
                    pass
    return None


# ---------------------------------------------------------------------------
# Environment-driven factory
# ---------------------------------------------------------------------------


def create_from_env(num_layers: int, num_experts: int) -> EapIntegration | None:
    """Create EAP integration if ``VLLM_EAP_ENABLE`` is set.

    Args:
        num_layers, num_experts: Model dimensions.
    """
    if os.environ.get("VLLM_EAP_ENABLE", "") in ("1", "true", "True"):
        logger.info(
            "EAP integration enabled (%d layers × %d experts)",
            num_layers, num_experts,
        )
        return EapIntegration(num_layers, num_experts)
    return None
