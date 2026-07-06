# SPDX-License-Identifier: Apache-2.0
"""EapAccumulator — per-request prefill expert frequency collector.

Attached to each request during its prefill phase.  Hooks into the
existing ``expert_trace_callback`` infrastructure to record expert
activations token-by-token.

When prefill completes, ``finalize()`` returns the compiled
:class:`EapProfile`, ready for serialisation alongside KV Cache.
"""

from __future__ import annotations

from .profile import EapProfile


class EapAccumulator:
    """Collects per-token expert activations during one request's prefill.

    Parameters
    ----------
    request_id:
        The request this accumulator is attached to.
    num_layers:
        Number of MoE layers in the model.
    num_experts:
        Number of routed experts per layer.
    """

    __slots__ = (
        "request_id",
        "profile",
        "_prefill_token_count",
        "_finalized",
    )

    def __init__(
        self, request_id: str, num_layers: int, num_experts: int
    ) -> None:
        self.request_id = request_id
        self.profile = EapProfile(num_layers, num_experts)
        self._prefill_token_count = 0
        self._finalized = False

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(
        self, layer_idx: int, expert_ids: list[int], *, is_prefill: bool
    ) -> None:
        """Record one (token, layer) expert selection.

        Called from the trace callback during forward.  Only prefill
        tokens are accumulated; decode tokens are silently ignored.

        Args:
            layer_idx: MoE layer index.
            expert_ids: Selected expert IDs for this (token, layer).
            is_prefill: Whether this token is in prefill phase.
        """
        if self._finalized:
            return
        if not is_prefill:
            return
        self.profile.record(layer_idx, expert_ids)

    def record_token_boundary(self) -> None:
        """Mark the end of one prefill token (for counting)."""
        if not self._finalized:
            self._prefill_token_count += 1

    # ------------------------------------------------------------------
    # Finalization
    # ------------------------------------------------------------------

    def finalize(self) -> EapProfile:
        """Mark prefill complete and return the compiled profile.

        After this call, further ``record()`` calls are silently ignored.
        """
        if not self._finalized:
            self._finalized = True
        return self.profile

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    @property
    def prefill_token_count(self) -> int:
        return self._prefill_token_count

    @property
    def is_finalized(self) -> bool:
        return self._finalized

    def __repr__(self) -> str:
        return (
            f"EapAccumulator(req={self.request_id}, "
            f"tokens={self._prefill_token_count}, "
            f"acts={self.profile.total_activations}, "
            f"finalized={self._finalized})"
        )
