# SPDX-License-Identifier: Apache-2.0
"""Expert trace collector — the primary public API.

An *ExpertTraceCollector* is attached to the GPU model runner and
receives expert-selection events via a callback that fires inside
every MoE router's ``select_experts()``.

Usage (standalone, with a running vLLM server)::

    import os
    os.environ["VLLM_EXPERT_TRACE_DIR"] = "./traces/run_001"

    # ... start vLLM server or LLM engine as usual ...
    # Traces are written automatically to ./traces/run_001/traces.jsonl
    # Metadata is written to ./traces/run_001/metadata.json at shutdown.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from .context import RequestMetadata, TraceContext
from .writer import TraceWriter

if TYPE_CHECKING:
    from collections.abc import Callable

logger = logging.getLogger(__name__)


class ExpertTraceCollector:
    """Collects MoE expert routing decisions across engine steps.

    The collector operates in two phases per engine step:

    1. **Before forward**: ``build_trace_context()`` is called with
       per-token metadata arrays from the model runner.
    2. **During forward**: The router callback (returned by
       ``make_step_callback()``) fires once per MoE layer, recording
       which experts were selected for every token.
    3. **After forward**: ``flush()`` is called periodically to write
       buffered records to disk.

    Parameters
    ----------
    output_dir:
        Directory where ``traces.jsonl`` and ``metadata.json`` will be
        written.  Created if it does not exist.
    all_moe_layers:
        Ordered list of MoE layer names (e.g. ``["model.layers.0.mlp",
        "model.layers.1.mlp", ...]``).  Used to build the
        ``layer_idx ↔ layer_name`` mapping stored in ``metadata.json``.
    buffer_size:
        Number of trace records to buffer in memory before flushing to
        disk.  Balances I/O overhead against memory usage.
    """

    def __init__(
        self,
        output_dir: str,
        all_moe_layers: list[str],
        buffer_size: int = 100_000,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Build layer_idx → layer_name mapping.
        self._layer_names = list(all_moe_layers)
        self._layer_idx_map: dict[str, int] = {
            name: idx for idx, name in enumerate(all_moe_layers)
        }

        # In-memory buffer for trace records.
        self._buffer: list[dict] = []
        self._buffer_size = buffer_size

        # Accumulated request metadata (populated during build_trace_context).
        self._request_meta: dict[str, RequestMetadata] = {}

        # Writer (created lazily on first flush).
        self._writer: TraceWriter | None = None
        self._step_count: int = 0
        self._total_records: int = 0

    # ------------------------------------------------------------------
    # Public API called from GPU model runner
    # ------------------------------------------------------------------

    def build_trace_context(
        self,
        request_ids: list[str],
        req_indices: np.ndarray,
        is_prefill_tokens: np.ndarray,
        token_indices: np.ndarray,
        token_ids: np.ndarray,
        prompt_lengths: dict[str, int],
    ) -> TraceContext:
        """Build the per-step trace context from model runner arrays.

        Called by the GPU model runner before each forward pass.

        Args:
            request_ids: Request IDs, length ``num_reqs``.
            req_indices: Per-token request index, length ``num_tokens``.
            is_prefill_tokens: Per-token prefill flag, length ``num_tokens``.
            token_indices: Per-token absolute sequence position, length
                ``num_tokens``.
            token_ids: Per-token vocabulary token ID, length ``num_tokens``.
                Use ``-1`` for positions where the token ID is not available
                (e.g. prompt embedding inputs).
            prompt_lengths: Map from ``request_id`` to its prompt length.

        Returns:
            A ``TraceContext`` ready to be passed to ``make_step_callback()``.
        """
        self._step_count += 1

        # Compute decode_token_indices.
        decode_token_indices: list[int] = []
        for i in range(len(req_indices)):
            req_idx = req_indices[i]
            req_id = request_ids[req_idx]
            if is_prefill_tokens[i]:
                decode_token_indices.append(-1)
            else:
                decode_token_indices.append(
                    int(token_indices[i]) - prompt_lengths.get(req_id, 0)
                )

        ctx = TraceContext(
            request_ids=tuple(request_ids),
            req_indices=tuple(int(x) for x in req_indices),
            is_prefill=tuple(bool(x) for x in is_prefill_tokens),
            token_indices=tuple(int(x) for x in token_indices),
            token_ids=tuple(int(x) for x in token_ids),
            decode_token_indices=tuple(decode_token_indices),
            timestamp=time.time(),
        )

        # Cache request metadata.
        for req_id in request_ids:
            if req_id not in self._request_meta:
                self._request_meta[req_id] = RequestMetadata(
                    request_id=req_id,
                    prompt_length=prompt_lengths.get(req_id, 0),
                )

        return ctx

    def make_step_callback(
        self, trace_context: TraceContext
    ) -> "Callable[[str, torch.Tensor], None]":
        """Return a callback suitable for ``ForwardContext.expert_trace_callback``.

        The returned callback has signature ``(layer_name, topk_ids)``
        where ``topk_ids`` is a ``[num_tokens, top_k]`` int tensor of
        selected expert IDs.

        The callback captures the *trace_context* by reference — the same
        context must not be shared across steps.
        """
        collector = self  # capture for use inside the callback closure
        layer_idx_map = self._layer_idx_map
        buffer = self._buffer
        buffer_size = self._buffer_size

        def _record_experts(layer_name: str, topk_ids: torch.Tensor) -> None:
            """Record expert selections for one MoE layer."""
            layer_idx = layer_idx_map.get(layer_name)
            if layer_idx is None:
                logger.warning(
                    "Unknown MoE layer '%s' — skipping trace for this layer",
                    layer_name,
                )
                return

            # Move to CPU as numpy — we are inside the forward pass so
            # synchronisation is unavoidable but acceptable for an
            # experimental collector.
            if topk_ids.is_cuda:
                ids_np = topk_ids.cpu().numpy()
            else:
                ids_np = topk_ids.numpy()

            num_tokens = ids_np.shape[0]
            ts = trace_context.timestamp

            # Build per-token records.
            for i in range(num_tokens):
                req_idx = trace_context.req_indices[i]
                buffer.append(
                    {
                        "request_id": trace_context.request_ids[req_idx],
                        "phase": "prefill" if trace_context.is_prefill[i] else "decode",
                        "token_idx": trace_context.token_indices[i],
                        "token_id": trace_context.token_ids[i],
                        "decode_token_idx": trace_context.decode_token_indices[i],
                        "layer_idx": layer_idx,
                        "experts": ids_np[i].tolist(),
                        "step": collector._step_count,
                        "timestamp": ts,
                    }
                )

            collector._total_records += num_tokens

            # Flush if buffer exceeds threshold.
            if len(buffer) >= buffer_size:
                collector.writer.write_many(buffer)
                buffer.clear()

        return _record_experts

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    @property
    def writer(self) -> TraceWriter:
        """Lazily create the JSONL writer on first access."""
        if self._writer is None:
            self._writer = TraceWriter(self.output_dir / "traces.jsonl")
        return self._writer

    def flush(self) -> None:
        """Write buffered records to disk."""
        if self._buffer:
            self.writer.write_many(self._buffer)
            self._buffer.clear()
        if self._writer is not None:
            self._writer.flush()

    def save_metadata(self) -> None:
        """Write ``metadata.json`` with layer mapping and request stats."""
        self.flush()

        metadata = {
            "layer_mapping": {
                str(idx): name for idx, name in enumerate(self._layer_names)
            },
            "num_layers": len(self._layer_names),
            "num_steps": self._step_count,
            "total_trace_records": self._total_records,
            "requests": {
                req_id: {
                    "prompt_length": meta.prompt_length,
                    "total_decode_tokens": meta.total_decode_tokens,
                }
                for req_id, meta in self._request_meta.items()
            },
        }

        path = self.output_dir / "metadata.json"
        path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False))
        logger.info("Trace metadata written to %s", path)

    def update_request_completion(
        self, request_id: str, total_decode_tokens: int
    ) -> None:
        """Record final decode length for a completed request."""
        if request_id in self._request_meta:
            self._request_meta[request_id].total_decode_tokens = total_decode_tokens

    def close(self) -> None:
        """Flush buffers, save metadata, close files."""
        self.save_metadata()
        if self._writer is not None:
            self._writer.close()
            self._writer = None
