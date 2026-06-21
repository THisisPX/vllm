# SPDX-License-Identifier: Apache-2.0
"""Data loaders for cold-start simulation.

Two modes, reachable via the common ``iter_requests()`` entry point:

- **JSONL** — load Phase 0 trace files (``traces.jsonl``).
- **Synthetic** — generate realistic MoE routing traces with
  configurable locality.

The common data contract is a ``RequestTrace``: a named tuple of
(request_id, records) where ``records`` is a list of ``TraceRecord``
sorted by (token_idx, layer_idx).  Each ``TraceRecord`` corresponds
to one (token, layer) expert selection event.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import numpy as np


# ---------------------------------------------------------------------------
# Data contract
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TraceRecord:
    """A single (token, layer) expert selection event."""

    token_idx: int
    phase: str  # "prefill" | "decode"
    decode_token_idx: int  # -1 for prefill
    layer_idx: int
    token_id: int
    experts: tuple[int, ...]


class RequestTrace(NamedTuple):
    request_id: str
    records: list[TraceRecord]


# ---------------------------------------------------------------------------
# JSONL loader
# ---------------------------------------------------------------------------


def load_jsonl_traces(path: str | Path) -> Iterator[RequestTrace]:
    """Load Phase 0 expert traces from a JSONL file.

    Records are grouped by ``request_id`` and sorted by
    (token_idx, layer_idx) to form a coherent per-request trace.

    Args:
        path: Path to a ``traces.jsonl`` file produced by
            ``ExpertTraceCollector``.

    Yields:
        One ``RequestTrace`` per unique request_id, in order of
        first appearance.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Trace file not found: {path}")

    # Read and group.
    groups: dict[str, list[TraceRecord]] = {}
    with open(path, encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_no}: {exc}") from exc

            rid = rec["request_id"]
            if rid not in groups:
                groups[rid] = []
            groups[rid].append(
                TraceRecord(
                    token_idx=int(rec["token_idx"]),
                    phase=str(rec["phase"]),
                    decode_token_idx=int(rec.get("decode_token_idx", -1)),
                    layer_idx=int(rec["layer_idx"]),
                    token_id=int(rec.get("token_id", -1)),
                    experts=tuple(int(e) for e in rec["experts"]),
                )
            )

    # Sort each group and yield.
    for rid, records in groups.items():
        records.sort(key=lambda r: (r.token_idx, r.layer_idx))
        yield RequestTrace(request_id=rid, records=records)


# ---------------------------------------------------------------------------
# Synthetic generator
# ---------------------------------------------------------------------------


def generate_synthetic_requests(
    num_requests: int = 50,
    prompt_len: int = 256,
    decode_len: int = 128,
    num_experts: int = 64,
    top_k: int = 6,
    num_layers: int = 26,
    locality_strength: float = 0.8,
    seed: int = 42,
) -> Iterator[RequestTrace]:
    """Generate synthetic MoE routing traces with tunable expert locality.

    Each request draws a *working set* of experts (size = 2 * top_k)
    per layer.  With probability ``locality_strength`` a token selects
    experts from within its working set; otherwise it falls back to
    uniform random selection across all experts.

    This models the observation that real experts show multi-token
    locality — a token's expert assignments are correlated with
    neighbouring tokens.

    Args:
        num_requests: Number of requests to generate.
        prompt_len: Number of prefill tokens per request.
        decode_len: Number of decode tokens per request.
        num_experts: Total number of experts.
        top_k: Experts selected per token.
        num_layers: Number of MoE layers.
        locality_strength: Probability [0, 1] that a token draws from
            its working set.  Higher values mean stronger locality.
        seed: RNG seed for reproducibility.

    Yields:
        ``RequestTrace`` for each synthetic request.
    """
    rng = np.random.default_rng(seed)
    working_set_size = max(2 * top_k, 4)

    for req_idx in range(num_requests):
        request_id = f"synthetic_{req_idx:04d}"

        # Each layer gets its own working set.
        working_sets: list[np.ndarray] = []
        for _ in range(num_layers):
            ws = rng.choice(num_experts, size=working_set_size, replace=False)
            working_sets.append(ws)

        records: list[TraceRecord] = []

        # Prefill tokens.
        for tok in range(prompt_len):
            for layer in range(num_layers):
                if rng.random() < locality_strength:
                    # Draw top_k from working set with replacement.
                    ws = working_sets[layer]
                    experts_np = rng.choice(ws, size=top_k, replace=True)
                else:
                    experts_np = rng.choice(num_experts, size=top_k, replace=False)
                records.append(
                    TraceRecord(
                        token_idx=tok,
                        phase="prefill",
                        decode_token_idx=-1,
                        layer_idx=layer,
                        token_id=tok + 100_000 * req_idx,
                        experts=tuple(int(e) for e in experts_np),
                    )
                )

        # Decode tokens.
        for dtok in range(decode_len):
            tok = prompt_len + dtok
            for layer in range(num_layers):
                # Decode tokens may show stronger locality.
                if rng.random() < min(locality_strength + 0.1, 1.0):
                    ws = working_sets[layer]
                    experts_np = rng.choice(ws, size=top_k, replace=True)
                else:
                    experts_np = rng.choice(num_experts, size=top_k, replace=False)
                records.append(
                    TraceRecord(
                        token_idx=tok,
                        phase="decode",
                        decode_token_idx=dtok,
                        layer_idx=layer,
                        token_id=tok + 100_000 * req_idx,
                        experts=tuple(int(e) for e in experts_np),
                    )
                )

        yield RequestTrace(request_id=request_id, records=records)


# ---------------------------------------------------------------------------
# Common entry point
# ---------------------------------------------------------------------------


def iter_requests(
    source: str | Path,
    **kwargs,
) -> Iterator[RequestTrace]:
    """Dispatch to the correct loader based on source.

    Args:
        source: Path to a ``.jsonl`` file, or the literal string
            ``"synthetic"`` to use the synthetic generator.  Any other
            string is treated as a JSONL path.
        **kwargs: Forwarded to ``generate_synthetic_requests()`` when
            ``source == "synthetic"``.  Ignored otherwise.

    Yields:
        ``RequestTrace`` objects, one per request.
    """
    source_str = str(source)
    if source_str == "synthetic":
        yield from generate_synthetic_requests(**kwargs)
    else:
        yield from load_jsonl_traces(source_str)
