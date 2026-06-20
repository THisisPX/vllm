# SPDX-License-Identifier: Apache-2.0
"""Trace data types for expert activation collection.

TraceContext is the per-engine-step metadata container passed from
the GPU model runner into the forward context. It provides per-token
mapping from the flat [num_tokens] batch layout back to individual
requests and sequence positions.
"""

from dataclasses import dataclass, field


@dataclass
class RequestMetadata:
    """Static per-request metadata recorded once when a request completes."""

    request_id: str
    prompt_length: int  # number of prompt tokens (including chunked prefill)
    # Added after request completes:
    total_decode_tokens: int | None = None


@dataclass
class TraceContext:
    """Per-step metadata for mapping token positions to requests.

    All sequences are length ``num_tokens`` (the total tokens scheduled
    in this engine step).  The i-th element describes the i-th token in
    the flat batch.

    Attributes:
        request_ids: Request IDs, length ``num_reqs``.
        req_indices: For each of the ``num_tokens`` scheduled tokens, the
            index into ``request_ids`` (and other per-request arrays) that
            owns this token.
        is_prefill: Per-token boolean — True if the token belongs to a
            request that is still in its prefill phase.  Computed as
            ``num_computed_tokens < num_prompt_tokens`` for the owning
            request.
        token_indices: Absolute position of each token within its sequence
            (0-based).  For decode tokens this equals the prompt length
            plus the number of already-generated tokens.
        token_ids: Vocabulary token ID for each token.  May be -1 for
            positions that use prompt embeddings rather than token IDs.
        decode_token_indices: For decode-phase tokens, the 0-based index
            relative to the start of decode (i.e. number of tokens
            generated so far, starting at 0 for the first decode token).
            For prefill-phase tokens the value is -1.
        timestamp: Wall-clock timestamp of this engine step (seconds since
            epoch, as returned by ``time.time()``).
    """

    request_ids: tuple[str, ...]
    req_indices: tuple[int, ...]
    is_prefill: tuple[bool, ...]
    token_indices: tuple[int, ...]
    token_ids: tuple[int, ...]
    decode_token_indices: tuple[int, ...]
    timestamp: float

    # Per-request metadata accumulated lazily.
    _request_meta: dict[str, RequestMetadata] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not (
            len(self.req_indices)
            == len(self.is_prefill)
            == len(self.token_indices)
            == len(self.token_ids)
            == len(self.decode_token_indices)
        ):
            raise ValueError(
                "All per-token sequences must have the same length, got "
                f"req_indices={len(self.req_indices)}, "
                f"is_prefill={len(self.is_prefill)}, "
                f"token_indices={len(self.token_indices)}, "
                f"token_ids={len(self.token_ids)}, "
                f"decode_token_indices={len(self.decode_token_indices)}"
            )

    def get_request_meta(self, request_id: str) -> RequestMetadata | None:
        """Return cached request metadata or None."""
        return self._request_meta.get(request_id)

    def set_request_meta(self, meta: RequestMetadata) -> None:
        """Cache request metadata."""
        self._request_meta[meta.request_id] = meta
