#!/usr/bin/env python3
"""Verify expert trace semantics across critical batching scenarios.

This script validates that the TraceContext computed by the GPU model
runner produces correct per-token metadata under three scenarios:

1. **Chunked Prefill**: Long prompts split across multiple forward passes.
2. **Continuous Batching**: Multiple requests at different decode positions
   mixed in one batch.
3. **Mixed Prefill/Decode**: Some requests prefill while others decode,
   all in the same batch.

Usage::

    python trace_collector/verify.py
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import numpy as np

from trace_collector.collector import ExpertTraceCollector
from trace_collector.context import TraceContext


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _schedule_step(
    collector: ExpertTraceCollector,
    request_ids: list[str],
    num_computed_tokens: dict[str, int],
    num_scheduled_tokens: dict[str, int],
    prompt_lengths: dict[str, int],
    token_ids: dict[str, list[int]],
) -> list[dict]:
    """Simulate one engine step and return the trace records that would be emitted.

    This mirrors the logic in ``GPUModelRunner.execute_model()`` for trace
    context construction.
    """
    # Flatten per-request arrays into per-token arrays.
    req_indices_list: list[int] = []
    token_indices_list: list[int] = []
    token_ids_list: list[int] = []
    is_prefill_list: list[bool] = []

    for req_idx, req_id in enumerate(request_ids):
        n_computed = num_computed_tokens.get(req_id, 0)
        n_tokens = num_scheduled_tokens.get(req_id, 0)
        prompt_len = prompt_lengths.get(req_id, 0)
        is_pref = n_computed < prompt_len

        for pos in range(n_tokens):
            req_indices_list.append(req_idx)
            token_indices_list.append(n_computed + pos)
            is_prefill_list.append(is_pref)
            tids = token_ids.get(req_id, [])
            col = n_computed + pos
            if col < len(tids):
                token_ids_list.append(tids[col])
            else:
                token_ids_list.append(-1)

    req_indices = np.array(req_indices_list, dtype=np.int32)
    is_prefill_tokens = np.array(is_prefill_list, dtype=bool)
    token_indices = np.array(token_indices_list, dtype=np.int32)
    token_ids_arr = np.array(token_ids_list, dtype=np.int32)

    if len(req_indices) == 0:
        return []

    trace_context = collector.build_trace_context(
        request_ids=request_ids,
        req_indices=req_indices,
        is_prefill_tokens=is_prefill_tokens,
        token_indices=token_indices,
        token_ids=token_ids_arr,
        prompt_lengths=prompt_lengths,
    )

    # Simulate what the callback would produce (without actual PyTorch tensors).
    records = _simulate_moe_forward(collector, trace_context)
    return records


def _simulate_moe_forward(
    collector: ExpertTraceCollector,
    trace_context: TraceContext,
    num_layers: int = 28,
    top_k: int = 2,
    rng: np.random.RandomState | None = None,
) -> list[dict]:
    """Simulate MoE routing for all layers, collecting trace records.

    Instead of importing PyTorch, we directly construct the records that
    the callback would produce.
    """
    if rng is None:
        rng = np.random.RandomState(42)

    records: list[dict] = []
    for layer_idx in range(num_layers):
        num_tokens = len(trace_context.req_indices)
        for i in range(num_tokens):
            req_idx = trace_context.req_indices[i]
            records.append(
                {
                    "request_id": trace_context.request_ids[req_idx],
                    "phase": "prefill" if trace_context.is_prefill[i] else "decode",
                    "token_idx": trace_context.token_indices[i],
                    "token_id": trace_context.token_ids[i],
                    "decode_token_idx": trace_context.decode_token_indices[i],
                    "layer_idx": layer_idx,
                    "experts": rng.randint(0, 64, size=top_k).tolist(),
                    "step": collector._step_count,
                    "timestamp": trace_context.timestamp,
                }
            )
    return records


# ---------------------------------------------------------------------------
# Scenario 1: Chunked Prefill
# ---------------------------------------------------------------------------


def test_chunked_prefill() -> None:
    """Verify token_indices are correct across chunked prefill steps.

    A single request with 10 prompt tokens, chunked into 3 steps:
    - Step 1: tokens [0..3]
    - Step 2: tokens [4..7]
    - Step 3: tokens [8..9]
    """
    print("=" * 60)
    print("SCENARIO 1: Chunked Prefill")
    print("=" * 60)

    collector = ExpertTraceCollector(
        output_dir=tempfile.mkdtemp(prefix="trace_verify_"),
        all_moe_layers=[f"layer_{i}" for i in range(2)],
    )

    req_id = "r0"
    prompt_ids = list(range(100, 110))  # 10 prompt tokens
    chunk_size = 4

    all_records: list[dict] = []
    for chunk_start in range(0, len(prompt_ids), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(prompt_ids))
        num_computed = chunk_start
        num_scheduled = chunk_end - chunk_start

        records = _schedule_step(
            collector,
            request_ids=[req_id],
            num_computed_tokens={req_id: num_computed},
            num_scheduled_tokens={req_id: num_scheduled},
            prompt_lengths={req_id: len(prompt_ids)},
            token_ids={req_id: prompt_ids},
        )
        all_records.extend(records)

    # Verify per-chunk token_indices.
    expected_indices = list(range(10)) * 2  # 10 tokens × 2 layers
    actual_indices = [r["token_idx"] for r in all_records]
    assert actual_indices == expected_indices, (
        f"Expected {expected_indices}, got {actual_indices}"
    )

    # Verify phase is always prefill.
    phases = {r["phase"] for r in all_records}
    assert phases == {"prefill"}, f"Expected only 'prefill', got {phases}"

    # Verify token_ids match.
    expected_tids = prompt_ids * 2
    actual_tids = [r["token_id"] for r in all_records]
    assert actual_tids == expected_tids, (
        f"Expected {expected_tids}, got {actual_tids}"
    )

    # Verify decode_token_indices are all -1.
    for r in all_records:
        assert r["decode_token_idx"] == -1, r

    print(f"  ✓ {len(all_records)} records, all prefill, correct token_indices")
    print(f"  ✓ Chunks: 0-3, 4-7, 8-9 → contiguous positions 0..9")
    print()


# ---------------------------------------------------------------------------
# Scenario 2: Continuous Batching
# ---------------------------------------------------------------------------


def test_continuous_batching() -> None:
    """Verify that multiple decode requests at different positions coexist.

    Two requests:
    - r0: prompt_length=5, already generated 3 tokens (decode_token_idx 3)
    - r1: prompt_length=8, already generated 12 tokens (decode_token_idx 12)
    Both decode in the same batch (1 token each).
    """
    print("=" * 60)
    print("SCENARIO 2: Continuous Batching (Decode)")
    print("=" * 60)

    collector = ExpertTraceCollector(
        output_dir=tempfile.mkdtemp(prefix="trace_verify_"),
        all_moe_layers=[f"layer_{i}" for i in range(2)],
    )

    prompt_lengths = {"r0": 5, "r1": 8}
    num_computed = {"r0": 8, "r1": 20}  # prompt + already decoded
    num_scheduled = {"r0": 1, "r1": 1}  # one decode token each
    token_ids = {"r0": list(range(100, 110)), "r1": list(range(200, 230))}

    records = _schedule_step(
        collector,
        request_ids=["r0", "r1"],
        num_computed_tokens=num_computed,
        num_scheduled_tokens=num_scheduled,
        prompt_lengths=prompt_lengths,
        token_ids=token_ids,
    )

    # r0: token at position 8 (5 prompt + 3 decoded), decode_token_idx=3
    # r1: token at position 20 (8 prompt + 12 decoded), decode_token_idx=12
    r0_recs = [r for r in records if r["request_id"] == "r0"]
    r1_recs = [r for r in records if r["request_id"] == "r1"]

    assert len(r0_recs) == 2, f"Expected 2 records for r0 (2 layers), got {len(r0_recs)}"
    assert len(r1_recs) == 2, f"Expected 2 records for r1 (2 layers), got {len(r1_recs)}"

    for rec in r0_recs:
        assert rec["phase"] == "decode", rec
        assert rec["token_idx"] == 8, f"Expected token_idx=8, got {rec['token_idx']}"
        assert rec["decode_token_idx"] == 3, (
            f"Expected decode_token_idx=3, got {rec['decode_token_idx']}"
        )
        assert rec["token_id"] == 108, f"Expected token_id=108, got {rec['token_id']}"

    for rec in r1_recs:
        assert rec["phase"] == "decode", rec
        assert rec["token_idx"] == 20, f"Expected token_idx=20, got {rec['token_idx']}"
        assert rec["decode_token_idx"] == 12, (
            f"Expected decode_token_idx=12, got {rec['decode_token_idx']}"
        )
        assert rec["token_id"] == 220, f"Expected token_id=220, got {rec['token_id']}"

    print(f"  ✓ r0: token_idx=8 (prompt 5 + decode 3), decode_token_idx=3")
    print(f"  ✓ r1: token_idx=20 (prompt 8 + decode 12), decode_token_idx=12")
    print()


# ---------------------------------------------------------------------------
# Scenario 3: Mixed Prefill/Decode
# ---------------------------------------------------------------------------


def test_mixed_prefill_decode() -> None:
    """Verify mixing prefill and decode in one batch.

    Three requests:
    - r0: prefill, chunks 0..4 of a 10-token prompt (step 1)
    - r1: decode, 1 token, 5th decode step
    - r2: prefill, chunks 5..9 of a 15-token prompt (step 2)
    """
    print("=" * 60)
    print("SCENARIO 3: Mixed Prefill/Decode")
    print("=" * 60)

    collector = ExpertTraceCollector(
        output_dir=tempfile.mkdtemp(prefix="trace_verify_"),
        all_moe_layers=[f"layer_{i}" for i in range(2)],
    )

    prompt_lengths = {"r0": 10, "r1": 20, "r2": 15}
    num_computed = {"r0": 0, "r1": 25, "r2": 5}  # r0 hasn't started, r1 mid-decode, r2 partial prefill
    num_scheduled = {"r0": 5, "r1": 1, "r2": 5}
    token_ids = {
        "r0": list(range(100, 120)),
        "r1": list(range(200, 250)),
        "r2": list(range(300, 320)),
    }

    records = _schedule_step(
        collector,
        request_ids=["r0", "r1", "r2"],
        num_computed_tokens=num_computed,
        num_scheduled_tokens=num_scheduled,
        prompt_lengths=prompt_lengths,
        token_ids=token_ids,
    )

    # r0: prefill, tokens 0..4, decode_token_idx=-1, req_idx=0
    r0_recs = [r for r in records if r["request_id"] == "r0"]
    assert all(r["phase"] == "prefill" for r in r0_recs), r0_recs
    assert [r["token_idx"] for r in r0_recs] == [0, 0, 1, 1, 2, 2, 3, 3, 4, 4]
    assert all(r["decode_token_idx"] == -1 for r in r0_recs)

    # r1: decode, token at position 25 (20 prompt + 5 decoded), req_idx=1
    r1_recs = [r for r in records if r["request_id"] == "r1"]
    assert all(r["phase"] == "decode" for r in r1_recs)
    assert r1_recs[0]["token_idx"] == 25
    assert r1_recs[0]["decode_token_idx"] == 5  # 25 - 20
    assert r1_recs[0]["token_id"] == 225

    # r2: prefill (still has 5 tokens to go), req_idx=2
    r2_recs = [r for r in records if r["request_id"] == "r2"]
    assert all(r["phase"] == "prefill" for r in r2_recs), r2_recs
    assert [r["token_idx"] for r in r2_recs] == [5, 5, 6, 6, 7, 7, 8, 8, 9, 9]
    assert all(r["decode_token_idx"] == -1 for r in r2_recs)

    # Verify req_indices ordering is correct.
    unique_req_ids = set()
    for r in records:
        unique_req_ids.add(r["request_id"])
    assert unique_req_ids == {"r0", "r1", "r2"}

    # Verify phases are correct by request.
    phases_by_req: dict[str, set] = {}
    for r in records:
        phases_by_req.setdefault(r["request_id"], set()).add(r["phase"])
    assert phases_by_req["r0"] == {"prefill"}
    assert phases_by_req["r1"] == {"decode"}
    assert phases_by_req["r2"] == {"prefill"}

    print(f"  ✓ r0 (prefill chunk 0): {len(r0_recs)} records, phase=prefill")
    print(f"  ✓ r1 (decode): token_idx=25, decode_token_idx=5")
    print(f"  ✓ r2 (prefill chunk 1): positions 5-9, phase=prefill")
    print(f"  ✓ All 3 requests correctly classified")
    print()


# ---------------------------------------------------------------------------
# Edge case: Transition from prefill to decode
# ---------------------------------------------------------------------------


def test_prefill_to_decode_transition() -> None:
    """Verify a request transitions from prefill to decode correctly.

    r0: prompt_length=5
    - Step 1: prefill tokens 0-4 (num_computed=0, 5 tokens)
    - Step 2: first decode token (num_computed=5)
    - Step 3: second decode token (num_computed=6)
    """
    print("=" * 60)
    print("EDGE CASE: Prefill → Decode Transition")
    print("=" * 60)

    collector = ExpertTraceCollector(
        output_dir=tempfile.mkdtemp(prefix="trace_verify_"),
        all_moe_layers=[f"layer_{i}" for i in range(2)],
    )

    prompt_ids = list(range(500, 505))

    # Step 1: full prefill
    records1 = _schedule_step(
        collector,
        request_ids=["r0"],
        num_computed_tokens={"r0": 0},
        num_scheduled_tokens={"r0": 5},
        prompt_lengths={"r0": 5},
        token_ids={"r0": prompt_ids},
    )
    phases1 = {r["phase"] for r in records1}
    assert phases1 == {"prefill"}, f"Step 1 should be prefill, got {phases1}"
    assert [r["decode_token_idx"] for r in records1] == [-1] * 10

    # Step 2: first decode token
    records2 = _schedule_step(
        collector,
        request_ids=["r0"],
        num_computed_tokens={"r0": 5},
        num_scheduled_tokens={"r0": 1},
        prompt_lengths={"r0": 5},
        token_ids={"r0": prompt_ids + [600]},
    )
    phases2 = {r["phase"] for r in records2}
    assert phases2 == {"decode"}, f"Step 2 should be decode, got {phases2}"

    r = records2[0]
    assert r["token_idx"] == 5, f"First decode should be at position 5, got {r['token_idx']}"
    assert r["decode_token_idx"] == 0, (
        f"First decode should have decode_token_idx=0, got {r['decode_token_idx']}"
    )
    assert r["token_id"] == 600, f"Expected token_id=600, got {r['token_id']}"

    # Step 3: second decode token
    records3 = _schedule_step(
        collector,
        request_ids=["r0"],
        num_computed_tokens={"r0": 6},
        num_scheduled_tokens={"r0": 1},
        prompt_lengths={"r0": 5},
        token_ids={"r0": prompt_ids + [600, 601]},
    )
    r = records3[0]
    assert r["phase"] == "decode"
    assert r["token_idx"] == 6
    assert r["decode_token_idx"] == 1  # 6 - 5
    assert r["token_id"] == 601

    print(f"  ✓ Step 1 (prefill): positions 0-4, phase=prefill, decode_token_idx=-1")
    print(f"  ✓ Step 2 (decode 0): position 5, phase=decode, decode_token_idx=0")
    print(f"  ✓ Step 3 (decode 1): position 6, phase=decode, decode_token_idx=1")
    print(f"  ✓ Transition correctly detected")
    print()


# ---------------------------------------------------------------------------
# Collector-level tests
# ---------------------------------------------------------------------------


def test_collector_metadata() -> None:
    """Verify collector metadata is correctly accumulated."""
    print("=" * 60)
    print("COLLECTOR: Metadata and Output")
    print("=" * 60)

    tmpdir = tempfile.mkdtemp(prefix="trace_verify_")
    collector = ExpertTraceCollector(
        output_dir=tmpdir,
        all_moe_layers=["model.layers.0.mlp", "model.layers.1.mlp"],
    )

    # Simulate 3 steps
    for step in range(3):
        records = _schedule_step(
            collector,
            request_ids=["r0", "r1"],
            num_computed_tokens={"r0": step, "r1": 0},
            num_scheduled_tokens={"r0": 1, "r1": 5},
            prompt_lengths={"r0": 5, "r1": 5},
            token_ids={
                "r0": list(range(100, 110)),
                "r1": list(range(200, 210)),
            },
        )

    collector.close()

    # Verify traces.jsonl exists and is valid JSONL.
    traces_path = Path(tmpdir) / "traces.jsonl"
    assert traces_path.exists(), f"traces.jsonl not found at {traces_path}"

    lines = traces_path.read_text().strip().split("\n")
    for i, line in enumerate(lines):
        record = json.loads(line)
        required_keys = {
            "request_id", "phase", "token_idx", "token_id",
            "decode_token_idx", "layer_idx", "experts", "step", "timestamp",
        }
        missing = required_keys - set(record.keys())
        assert not missing, f"Line {i}: missing keys {missing}"

    assert len(lines) == 3 * (2 * 6)  # 3 steps × (2 layers × (1+5) tokens)

    # Verify metadata.json.
    meta_path = Path(tmpdir) / "metadata.json"
    assert meta_path.exists(), f"metadata.json not found at {meta_path}"

    meta = json.loads(meta_path.read_text())
    assert meta["num_layers"] == 2
    assert meta["layer_mapping"] == {
        "0": "model.layers.0.mlp",
        "1": "model.layers.1.mlp",
    }
    assert meta["num_steps"] == 3
    assert meta["total_trace_records"] == 3 * (2 * 6)
    assert "r0" in meta["requests"]
    assert "r1" in meta["requests"]
    assert meta["requests"]["r0"]["prompt_length"] == 5
    assert meta["requests"]["r1"]["prompt_length"] == 5

    print(f"  ✓ traces.jsonl: {len(lines)} valid JSON records")
    print(f"  ✓ metadata.json: layer mapping and request stats correct")
    print()


def test_trace_context_validation() -> None:
    """Verify TraceContext invariants are enforced."""
    print("=" * 60)
    print("CONTEXT: Invariant Validation")
    print("=" * 60)

    # Mismatched lengths should raise.
    try:
        TraceContext(
            request_ids=("a",),
            req_indices=(0, 0),
            is_prefill=(True,),  # length 2 vs 1
            token_indices=(0, 0),
            token_ids=(100, 200),
            decode_token_indices=(-1, -1),
            timestamp=1.0,
        )
        assert False, "Should have raised ValueError"
    except ValueError:
        pass

    # Correct construction should work.
    ctx = TraceContext(
        request_ids=("a", "b"),
        req_indices=(0, 1),
        is_prefill=(True, False),
        token_indices=(0, 5),
        token_ids=(100, 200),
        decode_token_indices=(-1, 0),
        timestamp=1.0,
    )
    assert ctx.request_ids[0] == "a"
    assert ctx.decode_token_indices[1] == 0

    print(f"  ✓ Length mismatch detected")
    print(f"  ✓ Valid context constructed")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    print()
    print("╔══════════════════════════════════════════════════════════╗")
    print("║     Expert Trace Collector — Verification Suite          ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    test_trace_context_validation()
    test_chunked_prefill()
    test_continuous_batching()
    test_mixed_prefill_decode()
    test_prefill_to_decode_transition()
    test_collector_metadata()

    print("═" * 60)
    print("ALL VERIFICATION TESTS PASSED ✓")
    print("═" * 60)


if __name__ == "__main__":
    main()
