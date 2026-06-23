#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Controlled benchmark runner for expert trace collection.

Runs vLLM offline inference with the Phase 0 ``ExpertTraceCollector``
hooked in.  Supports two workload modes:

    random     — Synthetic prompts with exact control over prompt length
                 and decode length.  Best for Phase 1 (cold-start).
    sharegpt   — Real conversations from a ShareGPT JSON file.
                 Best for Phase 2 (recall analysis).

Traces are written to ``$VLLM_EXPERT_TRACE_DIR/traces.jsonl`` and
``metadata.json`` by the collector.

Usage
-----

    # Phase 1: cold-start measurement
    VLLM_EXPERT_TRACE_DIR="./traces/phase1" \\
    python trace_collector/bench_runner.py \\
        --mode random \\
        --num-prompts 50 \\
        --prompt-len 512 \\
        --decode-len 128 \\
        --model /path/to/DeepSeek-V2-Lite

    # Phase 2: recall analysis
    VLLM_EXPERT_TRACE_DIR="./traces/phase2" \\
    python trace_collector/bench_runner.py \\
        --mode sharegpt \\
        --num-prompts 100 \\
        --max-decode-len 256 \\
        --sharegpt-path /path/to/ShareGPT.json \\
        --model /path/to/DeepSeek-V2-Lite
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

# Ensure the trace_collector package is importable when run as a script.
_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))


def _make_random_prompt(prompt_len: int, tokenizer) -> tuple[str, list[int]]:
    """Generate a random prompt of approximately *prompt_len* tokens."""
    # Pick random tokens from the middle of the vocab (avoids BOS/EOS).
    vocab_size = tokenizer.vocab_size
    lo = max(1, vocab_size // 4)
    hi = min(vocab_size - 2, 3 * vocab_size // 4)
    token_ids = [random.randint(lo, hi) for _ in range(prompt_len)]
    text = tokenizer.decode(token_ids, skip_special_tokens=False)
    return text, token_ids


def run_random(
    model: str,
    num_prompts: int,
    prompt_len: int,
    decode_len: int,
    temperature: float,
    seed: int,
    **kwargs,
) -> None:
    """Run inference with random synthetic prompts."""
    from vllm import LLM, SamplingParams

    random.seed(seed)

    llm = LLM(
        model=model,
        trust_remote_code=kwargs.get("trust_remote_code", True),
        enforce_eager=kwargs.get("enforce_eager", True),
        gpu_memory_utilization=kwargs.get("gpu_memory_utilization", 0.50),
        max_model_len=prompt_len + decode_len + 256,
    )

    # Build prompts with a warm-up tokenizer call to determine actual
    # token counts (the LLM's own tokenizer may tokenise differently).
    tokenizer = llm.get_tokenizer()
    prompts: list[str] = []
    for i in range(num_prompts):
        _, tok_ids = _make_random_prompt(prompt_len, tokenizer)
        text = tokenizer.decode(tok_ids, skip_special_tokens=False)
        prompts.append(text)

    sp = SamplingParams(
        temperature=temperature,
        max_tokens=decode_len,
        ignore_eos=True,  # force exact decode_len for controlled experiment
    )

    print(f"Running {num_prompts} random prompts "
          f"(prompt_len≈{prompt_len}, decode_len={decode_len})...")
    outputs = llm.generate(prompts, sp)

    for o in outputs:
        n_out = len(o.outputs[0].token_ids)
        print(f"  {o.request_id}: prompt={len(o.prompt_token_ids)}t, "
              f"output={n_out}t")


def run_sharegpt(
    model: str,
    num_prompts: int,
    max_decode_len: int,
    sharegpt_path: str,
    temperature: float,
    seed: int,
    **kwargs,
) -> None:
    """Run inference with ShareGPT conversation prompts."""
    from vllm import LLM, SamplingParams

    # Load ShareGPT data.
    sharegpt_path = Path(sharegpt_path)
    if not sharegpt_path.exists():
        raise FileNotFoundError(f"ShareGPT file not found: {sharegpt_path}")
    with open(sharegpt_path, encoding="utf-8") as f:
        data = json.load(f)

    # Filter and shuffle.
    data = [
        e for e in data
        if "conversations" in e and len(e["conversations"]) >= 2
    ]
    random.seed(seed)
    random.shuffle(data)

    llm = LLM(
        model=model,
        trust_remote_code=kwargs.get("trust_remote_code", True),
        enforce_eager=kwargs.get("enforce_eager", True),
        gpu_memory_utilization=kwargs.get("gpu_memory_utilization", 0.50),
    )

    tokenizer = llm.get_tokenizer()
    sp = SamplingParams(
        temperature=temperature,
        max_tokens=max_decode_len,
    )

    prompts: list[str] = []
    prompt_lens: list[int] = []  # expected prompt length (tokens)
    for entry in data[:num_prompts]:
        prompt_text = entry["conversations"][0]["value"]
        prompts.append(prompt_text)
        prompt_lens.append(len(tokenizer(prompt_text).input_ids))

    print(f"Running {len(prompts)} ShareGPT prompts "
          f"(max_decode_len={max_decode_len})...")
    print(f"  Prompt lengths: min={min(prompt_lens)}, "
          f"median={sorted(prompt_lens)[len(prompt_lens)//2]}, "
          f"max={max(prompt_lens)}")

    outputs = llm.generate(prompts, sp)

    for o in outputs:
        n_out = len(o.outputs[0].token_ids)
        print(f"  {o.request_id}: prompt={len(o.prompt_token_ids)}t, "
              f"output={n_out}t, finish={o.outputs[0].finish_reason}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Benchmark runner for expert trace collection"
    )
    p.add_argument(
        "--mode", required=True, choices=["random", "sharegpt"],
        help="Workload mode: synthetic 'random' prompts or real 'sharegpt' data",
    )
    p.add_argument(
        "--model", required=True,
        help="Path or HF repo ID of the MoE model",
    )
    p.add_argument(
        "--num-prompts", type=int, default=50,
        help="Number of prompts to generate (default: 50)",
    )
    p.add_argument(
        "--temperature", type=float, default=0.0,
        help="Sampling temperature (default: 0.0 = greedy)",
    )
    p.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    # Random-mode flags.
    p.add_argument(
        "--prompt-len", type=int, default=512,
        help="[random] Target prompt length in tokens (default: 512)",
    )
    p.add_argument(
        "--decode-len", type=int, default=128,
        help="[random] Number of decode tokens per request (default: 128)",
    )
    # ShareGPT flags.
    p.add_argument(
        "--sharegpt-path",
        default="/workspace/volume/pengxiong/datasets/ShareGPT_V3_unfiltered_cleaned_split.json",
        help="Path to ShareGPT JSON file",
    )
    p.add_argument(
        "--max-decode-len", type=int, default=256,
        help="[sharegpt] Maximum decode tokens per request (default: 256)",
    )
    # Engine flags.
    p.add_argument(
        "--gpu-memory-utilization", type=float, default=0.50,
        help="GPU memory fraction for KV cache (default: 0.50)",
    )
    p.add_argument(
        "--no-enforce-eager", action="store_true",
        help="Disable enforce_eager (enables CUDA graphs if supported)",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    # Validate env.
    trace_dir = os.environ.get("VLLM_EXPERT_TRACE_DIR", "")
    if not trace_dir:
        print("⚠  VLLM_EXPERT_TRACE_DIR is not set — traces will NOT be collected!")
        print("   Set it before running, e.g.:")
        print("   export VLLM_EXPERT_TRACE_DIR=\"./traces/run_001\"")
        print()
    else:
        print(f"✓  Traces will be written to: {trace_dir}")
        print()

    engine_kwargs = {
        "trust_remote_code": True,
        "enforce_eager": not args.no_enforce_eager,
        "gpu_memory_utilization": args.gpu_memory_utilization,
    }

    if args.mode == "random":
        run_random(
            model=args.model,
            num_prompts=args.num_prompts,
            prompt_len=args.prompt_len,
            decode_len=args.decode_len,
            temperature=args.temperature,
            seed=args.seed,
            **engine_kwargs,
        )
    elif args.mode == "sharegpt":
        run_sharegpt(
            model=args.model,
            num_prompts=args.num_prompts,
            max_decode_len=args.max_decode_len,
            sharegpt_path=args.sharegpt_path,
            temperature=args.temperature,
            seed=args.seed,
            **engine_kwargs,
        )
    else:
        parser.error(f"Unknown mode: {args.mode}")

    # Confirm output.
    if trace_dir:
        trace_path = Path(trace_dir) / "traces.jsonl"
        meta_path = Path(trace_dir) / "metadata.json"
        if trace_path.exists():
            n_lines = len(trace_path.read_text().strip().splitlines())
            print(f"\n✓ traces.jsonl: {n_lines} records")
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            print(f"✓ metadata.json: {meta.get('num_layers')} layers, "
                  f"{meta.get('num_steps')} steps")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
