from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import statistics
import time

import torch
from torch.nn import functional as F

from .attention import ModelConfig, TinyCausalLM, attention, cached_decode, full_recomputation


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def measure(function, device: torch.device, warmup: int, repeats: int) -> dict:
    for _ in range(warmup):
        function()
    synchronize(device)
    samples = []
    for _ in range(repeats):
        synchronize(device)
        started = time.perf_counter_ns()
        function()
        synchronize(device)
        samples.append((time.perf_counter_ns() - started) / 1_000_000)
    ordered = sorted(samples)
    return {
        "median_ms": statistics.median(samples),
        "p95_ms": ordered[max(0, int(0.95 * len(ordered) + 0.999999) - 1)],
        "samples_ms": samples,
    }


def flash_probe() -> dict:
    if not torch.cuda.is_available():
        return {"status": "skipped", "reason": "CUDA unavailable; CPU/MPS SDPA is not a FlashAttention GPU result."}
    from torch.nn.attention import SDPBackend, sdpa_kernel
    try:
        q = torch.randn(1, 4, 128, 32, device="cuda", dtype=torch.float16)
        with torch.inference_mode(), sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            F.scaled_dot_product_attention(q, q, q, is_causal=True)
            torch.cuda.synchronize()
        return {"status": "passed", "scope": "forced CUDA FlashAttention backend smoke check only; not timed"}
    except RuntimeError as error:
        return {"status": "unsupported", "reason": str(error)}


def run(args: argparse.Namespace) -> dict:
    if args.prompt < 1 or args.steps < 1 or args.repeats < 1 or args.warmup < 0 or args.threads < 1:
        raise ValueError("prompt, steps, repeats and threads must be positive; warmup nonnegative")
    if args.device == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS was selected but is unavailable")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was selected but is unavailable")
    device = torch.device(args.device)
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed)
    config = ModelConfig(max_sequence=max(512, args.prompt + args.steps))
    model = TinyCausalLM(config).to(device).eval()
    tokens = torch.randint(0, config.vocab_size, (1, args.prompt + args.steps), device=device)
    timings = {}
    with torch.inference_mode():
        reference = full_recomputation(model, tokens, args.prompt, args.steps, "manual")
        correctness = {}
        for backend in ("manual", "sdpa"):
            full = full_recomputation(model, tokens, args.prompt, args.steps, backend)
            cached = cached_decode(model, tokens, args.prompt, args.steps, backend)
            torch.testing.assert_close(full, cached, atol=3e-6, rtol=3e-5)
            torch.testing.assert_close(full, reference, atol=3e-6, rtol=3e-5)
            correctness[backend] = {
                "cache_vs_full_max_abs_error": (cached - full).abs().max().item(),
                "vs_manual_full_max_abs_error": (full - reference).abs().max().item(),
                "atol": 3e-6,
                "rtol": 3e-5,
                "status": "passed",
            }
            full_timing = measure(lambda: full_recomputation(model, tokens, args.prompt, args.steps, backend), device, args.warmup, args.repeats)
            cache_timing = measure(lambda: cached_decode(model, tokens, args.prompt, args.steps, backend), device, args.warmup, args.repeats)
            timings[backend] = {
                "full_recomputation": full_timing,
                "preallocated_kv_cache": cache_timing,
                "full_median_divided_by_cached_median": full_timing["median_ms"] / cache_timing["median_ms"],
            }
        operator_timings = {}
        for phase, query_length, key_length in (("prefill", args.prompt, args.prompt), ("one_token_decode", 1, args.prompt + args.steps - 1)):
            q = torch.randn(1, config.heads, query_length, config.width // config.heads, device=device)
            k = torch.randn(1, config.heads, key_length, config.width // config.heads, device=device)
            v = torch.randn_like(k)
            offset = key_length - query_length
            manual = attention(q, k, v, "manual", offset)
            sdpa = attention(q, k, v, "sdpa", offset)
            torch.testing.assert_close(manual, sdpa, atol=3e-6, rtol=3e-5)
            operator_timings[phase] = {
                "shape_b_h_l_s_d": [1, config.heads, query_length, key_length, config.width // config.heads],
                "max_abs_error": (manual - sdpa).abs().max().item(),
                **{backend: measure(lambda backend=backend: attention(q, k, v, backend, offset), device, args.warmup, args.repeats) for backend in ("manual", "sdpa")},
            }
        cache = model.new_cache(1, args.prompt + args.steps)
    element_size = model.token_embedding.weight.element_size()
    formula_bytes = 2 * config.layers * 1 * config.heads * (args.prompt + args.steps) * (config.width // config.heads) * element_size
    return {
        "schema_version": 1,
        "kind": "tiny_random_model_correctness_and_local_timing",
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "environment": {
            "platform": platform.platform(), "machine": platform.machine(),
            "processor": platform.processor(), "python": platform.python_version(),
            "torch": torch.__version__, "device": str(device), "dtype": "float32",
            "intraop_threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(),
            "cuda_available": torch.cuda.is_available(), "mps_available": torch.backends.mps.is_available(),
        },
        "config": {"seed": args.seed, "prompt_tokens": args.prompt, "decode_steps": args.steps, "batch": 1, "warmup": args.warmup, "repeats": args.repeats, "model": asdict(config)},
        "timing_scope": "Each decode sample includes prefill, all decode steps, logits, and cache allocation when cached. Fixed teacher-forced tokens; no sampling, model downloads, batching, or serving scheduler. SDPA selects its local backend; no GPU backend inference is made from its name.",
        "correctness": correctness,
        "decode_timing": timings,
        "attention_operator_timing": operator_timings,
        "kv_memory": {
            "capacity_tokens": args.prompt + args.steps,
            "kv_tensor_payload_bytes": cache.allocated_tensor_bytes,
            "formula_bytes": formula_bytes,
            "formula": "2 * layers * batch * kv_heads * capacity * head_dim * dtype_bytes",
            "note": "Tensor element payload, validated by numel * element_size; excludes allocator metadata, model weights, temporary activations and process RSS. Dense tensor cache is separate from the C++ paged allocator.",
        },
        "forced_flash_attention_probe": flash_probe(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure causal attention correctness and CPU/MPS decode timings")
    parser.add_argument("--device", choices=("cpu", "mps", "cuda"), default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prompt", type=int, default=128)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("results/local_cpu.json"))
    args = parser.parse_args()
    report = run(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "correctness": report["correctness"], "median_decode_ms": {name: {"full": timing["full_recomputation"]["median_ms"], "cached": timing["preallocated_kv_cache"]["median_ms"]} for name, timing in report["decode_timing"].items()}, "flash_attention": report["forced_flash_attention_probe"]}, indent=2))


if __name__ == "__main__":
    main()
