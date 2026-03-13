"""
Portable benchmark runner for repo W4A16 kernels.

This script is designed to be easy to run on another CUDA machine, including
consumer GPUs such as an RTX 3070. It supports:

1. Synthetic kernel microbenchmarks.
2. Optional Hugging Face full-model regular-vs-quantized benchmarking.

Dtype policy:
- Dense baselines use BF16 when `torch.cuda.is_bf16_supported()` is true.
- Otherwise dense baselines fall back to FP16 on CUDA.
- The repo CUDA kernel itself still uses BF16 tensors because that is the
  extension's current contract, so quantized kernel paths explicitly report the
  dtype they use.

Examples:
  python portable_benchmark.py --mode synthetic --results-json portable_results.json
  python portable_benchmark.py --model-checkpoint meta-llama/Meta-Llama-3.1-8B --hf-token "$HF_TOKEN"
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load as torch_extension_load

from quantization import (
    CudaDirectQuantizedLinear4bit,
    CudaKernelQuantizedLinear4bit,
    dequantize_weights,
    load_w4a16_cuda_direct_extension,
    load_w4a16_cuda_extension,
    pack_rows_4,
    quantize_model_layers,
)


DEFAULT_OFEATURES = [4096, 8192, 16384, 32768]
AWQ_SOURCE_PATH = Path(__file__).with_name("awq_kernel.cu")



def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()



def warmup(fn, n: int = 10):
    for _ in range(n):
        fn()
    sync()



def measure_ms(fn, n: int) -> tuple[float, float]:
    times = []
    for _ in range(n):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        times.append((time.perf_counter() - t0) * 1_000)
    return statistics.mean(times), statistics.stdev(times) if len(times) > 1 else 0.0



def metric_to_json(metric: tuple[float, float]) -> dict[str, float]:
    return {"mean": metric[0], "std": metric[1]}



def dtype_name(dtype: torch.dtype | None) -> str | None:
    if dtype is None:
        return None
    if dtype == torch.bfloat16:
        return "bfloat16"
    if dtype == torch.float16:
        return "float16"
    if dtype == torch.float32:
        return "float32"
    return str(dtype)



def select_dense_dtype(device: torch.device) -> torch.dtype:
    if device.type != "cuda":
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16



def get_env_metadata(device: torch.device, selected_dtype: torch.dtype) -> dict[str, object]:
    metadata: dict[str, object] = {
        "benchmark_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "torch_version": torch.__version__,
        "device": str(device),
        "selected_dtype": dtype_name(selected_dtype),
        "repo_kernel_input_dtype": "bfloat16",
        "kernel_accumulation_dtype": "fp32",
    }
    if torch.cuda.is_available() and device.type == "cuda":
        metadata["gpu_name"] = torch.cuda.get_device_name(device)
        capability = torch.cuda.get_device_capability(device)
        metadata["cuda_capability"] = f"{capability[0]}.{capability[1]}"
        metadata["bf16_supported"] = torch.cuda.is_bf16_supported()
    else:
        metadata["gpu_name"] = None
        metadata["cuda_capability"] = None
        metadata["bf16_supported"] = False
    return metadata



def interleave_transposed_s_z(S: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
    OF, G = S.shape
    S_t = S.t().contiguous()
    Z_t = Z.t().contiguous()
    SZ = torch.empty((G, 2 * OF), device=S.device, dtype=S.dtype)
    SZ[:, 0::2] = S_t
    SZ[:, 1::2] = Z_t
    return SZ.contiguous()



def dequantize_from_sz(W_packed: torch.Tensor, SZ: torch.Tensor, group_size: int, in_features: int) -> torch.Tensor:
    of = W_packed.shape[0] * 4
    groups = in_features // group_size
    S = SZ[:, 0::2].t().contiguous().view(of, groups)
    Z = SZ[:, 1::2].t().contiguous().view(of, groups)
    return dequantize_weights(W_packed, S, Z, group_size, in_features)



def raw_cuda_w4a16(W, b, SZ, group_size, activations, cuda_ext):
    return cuda_ext.forward(W.contiguous(), b.contiguous(), SZ.contiguous(), activations.contiguous(), group_size)



def load_awq_extension(verbose: bool = False):
    if not AWQ_SOURCE_PATH.exists():
        raise FileNotFoundError(f"AWQ source not found at {AWQ_SOURCE_PATH}")
    return build_extension(
        name="awq_cuda_ext_portable_v1",
        sources=[str(AWQ_SOURCE_PATH)],
        extra_cuda_cflags=["-O3", "--use_fast_math"],
        verbose=verbose,
    )


def build_extension(name: str, sources: list[str], extra_cuda_cflags: list[str] | None = None, verbose: bool = False):
    kwargs = {"name": name, "sources": sources, "verbose": verbose}
    if extra_cuda_cflags is not None:
        kwargs["extra_cuda_cflags"] = extra_cuda_cflags
    return torch_extension_load(**kwargs)



def create_awq_inputs(of: int, ifeatures: int, group_size: int, device: torch.device):
    pack_factor = 8
    W_awq = torch.randint(0, 255, (of, ifeatures // 2), device=device, dtype=torch.uint8).contiguous()
    S_awq = torch.ones((of, ifeatures // group_size), device=device, dtype=torch.float16).contiguous()
    Z_int = torch.randint(0, 16, (of, ifeatures // group_size), device=device, dtype=torch.int32).contiguous()
    Z_awq_int32 = torch.zeros((of, (ifeatures // group_size) // pack_factor), device=device, dtype=torch.int32)
    for i in range(pack_factor):
        Z_awq_int32 |= (Z_int[:, i::pack_factor] & 0xF) << (i * 4)
    Z_awq = Z_awq_int32.view(torch.uint8).contiguous()
    return W_awq, Z_awq, S_awq



def run_synthetic_case(
    of: int,
    ifeatures: int,
    n_iters: int,
    group_size: int,
    device: torch.device,
    dense_dtype: torch.dtype,
    repo_cuda_ext,
    direct_cuda_ext,
    awq_ext=None,
):
    W_q = torch.randint(0, 16, (of, ifeatures), device=device, dtype=torch.uint8).contiguous()
    W_packed = pack_rows_4(W_q)

    bias_bf16 = torch.randn((of,), device=device, dtype=torch.bfloat16).contiguous()
    groups = ifeatures // group_size
    S_bf16 = torch.ones((of, groups), device=device, dtype=torch.bfloat16).contiguous()
    Z_bf16 = torch.randint(0, 16, (of, groups), device=device, dtype=torch.bfloat16).contiguous()
    SZ_bf16 = interleave_transposed_s_z(S_bf16, Z_bf16)

    act_kernel_col = torch.randn((ifeatures, 1), device=device, dtype=torch.bfloat16).contiguous()
    act_kernel_rank3 = act_kernel_col.t().contiguous().view(1, 1, ifeatures)

    bias_dense = bias_bf16.to(dense_dtype if dense_dtype != torch.float32 else torch.float32)
    act_dense = act_kernel_col.to(dense_dtype if dense_dtype != torch.float32 else torch.float32)
    dense_weight = torch.randn((of, ifeatures), device=device, dtype=act_dense.dtype).contiguous()

    dequant_ref = dequantize_from_sz(W_packed, SZ_bf16, group_size, ifeatures).to(torch.bfloat16)
    torch_ref = torch.matmul(dequant_ref, act_kernel_col) + bias_bf16[:, None]
    repo_out = raw_cuda_w4a16(W_packed, bias_bf16, SZ_bf16, group_size, act_kernel_col, repo_cuda_ext)
    direct_out = direct_cuda_ext.forward(W_packed, bias_bf16, SZ_bf16, act_kernel_rank3, group_size)
    direct_out_col = direct_out.reshape(-1, of).transpose(0, 1).contiguous()

    repo_diff = (repo_out - torch_ref).abs()
    direct_diff = (direct_out_col - torch_ref).abs()

    dense_ms = measure_ms(lambda: torch.matmul(dense_weight, act_dense) + bias_dense[:, None], n=n_iters)
    repo_ms = measure_ms(lambda: raw_cuda_w4a16(W_packed, bias_bf16, SZ_bf16, group_size, act_kernel_col, repo_cuda_ext), n=n_iters)
    direct_ms = measure_ms(lambda: direct_cuda_ext.forward(W_packed, bias_bf16, SZ_bf16, act_kernel_rank3, group_size), n=n_iters)

    awq_ms = None
    if awq_ext is not None:
        W_awq, Z_awq, S_awq = create_awq_inputs(of, ifeatures, group_size, device)
        act_awq = act_kernel_col.t().to(torch.float16).contiguous()
        awq_ms = measure_ms(lambda: awq_ext.forward(act_awq, W_awq, Z_awq, S_awq, ifeatures, of, group_size), n=n_iters)

    case = {
        "of": of,
        "if": ifeatures,
        "dense_selected_ms": metric_to_json(dense_ms),
        "repo_kernel_ms": metric_to_json(repo_ms),
        "direct_kernel_ms": metric_to_json(direct_ms),
        "correctness": {
            "repo_kernel": {
                "max_abs_err": float(repo_diff.max().item()),
                "mean_abs_err": float(repo_diff.mean().item()),
            },
            "direct_kernel": {
                "max_abs_err": float(direct_diff.max().item()),
                "mean_abs_err": float(direct_diff.mean().item()),
            },
        },
        "repo_kernel_speedup_vs_dense": dense_ms[0] / repo_ms[0] if repo_ms[0] else 0.0,
        "direct_kernel_speedup_vs_dense": dense_ms[0] / direct_ms[0] if direct_ms[0] else 0.0,
    }
    if awq_ms is not None:
        case["awq_reference_ms"] = metric_to_json(awq_ms)
        case["awq_reference_speedup_vs_dense"] = dense_ms[0] / awq_ms[0] if awq_ms[0] else 0.0
    return case



def bench_torch_model_variant(model: nn.Module, input_ids: torch.Tensor, n_runs: int, device: torch.device):
    model.to(device).eval()
    input_ids = input_ids.to(device)
    with torch.no_grad():
        warmup(lambda: model(input_ids=input_ids, use_cache=False), n=3)
        result = measure_ms(lambda: model(input_ids=input_ids, use_cache=False), n_runs)
    model.cpu()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return result



def load_auto_model_for_causal_lm():
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM



def run_full_model_benchmark(
    model_checkpoint: str,
    hf_token: str | None,
    group_size: int,
    n_runs: int,
    device: torch.device,
    dense_dtype: torch.dtype,
    repo_cuda_ext,
    enable_direct_kernel: bool,
):
    AutoModelForCausalLM = load_auto_model_for_causal_lm()
    base_model = AutoModelForCausalLM.from_pretrained(
        model_checkpoint,
        dtype=dense_dtype,
        token=hf_token,
    )
    vocab_size = base_model.config.vocab_size
    input_ids = torch.randint(0, vocab_size, (1, 1), dtype=torch.long)

    results: dict[str, tuple[float, float]] = {}
    results["regular_model"] = bench_torch_model_variant(base_model, input_ids, n_runs, device)

    _, quant_model = quantize_model_layers(
        base_model,
        group_size,
        linear_cls=CudaKernelQuantizedLinear4bit,
        cuda_ext=repo_cuda_ext,
    )
    results["repo_kernel_quantized_model"] = bench_torch_model_variant(quant_model, input_ids, n_runs, device)

    if enable_direct_kernel:
        _, direct_model = quantize_model_layers(
            base_model,
            group_size,
            linear_cls=CudaDirectQuantizedLinear4bit,
            cuda_ext=load_w4a16_cuda_direct_extension(verbose=False),
        )
        results["direct_kernel_quantized_model"] = bench_torch_model_variant(direct_model, input_ids, n_runs, device)

    del base_model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return results



def parse_ofeatures(args) -> list[int]:
    if args.ofeatures_list:
        return [int(item.strip()) for item in args.ofeatures_list.split(",") if item.strip()]
    if args.ofeatures is not None:
        return [args.ofeatures]
    return DEFAULT_OFEATURES



def resolve_mode(args) -> str:
    if args.mode is not None:
        return args.mode
    return "both" if args.model_checkpoint else "synthetic"



def default_results_path(mode: str) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return Path.cwd() / f"portable_benchmark_{mode}_{ts}.json"



def print_synthetic_summary(cases: list[dict]):
    print("\nSynthetic benchmark summary")
    print("-" * 72)
    for case in cases:
        parts = [
            f"OF={case['of']}",
            f"dense={case['dense_selected_ms']['mean']:.4f} ms",
            f"repo={case['repo_kernel_ms']['mean']:.4f} ms",
            f"direct={case['direct_kernel_ms']['mean']:.4f} ms",
        ]
        if "awq_reference_ms" in case:
            parts.append(f"awq={case['awq_reference_ms']['mean']:.4f} ms")
        print(", ".join(parts))



def print_model_summary(results: dict[str, tuple[float, float]]):
    print("\nFull-model benchmark summary")
    print("-" * 72)
    baseline = results["regular_model"][0]
    for name, metric in results.items():
        ratio = baseline / metric[0] if metric[0] else 0.0
        print(f"{name:<30} {metric[0]:>10.3f} ms  {ratio:>8.2f}x")



def parse_args():
    parser = argparse.ArgumentParser(description="Portable W4A16 benchmark runner.")
    parser.add_argument("--mode", choices=["synthetic", "full-model", "both"], default=None)
    parser.add_argument("--model-checkpoint", default=None)
    parser.add_argument("--hf-token", default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--ifeatures", type=int, default=8192)
    parser.add_argument("--ofeatures", type=int, default=None)
    parser.add_argument("--ofeatures-list", default=None)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--model-runs", type=int, default=30)
    parser.add_argument("--enable-direct-kernel", action="store_true")
    parser.add_argument("--enable-awq-reference", action="store_true")
    parser.add_argument("--results-json", default=None)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()



def main():
    args = parse_args()
    mode = resolve_mode(args)
    device = torch.device(args.device)
    selected_dtype = select_dense_dtype(device)
    torch.manual_seed(args.seed)

    results_path = Path(args.results_json) if args.results_json else default_results_path(mode)

    payload: dict[str, object] = {
        "mode": mode,
        **get_env_metadata(device, selected_dtype),
        "group_size": args.group_size,
        "seed": args.seed,
        "notes": [
            "Dense baselines use selected_dtype. Repo CUDA kernel paths use BF16 tensors because that is the current kernel contract.",
            "RTX 3070-class consumer GPUs commonly run best in FP16 for dense baselines, which is why the script auto-falls back when BF16 is not reported as supported.",
            "This script is based on the structure of previous_benchmarking.py, but rewritten to preserve the repo kernel contract, avoid extension-name mismatches, avoid hardcoded output paths, and provide a cleaner CLI.",
        ],
    }

    repo_cuda_ext = None
    direct_cuda_ext = None
    awq_ext = None

    if device.type == "cuda":
        repo_cuda_ext = load_w4a16_cuda_extension(verbose=False)
        if args.enable_direct_kernel:
            direct_cuda_ext = load_w4a16_cuda_direct_extension(verbose=False)
        else:
            direct_cuda_ext = load_w4a16_cuda_direct_extension(verbose=False)
        if args.enable_awq_reference:
            awq_ext = load_awq_extension(verbose=False)

    if mode in {"synthetic", "both"}:
        if repo_cuda_ext is None or direct_cuda_ext is None:
            raise RuntimeError("Synthetic mode requires CUDA and the repo kernel extensions to load successfully")
        ofeatures = parse_ofeatures(args)
        cases = []
        for of in ofeatures:
            case = run_synthetic_case(
                of=of,
                ifeatures=args.ifeatures,
                n_iters=args.iters,
                group_size=args.group_size,
                device=device,
                dense_dtype=selected_dtype,
                repo_cuda_ext=repo_cuda_ext,
                direct_cuda_ext=direct_cuda_ext,
                awq_ext=awq_ext,
            )
            cases.append(case)
        payload["cases"] = cases
        print_synthetic_summary(cases)

    if mode in {"full-model", "both"}:
        if not args.model_checkpoint:
            raise ValueError("--model-checkpoint is required for full-model mode")
        if repo_cuda_ext is None:
            raise RuntimeError("Full-model mode requires CUDA and the repo kernel extension")
        try:
            model_results = run_full_model_benchmark(
                model_checkpoint=args.model_checkpoint,
                hf_token=args.hf_token,
                group_size=args.group_size,
                n_runs=args.model_runs,
                device=device,
                dense_dtype=selected_dtype,
                repo_cuda_ext=repo_cuda_ext,
                enable_direct_kernel=args.enable_direct_kernel,
            )
            payload["full_model_results_ms"] = {name: metric_to_json(metric) for name, metric in model_results.items()}
            print_model_summary(model_results)
        except Exception as exc:
            payload["full_model_error"] = {
                "type": type(exc).__name__,
                "message": str(exc),
            }
            print(f"[WARN] Full-model benchmark failed: {type(exc).__name__}: {exc}")

    results_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {results_path}")


if __name__ == "__main__":
    main()
