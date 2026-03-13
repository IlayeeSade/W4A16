"""
Forward-pass speed benchmark for regular vs W4A16 model variants.

This script benchmarks:
1. A single linear layer.
2. A full causal LM forward pass for one decode token.
3. Optionally, one model-layer profile using real decode activations.
4. Optionally, an LMDeploy AWQ model for speed-only generation latency.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont

from quantization import (
    CudaDirectQuantizedLinear4bit,
    CudaKernelQuantizedLinear4bit,
    load_w4a16_cuda_direct_extension,
    load_w4a16_cuda_extension,
    quantize_model_layers,
)


VARIANT_LABELS = {
    "regular_bf16": "Regular BF16",
    "repo_cuda_kernel_w4a16": "Repo CUDA Kernel W4A16",
    "repo_direct_cuda_w4a16": "Repo Direct-Input CUDA W4A16",
    "lmdeploy_awq_pytorch": "LMDeploy AWQ (PyTorch)",
    "lmdeploy_awq_turbomind": "LMDeploy AWQ (TurboMind)",
}
ZERO_METRIC = (0.0, 0.0)


def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()



def warmup(fn, n: int = 5):
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



def breakdown_to_json(breakdown: dict[str, tuple[float, float]]) -> dict[str, dict[str, float]]:
    return {name: metric_to_json(metric) for name, metric in breakdown.items()}



def load_json(path: str | Path | None) -> dict | None:
    if not path:
        return None
    path = Path(path)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None



def resolve_baseline_payload(
    explicit_path: str | None,
    results_path: str | None,
    default_name: str,
) -> tuple[str | None, dict | None]:
    candidates = []
    if explicit_path:
        candidates.append(Path(explicit_path))
    if results_path:
        candidates.append(Path(results_path))
    candidates.append(Path(__file__).with_name(default_name))

    seen = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        payload = load_json(candidate)
        if payload is not None:
            return str(candidate), payload
    return None, None



def compare_result_sections(
    current: dict[str, tuple[float, float]],
    baseline: dict[str, dict[str, float]] | None,
) -> dict[str, dict[str, float]]:
    if not baseline:
        return {}

    comparison = {}
    for name, metric in current.items():
        baseline_metric = baseline.get(name)
        if not baseline_metric:
            continue
        baseline_mean = baseline_metric.get("mean")
        if baseline_mean is None:
            continue
        current_mean = metric[0]
        comparison[name] = {
            "current_mean": current_mean,
            "baseline_mean": baseline_mean,
            "delta_ms": current_mean - baseline_mean,
            "ratio_vs_baseline": (current_mean / baseline_mean) if baseline_mean else 0.0,
            "speedup_vs_baseline": (baseline_mean / current_mean) if current_mean else 0.0,
        }
    return comparison



def get_env_metadata(device: torch.device) -> dict[str, object]:
    metadata: dict[str, object] = {
        "kernel_accumulation_dtype": "fp32",
        "benchmark_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "torch_version": torch.__version__,
        "device": str(device),
    }
    if torch.cuda.is_available() and device.type == "cuda":
        metadata["gpu_name"] = torch.cuda.get_device_name(device)
        capability = torch.cuda.get_device_capability(device)
        metadata["cuda_capability"] = f"{capability[0]}.{capability[1]}"
    else:
        metadata["gpu_name"] = None
        metadata["cuda_capability"] = None
    return metadata



def unload_model(model: nn.Module):
    model.cpu()
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()



def load_auto_model_for_causal_lm():
    from transformers import AutoModelForCausalLM

    return AutoModelForCausalLM



def get_module_by_name(model: nn.Module, module_name: str) -> nn.Module:
    modules = dict(model.named_modules())
    if module_name not in modules:
        raise KeyError(f"Module '{module_name}' not found")
    return modules[module_name]



def pick_profile_layer_name(model: nn.Module, requested_name: str) -> str:
    if requested_name != "auto":
        get_module_by_name(model, requested_name)
        return requested_name

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            return name
    raise ValueError("Could not auto-select a linear layer to profile")



def capture_layer_input(
    model: nn.Module,
    layer_name: str,
    input_ids: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    model.to(device).eval()
    layer = get_module_by_name(model, layer_name)
    captured: dict[str, torch.Tensor] = {}

    def hook(_module, args):
        if args:
            captured["x"] = args[0].detach().clone()

    handle = layer.register_forward_pre_hook(hook)
    try:
        with torch.no_grad():
            model(input_ids=input_ids.to(device), use_cache=False)
    finally:
        handle.remove()

    if "x" not in captured:
        raise RuntimeError(f"Did not capture inputs for layer '{layer_name}'")
    return captured["x"]



def bench_regular_linear_breakdown(module: nn.Module, x: torch.Tensor, n_runs: int):
    warmup(lambda: module(x))
    end_to_end = measure_ms(lambda: module(x), n_runs)
    return {
        "input_prep": ZERO_METRIC,
        "kernel_call": end_to_end,
        "output_postprocess": ZERO_METRIC,
        "end_to_end": end_to_end,
    }



def bench_repo_cuda_breakdown(module: CudaKernelQuantizedLinear4bit, x: torch.Tensor, n_runs: int):
    prep_fn = lambda: module._kernel_input(x)
    warmup(prep_fn)
    input_prep = measure_ms(prep_fn, n_runs)
    prepared_input = prep_fn()

    kernel_fn = lambda: module.cuda_ext.forward(
        module.W_packed,
        module._kernel_bias(),
        module.SZ_packed,
        prepared_input,
        module.group_size,
    )
    warmup(kernel_fn)
    kernel_call = measure_ms(kernel_fn, n_runs)
    raw_out = kernel_fn()

    post_fn = lambda: raw_out.transpose(0, 1).contiguous().view(*x.shape[:-1], module.out_features)
    warmup(post_fn)
    output_postprocess = measure_ms(post_fn, n_runs)

    warmup(lambda: module(x))
    end_to_end = measure_ms(lambda: module(x), n_runs)

    return {
        "input_prep": input_prep,
        "kernel_call": kernel_call,
        "output_postprocess": output_postprocess,
        "end_to_end": end_to_end,
    }



def bench_direct_cuda_breakdown(module: CudaDirectQuantizedLinear4bit, x: torch.Tensor, n_runs: int):
    prep_fn = lambda: x if x.dtype == torch.bfloat16 and x.is_contiguous() else x.to(torch.bfloat16).contiguous()
    warmup(prep_fn)
    input_prep = measure_ms(prep_fn, n_runs)
    prepared_input = prep_fn()

    kernel_fn = lambda: module.cuda_ext.forward(
        module.W_packed,
        module._kernel_bias(),
        module.SZ_packed,
        prepared_input,
        module.group_size,
    )
    warmup(kernel_fn)
    kernel_call = measure_ms(kernel_fn, n_runs)

    warmup(lambda: module(x))
    end_to_end = measure_ms(lambda: module(x), n_runs)

    return {
        "input_prep": input_prep,
        "kernel_call": kernel_call,
        "output_postprocess": ZERO_METRIC,
        "end_to_end": end_to_end,
    }



def bench_single_linear(
    in_features: int,
    out_features: int,
    group_size: int,
    n_runs: int,
    device: torch.device,
    cuda_ext=None,
    direct_cuda_ext=None,
):
    print("\n" + "=" * 72)
    print("BENCHMARK: Single Linear Layer")
    print("=" * 72)

    regular = nn.Linear(in_features, out_features, bias=True).to(torch.bfloat16).to(device).eval()
    cuda_quant = None
    if cuda_ext is not None and device.type == "cuda":
        cuda_quant = CudaKernelQuantizedLinear4bit.from_linear(
            copy.deepcopy(regular).cpu(),
            group_size,
            cuda_ext=cuda_ext,
        ).to(device).eval()
    direct_cuda_quant = None
    if direct_cuda_ext is not None and device.type == "cuda":
        direct_cuda_quant = CudaDirectQuantizedLinear4bit.from_linear(
            copy.deepcopy(regular).cpu(),
            group_size,
            cuda_ext=direct_cuda_ext,
        ).to(device).eval()

    x = torch.randn(1, 1, in_features, dtype=torch.bfloat16, device=device)
    results: dict[str, tuple[float, float]] = {}
    breakdowns: dict[str, dict[str, tuple[float, float]]] = {}

    with torch.no_grad():
        breakdowns["regular_bf16"] = bench_regular_linear_breakdown(regular, x, n_runs)
        results["regular_bf16"] = breakdowns["regular_bf16"]["end_to_end"]

        if cuda_quant is not None:
            breakdowns["repo_cuda_kernel_w4a16"] = bench_repo_cuda_breakdown(cuda_quant, x, n_runs)
            results["repo_cuda_kernel_w4a16"] = breakdowns["repo_cuda_kernel_w4a16"]["end_to_end"]

        if direct_cuda_quant is not None:
            breakdowns["repo_direct_cuda_w4a16"] = bench_direct_cuda_breakdown(direct_cuda_quant, x, n_runs)
            results["repo_direct_cuda_w4a16"] = breakdowns["repo_direct_cuda_w4a16"]["end_to_end"]

    print_results_table(results)
    print_breakdown_table(breakdowns)
    return results, breakdowns



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



def bench_lmdeploy_variant(
    model_path: str,
    prompt: str,
    n_runs: int,
    backend: str,
):
    from lmdeploy import GenerationConfig, PytorchEngineConfig, TurbomindEngineConfig, pipeline

    if backend == "pytorch":
        backend_config = PytorchEngineConfig(dtype="bfloat16", device_type="cuda")
    else:
        backend_config = TurbomindEngineConfig(dtype="bfloat16")

    pipe = pipeline(model_path, backend_config=backend_config, log_level="ERROR")
    gen_config = GenerationConfig(max_new_tokens=1, do_sample=False)

    def run_once():
        pipe([prompt], gen_config=gen_config)

    warmup(run_once, n=2)
    result = measure_ms(run_once, n_runs)

    del pipe
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return result



def profile_model_layer(
    regular_model: nn.Module,
    input_ids: torch.Tensor,
    layer_name: str,
    n_runs: int,
    device: torch.device,
    cuda_quant_model: nn.Module | None = None,
    direct_cuda_quant_model: nn.Module | None = None,
):
    print(f"\nProfiling model layer: {layer_name}")
    captured_x = capture_layer_input(regular_model, layer_name, input_ids, device)
    regular_layer = get_module_by_name(regular_model, layer_name)
    results: dict[str, dict[str, tuple[float, float]]] = {
        "regular_bf16": bench_regular_linear_breakdown(regular_layer, captured_x, n_runs)
    }
    input_shape = list(captured_x.shape)

    regular_model.cpu()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()

    if cuda_quant_model is not None:
        cuda_quant_model.to(device).eval()
        quant_layer = get_module_by_name(cuda_quant_model, layer_name)
        results["repo_cuda_kernel_w4a16"] = bench_repo_cuda_breakdown(quant_layer, captured_x, n_runs)
        cuda_quant_model.cpu()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    if direct_cuda_quant_model is not None:
        direct_cuda_quant_model.to(device).eval()
        direct_layer = get_module_by_name(direct_cuda_quant_model, layer_name)
        results["repo_direct_cuda_w4a16"] = bench_direct_cuda_breakdown(direct_layer, captured_x, n_runs)
        direct_cuda_quant_model.cpu()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    print_breakdown_table(results)
    return {
        "layer_name": layer_name,
        "input_shape": input_shape,
        "variants": results,
    }



def bench_full_model(
    model_checkpoint: str,
    hf_token: str | None,
    group_size: int,
    n_runs: int,
    device: torch.device,
    cuda_ext=None,
    direct_cuda_ext=None,
    lmdeploy_model_path: str | None = None,
    lmdeploy_backend: str = "pytorch",
    profile_layer_name: str | None = None,
    profile_layer_runs: int = 50,
):
    print("\n" + "=" * 72)
    print("BENCHMARK: Full Model Decode Forward")
    print("=" * 72)
    print(f"model       : {model_checkpoint}")
    print(f"group_size  : {group_size}")
    print(f"device      : {device}")
    print(f"repetitions : {n_runs}")

    print("\nLoading model to CPU ...")
    AutoModelForCausalLM = load_auto_model_for_causal_lm()
    base_model = AutoModelForCausalLM.from_pretrained(
        model_checkpoint,
        dtype=torch.bfloat16,
        token=hf_token,
    )
    vocab_size = base_model.config.vocab_size
    print("Model loaded.")

    regular_model = base_model

    cuda_quant_model = None
    if cuda_ext is not None and device.type == "cuda":
        print("Quantizing CUDA-kernel W4A16 model ...")
        _, cuda_quant_model = quantize_model_layers(
            base_model,
            group_size,
            linear_cls=CudaKernelQuantizedLinear4bit,
            cuda_ext=cuda_ext,
        )

    direct_cuda_quant_model = None
    if direct_cuda_ext is not None and device.type == "cuda":
        print("Quantizing direct-input CUDA-kernel W4A16 model ...")
        _, direct_cuda_quant_model = quantize_model_layers(
            base_model,
            group_size,
            linear_cls=CudaDirectQuantizedLinear4bit,
            cuda_ext=direct_cuda_ext,
        )

    input_ids = torch.randint(0, vocab_size, (1, 1), dtype=torch.long)
    results = {}
    layer_profile = None

    if profile_layer_name:
        resolved_layer_name = pick_profile_layer_name(regular_model, profile_layer_name)
        layer_profile = profile_model_layer(
            regular_model=regular_model,
            input_ids=input_ids,
            layer_name=resolved_layer_name,
            n_runs=profile_layer_runs,
            device=device,
            cuda_quant_model=cuda_quant_model,
            direct_cuda_quant_model=direct_cuda_quant_model,
        )

    print("\nBenchmarking regular model ...")
    results["regular_bf16"] = bench_torch_model_variant(regular_model, input_ids, n_runs, device)

    if cuda_quant_model is not None:
        print("Benchmarking CUDA-kernel W4A16 model ...")
        results["repo_cuda_kernel_w4a16"] = bench_torch_model_variant(cuda_quant_model, input_ids, n_runs, device)
        del cuda_quant_model
        gc.collect()

    if direct_cuda_quant_model is not None:
        print("Benchmarking direct-input CUDA-kernel W4A16 model ...")
        results["repo_direct_cuda_w4a16"] = bench_torch_model_variant(
            direct_cuda_quant_model,
            input_ids,
            n_runs,
            device,
        )
        del direct_cuda_quant_model
        gc.collect()

    if lmdeploy_model_path:
        print(f"Benchmarking LMDeploy model from {lmdeploy_model_path} ...")
        results[f"lmdeploy_awq_{lmdeploy_backend}"] = bench_lmdeploy_variant(
            lmdeploy_model_path,
            prompt="Benchmark prompt",
            n_runs=n_runs,
            backend=lmdeploy_backend,
        )

    print_results_table(results)
    return results, layer_profile



def print_results_table(results: dict[str, tuple[float, float]]):
    print(f"\n{'Variant':<24} {'Mean (ms)':>12} {'Std (ms)':>12} {'vs regular':>12}")
    print("-" * 64)
    regular_mean = results["regular_bf16"][0]
    for name, (mean_ms, std_ms) in results.items():
        relative = regular_mean / mean_ms if mean_ms else 0.0
        label = VARIANT_LABELS.get(name, name.replace("_", " "))
        print(f"{label:<24} {mean_ms:>12.3f} {std_ms:>12.3f} {relative:>12.2f}x")



def print_breakdown_table(breakdowns: dict[str, dict[str, tuple[float, float]]]):
    print(
        f"\n{'Variant':<24} {'Prep (ms)':>12} {'Kernel (ms)':>12} "
        f"{'Post (ms)':>12} {'End-to-end':>12}"
    )
    print("-" * 76)
    for name, metrics in breakdowns.items():
        label = VARIANT_LABELS.get(name, name.replace("_", " "))
        print(
            f"{label:<24} "
            f"{metrics['input_prep'][0]:>12.3f} "
            f"{metrics['kernel_call'][0]:>12.3f} "
            f"{metrics['output_postprocess'][0]:>12.3f} "
            f"{metrics['end_to_end'][0]:>12.3f}"
        )



def draw_bar_chart(draw, area, title, result_group):
    x0, y0, x1, y1 = area
    padding = 24
    chart_x0 = x0 + padding
    chart_y0 = y0 + 48
    chart_x1 = x1 - padding
    chart_y1 = y1 - 24
    font = ImageFont.load_default()

    draw.text((x0 + padding, y0 + 16), title, fill="#111111", font=font)

    labels = list(result_group.keys())
    values = [result_group[key]["mean"] for key in labels]
    max_value = max(values) if values else 1.0
    bar_gap = 18
    slot_height = (chart_y1 - chart_y0) / max(len(labels), 1)
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for idx, key in enumerate(labels):
        label = VARIANT_LABELS.get(key, key)
        mean = result_group[key]["mean"]
        top = chart_y0 + idx * slot_height + 8
        bottom = top + max(slot_height - bar_gap, 18)
        width = 0 if max_value == 0 else (mean / max_value) * (chart_x1 - chart_x0 - 160)
        bar_x0 = chart_x0 + 140
        bar_x1 = bar_x0 + width
        draw.text((chart_x0, top), label, fill="#333333", font=font)
        draw.rounded_rectangle((bar_x0, top, bar_x1, bottom), radius=6, fill=colors[idx % len(colors)])
        draw.text((bar_x1 + 8, top), f"{mean:.3f} ms", fill="#111111", font=font)



def write_plot(path: str | None, payload: dict):
    if not path:
        return

    img = Image.new("RGB", (1200, 760), "#f8f6f1")
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()

    draw.text((40, 24), "W4A16 Forward Benchmark", fill="#111111", font=font)
    draw.text(
        (40, 48),
        "Regular BF16 vs this repo's CUDA-kernel W4A16"
        + (" vs LMDeploy AWQ" if any(k.startswith("lmdeploy_awq") for k in payload["full_model_results_ms"]) else ""),
        fill="#444444",
        font=font,
    )

    draw_bar_chart(draw, (20, 90, 1180, 360), "Single Layer Latency", payload["layer_results_ms"])
    draw_bar_chart(draw, (20, 390, 1180, 700), "Full Model Decode Latency", payload["full_model_results_ms"])

    footer = (
        "Regular BF16 = baseline Hugging Face model. "
        "Repo CUDA Kernel W4A16 = local CUDA extension path. "
        "Repo Direct-Input CUDA W4A16 = same kernel with input reshaping moved into the extension. "
        "LMDeploy AWQ = optional external speed-only comparison."
    )
    draw.text((40, 720), footer, fill="#444444", font=font)
    img.save(path)
    print(f"Wrote benchmark plot to {path}")



def maybe_load_cuda_extension(enable: bool):
    if not enable:
        return None
    try:
        print("Building/loading local CUDA extension ...")
        return load_w4a16_cuda_extension(verbose=False)
    except Exception as exc:
        print(f"[WARN] CUDA extension unavailable: {type(exc).__name__}: {exc}")
        return None



def maybe_load_direct_cuda_extension(enable: bool):
    if not enable:
        return None
    try:
        print("Building/loading direct-input CUDA extension ...")
        return load_w4a16_cuda_direct_extension(verbose=False)
    except Exception as exc:
        print(f"[WARN] Direct-input CUDA extension unavailable: {type(exc).__name__}: {exc}")
        return None



def write_results(path: str | None, payload: dict):
    if not path:
        return
    Path(path).write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote benchmark results to {path}")



def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark regular vs W4A16 model variants.")
    parser.add_argument("--model-checkpoint", default="meta-llama/Meta-Llama-3.1-8B")
    parser.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--layer-runs", type=int, default=200)
    parser.add_argument("--model-runs", type=int, default=30)
    parser.add_argument("--profile-layer-runs", type=int, default=50)
    parser.add_argument("--layer-in-features", type=int, default=4096)
    parser.add_argument("--layer-out-features", type=int, default=4096)
    parser.add_argument("--enable-cuda-kernel", action="store_true")
    parser.add_argument("--enable-direct-cuda-kernel", action="store_true")
    parser.add_argument("--lmdeploy-model-path", default=None)
    parser.add_argument("--lmdeploy-backend", choices=["pytorch", "turbomind"], default="pytorch")
    parser.add_argument("--profile-layer-name", default=None, help="Module name to profile, or 'auto'.")
    parser.add_argument("--baseline-json", default=None)
    parser.add_argument("--results-json", default=None)
    parser.add_argument("--plot-path", default=None)
    return parser.parse_args()



def main():
    args = parse_args()
    device = torch.device(args.device)
    cuda_ext = maybe_load_cuda_extension(args.enable_cuda_kernel)
    direct_cuda_ext = maybe_load_direct_cuda_extension(args.enable_direct_cuda_kernel)
    baseline_path, baseline_payload = resolve_baseline_payload(args.baseline_json, args.results_json, "benchmark_results.json")

    print(f"Running on: {device}")
    if args.lmdeploy_model_path:
        print(f"LMDeploy comparison enabled: {args.lmdeploy_model_path}")
    if baseline_path:
        print(f"Baseline comparison source: {baseline_path}")

    layer_results, layer_breakdown = bench_single_linear(
        in_features=args.layer_in_features,
        out_features=args.layer_out_features,
        group_size=args.group_size,
        n_runs=args.layer_runs,
        device=device,
        cuda_ext=cuda_ext,
        direct_cuda_ext=direct_cuda_ext,
    )

    model_results = {}
    profiled_layer = None
    full_model_error = None
    try:
        model_results, profiled_layer = bench_full_model(
            model_checkpoint=args.model_checkpoint,
            hf_token=args.hf_token,
            group_size=args.group_size,
            n_runs=args.model_runs,
            device=device,
            cuda_ext=cuda_ext,
            direct_cuda_ext=direct_cuda_ext,
            lmdeploy_model_path=args.lmdeploy_model_path,
            lmdeploy_backend=args.lmdeploy_backend,
            profile_layer_name=args.profile_layer_name,
            profile_layer_runs=args.profile_layer_runs,
        )
    except Exception as exc:
        full_model_error = {
            "type": type(exc).__name__,
            "message": str(exc),
        }
        print(f"[WARN] Full-model benchmark failed: {type(exc).__name__}: {exc}")

    payload = {
        "model_checkpoint": args.model_checkpoint,
        **get_env_metadata(device),
        "group_size": args.group_size,
        "layer_results_ms": {key: metric_to_json(value) for key, value in layer_results.items()},
        "layer_breakdown_ms": {key: breakdown_to_json(value) for key, value in layer_breakdown.items()},
        "full_model_results_ms": {key: metric_to_json(value) for key, value in model_results.items()},
        "variant_descriptions": {
            "regular_bf16": "Unquantized Hugging Face model in BF16.",
            "repo_cuda_kernel_w4a16": "This repo's W4A16 path using the local CUDA kernel from w4a16_cuda.cu.",
            "repo_direct_cuda_w4a16": "Same kernel structure, but wrapped by w4a16_cuda_direct.cu so Python forward passes the original activation tensor directly.",
            "lmdeploy_awq_pytorch": "LMDeploy AWQ model using LMDeploy's PyTorch backend.",
            "lmdeploy_awq_turbomind": "LMDeploy AWQ model using LMDeploy's TurboMind backend.",
        },
        "notes": [
            "LMDeploy comparison is speed-only and measures end-to-end generation latency for max_new_tokens=1.",
            "Regular BF16 / repo CUDA-kernel W4A16 / repo direct-input CUDA W4A16 comparisons measure raw torch forward latency for input_ids shape [1, 1].",
            "layer_breakdown_ms isolates input preparation, CUDA extension time, and output post-processing for the single-layer benchmark.",
        ],
    }
    if profiled_layer is not None:
        payload["profiled_model_layer_ms"] = {
            "layer_name": profiled_layer["layer_name"],
            "input_shape": profiled_layer["input_shape"],
            "variants": {
                key: breakdown_to_json(value) for key, value in profiled_layer["variants"].items()
            },
        }
    if full_model_error is not None:
        payload["full_model_error"] = full_model_error
    if baseline_payload is not None:
        payload["comparison_to_previous"] = {
            "baseline_path": baseline_path,
            "layer_results_ms": compare_result_sections(layer_results, baseline_payload.get("layer_results_ms")),
            "full_model_results_ms": compare_result_sections(model_results, baseline_payload.get("full_model_results_ms")),
        }

    write_results(args.results_json, payload)
    write_plot(args.plot_path, payload)


if __name__ == "__main__":
    main()
