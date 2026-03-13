"""
Forward-pass speed benchmark for regular vs W4A16 model variants.

This script benchmarks:
1. A single linear layer.
2. A full causal LM forward pass for one decode token.
3. Optionally, an LMDeploy AWQ model for speed-only generation latency.
"""

import argparse
import copy
import gc
import json
import os
import statistics
import time
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from transformers import AutoModelForCausalLM

from quantization import (
    CudaKernelQuantizedLinear4bit,
    load_w4a16_cuda_extension,
    quantize_model_layers,
)


VARIANT_LABELS = {
    "regular_bf16": "Regular BF16",
    "repo_cuda_kernel_w4a16": "Repo CUDA Kernel W4A16",
    "lmdeploy_awq_pytorch": "LMDeploy AWQ (PyTorch)",
    "lmdeploy_awq_turbomind": "LMDeploy AWQ (TurboMind)",
}


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


def unload_model(model: nn.Module):
    model.cpu()
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def bench_single_linear(
    in_features: int,
    out_features: int,
    group_size: int,
    n_runs: int,
    device: torch.device,
    cuda_ext=None,
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

    x = torch.randn(1, 1, in_features, dtype=torch.bfloat16, device=device)
    results = {}

    with torch.no_grad():
        warmup(lambda: regular(x))
        results["regular_bf16"] = measure_ms(lambda: regular(x), n_runs)

        if cuda_quant is not None:
            warmup(lambda: cuda_quant(x))
            results["repo_cuda_kernel_w4a16"] = measure_ms(lambda: cuda_quant(x), n_runs)

    print_results_table(results)
    return results


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


def bench_full_model(
    model_checkpoint: str,
    hf_token: str | None,
    group_size: int,
    n_runs: int,
    device: torch.device,
    cuda_ext=None,
    lmdeploy_model_path: str | None = None,
    lmdeploy_backend: str = "pytorch",
):
    print("\n" + "=" * 72)
    print("BENCHMARK: Full Model Decode Forward")
    print("=" * 72)
    print(f"model       : {model_checkpoint}")
    print(f"group_size  : {group_size}")
    print(f"device      : {device}")
    print(f"repetitions : {n_runs}")

    print("\nLoading model to CPU ...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_checkpoint,
        dtype=torch.bfloat16,
        token=hf_token,
        device_map="cpu",
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

    input_ids = torch.randint(0, vocab_size, (1, 1), dtype=torch.long)
    results = {}

    print("\nBenchmarking regular model ...")
    results["regular_bf16"] = bench_torch_model_variant(regular_model, input_ids, n_runs, device)

    if cuda_quant_model is not None:
        print("Benchmarking CUDA-kernel W4A16 model ...")
        results["repo_cuda_kernel_w4a16"] = bench_torch_model_variant(cuda_quant_model, input_ids, n_runs, device)
        del cuda_quant_model
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
    return results


def print_results_table(results: dict[str, tuple[float, float]]):
    print(f"\n{'Variant':<24} {'Mean (ms)':>12} {'Std (ms)':>12} {'vs regular':>12}")
    print("-" * 64)
    regular_mean = results["regular_bf16"][0]
    for name, (mean_ms, std_ms) in results.items():
        relative = regular_mean / mean_ms if mean_ms else 0.0
        label = VARIANT_LABELS.get(name, name.replace("_", " "))
        print(f"{label:<24} {mean_ms:>12.3f} {std_ms:>12.3f} {relative:>12.2f}x")


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
    parser.add_argument("--layer-in-features", type=int, default=4096)
    parser.add_argument("--layer-out-features", type=int, default=4096)
    parser.add_argument("--enable-cuda-kernel", action="store_true")
    parser.add_argument("--lmdeploy-model-path", default=None)
    parser.add_argument("--lmdeploy-backend", choices=["pytorch", "turbomind"], default="pytorch")
    parser.add_argument("--results-json", default=None)
    parser.add_argument("--plot-path", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    cuda_ext = maybe_load_cuda_extension(args.enable_cuda_kernel)

    print(f"Running on: {device}")
    if args.lmdeploy_model_path:
        print(f"LMDeploy comparison enabled: {args.lmdeploy_model_path}")

    layer_results = bench_single_linear(
        in_features=args.layer_in_features,
        out_features=args.layer_out_features,
        group_size=args.group_size,
        n_runs=args.layer_runs,
        device=device,
        cuda_ext=cuda_ext,
    )

    model_results = bench_full_model(
        model_checkpoint=args.model_checkpoint,
        hf_token=args.hf_token,
        group_size=args.group_size,
        n_runs=args.model_runs,
        device=device,
        cuda_ext=cuda_ext,
        lmdeploy_model_path=args.lmdeploy_model_path,
        lmdeploy_backend=args.lmdeploy_backend,
    )

    payload = {
        "model_checkpoint": args.model_checkpoint,
        "device": str(device),
        "group_size": args.group_size,
        "layer_results_ms": {key: {"mean": value[0], "std": value[1]} for key, value in layer_results.items()},
        "full_model_results_ms": {key: {"mean": value[0], "std": value[1]} for key, value in model_results.items()},
        "variant_descriptions": {
            "regular_bf16": "Unquantized Hugging Face model in BF16.",
            "repo_cuda_kernel_w4a16": "This repo's W4A16 path using the local CUDA kernel from w4a16_cuda.cu.",
            "lmdeploy_awq_pytorch": "LMDeploy AWQ model using LMDeploy's PyTorch backend.",
            "lmdeploy_awq_turbomind": "LMDeploy AWQ model using LMDeploy's TurboMind backend.",
        },
        "notes": [
            "LMDeploy comparison is speed-only and measures end-to-end generation latency for max_new_tokens=1.",
            "Regular BF16 / repo CUDA-kernel W4A16 comparisons measure raw torch forward latency for input_ids shape [1, 1].",
        ],
    }
    write_results(args.results_json, payload)
    write_plot(args.plot_path, payload)


if __name__ == "__main__":
    main()
