import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

from quantization import (
    load_w4a16_cuda_direct_extension,
    load_w4a16_cuda_extension,
    pack_rows_4,
)


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GROUP_SIZE = 64
OUT_PATH = Path(__file__).with_name("kernel_only_results.json")



def sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()



def interleave_transposed_s_z(S: torch.Tensor, Z: torch.Tensor) -> torch.Tensor:
    OF, G = S.shape
    S_t = S.t().contiguous()
    Z_t = Z.t().contiguous()
    SZ = torch.empty((G, 2 * OF), device=S.device, dtype=S.dtype)
    SZ[:, 0::2] = S_t
    SZ[:, 1::2] = Z_t
    return SZ.contiguous()



def measure_ms(fn, n: int = 100):
    for _ in range(10):
        fn()
    sync()

    times = []
    for _ in range(n):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        times.append((time.perf_counter() - t0) * 1_000)
    return statistics.mean(times), statistics.stdev(times) if len(times) > 1 else 0.0



def metric_to_json(metric):
    return {"mean": metric[0], "std": metric[1]}



def load_baseline(path: Path):
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return None



def compare_cases(current_cases, baseline_cases):
    if not baseline_cases:
        return []

    baseline_map = {(case.get("of"), case.get("if")): case for case in baseline_cases}
    comparisons = []
    metric_keys = ["dense_bf16_ms", "repo_kernel_ms", "direct_kernel_ms"]
    for case in current_cases:
        key = (case.get("of"), case.get("if"))
        baseline_case = baseline_map.get(key)
        if not baseline_case:
            continue
        metric_comparison = {}
        for metric_key in metric_keys:
            current_metric = case.get(metric_key)
            baseline_metric = baseline_case.get(metric_key)
            if not current_metric or not baseline_metric:
                continue
            current_mean = current_metric["mean"]
            baseline_mean = baseline_metric["mean"]
            metric_comparison[metric_key] = {
                "current_mean": current_mean,
                "baseline_mean": baseline_mean,
                "delta_ms": current_mean - baseline_mean,
                "ratio_vs_baseline": (current_mean / baseline_mean) if baseline_mean else 0.0,
                "speedup_vs_baseline": (baseline_mean / current_mean) if current_mean else 0.0,
            }
        comparisons.append({"of": key[0], "if": key[1], "metrics": metric_comparison})
    return comparisons



def get_env_metadata() -> dict[str, object]:
    metadata: dict[str, object] = {
        "kernel_accumulation_dtype": "fp32",
        "benchmark_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "torch_version": torch.__version__,
        "device": str(DEVICE),
    }
    if torch.cuda.is_available() and DEVICE.type == "cuda":
        metadata["gpu_name"] = torch.cuda.get_device_name(0)
        capability = torch.cuda.get_device_capability(0)
        metadata["cuda_capability"] = f"{capability[0]}.{capability[1]}"
    else:
        metadata["gpu_name"] = None
        metadata["cuda_capability"] = None
    return metadata



def run_case(of: int, ifeatures: int, n_iters: int, colmajor_ext, direct_ext):
    W_q = torch.randint(0, 16, (of, ifeatures), device=DEVICE, dtype=torch.uint8).contiguous()
    W_packed = pack_rows_4(W_q)
    bias = torch.randn((of,), device=DEVICE, dtype=torch.bfloat16).contiguous()
    groups = ifeatures // GROUP_SIZE
    S = torch.ones((of, groups), device=DEVICE, dtype=torch.bfloat16).contiguous()
    Z = torch.randint(0, 16, (of, groups), device=DEVICE, dtype=torch.bfloat16).contiguous()
    SZ = interleave_transposed_s_z(S, Z)

    act_col = torch.randn((ifeatures, 1), device=DEVICE, dtype=torch.bfloat16).contiguous()
    act_rank3 = torch.randn((1, 1, ifeatures), device=DEVICE, dtype=torch.bfloat16).contiguous()
    dense_weight = torch.randn((of, ifeatures), device=DEVICE, dtype=torch.bfloat16).contiguous()

    dense_ms = measure_ms(lambda: torch.matmul(dense_weight, act_col) + bias[:, None], n=n_iters)
    colmajor_ms = measure_ms(lambda: colmajor_ext.forward(W_packed, bias, SZ, act_col, GROUP_SIZE), n=n_iters)
    direct_ms = measure_ms(lambda: direct_ext.forward(W_packed, bias, SZ, act_rank3, GROUP_SIZE), n=n_iters)

    dense = dense_ms[0]
    colmajor = colmajor_ms[0]
    direct = direct_ms[0]

    return {
        "of": of,
        "if": ifeatures,
        "dense_bf16_ms": metric_to_json(dense_ms),
        "repo_kernel_ms": metric_to_json(colmajor_ms),
        "direct_kernel_ms": metric_to_json(direct_ms),
        "repo_kernel_speedup_vs_dense": dense / colmajor if colmajor else 0.0,
        "direct_kernel_speedup_vs_dense": dense / direct if direct else 0.0,
    }



def main():
    torch.manual_seed(0)

    baseline_payload = load_baseline(OUT_PATH)
    colmajor_ext = load_w4a16_cuda_extension(verbose=False)
    direct_ext = load_w4a16_cuda_direct_extension(verbose=False)

    cases = []
    for of in [4096, 8192, 16384, 32768]:
        result = run_case(of=of, ifeatures=8192, n_iters=100, colmajor_ext=colmajor_ext, direct_ext=direct_ext)
        cases.append(result)
        print(
            f"OF={of}: dense_bf16={result['dense_bf16_ms']['mean']:.4f} ms, "
            f"repo_kernel={result['repo_kernel_ms']['mean']:.4f} ms, "
            f"direct_kernel={result['direct_kernel_ms']['mean']:.4f} ms"
        )

    payload = {
        **get_env_metadata(),
        "group_size": GROUP_SIZE,
        "cases": cases,
    }
    if baseline_payload is not None:
        payload["comparison_to_previous"] = {
            "baseline_path": str(OUT_PATH),
            "cases": compare_cases(cases, baseline_payload.get("cases")),
        }
    OUT_PATH.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
