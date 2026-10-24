#!/usr/bin/env python3
"""Measure GPU matrix-multiply throughput (TFLOPS) at different precisions.

The benchmark times a large square GEMM, C = A @ B with shape (N, N) x (N, N).
GEMM is compute-bound and runs on tensor cores / matrix units, so it is the
closest practical probe of peak FLOPS. Each GEMM performs 2 * N^3 floating-point
operations (one multiply and one add per inner-product term).

Precisions:
  fp64  float64            (CUDA and CPU only; Apple MPS has no float64)
  fp32  float32, TF32 disabled
  tf32  float32 inputs, TF32 tensor-core math enabled (CUDA only, Ampere+)
  fp16  float16
  bf16  bfloat16

Examples:
  python gpu_flops.py
  python gpu_flops.py --size 4096 --precisions fp16 bf16
  python gpu_flops.py --device cuda:1 --json
"""
import argparse
import json
import platform
import statistics
import sys
import time

import torch
from torch.autograd import DeviceType
from torch.profiler import ProfilerActivity, profile

# name -> (dtype, requires TF32 flag on CUDA matmul, or None)
PRECISIONS = {
    "fp64": (torch.float64, None),
    "fp32": (torch.float32, False),
    "tf32": (torch.float32, True),
    "fp16": (torch.float16, None),
    "bf16": (torch.bfloat16, None),
}

# Substrings of cuBLAS kernel names that run on tensor cores:
# s884/s1688/s16816 = HMMA tile shapes (Volta/Turing/Ampere), xmma/gmma/wgmma/warpgroup = Hopper
# and newer, hmma/wmma/tensorop = generic markers.
TENSOR_CORE_KERNEL_MARKERS = (
    "s884", "s1688", "s16816", "h884", "h1688", "h16816", "hmma", "wmma", "tensorop",
    "xmma", "gmma", "wgmma", "warpgroup", "nvjet",
)

# Substrings of kernel names that are plain SIMT GEMMs (no tensor cores), checked first
# so that e.g. an fp32 "sgemm" or "ffma" kernel is never mistaken for a tensor-core kernel.
NON_TENSOR_CORE_KERNEL_MARKERS = ("sgemm", "dgemm", "gemv", "ffma")

# Tensor cores need the GEMM dimensions to be multiples of 8 for fp16/bf16 (16 bytes).
TENSOR_CORE_ALIGNMENT = 8

# Theoretical peak TFLOPS from NVIDIA datasheets, dense (no sparsity). Each entry is
# (model, variant, peaks): the model must appear in the device name, and the variant
# (if not None) must too. More specific entries come first.
THEORETICAL_PEAK_TFLOPS = [
    ("H200", None, {"fp64": 67, "fp32": 67, "tf32": 495, "fp16": 989, "bf16": 989}),
    ("H100", "PCIe", {"fp64": 51, "fp32": 51, "tf32": 378, "fp16": 756, "bf16": 756}),
    ("H100", None, {"fp64": 67, "fp32": 67, "tf32": 495, "fp16": 989, "bf16": 989}),
    ("A100", "PCIe", {"fp64": 19.5, "fp32": 19.5, "tf32": 156, "fp16": 312, "bf16": 312}),
    ("A100", None, {"fp64": 19.5, "fp32": 19.5, "tf32": 156, "fp16": 312, "bf16": 312}),
    ("L40S", None, {"fp64": 1.4, "fp32": 91.6, "tf32": 183, "fp16": 362, "bf16": 362}),
    ("V100", None, {"fp64": 7.8, "fp32": 15.7, "fp16": 125}),
]


def pick_device(name):
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def device_name(device):
    if device.type == "cuda":
        return torch.cuda.get_device_name(device)
    if device.type == "mps":
        return f"Apple MPS ({platform.processor() or platform.machine()})"
    return f"CPU ({platform.processor() or platform.machine()})"


def theoretical_peak(device):
    """Return {precision: peak TFLOPS} from the datasheet table, or {} if the GPU is unknown."""
    if device.type != "cuda":
        return {}
    name = torch.cuda.get_device_name(device)
    for model, variant, peaks in THEORETICAL_PEAK_TFLOPS:
        if model in name and (variant is None or variant in name):
            return peaks
    return {}


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def is_tensor_core_kernel(name):
    if any(marker in name for marker in NON_TENSOR_CORE_KERNEL_MARKERS):
        return False
    return any(marker in name for marker in TENSOR_CORE_KERNEL_MARKERS)


def uses_tensor_cores(dtype, n, device):
    """Return (tensor_cores, kernel_names). tensor_cores is True/False, or None if unknown.

    Profiles one matmul and inspects the CUDA kernel names. Only CUDA exposes this;
    MPS and CPU do not report which matrix units a kernel uses, so the result is None.
    """
    if device.type != "cuda":
        return None, []

    a = torch.randn(n, n, device=device, dtype=dtype)
    b = torch.randn(n, n, device=device, dtype=dtype)
    torch.mm(a, b)  # warm up so cuBLAS has selected its kernel before profiling
    torch.cuda.synchronize(device)

    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        torch.mm(a, b)
        torch.cuda.synchronize(device)

    kernels = [e.name for e in prof.events() if e.device_type == DeviceType.CUDA]
    if not kernels:
        return None, []
    return any(is_tensor_core_kernel(k.lower()) for k in kernels), kernels


def time_matmul(dtype, n, device, warmup, iters):
    a = torch.randn(n, n, device=device, dtype=dtype)
    b = torch.randn(n, n, device=device, dtype=dtype)
    c = torch.empty(n, n, device=device, dtype=dtype)

    for _ in range(warmup):
        torch.mm(a, b, out=c)
    sync(device)

    times = []
    for _ in range(iters):
        start = time.perf_counter()
        torch.mm(a, b, out=c)
        sync(device)
        times.append(time.perf_counter() - start)
    return times


def run_precision(name, n, device, warmup, iters):
    dtype, tf32 = PRECISIONS[name]
    if tf32 is not None:
        torch.backends.cuda.matmul.allow_tf32 = tf32

    flops = 2 * n**3
    try:
        times = time_matmul(dtype, n, device, warmup, iters)
        tensor_cores, kernels = uses_tensor_cores(dtype, n, device)
    except (RuntimeError, NotImplementedError) as err:
        return {"precision": name, "error": str(err).splitlines()[0]}

    mean_s = statistics.mean(times)
    tflops = [flops / t / 1e12 for t in times]
    return {
        "precision": name,
        "tensor_cores": tensor_cores,
        "kernels": kernels,
        "tflops_mean": statistics.mean(tflops),
        "tflops_best": max(tflops),
        "tflops_worst": min(tflops),
        "tflops_cv_percent": 100 * statistics.stdev(tflops) / statistics.mean(tflops)
        if len(tflops) > 1
        else 0.0,
        "ms_mean": mean_s * 1e3,
    }


def format_tensor_cores(value):
    return {True: "yes", False: "NO", None: "n/a"}[value]


def print_table(results, peaks, show_kernels):
    print(
        "TFLOPS columns: best = fastest iteration, worst = slowest, "
        "CV = std dev / mean across iterations, in percent"
    )
    print(
        f"{'Precision':<10}{'Mean':>10}{'Best':>10}{'Worst':>10}{'CV %':>10}"
        f"{'ms/matmul':>11}{'Tensor cores':>14}{'Peak':>10}{'% of peak':>11}"
    )
    print("-" * 96)
    for r in results:
        if "error" in r:
            print(f"{r['precision']:<10}  skipped: {r['error']}")
            continue
        peak = peaks.get(r["precision"])
        peak_text = f"{peak:.1f}" if peak is not None else "n/a"
        percent_text = f"{100 * r['tflops_mean'] / peak:.1f}%" if peak is not None else "n/a"
        print(
            f"{r['precision']:<10}{r['tflops_mean']:>10.2f}"
            f"{r['tflops_best']:>10.2f}{r['tflops_worst']:>10.2f}{r['tflops_cv_percent']:>10.2f}"
            f"{r['ms_mean']:>11.2f}"
            f"{format_tensor_cores(r['tensor_cores']):>14}"
            f"{peak_text:>10}{percent_text:>11}"
        )
        if show_kernels:
            for kernel in r["kernels"]:
                print(f"    kernel: {kernel}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--device", default="auto", help="auto, cuda, cuda:N, mps or cpu")
    parser.add_argument("--size", type=int, default=8192, help="matrix dimension N (default 8192)")
    parser.add_argument("--warmup", type=int, default=5, help="untimed iterations per precision")
    parser.add_argument("--iters", type=int, default=20, help="timed iterations per precision")
    parser.add_argument(
        "--precisions",
        nargs="+",
        default=list(PRECISIONS),
        choices=list(PRECISIONS),
        help="precisions to test (default: all)",
    )
    parser.add_argument("--json", action="store_true", help="print results as JSON")
    parser.add_argument(
        "--show-kernels",
        action="store_true",
        help="list the CUDA kernel names behind each result (useful when checking tensor cores)",
    )
    args = parser.parse_args()

    device = pick_device(args.device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())

    if args.size % TENSOR_CORE_ALIGNMENT:
        print(
            f"warning: --size {args.size} is not a multiple of {TENSOR_CORE_ALIGNMENT}; "
            "tensor cores may not be used",
            file=sys.stderr,
        )

    original_tf32 = torch.backends.cuda.matmul.allow_tf32
    results = []
    try:
        for name in args.precisions:
            if name == "tf32" and device.type != "cuda":
                results.append({"precision": name, "error": "TF32 is CUDA-only"})
                continue
            if name == "fp64" and device.type == "mps":
                results.append({"precision": name, "error": "MPS has no float64"})
                continue
            results.append(run_precision(name, args.size, device, args.warmup, args.iters))
    finally:
        torch.backends.cuda.matmul.allow_tf32 = original_tf32

    peaks = theoretical_peak(device)
    info = {
        "device": str(device),
        "device_name": device_name(device),
        "torch": torch.__version__,
        "matrix_size": args.size,
        "theoretical_peak_tflops": peaks,
        "results": results,
    }

    if args.json:
        print(json.dumps(info, indent=2))
    else:
        print(f"Device:      {info['device_name']} [{info['device']}]")
        print(f"PyTorch:     {info['torch']}")
        print(f"GEMM shape:  {args.size} x {args.size} x {args.size}")
        if peaks:
            print("Theoretical peak TFLOPS (dense, datasheet):")
            print("  " + ", ".join(f"{p}={v:g}" for p, v in peaks.items()))
        else:
            print("Theoretical peak: unknown for this device (not in the datasheet table)")
        print()
        print_table(results, peaks, args.show_kernels)


if __name__ == "__main__":
    main()
