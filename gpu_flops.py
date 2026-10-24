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
import time

import torch

# name -> (dtype, requires TF32 flag on CUDA matmul, or None)
PRECISIONS = {
    "fp64": (torch.float64, None),
    "fp32": (torch.float32, False),
    "tf32": (torch.float32, True),
    "fp16": (torch.float16, None),
    "bf16": (torch.bfloat16, None),
}


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


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


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
    except (RuntimeError, NotImplementedError) as err:
        return {"precision": name, "error": str(err).splitlines()[0]}

    median_s = sorted(times)[len(times) // 2]
    best_s = min(times)
    return {
        "precision": name,
        "tflops_median": flops / median_s / 1e12,
        "tflops_best": flops / best_s / 1e12,
        "ms_median": median_s * 1e3,
    }


def print_table(results):
    print(f"{'Precision':<10}{'TFLOPS (median)':>18}{'TFLOPS (best)':>16}{'ms/matmul':>12}")
    print("-" * 56)
    for r in results:
        if "error" in r:
            print(f"{r['precision']:<10}  skipped: {r['error']}")
            continue
        print(
            f"{r['precision']:<10}{r['tflops_median']:>18.2f}"
            f"{r['tflops_best']:>16.2f}{r['ms_median']:>12.2f}"
        )


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
    args = parser.parse_args()

    device = pick_device(args.device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())

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

    info = {
        "device": str(device),
        "device_name": device_name(device),
        "torch": torch.__version__,
        "matrix_size": args.size,
        "results": results,
    }

    if args.json:
        print(json.dumps(info, indent=2))
    else:
        print(f"Device:      {info['device_name']} [{info['device']}]")
        print(f"PyTorch:     {info['torch']}")
        print(f"GEMM shape:  {args.size} x {args.size} x {args.size}")
        print()
        print_table(results)


if __name__ == "__main__":
    main()
