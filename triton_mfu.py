#!/usr/bin/env python3
"""Find the Triton GEMM configuration that maximizes model FLOPs utilization (MFU) on a CUDA GPU.

MFU is the achieved matmul throughput divided by the theoretical dense peak:
    MFU = (2 * M * N * K) / time / peak_flops

The benchmark uses synthetic random inputs. It sweeps a set of Triton tile
configurations (BLOCK_M, BLOCK_N, BLOCK_K, num_warps, num_stages), keeps the one
with the lowest median time, checks its output against torch.matmul (cuBLAS), and
reports both as MFU. cuBLAS is the reference ceiling: a Triton kernel that beats it
is a sign the sweep found a better tiling than the default library heuristic.

Precisions:
  fp16  float16 inputs, float32 accumulation
  bf16  bfloat16 inputs, float32 accumulation

Requires a CUDA GPU, PyTorch and Triton. Peak TFLOPS come from the datasheet
table in gpu_flops.py; GPUs missing from that table are benchmarked without an MFU.

Examples:
  python triton_mfu.py
  python triton_mfu.py --size 4096 --precisions fp16
  python triton_mfu.py --device cuda:1 --iters 50
"""
import argparse
import statistics
import sys
from functools import partial

import torch
import triton
import triton.language as tl

from gpu_flops import theoretical_peak

DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16}

# Tile configurations to sweep. Block sizes must divide M, N and K, so the kernel
# needs no boundary masks; configs that do not divide the chosen size are skipped.
CONFIGS = [
    {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 8, "num_warps": 8, "num_stages": 3},
    {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8, "num_warps": 8, "num_stages": 3},
    {"BLOCK_M": 128, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 8, "num_warps": 8, "num_stages": 4},
    {"BLOCK_M": 256, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8, "num_warps": 8, "num_stages": 4},
    {"BLOCK_M": 256, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 8, "num_warps": 8, "num_stages": 3},
    {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 64, "GROUP_M": 8, "num_warps": 4, "num_stages": 4},
    {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 32, "GROUP_M": 8, "num_warps": 4, "num_stages": 5},
    {"BLOCK_M": 128, "BLOCK_N": 128, "BLOCK_K": 128, "GROUP_M": 8, "num_warps": 8, "num_stages": 3},
    {"BLOCK_M": 64, "BLOCK_N": 256, "BLOCK_K": 64, "GROUP_M": 8, "num_warps": 4, "num_stages": 4},
    {"BLOCK_M": 256, "BLOCK_N": 64, "BLOCK_K": 64, "GROUP_M": 8, "num_warps": 4, "num_stages": 4},
]


@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    # Group programs so that neighbouring CTAs reuse the same rows of A and columns of B in L2.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc = tl.dot(a, b, acc)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    c = acc.to(c_ptr.dtype.element_ty)
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    tl.store(c_ptrs, c)


def launch(a, b, c, cfg):
    M, K = a.shape
    N = b.shape[1]
    grid = (triton.cdiv(M, cfg["BLOCK_M"]) * triton.cdiv(N, cfg["BLOCK_N"]),)
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=cfg["BLOCK_M"], BLOCK_N=cfg["BLOCK_N"], BLOCK_K=cfg["BLOCK_K"],
        GROUP_M=cfg["GROUP_M"],
        num_warps=cfg["num_warps"], num_stages=cfg["num_stages"],
    )
    return c


def divides_evenly(cfg, m, n, k):
    return m % cfg["BLOCK_M"] == 0 and n % cfg["BLOCK_N"] == 0 and k % cfg["BLOCK_K"] == 0


def time_ms(fn, warmup, iters):
    """Return per-iteration GPU times in milliseconds, timed with CUDA events."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(iters)]
    for start, end in zip(starts, ends):
        start.record()
        fn()
        end.record()
    torch.cuda.synchronize()
    return [start.elapsed_time(end) for start, end in zip(starts, ends)]


def mfu_percent(ms, flops, peak_tflops):
    return 100 * flops / (ms * 1e-3) / (peak_tflops * 1e12)


def check_correctness(a, b, c):
    """Return the max abs error of c relative to torch.matmul, as a fraction of the reference's max."""
    ref = torch.matmul(a, b)
    return ((c.float() - ref.float()).abs().max() / ref.float().abs().max()).item()


def sweep(a, b, c, args):
    """Time every config that divides the problem size. Returns [(median_ms, cfg)] sorted fastest first."""
    m, k = a.shape
    n = b.shape[1]
    results = []
    for cfg in CONFIGS:
        label = describe(cfg)
        if not divides_evenly(cfg, m, n, k):
            print(f"  skipped {label}: block sizes do not divide {m}x{n}x{k}")
            continue
        try:
            times = time_ms(partial(launch, a, b, c, cfg), args.warmup, args.iters)
        except Exception as err:  # Triton raises different errors when a config exceeds shared memory or registers.
            print(f"  skipped {label}: {str(err).splitlines()[0]}")
            continue
        median = statistics.median(times)
        print(f"  {label}: median {median:.3f} ms")
        results.append((median, cfg))
    return sorted(results, key=lambda r: r[0])


def describe(cfg):
    return (
        f"BM={cfg['BLOCK_M']} BN={cfg['BLOCK_N']} BK={cfg['BLOCK_K']} "
        f"warps={cfg['num_warps']} stages={cfg['num_stages']}"
    )


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--device", default="cuda", help="CUDA device, e.g. cuda or cuda:1")
    parser.add_argument("--size", type=int, default=8192, help="matrix dimension N for M=N=K (default 8192)")
    parser.add_argument("--warmup", type=int, default=5, help="untimed iterations per config")
    parser.add_argument("--iters", type=int, default=30, help="timed iterations per config")
    parser.add_argument(
        "--precisions", nargs="+", default=list(DTYPES), choices=list(DTYPES),
        help=f"precisions to tune (default: {' '.join(DTYPES)})",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        sys.exit("error: triton_mfu.py needs a CUDA GPU; Triton does not run on MPS or CPU")

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    peaks = theoretical_peak(device)
    n = args.size

    print(f"PyTorch: {torch.__version__}   Triton: {triton.__version__}")
    print(f"Device:  {torch.cuda.get_device_name(device)} [{device}]")
    print(f"GEMM:    {n} x {n} x {n}   ({2 * n**3 / 1e12:.3f} TFLOP per matmul)")

    print()
    print(f"{'Precision':<10}{'Kernel':<22}{'ms (median)':>13}{'TFLOPS':>10}{'MFU %':>9}")
    print("-" * 64)

    for precision in args.precisions:
        dtype = DTYPES[precision]
        peak = peaks.get(precision)
        flops = 2 * n**3

        # Synthetic inputs: standard normal values, so the output stays well within fp16 range.
        a = torch.randn(n, n, device=device, dtype=dtype)
        b = torch.randn(n, n, device=device, dtype=dtype)
        c = torch.empty(n, n, device=device, dtype=dtype)

        print(f"Sweeping {len(CONFIGS)} Triton configs for {precision}...")
        ranked = sweep(a, b, c, args)
        if not ranked:
            print(f"{precision}: no config could run at size {n}")
            print()
            continue

        best_ms, best_cfg = ranked[0]
        launch(a, b, c, best_cfg)
        error = check_correctness(a, b, c)

        # cuBLAS baseline with the same timing method, for comparison.
        ref_out = torch.empty(n, n, device=device, dtype=dtype)
        cublas_ms = statistics.median(time_ms(lambda: torch.matmul(a, b, out=ref_out), args.warmup, args.iters))

        def row(name, ms):
            tflops = flops / (ms * 1e-3) / 1e12
            mfu = f"{mfu_percent(ms, flops, peak):.1f}" if peak else "n/a"
            print(f"{precision:<10}{name:<22}{ms:>13.3f}{tflops:>10.1f}{mfu:>9}")

        row("triton (best)", best_ms)
        row("cuBLAS (torch.matmul)", cublas_ms)
        print(f"  best config: {describe(best_cfg)}")
        print(f"  max relative error vs torch.matmul: {error:.2e}")
        if error > 1e-2:
            print("  warning: error is high; treat the timings above as suspect")
        if peak is None:
            print("  note: no theoretical peak for this GPU in gpu_flops.py, so MFU is n/a")
        print()


if __name__ == "__main__":
    main()
