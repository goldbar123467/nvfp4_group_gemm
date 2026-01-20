"""
Utility functions for NVFP4 Group GEMM
"""

import torch
from typing import Callable, Any


def make_match_reference(ref_fn: Callable, rtol: float = 1e-3, atol: float = 1e-3):
    """
    Create a checker function that compares implementation output against reference.

    Args:
        ref_fn: Reference implementation function
        rtol: Relative tolerance for comparison
        atol: Absolute tolerance for comparison

    Returns:
        A checker function that returns True if outputs match within tolerance
    """
    def check_implementation(impl_fn: Callable, data: Any) -> bool:
        ref_output = ref_fn(data)
        impl_output = impl_fn(data)

        if len(ref_output) != len(impl_output):
            print(f"Output length mismatch: ref={len(ref_output)}, impl={len(impl_output)}")
            return False

        for i, (ref_c, impl_c) in enumerate(zip(ref_output, impl_output)):
            if not torch.allclose(ref_c, impl_c, rtol=rtol, atol=atol):
                max_diff = (ref_c - impl_c).abs().max().item()
                print(f"Group {i} mismatch: max_diff={max_diff}")
                return False

        return True

    return check_implementation


def ceil_div(a: int, b: int) -> int:
    """Ceiling division"""
    return (a + b - 1) // b


def benchmark_kernel(fn: Callable, data: Any, warmup: int = 10, iters: int = 100) -> float:
    """
    Benchmark a kernel function.

    Args:
        fn: Function to benchmark
        data: Input data
        warmup: Number of warmup iterations
        iters: Number of timed iterations

    Returns:
        Average execution time in microseconds
    """
    # Warmup
    for _ in range(warmup):
        fn(data)

    torch.cuda.synchronize()

    # Timed iterations
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        fn(data)
    end.record()

    torch.cuda.synchronize()

    return start.elapsed_time(end) * 1000 / iters  # Convert ms to us
