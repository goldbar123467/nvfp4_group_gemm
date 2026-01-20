"""
Test suite for NVFP4 Block-Scaled Group GEMM
"""

import torch
from reference import generate_input, ref_kernel, check_implementation
from solution import custom_kernel
from utils import benchmark_kernel


def test_correctness():
    """Test that custom kernel produces correct results."""
    print("Testing correctness...")

    # Test case 1: Small matrices
    m = (128, 256)
    n = (4096, 4096)
    k = (256, 512)
    g = 2
    data = generate_input(m, n, k, g, seed=42)

    # Get reference output
    ref_out = ref_kernel(data)

    # Regenerate data (since ref_kernel modifies c in-place)
    data = generate_input(m, n, k, g, seed=42)

    # Get custom kernel output
    custom_out = custom_kernel(data)

    # Compare
    for i, (ref_c, custom_c) in enumerate(zip(ref_out, custom_out)):
        if torch.allclose(ref_c, custom_c, rtol=1e-3, atol=1e-3):
            print(f"  Group {i}: PASS")
        else:
            max_diff = (ref_c - custom_c).abs().max().item()
            print(f"  Group {i}: FAIL (max_diff={max_diff})")

    print()


def test_benchmark_cases():
    """Benchmark against the speed of light reference cases."""
    print("Benchmarking against SOL cases...")
    print("=" * 80)

    # SOL reference cases from the problem
    sol_cases = [
        {
            "name": "Case 1 (G=8)",
            "m": (80, 176, 128, 72, 64, 248, 96, 160),
            "n": (4096,) * 8,
            "k": (7168,) * 8,
            "g": 8,
            "sol_us": 18.833,
        },
        {
            "name": "Case 2 (G=8)",
            "m": (40, 76, 168, 72, 164, 148, 196, 160),
            "n": (7168,) * 8,
            "k": (2048,) * 8,
            "g": 8,
            "sol_us": 10.667,
        },
        {
            "name": "Case 3 (G=2)",
            "m": (192, 320),
            "n": (3072, 3072),
            "k": (4096, 4096),
            "g": 2,
            "sol_us": 2.406,
        },
        {
            "name": "Case 4 (G=2)",
            "m": (128, 384),
            "n": (4096, 4096),
            "k": (1536, 1536),
            "g": 2,
            "sol_us": 1.525,
        },
    ]

    for case in sol_cases:
        data = generate_input(case["m"], case["n"], case["k"], case["g"], seed=123)

        # Warmup and benchmark
        time_us = benchmark_kernel(custom_kernel, data, warmup=10, iters=100)

        ratio = time_us / case["sol_us"]
        print(f"{case['name']}:")
        print(f"  Achieved: {time_us:.3f} us")
        print(f"  SOL:      {case['sol_us']:.3f} us")
        print(f"  Ratio:    {ratio:.2f}x SOL")
        print()

    print("=" * 80)


def main():
    """Run all tests."""
    print("NVFP4 Block-Scaled Group GEMM Test Suite")
    print("=" * 80)
    print()

    # Check CUDA availability
    if not torch.cuda.is_available():
        print("CUDA not available!")
        return

    device = torch.cuda.get_device_name(0)
    print(f"Running on: {device}")
    print()

    test_correctness()
    test_benchmark_cases()


if __name__ == "__main__":
    main()
