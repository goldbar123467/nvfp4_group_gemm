# NVFP4 Block-Scaled Group GEMM for NVIDIA B200

Implementation of a block-scaled group matrix-matrix multiplication kernel optimized for NVIDIA B200 GPUs.

## Problem Overview

Implement a group GEMM kernel with:
- **Input A**: `[M, K//2, L]` in `float4_e2m1fn_x2` format
- **Input B**: `[N, K//2, L]` in `float4_e2m1fn_x2` format
- **Scale factors**: `float8_e4m3fnuz` with blocked layout
- **Output C**: `[M, N, L]` in `float16`

## B200 Target Specifications

- FP4 Tensor Core throughput: ~10 PFLOPS
- Memory bandwidth: ~8 TB/s HBM3e
- 192 SMs

## Speed of Light Targets

| Groups | M values | N | K | Target (μs) |
|--------|----------|---|---|-------------|
| 8 | 80-248 | 4096 | 7168 | 18.833 |
| 8 | 40-196 | 7168 | 2048 | 10.667 |
| 2 | 192, 320 | 3072 | 4096 | 2.406 |
| 2 | 128, 384 | 4096 | 1536 | 1.525 |

## Files

- `task.py` - Type definitions
- `reference.py` - Reference implementation with `torch._scaled_mm`
- `solution.py` - Basic optimized kernel
- `solution_advanced.py` - Advanced optimizations (streams, fusion, graphs)
- `test.py` - Test suite and benchmarks
- `utils.py` - Utility functions

## Key Optimizations

1. **Pre-computed scale factor layouts** - Avoid repeated blocking computation
2. **CUDA streams** - Parallel execution of independent GEMMs
3. **Fused execution** - Minimize Python overhead
4. **CUDA graphs** - Eliminate kernel launch overhead

## Usage

```python
from reference import generate_input
from solution import custom_kernel

# Generate test data
data = generate_input(
    m=(128, 256),
    n=(4096, 4096),
    k=(7168, 7168),
    g=2,
    seed=42
)

# Run kernel
outputs = custom_kernel(data)
```

## Testing

```bash
python test.py
```

## Requirements

- PyTorch with CUDA support
- NVIDIA B200 GPU (or compatible)
- Python 3.8+
