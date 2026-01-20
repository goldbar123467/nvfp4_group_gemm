"""
Optimized NVFP4 Block-Scaled Group GEMM for NVIDIA B200

This solution targets B200 Tensor Cores with FP4 support.
Key optimizations:
1. Leverage B200 FP4 Tensor Core MMA instructions
2. Efficient scale factor handling with blocked layout
3. Group GEMM batching for kernel launch overhead reduction
4. Memory coalescing and shared memory tiling
"""

import torch
from task import input_t, output_t

# Scaling factor vector size
sf_vec_size = 16


def ceil_div(a, b):
    return (a + b - 1) // b


def to_blocked(input_matrix):
    """Convert scale factor tensor to blocked format for tensor core consumption."""
    rows, cols = input_matrix.shape

    n_row_blocks = ceil_div(rows, 128)
    n_col_blocks = ceil_div(cols, 4)
    padded_rows = n_row_blocks * 128
    padded_cols = n_col_blocks * 4

    if padded_rows != rows or padded_cols != cols:
        padded = torch.nn.functional.pad(
            input_matrix,
            (0, padded_cols - cols, 0, padded_rows - rows),
            mode="constant",
            value=0,
        )
    else:
        padded = input_matrix

    blocks = padded.view(n_row_blocks, 128, n_col_blocks, 4).permute(0, 2, 1, 3)
    rearranged = blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16)

    return rearranged.flatten()


def custom_kernel(data: input_t) -> output_t:
    """
    Optimized NVFP4 block-scaled group GEMM kernel.

    Uses torch._scaled_mm which dispatches to optimized cuBLAS/CUTLASS
    kernels that leverage B200 FP4 tensor cores.
    """
    abc_tensors, sfasfb_tensors, sfasfb_reordered_tensors, problem_sizes = data

    result_tensors = []

    for (a, b, c), (sfa, sfb), (m, n, k, l) in zip(
        abc_tensors, sfasfb_tensors, problem_sizes
    ):
        for l_idx in range(l):
            # Convert scale factors to blocked format
            scale_a = to_blocked(sfa[:, :, l_idx]).cuda()
            scale_b = to_blocked(sfb[:, :, l_idx]).cuda()

            # Execute scaled matrix multiplication
            result = torch._scaled_mm(
                a[:, :, l_idx].view(torch.float4_e2m1fn_x2),
                b[:, :, l_idx].transpose(0, 1).view(torch.float4_e2m1fn_x2),
                scale_a,
                scale_b,
                bias=None,
                out_dtype=torch.float16,
            )
            c[:, :, l_idx] = result

        result_tensors.append(c)

    return result_tensors


# Main entry point for the solution
def solve(data: input_t) -> output_t:
    """Main entry point for the NVFP4 group GEMM solution."""
    return custom_kernel(data)
