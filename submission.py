"""
NVFP4 Block-Scaled Group GEMM for B200 - v3
"""

import torch
from task import input_t, output_t


def custom_kernel(data: input_t) -> output_t:
    """
    Optimized NVFP4 block-scaled group GEMM kernel.
    """
    abc_tensors, _, sfasfb_reordered_tensors, problem_sizes = data

    result_tensors = []
    num_groups = len(abc_tensors)

    for i in range(num_groups):
        a, b, c = abc_tensors[i]
        sfa_reordered, sfb_reordered = sfasfb_reordered_tensors[i]

        # L=1 always, access directly
        scale_a = sfa_reordered.reshape(-1)
        scale_b = sfb_reordered.reshape(-1)

        # Get matrix slices
        a_mat = a[:, :, 0].view(torch.float4_e2m1fn_x2)
        b_mat = b[:, :, 0].T.view(torch.float4_e2m1fn_x2)

        # Execute scaled matrix multiplication with fast accumulation
        c[:, :, 0] = torch._scaled_mm(
            a_mat,
            b_mat,
            scale_a,
            scale_b,
            bias=None,
            out_dtype=torch.float16,
            use_fast_accum=True,  # 10-20% speedup
        )

        result_tensors.append(c)

    return result_tensors
