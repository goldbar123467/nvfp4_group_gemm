"""
NVFP4 Block-Scaled Group GEMM Task Definition

Input/Output type definitions for the group GEMM kernel.
"""

import torch
from typing import List, Tuple

# Input type: (abc_tensors, sfasfb_tensors, sfasfb_reordered_tensors, problem_sizes)
# abc_tensors: List of tuples (a, b, c)
#   a: torch.Tensor[float4e2m1fn_x2] of shape [M, K // 2, L]
#   b: torch.Tensor[float4e2m1fn_x2] of shape [N, K // 2, L]
#   c: torch.Tensor[float16] of shape [M, N, L]
# sfasfb_tensors: List of tuples (sfa, sfb)
#   sfa: torch.Tensor[float8_e4m3fnuz] of shape [M, K // 16, L]
#   sfb: torch.Tensor[float8_e4m3fnuz] of shape [N, K // 16, L]
# sfasfb_reordered_tensors: List of tuples (sfa_reordered, sfb_reordered)
#   Reordered scale factors for kernel consumption
# problem_sizes: List of tuples (M, N, K, L)

input_t = Tuple[
    List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]],  # abc_tensors
    List[Tuple[torch.Tensor, torch.Tensor]],                 # sfasfb_tensors
    List[Tuple[torch.Tensor, torch.Tensor]],                 # sfasfb_reordered_tensors
    List[Tuple[int, int, int, int]]                          # problem_sizes
]

# Output type: List of output tensors C for each group
output_t = List[torch.Tensor]
