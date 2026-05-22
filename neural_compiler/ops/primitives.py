############################################################
#
# Neural Compiler: Compiling scheme into composable and
#                  differentiable neural network representations
#
# primitives.py: Fixed-weight implementations of 51 primitive operations with tensor broadcasting
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Primitive operation implementations.

Each operation takes a list of input tensors and returns a single output tensor.
Scalar operations broadcast over vectors and matrices via PyTorch semantics.
Vector/matrix operations use dim=-1 / dim=-2 conventions for batch compatibility.
"""

from __future__ import annotations
import torch


def _op_add(args: list[torch.Tensor]) -> torch.Tensor:
    return args[0] + args[1]


def _op_sub(args: list[torch.Tensor]) -> torch.Tensor:
    return args[0] - args[1]


def _op_mul(args: list[torch.Tensor]) -> torch.Tensor:
    return args[0] * args[1]


def _op_div(args: list[torch.Tensor]) -> torch.Tensor:
    return args[0] / args[1]


def _op_modulo(args: list[torch.Tensor]) -> torch.Tensor:
    return torch.fmod(args[0], args[1])


def _op_remainder(args: list[torch.Tensor]) -> torch.Tensor:
    return torch.remainder(args[0], args[1])


def _op_abs(args: list[torch.Tensor]) -> torch.Tensor:
    return torch.abs(args[0])


def _op_min(args: list[torch.Tensor]) -> torch.Tensor:
    return torch.minimum(args[0], args[1])


def _op_max(args: list[torch.Tensor]) -> torch.Tensor:
    return torch.maximum(args[0], args[1])


def _op_eq(args: list[torch.Tensor]) -> torch.Tensor:
    return (args[0] == args[1]).float()


def _op_lt(args: list[torch.Tensor]) -> torch.Tensor:
    return (args[0] < args[1]).float()


def _op_gt(args: list[torch.Tensor]) -> torch.Tensor:
    return (args[0] > args[1]).float()


def _op_le(args: list[torch.Tensor]) -> torch.Tensor:
    return (args[0] <= args[1]).float()


def _op_ge(args: list[torch.Tensor]) -> torch.Tensor:
    return (args[0] >= args[1]).float()


def _op_not(args: list[torch.Tensor]) -> torch.Tensor:
    return (args[0] == 0.0).float()


def _op_and(args: list[torch.Tensor]) -> torch.Tensor:
    return ((args[0] != 0.0) & (args[1] != 0.0)).float()


def _op_or(args: list[torch.Tensor]) -> torch.Tensor:
    return ((args[0] != 0.0) | (args[1] != 0.0)).float()


def _op_sin(args: list[torch.Tensor]) -> torch.Tensor:
    return torch.sin(args[0])


def _op_cos(args: list[torch.Tensor]) -> torch.Tensor:
    return torch.cos(args[0])


def _op_exp(args: list[torch.Tensor]) -> torch.Tensor:
    return torch.exp(args[0])


def _op_sqrt(args: list[torch.Tensor]) -> torch.Tensor:
    return torch.sqrt(torch.clamp(args[0], min=1e-8))


def _op_log(args: list[torch.Tensor]) -> torch.Tensor:
    return torch.log(torch.clamp(args[0], min=1e-8))


def _op_pow(args: list[torch.Tensor]) -> torch.Tensor:
    return torch.pow(args[0], args[1])


def _op_if(args: list[torch.Tensor]) -> torch.Tensor:
    """MUX: if test != 0, return then_val, else return else_val."""
    test, then_val, else_val = args
    sel = (test != 0.0).float()
    while sel.dim() < then_val.dim():
        sel = sel.unsqueeze(-1)
    return sel * then_val + (1.0 - sel) * else_val


# ---------------------------------------------------------------------------
# Vector operations
# ---------------------------------------------------------------------------

def _op_vec(args: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack(args, dim=-1)


def _op_ref(args: list[torch.Tensor]) -> torch.Tensor:
    vec, idx = args[0], args[1]
    i = idx.long()
    if vec.dim() == 1:
        return vec[i]
    return torch.gather(vec, -1, i.unsqueeze(-1)).squeeze(-1)


def _op_dot(args: list[torch.Tensor]) -> torch.Tensor:
    return (args[0] * args[1]).sum(dim=-1)


def _op_cross(args: list[torch.Tensor]) -> torch.Tensor:
    return torch.linalg.cross(args[0], args[1], dim=-1)


def _op_norm(args: list[torch.Tensor]) -> torch.Tensor:
    return torch.norm(args[0], dim=-1)


def _op_normalize(args: list[torch.Tensor]) -> torch.Tensor:
    n = torch.norm(args[0], dim=-1, keepdim=True).clamp(min=1e-8)
    return args[0] / n


def _op_vsum(args: list[torch.Tensor]) -> torch.Tensor:
    return args[0].sum(dim=-1)


def _op_vlen(args: list[torch.Tensor]) -> torch.Tensor:
    return torch.tensor(args[0].shape[-1], dtype=torch.float32)


def _op_scale(args: list[torch.Tensor]) -> torch.Tensor:
    return args[0] * args[1]


# ---------------------------------------------------------------------------
# Matrix operations
# ---------------------------------------------------------------------------

def _op_mat(args: list[torch.Tensor]) -> torch.Tensor:
    return torch.stack(args, dim=-2)


def _op_matmul(args: list[torch.Tensor]) -> torch.Tensor:
    return torch.matmul(args[0], args[1])


def _op_matvec(args: list[torch.Tensor]) -> torch.Tensor:
    return torch.matmul(args[0], args[1].unsqueeze(-1)).squeeze(-1)


def _op_transpose(args: list[torch.Tensor]) -> torch.Tensor:
    return args[0].transpose(-2, -1)


def _op_trace(args: list[torch.Tensor]) -> torch.Tensor:
    return torch.diagonal(args[0], dim1=-2, dim2=-1).sum(dim=-1)


def _op_det(args: list[torch.Tensor]) -> torch.Tensor:
    return torch.linalg.det(args[0])


def _op_inv(args: list[torch.Tensor]) -> torch.Tensor:
    return torch.linalg.inv(args[0])


def _op_outer(args: list[torch.Tensor]) -> torch.Tensor:
    return args[0].unsqueeze(-1) * args[1].unsqueeze(-2)


def _op_eye(args: list[torch.Tensor]) -> torch.Tensor:
    n = int(args[0].reshape(-1)[0].item())
    return torch.eye(n, dtype=torch.float32, device=args[0].device)


def _op_zeros(args: list[torch.Tensor]) -> torch.Tensor:
    n = int(args[0].reshape(-1)[0].item())
    return torch.zeros(n, dtype=torch.float32, device=args[0].device)


def _op_ones(args: list[torch.Tensor]) -> torch.Tensor:
    n = int(args[0].reshape(-1)[0].item())
    return torch.ones(n, dtype=torch.float32, device=args[0].device)


OP_TABLE: dict[str, callable] = {
    "+": _op_add,
    "-": _op_sub,
    "*": _op_mul,
    "/": _op_div,
    "modulo": _op_modulo,
    "remainder": _op_remainder,
    "abs": _op_abs,
    "min": _op_min,
    "max": _op_max,
    "=": _op_eq,
    "<": _op_lt,
    ">": _op_gt,
    "<=": _op_le,
    ">=": _op_ge,
    "not": _op_not,
    "and": _op_and,
    "or": _op_or,
    "sin": _op_sin,
    "cos": _op_cos,
    "exp": _op_exp,
    "sqrt": _op_sqrt,
    "log": _op_log,
    "pow": _op_pow,
    "if": _op_if,
    # Vector ops
    "vec": _op_vec,
    "ref": _op_ref,
    "dot": _op_dot,
    "cross": _op_cross,
    "norm": _op_norm,
    "normalize": _op_normalize,
    "vsum": _op_vsum,
    "vlen": _op_vlen,
    "scale": _op_scale,
    # Matrix ops
    "mat": _op_mat,
    "matmul": _op_matmul,
    "matvec": _op_matvec,
    "transpose": _op_transpose,
    "trace": _op_trace,
    "det": _op_det,
    "inv": _op_inv,
    "outer": _op_outer,
    "eye": _op_eye,
    "zeros": _op_zeros,
    "ones": _op_ones,
}


def evaluate_op(op_type: str, args: list[torch.Tensor]) -> torch.Tensor:
    """Evaluate a primitive operation on input tensors."""
    if op_type not in OP_TABLE:
        raise ValueError(f"Unknown operation: {op_type}")
    return OP_TABLE[op_type](args)
