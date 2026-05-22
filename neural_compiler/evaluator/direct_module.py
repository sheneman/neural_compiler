############################################################
#
# Neural Compiler: Compiling scheme into composable and
#                  differentiable neural network representations
#
# direct_module.py: PyTorch nn.Module for flat instruction-sequence evaluation in topological order
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Direct-execution module for compiled programs.

Compiles the compute graph into a flat instruction sequence executed in
topological order using raw PyTorch operations — just arithmetic.
"""

from __future__ import annotations
from typing import Callable
import torch
import torch.nn as nn
from neural_compiler.graph.builder import ComputeGraph
from neural_compiler.ops.primitives import OP_TABLE

DEFAULT_MAX_ITERATIONS = 10000
DEFAULT_MAX_RECURSION_DEPTH = 10000

_OP_FN: dict[str, Callable] = {
    "+": torch.add,
    "-": torch.sub,
    "*": torch.mul,
    "/": torch.div,
    "pow": torch.pow,
    "sin": torch.sin,
    "cos": torch.cos,
    "exp": torch.exp,
    "abs": torch.abs,
}

def _make_sqrt(x: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.clamp(x, min=1e-8))

def _make_log(x: torch.Tensor) -> torch.Tensor:
    return torch.log(torch.clamp(x, min=1e-8))


# Instruction types for the flat execution loop
_INST_CONST = 0
_INST_INPUT = 1
_INST_UNARY = 2
_INST_BINARY = 3
_INST_IF = 4
_INST_LOOP = 5
_INST_CALL = 6
_INST_RECUR = 7
_INST_VARIADIC = 8

_UNARY_DISPATCH: dict[str, Callable] = {
    "sin": torch.sin,
    "cos": torch.cos,
    "exp": torch.exp,
    "abs": torch.abs,
    "sqrt": _make_sqrt,
    "log": _make_log,
    "not": lambda x: (x == 0.0).float(),
    "norm": lambda x: torch.norm(x, dim=-1),
    "normalize": lambda x: x / torch.norm(x, dim=-1, keepdim=True).clamp(min=1e-8),
    "transpose": lambda x: x.transpose(-2, -1),
    "trace": lambda x: torch.diagonal(x, dim1=-2, dim2=-1).sum(dim=-1),
    "det": lambda x: torch.linalg.det(x),
    "inv": lambda x: torch.linalg.inv(x),
    "vsum": lambda x: x.sum(dim=-1),
    "vlen": lambda x: torch.tensor(x.shape[-1], dtype=torch.float32, device=x.device),
    "eye": lambda x: torch.eye(int(x.reshape(-1)[0].item()), dtype=torch.float32, device=x.device),
    "zeros": lambda x: torch.zeros(int(x.reshape(-1)[0].item()), dtype=torch.float32, device=x.device),
    "ones": lambda x: torch.ones(int(x.reshape(-1)[0].item()), dtype=torch.float32, device=x.device),
}

_BINARY_DISPATCH: dict[str, Callable] = {
    "+": torch.add,
    "-": torch.sub,
    "*": torch.mul,
    "/": torch.div,
    "pow": torch.pow,
    "modulo": torch.fmod,
    "remainder": torch.remainder,
    "min": torch.minimum,
    "max": torch.maximum,
    "=": lambda a, b: (a == b).float(),
    "<": lambda a, b: (a < b).float(),
    ">": lambda a, b: (a > b).float(),
    "<=": lambda a, b: (a <= b).float(),
    ">=": lambda a, b: (a >= b).float(),
    "and": lambda a, b: ((a != 0.0) & (b != 0.0)).float(),
    "or": lambda a, b: ((a != 0.0) | (b != 0.0)).float(),
    "dot": lambda a, b: (a * b).sum(dim=-1),
    "cross": lambda a, b: torch.linalg.cross(a, b, dim=-1),
    "scale": lambda a, b: a * b,
    "matvec": lambda a, b: torch.matmul(a, b.unsqueeze(-1)).squeeze(-1),
    "matmul": lambda a, b: torch.matmul(a, b),
    "outer": lambda a, b: a.unsqueeze(-1) * b.unsqueeze(-2),
    "ref": lambda v, i: (v[..., i.long()] if v.dim() == 1
                         else torch.gather(v, -1, i.long().unsqueeze(-1)).squeeze(-1)),
}

_VARIADIC_DISPATCH: dict[str, Callable] = {
    "vec": lambda args: torch.stack(args, dim=-1),
    "mat": lambda args: torch.stack(args, dim=-2),
}

_VARIADIC_OPS: set[str] = set(_VARIADIC_DISPATCH.keys())


def _compile_instructions(graph: ComputeGraph):
    """Pre-compile a compute graph into a flat instruction list.

    Returns:
        instructions: list of (inst_type, output_slot, payload)
        slot_for: dict mapping node_id -> slot index
        input_slots: dict mapping input_name -> slot index
        num_slots: total number of value slots
        root_slot: slot index of the root/output node
        const_values: list of (slot, value) for constant nodes
    """
    order = graph.topological_order()
    slot_for: dict[int, int] = {}
    instructions = []
    input_slots: dict[str, int] = {}
    const_values: list[tuple[int, float]] = []
    num_slots = 0

    for nid in order:
        node = graph.nodes[nid]
        slot = num_slots
        slot_for[nid] = slot
        num_slots += 1

        if node.op_type == "const":
            instructions.append((_INST_CONST, slot, node.value))
            const_values.append((slot, node.value))

        elif node.op_type in ("input", "loop_param", "func_param"):
            input_slots[node.name] = slot
            instructions.append((_INST_INPUT, slot, node.name))

        elif node.op_type == "recur":
            in_slots = [slot_for[e] for e in node.input_edges]
            instructions.append((_INST_RECUR, slot, in_slots))

        elif node.op_type == "if":
            test_slot = slot_for[node.input_edges[0]]
            then_slot = slot_for[node.input_edges[1]]
            else_slot = slot_for[node.input_edges[2]]
            instructions.append((_INST_IF, slot, (test_slot, then_slot, else_slot)))

        elif node.op_type == "loop":
            in_slots = [slot_for[e] for e in node.input_edges]
            instructions.append((_INST_LOOP, slot, (nid, in_slots)))

        elif node.op_type == "call":
            in_slots = [slot_for[e] for e in node.input_edges]
            instructions.append((_INST_CALL, slot, (node.call_target, in_slots)))

        elif node.op_type in _VARIADIC_OPS:
            fn = _VARIADIC_DISPATCH[node.op_type]
            in_slots = [slot_for[e] for e in node.input_edges]
            instructions.append((_INST_VARIADIC, slot, (fn, in_slots)))

        elif node.op_type in _UNARY_DISPATCH:
            fn = _UNARY_DISPATCH[node.op_type]
            in_slot = slot_for[node.input_edges[0]]
            instructions.append((_INST_UNARY, slot, (fn, in_slot)))

        elif node.op_type in _BINARY_DISPATCH:
            fn = _BINARY_DISPATCH[node.op_type]
            a_slot = slot_for[node.input_edges[0]]
            b_slot = slot_for[node.input_edges[1]]
            instructions.append((_INST_BINARY, slot, (fn, a_slot, b_slot)))

        else:
            raise ValueError(f"Unknown op type: {node.op_type}")

    root_slot = slot_for[graph.root_id] if graph.root_id is not None else 0
    return instructions, slot_for, input_slots, num_slots, root_slot, const_values


class _LoopBodyDirect:
    """Pre-compiled loop body for direct execution."""

    def __init__(self, body_graph: ComputeGraph, params: tuple[str, ...],
                 captures: tuple[str, ...] = ()) -> None:
        self.body_graph = body_graph
        self.params = params
        self.captures = captures
        compiled = _compile_instructions(body_graph)
        self.instructions = compiled[0]
        self.slot_for = compiled[1]
        self.input_slots = compiled[2]
        self.num_slots = compiled[3]
        self.root_slot = compiled[4]
        self.const_values = compiled[5]


class DirectModule(nn.Module):
    """Direct-execution compiled program module.

    Evaluates a compiled Scheme program by walking a pre-computed
    instruction list in topological order. Each instruction is a
    single PyTorch operation.

    Supports forward() for scalar inputs and forward_batch() for batched inputs.
    """

    def __init__(self, graph: ComputeGraph,
                 max_iter: int = DEFAULT_MAX_ITERATIONS,
                 max_depth: int = DEFAULT_MAX_RECURSION_DEPTH) -> None:
        super().__init__()
        self.graph = graph
        self.max_iter = max_iter
        self.max_depth = max_depth

        compiled = _compile_instructions(graph)
        self._instructions = compiled[0]
        self._slot_for = compiled[1]
        self._input_slots = compiled[2]
        self._num_slots = compiled[3]
        self._root_slot = compiled[4]

        const_vals = torch.zeros(self._num_slots, dtype=torch.float32)
        const_mask = torch.zeros(self._num_slots, dtype=torch.bool)
        for slot, val in compiled[5]:
            const_vals[slot] = val
            const_mask[slot] = True
        self.register_buffer("_const_vals", const_vals)
        self.register_buffer("_const_mask", const_mask)

        self._loop_bodies: dict[int, _LoopBodyDirect] = {}
        for loop_nid, loop_body in graph.loops.items():
            self._loop_bodies[loop_nid] = _LoopBodyDirect(
                loop_body.body_graph, loop_body.params, loop_body.captures
            )

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Evaluate the compiled program on scalar inputs.

        Args:
            inputs: dict mapping input variable names to scalar tensors.

        Returns:
            Scalar tensor with the program result.
        """
        device = next(iter(inputs.values())).device if inputs else self._const_vals.device

        if self.graph.has_functions:
            return self._forward_with_functions(inputs, device)

        values = [None] * self._num_slots
        self._exec_instructions(self._instructions, values, inputs, device)
        return values[self._root_slot]

    def forward_batch_with_intermediates(
        self, inputs: dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Evaluate and return (output, all_intermediates).

        The intermediates tensor has shape [batch_size, num_slots] containing
        every computed value in topological order — useful for structural
        feature extraction.
        """
        if not inputs:
            raise ValueError("forward_batch_with_intermediates requires at least one input")
        device = next(iter(inputs.values())).device
        batch_size = next(iter(inputs.values())).shape[0]
        values = [None] * self._num_slots
        self._exec_instructions_batch(
            self._instructions, values, inputs, device, batch_size
        )
        intermediates = torch.stack(
            [v if v is not None else torch.zeros(batch_size, device=device)
             for v in values],
            dim=1,
        )
        return values[self._root_slot], intermediates

    def forward_batch(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Evaluate the compiled program on a batch of inputs.

        Args:
            inputs: dict mapping input variable names to 1-D tensors
                    of shape [batch_size].

        Returns:
            1-D tensor of shape [batch_size].
        """
        if self.graph.has_functions:
            raise NotImplementedError(
                "Batched execution does not support general recursion"
            )
        if not inputs:
            raise ValueError("forward_batch requires at least one input")

        device = next(iter(inputs.values())).device
        batch_size = next(iter(inputs.values())).shape[0]

        if self.graph.has_loops:
            # For graphs with loops, execute the outer instructions which
            # will dispatch to _eval_loop_batch for loop nodes
            values = [None] * self._num_slots
            self._exec_instructions_batch(
                self._instructions, values, inputs, device, batch_size
            )
            return values[self._root_slot]

        values = [None] * self._num_slots
        self._exec_instructions_batch(
            self._instructions, values, inputs, device, batch_size
        )
        return values[self._root_slot]

    def _exec_instructions(self, instructions, values, inputs, device):
        """Execute a flat instruction list for scalar evaluation."""
        for inst_type, slot, payload in instructions:
            if inst_type == _INST_CONST:
                values[slot] = torch.tensor(payload, dtype=torch.float32, device=device)

            elif inst_type == _INST_INPUT:
                name = payload
                values[slot] = inputs[name].to(device) if isinstance(inputs[name], torch.Tensor) else torch.tensor(inputs[name], dtype=torch.float32, device=device)

            elif inst_type == _INST_UNARY:
                fn, in_slot = payload
                values[slot] = fn(values[in_slot])

            elif inst_type == _INST_BINARY:
                fn, a_slot, b_slot = payload
                values[slot] = fn(values[a_slot], values[b_slot])

            elif inst_type == _INST_IF:
                test_slot, then_slot, else_slot = payload
                sel = (values[test_slot] != 0.0).float()
                then_val = values[then_slot]
                while sel.dim() < then_val.dim():
                    sel = sel.unsqueeze(-1)
                values[slot] = sel * then_val + (1.0 - sel) * values[else_slot]

            elif inst_type == _INST_LOOP:
                loop_nid, in_slots = payload
                values[slot] = self._eval_loop_scalar(
                    loop_nid, values, in_slots, device
                )

            elif inst_type == _INST_RECUR:
                values[slot] = torch.tensor(0.0, device=device)

            elif inst_type == _INST_VARIADIC:
                fn, in_slots = payload
                values[slot] = fn([values[s] for s in in_slots])

            elif inst_type == _INST_CALL:
                pass

    def _exec_instructions_batch(self, instructions, values, inputs, device,
                                  batch_size):
        """Execute a flat instruction list for batched evaluation."""
        for inst_type, slot, payload in instructions:
            if inst_type == _INST_CONST:
                values[slot] = torch.full((batch_size,), payload, dtype=torch.float32, device=device)

            elif inst_type == _INST_INPUT:
                name = payload
                values[slot] = inputs[name].to(device)

            elif inst_type == _INST_UNARY:
                fn, in_slot = payload
                values[slot] = fn(values[in_slot])

            elif inst_type == _INST_BINARY:
                fn, a_slot, b_slot = payload
                values[slot] = fn(values[a_slot], values[b_slot])

            elif inst_type == _INST_IF:
                test_slot, then_slot, else_slot = payload
                sel = (values[test_slot] != 0.0).float()
                then_val = values[then_slot]
                while sel.dim() < then_val.dim():
                    sel = sel.unsqueeze(-1)
                values[slot] = sel * then_val + (1.0 - sel) * values[else_slot]

            elif inst_type == _INST_LOOP:
                loop_nid, in_slots = payload
                values[slot] = self._eval_loop_batch(
                    loop_nid, values, in_slots, device, batch_size
                )

            elif inst_type == _INST_RECUR:
                values[slot] = torch.zeros(batch_size, device=device)

            elif inst_type == _INST_VARIADIC:
                fn, in_slots = payload
                values[slot] = fn([values[s] for s in in_slots])

    def _eval_loop_scalar(self, loop_nid, outer_values, in_slots, device):
        """Scalar loop evaluation."""
        body = self._loop_bodies[loop_nid]
        current = {}
        for i, param in enumerate(body.params):
            current[param] = outer_values[in_slots[i]]
        for i, cap_name in enumerate(body.captures):
            current[cap_name] = outer_values[in_slots[len(body.params) + i]]

        if body.body_graph.has_functions or body.body_graph.has_loops:
            return self._run_loop_sequential_fallback(
                body.body_graph, body.params, current, device
            )

        root_node = body.body_graph.nodes[body.body_graph.root_id]

        for _ in range(self.max_iter):
            values = [None] * body.num_slots
            self._exec_instructions(body.instructions, values, current, device)

            if root_node.op_type == "recur":
                for i, param in enumerate(body.params):
                    arg_slot = body.slot_for[root_node.input_edges[i]]
                    current[param] = values[arg_slot]
                continue

            if root_node.op_type == "if":
                recurred, new_params = self._check_if_recurs_direct(
                    body.body_graph.root_id, body.body_graph,
                    values, body.slot_for, body.params
                )
                if recurred:
                    current.update(new_params)
                    continue

            return values[body.root_slot]

        raise RuntimeError(f"Loop did not terminate after {self.max_iter} iterations")

    def _eval_loop_batch(self, loop_nid, outer_values, in_slots, device,
                          batch_size):
        """Batched loop evaluation with masking."""
        body = self._loop_bodies[loop_nid]
        current = {}
        for i, param in enumerate(body.params):
            current[param] = outer_values[in_slots[i]]
        for i, cap_name in enumerate(body.captures):
            current[cap_name] = outer_values[in_slots[len(body.params) + i]]

        if body.body_graph.has_functions or body.body_graph.has_loops:
            raise NotImplementedError(
                "Batched loop does not support nested loops or function calls"
            )

        root_node = body.body_graph.nodes[body.body_graph.root_id]
        result = None
        active = torch.ones(batch_size, dtype=torch.bool, device=device)

        for _ in range(self.max_iter):
            if not active.any():
                break

            values = [None] * body.num_slots
            self._exec_instructions_batch(
                body.instructions, values, current, device, batch_size
            )

            if root_node.op_type == "if":
                test_nid = root_node.input_edges[0]
                test_slot = body.slot_for[test_nid]
                test_vals = values[test_slot]

                then_nid = root_node.input_edges[1]
                else_nid = root_node.input_edges[2]
                then_node = body.body_graph.nodes[then_nid]
                else_node = body.body_graph.nodes[else_nid]

                cond = test_vals != 0.0

                if then_node.op_type == "recur":
                    recur_node, result_nid = then_node, else_nid
                    recur_mask = cond & active
                    done_mask = (~cond) & active
                elif else_node.op_type == "recur":
                    recur_node, result_nid = else_node, then_nid
                    recur_mask = (~cond) & active
                    done_mask = cond & active
                else:
                    raise RuntimeError("Loop body if-expression has no recur branch")

                if done_mask.any():
                    res_slot = body.slot_for[result_nid]
                    res_val = values[res_slot]
                    if result is None:
                        result = torch.zeros_like(res_val)
                    result[done_mask] = res_val[done_mask]
                    active[done_mask] = False

                if recur_mask.any():
                    for i, param in enumerate(body.params):
                        arg_slot = body.slot_for[recur_node.input_edges[i]]
                        mask = active
                        while mask.dim() < values[arg_slot].dim():
                            mask = mask.unsqueeze(-1)
                        current[param] = torch.where(
                            mask, values[arg_slot], current[param]
                        )

            elif root_node.op_type == "recur":
                for i, param in enumerate(body.params):
                    arg_slot = body.slot_for[root_node.input_edges[i]]
                    current[param] = values[arg_slot]

            else:
                res_val = values[body.root_slot]
                if result is None:
                    result = torch.zeros_like(res_val)
                result[active] = res_val[active]
                return result

        if active.any():
            raise RuntimeError(
                f"Loop did not terminate for {active.sum().item()} batch elements"
            )
        return result

    def _check_if_recurs_direct(self, nid, graph, values, slot_for, params):
        """Check if an if-expression result is a recur in direct mode."""
        node = graph.nodes[nid]
        if node.op_type != "if":
            return False, {}

        test_slot = slot_for[node.input_edges[0]]
        test_val = values[test_slot].item()
        taken_nid = node.input_edges[1] if test_val != 0.0 else node.input_edges[2]
        taken_node = graph.nodes[taken_nid]

        if taken_node.op_type == "recur":
            new_params = {}
            for i, param in enumerate(params):
                arg_slot = slot_for[taken_node.input_edges[i]]
                new_params[param] = values[arg_slot]
            return True, new_params

        if taken_node.op_type == "if":
            return self._check_if_recurs_direct(
                taken_nid, graph, values, slot_for, params
            )

        return False, {}

    def _forward_with_functions(self, inputs, device):
        """Forward pass for graphs with recursive functions (lazy eval)."""
        from neural_compiler.ops.primitives import evaluate_op

        input_map = {}
        for name in self.graph.input_names:
            input_map[name] = inputs[name].to(device) if isinstance(inputs[name], torch.Tensor) else torch.tensor(inputs[name], dtype=torch.float32, device=device)

        memo: dict[int, torch.Tensor] = {}
        return self._eval_lazy(
            self.graph, self.graph.root_id, input_map, memo, device, 0
        )

    def _eval_lazy(self, graph, nid, inputs, memo, device, depth):
        """Demand-driven evaluation for recursive functions."""
        from neural_compiler.ops.primitives import evaluate_op

        if nid in memo:
            return memo[nid]

        node = graph.nodes[nid]

        if node.op_type == "const":
            result = torch.tensor(node.value, dtype=torch.float32, device=device)
        elif node.op_type in ("func_param", "input", "loop_param"):
            val = inputs[node.name]
            result = val.to(device) if isinstance(val, torch.Tensor) else torch.tensor(val, dtype=torch.float32, device=device)
        elif node.op_type == "if":
            test_val = self._eval_lazy(graph, node.input_edges[0], inputs, memo, device, depth)
            if test_val.item() != 0.0:
                result = self._eval_lazy(graph, node.input_edges[1], inputs, memo, device, depth)
            else:
                result = self._eval_lazy(graph, node.input_edges[2], inputs, memo, device, depth)
        elif node.op_type == "loop":
            loop_body = graph.loops[nid]
            params = loop_body.params
            init_vals = {}
            for i, param in enumerate(params):
                init_vals[param] = self._eval_lazy(graph, node.input_edges[i], inputs, memo, device, depth)
            for i, cap_name in enumerate(loop_body.captures):
                init_vals[cap_name] = self._eval_lazy(
                    graph, node.input_edges[len(params) + i], inputs, memo, device, depth
                )
            result = self._run_loop_sequential_fallback(
                loop_body.body_graph, params, init_vals, device, depth
            )
        elif node.op_type == "call":
            if depth >= self.max_depth:
                raise RuntimeError(f"Recursion depth exceeded {self.max_depth}")
            func_body = graph.functions[node.call_target]
            args = {}
            for i, param in enumerate(func_body.params):
                args[param] = self._eval_lazy(graph, node.input_edges[i], inputs, memo, device, depth)
            body_memo: dict[int, torch.Tensor] = {}
            result = self._eval_lazy(
                func_body.body_graph, func_body.body_graph.root_id,
                args, body_memo, device, depth + 1
            )
        else:
            arg_vals = [
                self._eval_lazy(graph, e, inputs, memo, device, depth)
                for e in node.input_edges
            ]
            result = evaluate_op(node.op_type, arg_vals)

        memo[nid] = result
        return result

    def _run_loop_sequential_fallback(self, body_graph, params, init_vals,
                                       device, depth=0):
        """Sequential loop fallback for complex bodies."""
        from neural_compiler.ops.primitives import evaluate_op

        current = dict(init_vals)
        body_topo = body_graph.topological_order()

        for _ in range(self.max_iter):
            bv: dict[int, torch.Tensor] = {}
            for nid in body_topo:
                node = body_graph.nodes[nid]
                if node.op_type == "const":
                    bv[nid] = torch.tensor(node.value, dtype=torch.float32, device=device)
                elif node.op_type in ("loop_param", "input"):
                    bv[nid] = current[node.name]
                elif node.op_type == "recur":
                    bv[nid] = torch.tensor(0.0, device=device)
                elif node.op_type == "if":
                    arg_tensors = [bv[eid] for eid in node.input_edges]
                    bv[nid] = evaluate_op("if", arg_tensors)
                elif node.op_type == "call":
                    func_body = body_graph.functions[node.call_target]
                    args = {}
                    for i, p in enumerate(func_body.params):
                        args[p] = bv[node.input_edges[i]]
                    body_memo: dict[int, torch.Tensor] = {}
                    bv[nid] = self._eval_lazy(
                        func_body.body_graph, func_body.body_graph.root_id,
                        args, body_memo, device, depth + 1
                    )
                elif node.op_type == "loop":
                    inner = body_graph.loops[nid]
                    inner_init = {}
                    for i, p in enumerate(inner.params):
                        inner_init[p] = bv[node.input_edges[i]]
                    for i, cap_name in enumerate(inner.captures):
                        inner_init[cap_name] = bv[node.input_edges[len(inner.params) + i]]
                    bv[nid] = self._run_loop_sequential_fallback(
                        inner.body_graph, inner.params, inner_init, device, depth
                    )
                else:
                    arg_tensors = [bv[eid] for eid in node.input_edges]
                    bv[nid] = evaluate_op(node.op_type, arg_tensors)

            root_node = body_graph.nodes[body_graph.root_id]
            if root_node.op_type == "recur":
                for i, param in enumerate(params):
                    current[param] = bv[root_node.input_edges[i]]
                continue

            if root_node.op_type == "if":
                test_val = bv[root_node.input_edges[0]].item()
                taken_nid = root_node.input_edges[1] if test_val != 0.0 else root_node.input_edges[2]
                taken_node = body_graph.nodes[taken_nid]
                if taken_node.op_type == "recur":
                    for i, param in enumerate(params):
                        current[param] = bv[taken_node.input_edges[i]]
                    continue

            return bv[body_graph.root_id]

        raise RuntimeError(f"Loop did not terminate after {self.max_iter} iterations")
