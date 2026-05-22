############################################################
#
# Neural Compiler: Compiling scheme into composable and
#                  differentiable neural network representations
#
# engine.py: Sequential evaluation engine walking compute graphs with primitive operations
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Evaluation engine: execute a ComputeGraph with concrete inputs.

Walks the graph in topological order. Each node computes its value
from its input edges using the fixed-weight primitive operations.

For loops, the engine iteratively evaluates the loop body graph
until it produces a non-recur result. Each iteration feeds the
new parameter values back as inputs to the body graph.
"""

from __future__ import annotations
import torch
from neural_compiler.graph.builder import ComputeGraph
from neural_compiler.ops.primitives import evaluate_op

DEFAULT_MAX_ITERATIONS = 10000
DEFAULT_MAX_RECURSION_DEPTH = 10000


def _to_tensor(val) -> torch.Tensor:
    if isinstance(val, torch.Tensor):
        return val
    return torch.tensor(val, dtype=torch.float32)


def _eval_graph(
    graph: ComputeGraph,
    inputs: dict,
    outer_values: dict[int, torch.Tensor] | None = None,
    max_iter: int = DEFAULT_MAX_ITERATIONS,
    max_depth: int = DEFAULT_MAX_RECURSION_DEPTH,
    depth: int = 0,
) -> dict[int, torch.Tensor]:
    """Evaluate a compute graph, returning values for all nodes."""
    values: dict[int, torch.Tensor] = {}
    order = graph.topological_order()

    for nid in order:
        node = graph.nodes[nid]

        if node.op_type == "const":
            values[nid] = torch.tensor(node.value, dtype=torch.float32)

        elif node.op_type == "input":
            values[nid] = _to_tensor(inputs[node.name])

        elif node.op_type == "loop_param":
            values[nid] = _to_tensor(inputs[node.name])

        elif node.op_type == "func_param":
            values[nid] = _to_tensor(inputs[node.name])

        elif node.op_type == "if":
            arg_tensors = [values[eid] for eid in node.input_edges]
            values[nid] = evaluate_op("if", arg_tensors)

        elif node.op_type == "recur":
            values[nid] = torch.tensor(0.0)

        elif node.op_type == "loop":
            values[nid] = _eval_loop(node, graph, values, max_iter)

        elif node.op_type == "call":
            values[nid] = _eval_call_from_eager(node, graph, values, max_iter, max_depth, depth)

        else:
            arg_tensors = [values[eid] for eid in node.input_edges]
            values[nid] = evaluate_op(node.op_type, arg_tensors)

    return values


def _run_loop_body(
    body_graph: ComputeGraph,
    params: tuple[str, ...],
    current_params: dict[str, float],
    max_iter: int,
    max_depth: int = DEFAULT_MAX_RECURSION_DEPTH,
    depth: int = 0,
) -> torch.Tensor:
    """Run a loop body iteratively until termination."""
    for _ in range(max_iter):
        body_values = _eval_graph(
            body_graph, current_params, max_iter=max_iter, max_depth=max_depth, depth=depth
        )
        root_nid = body_graph.root_id
        root_node = body_graph.nodes[root_nid]

        if root_node.op_type == "recur":
            for i, param in enumerate(params):
                arg_nid = root_node.input_edges[i]
                current_params[param] = body_values[arg_nid]
        elif root_node.op_type == "if":
            result = body_values[root_nid]
            if _check_if_recurs(root_nid, body_graph, body_values, params, current_params):
                continue
            return result
        else:
            return body_values[root_nid]

    raise RuntimeError(f"Loop did not terminate after {max_iter} iterations")


def _eval_loop(
    loop_node,
    outer_graph: ComputeGraph,
    outer_values: dict[int, torch.Tensor],
    max_iter: int,
) -> torch.Tensor:
    """Evaluate a loop by iterating the body graph."""
    loop_body = outer_graph.loops[loop_node.node_id]
    params = loop_body.params

    current_params = {}
    for i, param in enumerate(params):
        init_nid = loop_node.input_edges[i]
        current_params[param] = outer_values[init_nid]
    for i, cap_name in enumerate(loop_body.captures):
        current_params[cap_name] = outer_values[loop_node.input_edges[len(params) + i]]

    return _run_loop_body(loop_body.body_graph, params, current_params, max_iter)


def _check_if_recurs(
    nid: int,
    graph: ComputeGraph,
    values: dict[int, torch.Tensor],
    params: tuple[str, ...],
    current_params: dict,
) -> bool:
    """Check if the result of an if-expression is a recur node.

    When the loop body is: (if test (recur ...) result) or (if test result (recur ...)),
    we need to determine which branch was taken and extract recur args if applicable.
    """
    node = graph.nodes[nid]
    if node.op_type != "if":
        return False

    test_nid = node.input_edges[0]
    then_nid = node.input_edges[1]
    else_nid = node.input_edges[2]

    test_val = values[test_nid].item()
    taken_nid = then_nid if test_val != 0.0 else else_nid
    taken_node = graph.nodes[taken_nid]

    if taken_node.op_type == "recur":
        for i, param in enumerate(params):
            arg_nid = taken_node.input_edges[i]
            current_params[param] = values[arg_nid]
        return True

    if taken_node.op_type == "if":
        return _check_if_recurs(taken_nid, graph, values, params, current_params)

    return False


def _eval_lazy(
    nid: int,
    graph: ComputeGraph,
    inputs: dict,
    memo: dict[int, torch.Tensor],
    max_iter: int,
    max_depth: int,
    depth: int,
) -> torch.Tensor:
    """Demand-driven evaluation: only compute nodes actually needed.

    Unlike topological evaluation, if-nodes evaluate lazily — only the taken
    branch is evaluated. This prevents infinite recursion in base cases.
    """
    if nid in memo:
        return memo[nid]

    node = graph.nodes[nid]

    if node.op_type == "const":
        result = torch.tensor(node.value, dtype=torch.float32)

    elif node.op_type in ("func_param", "input", "loop_param"):
        result = _to_tensor(inputs[node.name])

    elif node.op_type == "if":
        test_val = _eval_lazy(node.input_edges[0], graph, inputs, memo, max_iter, max_depth, depth)
        if test_val.item() != 0.0:
            result = _eval_lazy(node.input_edges[1], graph, inputs, memo, max_iter, max_depth, depth)
        else:
            result = _eval_lazy(node.input_edges[2], graph, inputs, memo, max_iter, max_depth, depth)

    elif node.op_type == "loop":
        loop_body = graph.loops[nid]
        params = loop_body.params
        current_params = {}
        for i, param in enumerate(params):
            current_params[param] = _eval_lazy(
                node.input_edges[i], graph, inputs, memo, max_iter, max_depth, depth
            )
        for i, cap_name in enumerate(loop_body.captures):
            current_params[cap_name] = _eval_lazy(
                node.input_edges[len(params) + i], graph, inputs, memo, max_iter, max_depth, depth
            )
        result = _run_loop_body(
            loop_body.body_graph, params, current_params, max_iter, max_depth, depth
        )

    elif node.op_type == "call":
        result = _eval_call(node, graph, inputs, memo, max_iter, max_depth, depth)

    elif node.op_type == "recur":
        result = torch.tensor(0.0)

    else:
        arg_vals = [_eval_lazy(e, graph, inputs, memo, max_iter, max_depth, depth) for e in node.input_edges]
        result = evaluate_op(node.op_type, arg_vals)

    memo[nid] = result
    return result


def _eval_call_from_eager(
    call_node,
    graph: ComputeGraph,
    values: dict[int, torch.Tensor],
    max_iter: int,
    max_depth: int,
    depth: int,
) -> torch.Tensor:
    """Bridge from eager topological eval into lazy function body eval."""
    if depth >= max_depth:
        raise RuntimeError(f"Recursion depth exceeded {max_depth}")

    func_name = call_node.call_target
    func_body = graph.functions[func_name]
    body_graph = func_body.body_graph

    args = {}
    for i, param in enumerate(func_body.params):
        arg_nid = call_node.input_edges[i]
        args[param] = values[arg_nid]

    body_memo: dict[int, torch.Tensor] = {}
    return _eval_lazy(
        body_graph.root_id, body_graph, args, body_memo, max_iter, max_depth, depth + 1
    )


def _eval_call(
    call_node,
    graph: ComputeGraph,
    inputs: dict,
    memo: dict[int, torch.Tensor],
    max_iter: int,
    max_depth: int,
    depth: int,
) -> torch.Tensor:
    """Evaluate a recursive function call within lazy evaluation."""
    if depth >= max_depth:
        raise RuntimeError(f"Recursion depth exceeded {max_depth}")

    func_name = call_node.call_target
    func_body = graph.functions[func_name]
    body_graph = func_body.body_graph

    args = {}
    for i, param in enumerate(func_body.params):
        arg_nid = call_node.input_edges[i]
        args[param] = _eval_lazy(arg_nid, graph, inputs, memo, max_iter, max_depth, depth)

    body_memo: dict[int, torch.Tensor] = {}
    return _eval_lazy(
        body_graph.root_id, body_graph, args, body_memo, max_iter, max_depth, depth + 1
    )


def evaluate(
    graph: ComputeGraph,
    inputs: dict = None,
    max_iter: int = DEFAULT_MAX_ITERATIONS,
    max_depth: int = DEFAULT_MAX_RECURSION_DEPTH,
):
    """Evaluate a compute graph with the given input values.

    Args:
        graph: The compiled compute graph.
        inputs: Map of input variable names to values (float or tensor).
        max_iter: Maximum loop iterations before raising an error.

    Returns:
        Scalar float for scalar programs, or torch.Tensor for vector/matrix programs.
    """
    inputs = inputs or {}
    for name in graph.input_names:
        if name not in inputs:
            raise ValueError(f"Missing input: {name}")

    values = _eval_graph(graph, inputs, max_iter=max_iter, max_depth=max_depth)

    if graph.root_id is None:
        raise ValueError("Graph has no root node")

    result = values[graph.root_id]
    if result.dim() == 0:
        return result.item()
    return result
