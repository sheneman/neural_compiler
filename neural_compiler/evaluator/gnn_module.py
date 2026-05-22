############################################################
#
# Neural Compiler: Compiling scheme into composable and
#                  differentiable neural network representations
#
# gnn_module.py: PyTorch nn.Module wrapper using fixed-weight message passing for graph evaluation
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""PyTorch nn.Module wrapper for the compute graph.

This wraps the compiled graph as a standard torch.nn.Module,
making it usable in PyTorch pipelines. The forward pass performs
deterministic message passing with fixed (non-learned) weights.

For loops, the module iteratively evaluates the loop body until
the result is not a recur node.
"""

from __future__ import annotations
import torch
import torch.nn as nn
from neural_compiler.graph.builder import ComputeGraph
from neural_compiler.ops.primitives import evaluate_op

DEFAULT_MAX_ITERATIONS = 10000
DEFAULT_MAX_RECURSION_DEPTH = 10000


class SchemeGNN(nn.Module):
    """A GNN that executes a compiled Scheme program.

    The graph topology encodes the program's dataflow structure.
    Node operations are fixed (not learned). The forward pass
    propagates values from input/const leaves to the root node
    in topological order.
    """

    def __init__(self, graph: ComputeGraph, max_iter: int = DEFAULT_MAX_ITERATIONS, max_depth: int = DEFAULT_MAX_RECURSION_DEPTH) -> None:
        super().__init__()
        self.graph = graph
        self.max_iter = max_iter
        self.max_depth = max_depth
        self._topo_order = graph.topological_order()

        for nid in self._topo_order:
            node = graph.nodes[nid]
            if node.op_type == "const":
                self.register_buffer(
                    f"const_{nid}",
                    torch.tensor(node.value, dtype=torch.float32),
                )

    def forward(self, inputs: dict[str, torch.Tensor]) -> torch.Tensor:
        values: dict[int, torch.Tensor] = {}

        for nid in self._topo_order:
            node = self.graph.nodes[nid]

            if node.op_type == "const":
                values[nid] = getattr(self, f"const_{nid}")
            elif node.op_type == "input":
                values[nid] = inputs[node.name]
            elif node.op_type == "loop":
                values[nid] = self._eval_loop(nid, values)
            elif node.op_type == "call":
                values[nid] = self._eval_call_from_eager(node, values)
            elif node.op_type in ("recur", "loop_param", "func_param"):
                values[nid] = torch.tensor(0.0)
            else:
                arg_tensors = [values[eid] for eid in node.input_edges]
                values[nid] = evaluate_op(node.op_type, arg_tensors)

        return values[self.graph.root_id]

    def _eval_loop(self, loop_nid: int, outer_values: dict[int, torch.Tensor]) -> torch.Tensor:
        loop_node = self.graph.nodes[loop_nid]
        loop_body = self.graph.loops[loop_nid]
        params = loop_body.params

        init_vals = {}
        for i, param in enumerate(params):
            init_vals[param] = outer_values[loop_node.input_edges[i]]
        for i, cap_name in enumerate(loop_body.captures):
            init_vals[cap_name] = outer_values[loop_node.input_edges[len(params) + i]]

        return self._run_loop(loop_body.body_graph, params, init_vals)

    def _run_loop(
        self,
        body_graph: ComputeGraph,
        params: tuple[str, ...],
        init_vals: dict[str, torch.Tensor],
        depth: int = 0,
    ) -> torch.Tensor:
        """Run a loop body iteratively. Handles call and nested loop nodes."""
        current = dict(init_vals)
        body_topo = body_graph.topological_order()

        for _ in range(self.max_iter):
            bv: dict[int, torch.Tensor] = {}
            for nid in body_topo:
                node = body_graph.nodes[nid]
                if node.op_type == "const":
                    bv[nid] = torch.tensor(node.value, dtype=torch.float32)
                elif node.op_type in ("loop_param", "input"):
                    bv[nid] = current[node.name]
                elif node.op_type == "recur":
                    bv[nid] = torch.tensor(0.0)
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
                        args, body_memo, depth + 1,
                    )
                elif node.op_type == "loop":
                    inner = body_graph.loops[nid]
                    inner_init = {}
                    for i, p in enumerate(inner.params):
                        inner_init[p] = bv[node.input_edges[i]]
                    for i, cap_name in enumerate(inner.captures):
                        inner_init[cap_name] = bv[node.input_edges[len(inner.params) + i]]
                    bv[nid] = self._run_loop(inner.body_graph, inner.params, inner_init, depth)
                else:
                    arg_tensors = [bv[eid] for eid in node.input_edges]
                    bv[nid] = evaluate_op(node.op_type, arg_tensors)

            root_node = body_graph.nodes[body_graph.root_id]
            if root_node.op_type == "recur":
                for i, param in enumerate(params):
                    current[param] = bv[root_node.input_edges[i]]
                continue

            if root_node.op_type == "if":
                recurred, new_params = self._check_if_recurs(
                    body_graph.root_id, body_graph, bv, params
                )
                if recurred:
                    current.update(new_params)
                    continue

            return bv[body_graph.root_id]

        raise RuntimeError(f"Loop did not terminate after {self.max_iter} iterations")

    def _eval_call_from_eager(
        self, call_node, values: dict[int, torch.Tensor], depth: int = 0
    ) -> torch.Tensor:
        if depth >= self.max_depth:
            raise RuntimeError(f"Recursion depth exceeded {self.max_depth}")
        func_body = self.graph.functions[call_node.call_target]
        args = {}
        for i, param in enumerate(func_body.params):
            args[param] = values[call_node.input_edges[i]]
        memo: dict[int, torch.Tensor] = {}
        return self._eval_lazy(func_body.body_graph, func_body.body_graph.root_id, args, memo, depth + 1)

    def _eval_lazy(
        self,
        graph: ComputeGraph,
        nid: int,
        inputs: dict[str, torch.Tensor],
        memo: dict[int, torch.Tensor],
        depth: int,
    ) -> torch.Tensor:
        if nid in memo:
            return memo[nid]
        node = graph.nodes[nid]

        if node.op_type == "const":
            result = torch.tensor(node.value, dtype=torch.float32)
        elif node.op_type in ("func_param", "input", "loop_param"):
            result = inputs[node.name]
        elif node.op_type == "if":
            test_val = self._eval_lazy(graph, node.input_edges[0], inputs, memo, depth)
            if test_val.item() != 0.0:
                result = self._eval_lazy(graph, node.input_edges[1], inputs, memo, depth)
            else:
                result = self._eval_lazy(graph, node.input_edges[2], inputs, memo, depth)
        elif node.op_type == "loop":
            loop_body = graph.loops[nid]
            params = loop_body.params
            init_vals = {}
            for i, param in enumerate(params):
                init_vals[param] = self._eval_lazy(graph, node.input_edges[i], inputs, memo, depth)
            for i, cap_name in enumerate(loop_body.captures):
                init_vals[cap_name] = self._eval_lazy(graph, node.input_edges[len(params) + i], inputs, memo, depth)
            result = self._run_loop(loop_body.body_graph, params, init_vals, depth)
        elif node.op_type == "call":
            if depth >= self.max_depth:
                raise RuntimeError(f"Recursion depth exceeded {self.max_depth}")
            func_body = graph.functions[node.call_target]
            args = {}
            for i, param in enumerate(func_body.params):
                args[param] = self._eval_lazy(graph, node.input_edges[i], inputs, memo, depth)
            body_memo: dict[int, torch.Tensor] = {}
            result = self._eval_lazy(func_body.body_graph, func_body.body_graph.root_id, args, body_memo, depth + 1)
        else:
            arg_vals = [self._eval_lazy(graph, e, inputs, memo, depth) for e in node.input_edges]
            result = evaluate_op(node.op_type, arg_vals)

        memo[nid] = result
        return result

    def _check_if_recurs(
        self,
        nid: int,
        graph: ComputeGraph,
        values: dict[int, torch.Tensor],
        params: tuple[str, ...],
    ) -> tuple[bool, dict[str, torch.Tensor]]:
        node = graph.nodes[nid]
        if node.op_type != "if":
            return False, {}

        test_val = values[node.input_edges[0]].item()
        taken_nid = node.input_edges[1] if test_val != 0.0 else node.input_edges[2]
        taken_node = graph.nodes[taken_nid]

        if taken_node.op_type == "recur":
            new_params = {}
            for i, param in enumerate(params):
                new_params[param] = values[taken_node.input_edges[i]]
            return True, new_params

        if taken_node.op_type == "if":
            return self._check_if_recurs(taken_nid, graph, values, params)

        return False, {}
