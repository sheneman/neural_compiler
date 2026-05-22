############################################################
#
# Neural Compiler: Compiling scheme into composable and
#                  differentiable neural network representations
#
# builder.py: Builds PyTorch-compatible dataflow graphs from ANF representation
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Build a dataflow graph from ANF.

Each ANF let-binding becomes a node. Edges flow from producers to consumers
(from arguments to the operation that uses them). The graph is a DAG for
non-recursive expressions.

For loop/recur constructs, the loop body is compiled into a separate
ComputeGraph (the LoopBody). At runtime, the loop body graph is
evaluated iteratively until it produces a non-recur result.

For letrec constructs, each function body is compiled into a separate
ComputeGraph (FunctionBody). Call nodes invoke these bodies lazily.

All composition patterns are supported: loops inside function bodies,
letrec inside loop bodies, nested loops, etc.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from neural_compiler.anf.anf_nodes import (
    ANFNode,
    ANFConst,
    ANFVar,
    ANFLet,
    ANFIf,
    ANFApp,
    ANFLambda,
    ANFLetrec,
    ANFLoop,
    ANFRecur,
)
from neural_compiler.parser.ast_nodes import PRIMITIVES


@dataclass
class GraphNode:
    """A node in the compute graph."""

    node_id: int
    name: str
    op_type: str  # "const", "var", "input", "if", "loop", "call", or a primitive
    value: float | bool | None = None
    input_edges: list[int] = field(default_factory=list)
    input_names: list[str] = field(default_factory=list)
    call_target: str | None = None


@dataclass
class LoopBody:
    """A loop body: a compute graph that runs iteratively.

    The body graph takes loop variables as inputs and produces either:
    - A result value (when the loop terminates)
    - New loop variable values + a recur flag (when the loop continues)

    The state vector layout:
      [param0, param1, ..., paramN, result, recur_flag]
    """

    params: tuple[str, ...]
    body_graph: ComputeGraph
    captures: tuple[str, ...] = ()
    recur_flag_id: int | None = None
    result_id: int | None = None
    recur_arg_ids: dict[str, list[tuple[list[int], list[str]]]] = field(
        default_factory=dict
    )


@dataclass
class FunctionBody:
    """A named recursive function: a compute graph invoked by call nodes."""

    name: str
    params: tuple[str, ...]
    body_graph: ComputeGraph


@dataclass
class ComputeGraph:
    """A dataflow graph ready for GNN conversion."""

    nodes: dict[int, GraphNode] = field(default_factory=dict)
    name_to_id: dict[str, int] = field(default_factory=dict)
    root_id: int | None = None
    input_names: list[str] = field(default_factory=list)
    loops: dict[int, LoopBody] = field(default_factory=dict)
    functions: dict[str, FunctionBody] = field(default_factory=dict)
    _next_id: int = 0
    _outer_scope: ComputeGraph | None = field(default=None, repr=False)
    _captures: dict[str, int] = field(default_factory=dict, repr=False)

    def add_node(
        self,
        name: str,
        op_type: str,
        value: float | bool | None = None,
        input_edges: list[int] | None = None,
        input_names: list[str] | None = None,
    ) -> int:
        nid = self._next_id
        self._next_id += 1
        node = GraphNode(
            node_id=nid,
            name=name,
            op_type=op_type,
            value=value,
            input_edges=input_edges or [],
            input_names=input_names or [],
        )
        self.nodes[nid] = node
        self.name_to_id[name] = nid
        return nid

    def resolve(self, name: str) -> int:
        if name in self.name_to_id:
            return self.name_to_id[name]
        if self._outer_scope is not None and name in self._outer_scope.name_to_id:
            nid = self.add_node(name, "input")
            self._captures[name] = self._outer_scope.name_to_id[name]
            return nid
        raise KeyError(f"Undefined variable: {name}")

    def topological_order(self) -> list[int]:
        """Return node IDs in topological order (inputs first)."""
        visited: set[int] = set()
        order: list[int] = []

        def dfs(nid: int) -> None:
            if nid in visited:
                return
            visited.add(nid)
            for dep in self.nodes[nid].input_edges:
                dfs(dep)
            order.append(nid)

        for nid in self.nodes:
            dfs(nid)
        return order

    def depth(self) -> int:
        """Compute the longest path length (number of message passing rounds needed)."""
        memo: dict[int, int] = {}

        def node_depth(nid: int) -> int:
            if nid in memo:
                return memo[nid]
            node = self.nodes[nid]
            if not node.input_edges:
                memo[nid] = 0
                return 0
            d = 1 + max(node_depth(e) for e in node.input_edges)
            memo[nid] = d
            return d

        if self.root_id is None:
            return 0
        return node_depth(self.root_id)

    @property
    def has_loops(self) -> bool:
        return len(self.loops) > 0

    @property
    def has_functions(self) -> bool:
        return len(self.functions) > 0


def _resolve_trivial(node: ANFNode, graph: ComputeGraph) -> int:
    """Resolve a trivial ANF node (Const or Var) to a graph node ID."""
    if isinstance(node, ANFConst):
        name = f"__const_{id(node)}"
        return graph.add_node(name, "const", value=float(node.value) if not isinstance(node.value, bool) else (1.0 if node.value else 0.0))
    if isinstance(node, ANFVar):
        return graph.resolve(node.name)
    raise TypeError(f"Non-trivial node in argument position: {type(node)}")


def _build_general(
    node: ANFNode,
    graph: ComputeGraph,
    loop_params: tuple[str, ...] | None = None,
    func_names: frozenset[str] = frozenset(),
) -> int:
    """Unified graph builder. Handles all ANF node types in any context.

    Args:
        node: The ANF node to compile.
        graph: The graph to add nodes to.
        loop_params: If not None, we're inside a loop body and ANFRecur is valid.
        func_names: Function names that should be compiled as call nodes.
    """
    if isinstance(node, ANFConst):
        name = f"__const_{id(node)}"
        val = float(node.value) if not isinstance(node.value, bool) else (1.0 if node.value else 0.0)
        return graph.add_node(name, "const", value=val)

    if isinstance(node, ANFVar):
        return graph.resolve(node.name)

    if isinstance(node, ANFLet):
        rhs_id = _build_general(node.rhs, graph, loop_params, func_names)
        graph.name_to_id[node.name] = rhs_id
        graph.nodes[rhs_id].name = node.name
        return _build_general(node.body, graph, loop_params, func_names)

    if isinstance(node, ANFApp):
        if isinstance(node.func, ANFVar) and node.func.name in PRIMITIVES:
            op_name = node.func.name
            arg_ids = [_resolve_trivial(a, graph) for a in node.args]
            arg_labels = [f"arg{i}" for i in range(len(arg_ids))]
            name = f"__op_{id(node)}"
            return graph.add_node(
                name, op_name, input_edges=arg_ids, input_names=arg_labels
            )
        if isinstance(node.func, ANFVar) and node.func.name in func_names:
            arg_ids = [_resolve_trivial(a, graph) for a in node.args]
            arg_labels = [f"arg{i}" for i in range(len(arg_ids))]
            call_nid = graph.add_node(
                f"__call_{node.func.name}_{id(node)}",
                "call",
                input_edges=arg_ids,
                input_names=arg_labels,
            )
            graph.nodes[call_nid].call_target = node.func.name
            return call_nid
        raise NotImplementedError(
            f"Non-primitive function application: {node.func}"
        )

    if isinstance(node, ANFIf):
        test_id = _resolve_trivial(node.test, graph)
        then_id = _build_general(node.then_, graph, loop_params, func_names)
        else_id = _build_general(node.else_, graph, loop_params, func_names)
        name = f"__if_{id(node)}"
        return graph.add_node(
            name,
            "if",
            input_edges=[test_id, then_id, else_id],
            input_names=["test", "then", "else"],
        )

    if isinstance(node, ANFLoop):
        return _build_loop(node, graph, func_names)

    if isinstance(node, ANFRecur):
        if loop_params is None:
            raise SyntaxError("'recur' outside of loop")
        arg_ids = [_resolve_trivial(a, graph) for a in node.args]
        if len(arg_ids) != len(loop_params):
            raise SyntaxError(
                f"'recur' has {len(arg_ids)} args but loop has {len(loop_params)} params"
            )
        return graph.add_node(
            f"__recur_{id(node)}",
            "recur",
            input_edges=arg_ids,
            input_names=[f"next_{p}" for p in loop_params],
        )

    if isinstance(node, ANFLetrec):
        return _build_letrec(node, graph, loop_params, func_names)

    if isinstance(node, ANFLambda):
        raise NotImplementedError("Lambda compilation not yet supported")

    raise TypeError(f"Unknown ANF node type: {type(node)}")


def _build_loop(
    node: ANFLoop,
    outer_graph: ComputeGraph,
    func_names: frozenset[str] = frozenset(),
) -> int:
    """Build a loop construct with a separate body subgraph."""
    body_graph = ComputeGraph()
    body_graph._outer_scope = outer_graph

    for param in node.params:
        body_graph.add_node(param, "loop_param")
        body_graph.input_names.append(param)

    result_id = _build_general(
        node.body, body_graph, loop_params=node.params, func_names=func_names
    )
    body_graph.root_id = result_id

    captures = body_graph._captures
    body_graph._outer_scope = None
    body_graph._captures = {}

    # Copy outer functions to body graph for calls from enclosing letrec scope
    for fname in func_names:
        if fname in outer_graph.functions and fname not in body_graph.functions:
            body_graph.functions[fname] = outer_graph.functions[fname]

    init_ids = [_resolve_trivial(init, outer_graph) for init in node.inits]
    capture_names = list(captures.keys())
    capture_outer_nids = [captures[n] for n in capture_names]

    loop_nid = outer_graph.add_node(
        f"__loop_{id(node)}",
        "loop",
        input_edges=init_ids + capture_outer_nids,
        input_names=[f"init_{p}" for p in node.params] + [f"capture_{n}" for n in capture_names],
    )

    loop_body = LoopBody(
        params=node.params,
        body_graph=body_graph,
        captures=tuple(capture_names),
    )
    outer_graph.loops[loop_nid] = loop_body

    return loop_nid


def _build_letrec(
    node: ANFLetrec,
    outer_graph: ComputeGraph,
    loop_params: tuple[str, ...] | None = None,
    outer_func_names: frozenset[str] = frozenset(),
) -> int:
    """Build a letrec construct: compile each function body into a subgraph."""
    new_func_names = frozenset(name for name, _ in node.bindings)
    all_func_names = new_func_names | outer_func_names

    for name, lam in node.bindings:
        body_graph = ComputeGraph()

        for param in lam.params:
            body_graph.add_node(param, "func_param")
            body_graph.input_names.append(param)

        result_id = _build_general(lam.body, body_graph, func_names=all_func_names)
        body_graph.root_id = result_id

        func_body = FunctionBody(
            name=name,
            params=lam.params,
            body_graph=body_graph,
        )
        outer_graph.functions[name] = func_body

    # Share functions dict across all function body subgraphs
    for fb in outer_graph.functions.values():
        fb.body_graph.functions = outer_graph.functions
        # Propagate to loop bodies within function bodies
        for lb in fb.body_graph.loops.values():
            lb.body_graph.functions = outer_graph.functions

    return _build_general(
        node.body, outer_graph, loop_params=loop_params, func_names=all_func_names
    )


def build_graph(anf: ANFNode, inputs: dict[str, None] | None = None) -> ComputeGraph:
    """Build a ComputeGraph from an ANF expression.

    Args:
        anf: The ANF expression to compile.
        inputs: Dict of input variable names (values are ignored at build time).
                These become 'input' nodes in the graph.
    """
    graph = ComputeGraph()
    inputs = inputs or {}

    for name in inputs:
        graph.add_node(name, "input")
        graph.input_names.append(name)

    root_id = _build_general(anf, graph)
    graph.root_id = root_id
    return graph
