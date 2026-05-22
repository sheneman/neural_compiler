############################################################
#
# Neural Compiler: Compiling scheme into composable and
#                  differentiable neural network representations
#
# test_graph.py: Unit tests for graph construction including constants, inputs, and dataflow
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Unit tests for the graph builder."""

import pytest
from neural_compiler.parser import parse
from neural_compiler.anf import to_anf
from neural_compiler.graph import build_graph


class TestGraphConstruction:
    def test_const_only(self):
        anf = to_anf(parse("42"))
        graph = build_graph(anf)
        assert graph.root_id is not None
        root = graph.nodes[graph.root_id]
        assert root.op_type == "const"
        assert root.value == 42.0

    def test_input_var(self):
        anf = to_anf(parse("x"))
        graph = build_graph(anf, inputs={"x": None})
        root = graph.nodes[graph.root_id]
        assert root.op_type == "input"
        assert root.name == "x"

    def test_simple_add(self):
        anf = to_anf(parse("(+ 1 2)"))
        graph = build_graph(anf)
        root = graph.nodes[graph.root_id]
        assert root.op_type == "+"
        assert len(root.input_edges) == 2

    def test_nested_creates_intermediate_nodes(self):
        anf = to_anf(parse("(+ (* 3 x) (- y 1))"))
        graph = build_graph(anf, inputs={"x": None, "y": None})
        root = graph.nodes[graph.root_id]
        assert root.op_type == "+"
        op_types = {n.op_type for n in graph.nodes.values()}
        assert "*" in op_types
        assert "-" in op_types

    def test_if_node(self):
        anf = to_anf(parse("(if #t 1 0)"))
        graph = build_graph(anf)
        root = graph.nodes[graph.root_id]
        assert root.op_type == "if"
        assert len(root.input_edges) == 3
        assert root.input_names == ["test", "then", "else"]


class TestGraphTopology:
    def test_topological_order_respects_deps(self):
        anf = to_anf(parse("(+ (* 3 x) 1)"))
        graph = build_graph(anf, inputs={"x": None})
        order = graph.topological_order()
        positions = {nid: i for i, nid in enumerate(order)}
        for nid in order:
            node = graph.nodes[nid]
            for dep in node.input_edges:
                assert positions[dep] < positions[nid], (
                    f"Node {nid} ({node.op_type}) appears before dependency {dep}"
                )

    def test_depth_simple(self):
        anf = to_anf(parse("(+ 1 2)"))
        graph = build_graph(anf)
        assert graph.depth() == 1

    def test_depth_nested(self):
        anf = to_anf(parse("(+ (* 3 x) (- y 1))"))
        graph = build_graph(anf, inputs={"x": None, "y": None})
        assert graph.depth() == 2

    def test_depth_deeply_nested(self):
        anf = to_anf(parse("(+ (+ (+ 1 2) 3) 4)"))
        graph = build_graph(anf)
        assert graph.depth() == 3

    def test_dag_no_duplicate_input_nodes(self):
        anf = to_anf(parse("(+ x x)"))
        graph = build_graph(anf, inputs={"x": None})
        input_nodes = [n for n in graph.nodes.values() if n.op_type == "input"]
        assert len(input_nodes) == 1


class TestGraphErrors:
    def test_undefined_variable(self):
        anf = to_anf(parse("(+ x 1)"))
        with pytest.raises(KeyError, match="Undefined variable"):
            build_graph(anf)
