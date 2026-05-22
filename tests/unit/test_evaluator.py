############################################################
#
# Neural Compiler: Compiling scheme into composable and
#                  differentiable neural network representations
#
# test_evaluator.py: Unit tests for the sequential evaluation engine on arithmetic, conditionals, and functions
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Unit tests for the evaluation engine."""

import pytest
from neural_compiler.parser import parse
from neural_compiler.anf import to_anf
from neural_compiler.graph import build_graph
from neural_compiler.evaluator import evaluate


def _eval(source, inputs=None):
    inputs = inputs or {}
    input_decl = {k: None for k in inputs}
    anf = to_anf(parse(source))
    graph = build_graph(anf, inputs=input_decl)
    return evaluate(graph, inputs)


class TestEvalConst:
    def test_integer(self):
        assert _eval("42") == 42.0

    def test_float(self):
        assert _eval("3.14") == pytest.approx(3.14)

    def test_boolean_true(self):
        assert _eval("#t") == 1.0

    def test_boolean_false(self):
        assert _eval("#f") == 0.0


class TestEvalArithmetic:
    def test_add(self):
        assert _eval("(+ 3 4)") == 7.0

    def test_sub(self):
        assert _eval("(- 10 3)") == 7.0

    def test_mul(self):
        assert _eval("(* 3 4)") == 12.0

    def test_div(self):
        assert _eval("(/ 10 4)") == 2.5

    def test_nested(self):
        assert _eval("(+ (* 3 4) (- 10 3))") == 19.0


class TestEvalWithInputs:
    def test_single_input(self):
        assert _eval("(+ x 1)", {"x": 5.0}) == 6.0

    def test_two_inputs(self):
        assert _eval("(+ x y)", {"x": 3.0, "y": 4.0}) == 7.0

    def test_input_used_twice(self):
        assert _eval("(* x x)", {"x": 5.0}) == 25.0

    def test_complex_expression(self):
        assert _eval("(+ (* 3 x) (- y 1))", {"x": 4.0, "y": 7.0}) == 18.0


class TestEvalIf:
    def test_if_true_literal(self):
        assert _eval("(if #t 10 20)") == 10.0

    def test_if_false_literal(self):
        assert _eval("(if #f 10 20)") == 20.0

    def test_if_with_comparison(self):
        assert _eval("(if (> x 0) x (- 0 x))", {"x": 5.0}) == 5.0
        assert _eval("(if (> x 0) x (- 0 x))", {"x": -5.0}) == 5.0

    def test_nested_if(self):
        source = "(if (> x 0) (if (< x 10) 1 2) 0)"
        assert _eval(source, {"x": 5.0}) == 1.0
        assert _eval(source, {"x": 15.0}) == 2.0
        assert _eval(source, {"x": -1.0}) == 0.0


class TestEvalLet:
    def test_simple_let(self):
        assert _eval("(let ((a 3)) (+ a 1))") == 4.0

    def test_let_compound_rhs(self):
        assert _eval("(let ((a (+ 1 2))) (* a a))") == 9.0

    def test_multiple_bindings(self):
        assert _eval("(let ((a 3) (b 4)) (+ a b))") == 7.0

    def test_nested_let(self):
        source = "(let ((a 3)) (let ((b (* a a))) (+ b 1)))"
        assert _eval(source) == 10.0


class TestEvalErrors:
    def test_missing_input(self):
        anf = to_anf(parse("(+ x 1)"))
        graph = build_graph(anf, inputs={"x": None})
        with pytest.raises(ValueError, match="Missing input"):
            evaluate(graph, {})
