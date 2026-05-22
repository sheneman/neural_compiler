############################################################
#
# Neural Compiler: Compiling scheme into composable and
#                  differentiable neural network representations
#
# test_anf.py: Unit tests for A-Normal Form transformation on trivial and compound expressions
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Unit tests for the ANF transform."""

import pytest
from neural_compiler.parser import parse
from neural_compiler.anf import (
    to_anf,
    ANFConst,
    ANFVar,
    ANFLet,
    ANFIf,
    ANFApp,
    ANFLambda,
)


def _is_trivial(node):
    return isinstance(node, (ANFConst, ANFVar))


class TestANFTrivial:
    def test_const(self):
        result = to_anf(parse("42"))
        assert result == ANFConst(42)

    def test_var(self):
        result = to_anf(parse("x"))
        assert result == ANFVar("x")

    def test_boolean(self):
        result = to_anf(parse("#t"))
        assert result == ANFConst(True)


class TestANFSimpleApp:
    def test_primitive_trivial_args(self):
        result = to_anf(parse("(+ x y)"))
        assert isinstance(result, ANFApp)
        assert result.func == ANFVar("+")
        assert all(_is_trivial(a) for a in result.args)

    def test_primitive_const_args(self):
        result = to_anf(parse("(+ 1 2)"))
        assert isinstance(result, ANFApp)
        assert result.args == (ANFConst(1), ANFConst(2))


class TestANFNested:
    def test_nested_args_get_let_bound(self):
        result = to_anf(parse("(+ (* 3 x) (- y 1))"))
        assert isinstance(result, ANFLet)
        assert isinstance(result.rhs, ANFApp)
        assert result.rhs.func == ANFVar("*")
        inner = result.body
        assert isinstance(inner, ANFLet)
        assert isinstance(inner.rhs, ANFApp)
        assert inner.rhs.func == ANFVar("-")
        app = inner.body
        assert isinstance(app, ANFApp)
        assert app.func == ANFVar("+")
        assert all(_is_trivial(a) for a in app.args)

    def test_deeply_nested(self):
        result = to_anf(parse("(+ (+ (+ 1 2) 3) 4)"))
        # The outermost form should be a let binding for the nested subexpr
        assert isinstance(result, ANFLet)
        # The ANF invariant (all App args trivial) is verified separately;
        # here just confirm structure is a chain of lets ending in an App
        node = result
        depth = 0
        while isinstance(node, ANFLet):
            depth += 1
            node = node.body
        assert isinstance(node, ANFApp)
        assert all(_is_trivial(a) for a in node.args)
        assert depth >= 1


class TestANFIf:
    def test_trivial_test(self):
        result = to_anf(parse("(if x 1 0)"))
        assert isinstance(result, ANFIf)
        assert result.test == ANFVar("x")

    def test_compound_test_gets_let_bound(self):
        result = to_anf(parse("(if (> x 0) 1 0)"))
        assert isinstance(result, ANFLet)
        assert isinstance(result.rhs, ANFApp)
        assert isinstance(result.body, ANFIf)
        assert _is_trivial(result.body.test)


class TestANFLet:
    def test_let_preserved(self):
        result = to_anf(parse("(let ((a 1)) (+ a 2))"))
        assert isinstance(result, ANFLet)
        assert result.name == "a"
        assert result.rhs == ANFConst(1)
        assert isinstance(result.body, ANFApp)

    def test_let_with_compound_rhs(self):
        result = to_anf(parse("(let ((a (+ 1 2))) (* a a))"))
        assert isinstance(result, ANFLet)
        assert result.name == "a"
        assert isinstance(result.rhs, ANFApp)
        assert isinstance(result.body, ANFApp)

    def test_multiple_bindings(self):
        result = to_anf(parse("(let ((a 1) (b 2)) (+ a b))"))
        assert isinstance(result, ANFLet)
        assert result.name == "a"
        assert isinstance(result.body, ANFLet)
        assert result.body.name == "b"


class TestANFLambda:
    def test_lambda(self):
        result = to_anf(parse("(lambda (x y) (+ x y))"))
        assert isinstance(result, ANFLambda)
        assert result.params == ("x", "y")
        assert isinstance(result.body, ANFApp)


class TestANFInvariant:
    """Verify the core ANF invariant: all App arguments are trivial."""

    def _check_all_args_trivial(self, node):
        if isinstance(node, ANFApp):
            for arg in node.args:
                assert _is_trivial(arg), f"Non-trivial arg in App: {arg}"
        if isinstance(node, ANFLet):
            self._check_all_args_trivial(node.rhs)
            self._check_all_args_trivial(node.body)
        if isinstance(node, ANFIf):
            assert _is_trivial(node.test), f"Non-trivial test in If: {node.test}"
            self._check_all_args_trivial(node.then_)
            self._check_all_args_trivial(node.else_)
        if isinstance(node, ANFLambda):
            self._check_all_args_trivial(node.body)

    @pytest.mark.parametrize(
        "source",
        [
            "(+ 1 2)",
            "(+ (* 3 x) (- y 1))",
            "(if (> x 0) (+ x 1) (- 0 x))",
            "(let ((a (+ 1 2))) (* a (+ a 3)))",
            "(+ (+ (+ 1 2) 3) 4)",
            "(* (+ a b) (- c (/ d e)))",
        ],
    )
    def test_anf_invariant(self, source):
        result = to_anf(parse(source))
        self._check_all_args_trivial(result)
