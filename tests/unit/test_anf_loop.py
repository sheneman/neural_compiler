############################################################
#
# Neural Compiler: Compiling scheme into composable and
#                  differentiable neural network representations
#
# test_anf_loop.py: Unit tests for ANF transformation of loop/recur constructs
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Unit tests for ANF transform of loop/recur."""

import pytest
from neural_compiler.parser import parse
from neural_compiler.anf import to_anf, ANFLoop, ANFRecur, ANFIf, ANFLet, ANFConst, ANFVar


def _is_trivial(node):
    return isinstance(node, (ANFConst, ANFVar))


class TestANFLoop:
    def test_simple_loop(self):
        result = to_anf(parse("(loop ((n 5)) (recur n))"))
        assert isinstance(result, ANFLoop)
        assert result.params == ("n",)
        assert all(_is_trivial(i) for i in result.inits)

    def test_loop_with_compound_init(self):
        result = to_anf(parse("(loop ((n (+ 2 3))) (recur n))"))
        assert isinstance(result, ANFLet)
        assert isinstance(result.body, ANFLoop)

    def test_loop_body_normalized(self):
        result = to_anf(parse(
            "(loop ((n 5) (acc 1)) (if (= n 0) acc (recur (- n 1) (* acc n))))"
        ))
        assert isinstance(result, ANFLoop)
        assert isinstance(result.body, ANFIf) or isinstance(result.body, ANFLet)

    def test_recur_args_trivial(self):
        result = to_anf(parse(
            "(loop ((n 5)) (recur (- n 1)))"
        ))
        assert isinstance(result, ANFLoop)
        body = result.body
        # recur args with compound expr get let-bound
        if isinstance(body, ANFLet):
            while isinstance(body, ANFLet):
                body = body.body
        if isinstance(body, ANFRecur):
            assert all(_is_trivial(a) for a in body.args)
