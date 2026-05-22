############################################################
#
# Neural Compiler: Compiling scheme into composable and
#                  differentiable neural network representations
#
# test_parser_loop.py: Unit tests for parsing loop/recur tail-recursive constructs
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Unit tests for loop/recur parsing."""

import pytest
from neural_compiler.parser import parse, Loop, Recur, Const, Var, If, App


class TestParseLoop:
    def test_simple_loop(self):
        result = parse("(loop ((n 5) (acc 1)) (recur (- n 1) (* acc n)))")
        assert isinstance(result, Loop)
        assert len(result.bindings) == 2
        assert result.bindings[0][0] == "n"
        assert result.bindings[1][0] == "acc"

    def test_loop_with_if(self):
        result = parse("(loop ((n 5) (acc 1)) (if (= n 0) acc (recur (- n 1) (* acc n))))")
        assert isinstance(result, Loop)
        assert isinstance(result.body, If)

    def test_recur_in_body(self):
        result = parse("(loop ((x 10)) (recur (- x 1)))")
        assert isinstance(result, Loop)
        body = result.body
        assert isinstance(body, Recur)
        assert len(body.args) == 1


class TestParseLoopErrors:
    def test_loop_wrong_arity(self):
        with pytest.raises(SyntaxError):
            parse("(loop ((n 5)))")  # missing body

    def test_loop_bad_bindings(self):
        with pytest.raises(SyntaxError):
            parse("(loop (n 5) (recur n))")

    def test_recur_no_args(self):
        with pytest.raises(SyntaxError):
            parse("(recur)")
