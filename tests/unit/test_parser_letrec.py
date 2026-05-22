############################################################
#
# Neural Compiler: Compiling scheme into composable and
#                  differentiable neural network representations
#
# test_parser_letrec.py: Unit tests for parsing letrec recursive function bindings
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Unit tests for letrec parsing."""

import pytest
from neural_compiler.parser import parse, Letrec, Lambda, If, App, Var


class TestParseLetrec:
    def test_simple_letrec(self):
        result = parse(
            "(letrec ((f (lambda (n) (if (= n 0) 1 (* n (f (- n 1))))))) (f 5))"
        )
        assert isinstance(result, Letrec)
        assert len(result.bindings) == 1
        assert result.bindings[0][0] == "f"
        assert isinstance(result.bindings[0][1], Lambda)

    def test_mutual_recursion(self):
        result = parse("""
            (letrec ((even? (lambda (n) (if (= n 0) 1 (odd? (- n 1)))))
                     (odd?  (lambda (n) (if (= n 0) 0 (even? (- n 1))))))
              (even? 10))
        """)
        assert isinstance(result, Letrec)
        assert len(result.bindings) == 2
        assert result.bindings[0][0] == "even?"
        assert result.bindings[1][0] == "odd?"

    def test_body_is_application(self):
        result = parse("(letrec ((f (lambda (x) x))) (f 42))")
        assert isinstance(result, Letrec)
        assert isinstance(result.body, App)

    def test_lambda_params(self):
        result = parse(
            "(letrec ((g (lambda (a b) (+ a b)))) (g 1 2))"
        )
        lam = result.bindings[0][1]
        assert isinstance(lam, Lambda)
        assert lam.params == ("a", "b")


class TestParseLetrecErrors:
    def test_missing_body(self):
        with pytest.raises(SyntaxError):
            parse("(letrec ((f (lambda (n) n))))")

    def test_bad_bindings_format(self):
        with pytest.raises(SyntaxError):
            parse("(letrec (f (lambda (n) n)) (f 5))")

    def test_non_lambda_binding(self):
        with pytest.raises(SyntaxError):
            parse("(letrec ((x 5)) (+ x 1))")

    def test_extra_parts(self):
        with pytest.raises(SyntaxError):
            parse("(letrec ((f (lambda (n) n))) (f 1) (f 2))")
