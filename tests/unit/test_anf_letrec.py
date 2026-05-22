############################################################
#
# Neural Compiler: Compiling scheme into composable and
#                  differentiable neural network representations
#
# test_anf_letrec.py: Unit tests for ANF transformation of letrec recursive function definitions
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Unit tests for ANF transform of letrec."""

import pytest
from neural_compiler.parser import parse
from neural_compiler.anf import to_anf, ANFLetrec, ANFLambda, ANFIf, ANFLet, ANFApp


class TestANFLetrec:
    def test_simple_letrec(self):
        result = to_anf(parse(
            "(letrec ((f (lambda (n) (if (= n 0) 1 (* n (f (- n 1))))))) (f 5))"
        ))
        assert isinstance(result, ANFLetrec)
        assert len(result.bindings) == 1
        assert result.bindings[0][0] == "f"
        assert isinstance(result.bindings[0][1], ANFLambda)

    def test_lambda_body_normalized(self):
        result = to_anf(parse(
            "(letrec ((f (lambda (n) (* (+ n 1) (- n 1))))) (f 5))"
        ))
        assert isinstance(result, ANFLetrec)
        lam = result.bindings[0][1]
        assert isinstance(lam, ANFLambda)
        # Body should have let-bindings for the compound args to *
        body = lam.body
        assert isinstance(body, ANFLet)

    def test_mutual_recursion(self):
        result = to_anf(parse("""
            (letrec ((even? (lambda (n) (if (= n 0) 1 (odd? (- n 1)))))
                     (odd?  (lambda (n) (if (= n 0) 0 (even? (- n 1))))))
              (even? 10))
        """))
        assert isinstance(result, ANFLetrec)
        assert len(result.bindings) == 2

    def test_body_normalized(self):
        result = to_anf(parse(
            "(letrec ((f (lambda (n) n))) (+ (f 1) (f 2)))"
        ))
        assert isinstance(result, ANFLetrec)
        # Body should be let-bound since (f 1) and (f 2) are compound
        body = result.body
        assert isinstance(body, ANFLet)
