############################################################
#
# Neural Compiler: Compiling scheme into composable and
#                  differentiable neural network representations
#
# test_tco.py: Unit tests for tail-call optimization eligibility and correctness
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Unit tests for the tail-call optimization pass."""

import pytest
from neural_compiler.parser import parse
from neural_compiler.anf import to_anf
from neural_compiler.anf.tco import optimize_tco
from neural_compiler.anf.anf_nodes import (
    ANFLoop, ANFLetrec, ANFRecur, ANFLet, ANFIf, ANFApp, ANFVar,
)


def _anf(source):
    return to_anf(parse(source))


class TestTCOEligibility:
    """Test which letrec forms are eligible for TCO."""

    def test_simple_tail_recursive_transforms(self):
        anf = _anf("(letrec ((f (lambda (n acc) (if (= n 0) acc (f (- n 1) (* acc n)))))) (f 5 1))")
        result = optimize_tco(anf)
        assert isinstance(result, ANFLoop)

    def test_non_tail_recursive_stays(self):
        anf = _anf("(letrec ((f (lambda (n) (if (= n 0) 1 (* n (f (- n 1))))))) (f 5))")
        result = optimize_tco(anf)
        assert isinstance(result, ANFLetrec)

    def test_mutual_recursion_stays(self):
        anf = _anf("""
            (letrec ((even? (lambda (n) (if (= n 0) 1 (odd? (- n 1)))))
                     (odd?  (lambda (n) (if (= n 0) 0 (even? (- n 1))))))
              (even? 10))
        """)
        result = optimize_tco(anf)
        assert isinstance(result, ANFLetrec)

    def test_no_self_calls_stays(self):
        anf = _anf("(letrec ((f (lambda (n) n))) (f 5))")
        result = optimize_tco(anf)
        # No self-calls → no transformation needed (but also not harmful)
        # The function is trivially tail-recursive with zero calls.
        # Either ANFLoop or ANFLetrec is acceptable.
        assert isinstance(result, (ANFLoop, ANFLetrec))

    def test_both_if_branches_tail_call(self):
        anf = _anf("(letrec ((f (lambda (n) (if (> n 0) (f (- n 1)) (f (+ n 1)))))) (f 5))")
        result = optimize_tco(anf)
        assert isinstance(result, ANFLoop)

    def test_call_in_if_test_not_eligible(self):
        """Self-call in the test position of an if is non-tail."""
        anf = _anf("(letrec ((f (lambda (n) (if (f n) 1 0)))) (f 5))")
        result = optimize_tco(anf)
        assert isinstance(result, ANFLetrec)


class TestTCOTransformation:
    """Test the structure of the transformed ANF."""

    def test_direct_call_produces_loop(self):
        anf = _anf("(letrec ((f (lambda (n acc) (if (= n 0) acc (f (- n 1) (* acc n)))))) (f 5 1))")
        result = optimize_tco(anf)
        assert isinstance(result, ANFLoop)
        assert result.params == ("n", "acc")

    def test_loop_inits_from_call_args(self):
        anf = _anf("(letrec ((f (lambda (n acc) (if (= n 0) acc (f (- n 1) (* acc n)))))) (f 5 1))")
        result = optimize_tco(anf)
        assert isinstance(result, ANFLoop)
        assert len(result.inits) == 2

    def test_body_contains_recur(self):
        anf = _anf("(letrec ((f (lambda (n acc) (if (= n 0) acc (f (- n 1) (* acc n)))))) (f 5 1))")
        result = optimize_tco(anf)
        assert isinstance(result, ANFLoop)

        def _find_recur(node):
            if isinstance(node, ANFRecur):
                return True
            if isinstance(node, ANFIf):
                return _find_recur(node.then_) or _find_recur(node.else_)
            if isinstance(node, ANFLet):
                return _find_recur(node.rhs) or _find_recur(node.body)
            return False

        assert _find_recur(result.body)

    def test_no_letrec_in_output(self):
        anf = _anf("(letrec ((f (lambda (n acc) (if (= n 0) acc (f (- n 1) (* acc n)))))) (f 5 1))")
        result = optimize_tco(anf)

        def _has_letrec(node):
            if isinstance(node, ANFLetrec):
                return True
            if isinstance(node, ANFLet):
                return _has_letrec(node.rhs) or _has_letrec(node.body)
            if isinstance(node, ANFIf):
                return _has_letrec(node.then_) or _has_letrec(node.else_)
            if isinstance(node, ANFLoop):
                return _has_letrec(node.body)
            return False

        assert not _has_letrec(result)

    def test_indirect_use_keeps_letrec(self):
        """When letrec body is not a direct call, letrec is kept with loop-ified lambda."""
        anf = _anf("(letrec ((f (lambda (n acc) (if (= n 0) acc (f (- n 1) (* acc n)))))) (+ (f 5 1) 1))")
        result = optimize_tco(anf)
        # The function is tail-recursive but the call is indirect → keep letrec
        assert isinstance(result, (ANFLetrec, ANFLet))


class TestTCONestedStructures:
    """Test TCO with nested let/if structures."""

    def test_letrec_inside_let(self):
        anf = _anf("(let ((x 1)) (letrec ((f (lambda (n) (if (= n 0) x (f (- n 1)))))) (f 5)))")
        result = optimize_tco(anf)

        def _has_loop(node):
            if isinstance(node, ANFLoop):
                return True
            if isinstance(node, ANFLet):
                return _has_loop(node.rhs) or _has_loop(node.body)
            if isinstance(node, ANFIf):
                return _has_loop(node.then_) or _has_loop(node.else_)
            return False

        assert _has_loop(result)

    def test_non_letrec_nodes_unchanged(self):
        anf = _anf("(+ 1 2)")
        result = optimize_tco(anf)
        assert isinstance(result, ANFApp)
