############################################################
#
# Neural Compiler: Compiling scheme into composable and
#                  differentiable neural network representations
#
# test_tco.py: Integration tests for tail-call optimization correctness and stress tests
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Integration tests for tail-call optimization: correctness and large-input stress tests."""

import pytest
import torch
from neural_compiler.compiler import compile_scheme, run_scheme
from neural_compiler.evaluator import evaluate, SchemeGNN, DirectModule


def _all_eval(source, inputs=None):
    """Evaluate with all three evaluators."""
    inputs = inputs or {}
    input_decl = {k: None for k in inputs}
    graph = compile_scheme(source, inputs=input_decl)

    seq = evaluate(graph, inputs)
    gnn = SchemeGNN(graph)(
        {k: torch.tensor(v) for k, v in inputs.items()}
    ).item()
    pyg = DirectModule(graph)(
        {k: torch.tensor(v) for k, v in inputs.items()}
    ).item()

    return seq, gnn, pyg


class TestTCOCorrectness:
    """Tail-recursive letrec should produce identical results to hand-written loop/recur."""

    def test_factorial_matches_loop(self):
        letrec_src = "(letrec ((f (lambda (n acc) (if (= n 0) acc (f (- n 1) (* acc n)))))) (f 10 1))"
        loop_src = "(loop ((n 10) (acc 1)) (if (= n 0) acc (recur (- n 1) (* acc n))))"
        assert run_scheme(letrec_src) == run_scheme(loop_src)

    def test_gcd_matches_loop(self):
        letrec_src = "(letrec ((gcd (lambda (a b) (if (= b 0) a (gcd b (modulo a b)))))) (gcd 48 18))"
        loop_src = "(loop ((a 48) (b 18)) (if (= b 0) a (recur b (modulo a b))))"
        assert run_scheme(letrec_src) == run_scheme(loop_src)

    def test_sum_matches_loop(self):
        letrec_src = "(letrec ((sum (lambda (n acc) (if (= n 0) acc (sum (- n 1) (+ acc n)))))) (sum 100 0))"
        loop_src = "(loop ((n 100) (acc 0)) (if (= n 0) acc (recur (- n 1) (+ acc n))))"
        assert run_scheme(letrec_src) == run_scheme(loop_src)

    def test_fibonacci_accumulator(self):
        src = "(letrec ((fib (lambda (n a b) (if (= n 0) a (fib (- n 1) b (+ a b)))))) (fib 20 0 1))"
        seq, gnn, pyg = _all_eval(src)
        assert seq == pytest.approx(6765.0)
        assert gnn == pytest.approx(6765.0)
        assert pyg == pytest.approx(6765.0)

    @pytest.mark.parametrize("n,expected", [
        (0, 1.0),
        (1, 1.0),
        (5, 120.0),
        (10, 3628800.0),
    ])
    def test_factorial_all_evaluators(self, n, expected):
        src = "(letrec ((f (lambda (n acc) (if (= n 0) acc (f (- n 1) (* acc n)))))) (f N 1))"
        seq, gnn, pyg = _all_eval(src, {"N": float(n)})
        assert seq == pytest.approx(expected)
        assert gnn == pytest.approx(expected)
        assert pyg == pytest.approx(expected)


class TestTCOLargeInputs:
    """Programs that would exceed max_depth without TCO should now work."""

    def test_countdown_large(self):
        src = "(letrec ((f (lambda (n) (if (= n 0) 0 (f (- n 1)))))) (f 50000))"
        graph = compile_scheme(src)
        assert graph.has_loops
        assert not graph.has_functions
        result = evaluate(graph, {}, max_iter=100000)
        assert result == 0.0

    def test_gcd_large(self):
        src = "(letrec ((gcd (lambda (a b) (if (= b 0) a (gcd b (modulo a b)))))) (gcd 1000000 999999))"
        result = run_scheme(src)
        assert result == 1.0

    def test_sum_large(self):
        """Sum to 1000 — verifiable within float32 precision."""
        src = "(letrec ((sum (lambda (n acc) (if (= n 0) acc (sum (- n 1) (+ acc n)))))) (sum 1000 0))"
        result = run_scheme(src)
        assert result == pytest.approx(500500.0)

    def test_graph_is_loop_not_function(self):
        """After TCO, graph should use loop/recur, not recursive functions."""
        src = "(letrec ((f (lambda (n acc) (if (= n 0) acc (f (- n 1) (* acc n)))))) (f 5 1))"
        graph = compile_scheme(src)
        assert graph.has_loops
        assert not graph.has_functions


class TestNonTailRecursiveUnchanged:
    """Non-tail-recursive programs should still work (just not optimized)."""

    def test_tree_fibonacci_still_works(self):
        src = "(letrec ((fib (lambda (n) (if (< n 2) n (+ (fib (- n 1)) (fib (- n 2))))))) (fib 10))"
        seq, gnn, pyg = _all_eval(src)
        assert seq == pytest.approx(55.0)
        assert gnn == pytest.approx(55.0)
        assert pyg == pytest.approx(55.0)

    def test_non_tail_is_still_function(self):
        """Non-tail-recursive letrec should keep using functions, not loops."""
        src = "(letrec ((f (lambda (n) (if (= n 0) 1 (* n (f (- n 1))))))) (f 5))"
        graph = compile_scheme(src)
        assert graph.has_functions
        assert not graph.has_loops

    def test_mutual_recursion_still_works(self):
        src = """
        (letrec ((even? (lambda (n) (if (= n 0) 1 (odd? (- n 1)))))
                 (odd?  (lambda (n) (if (= n 0) 0 (even? (- n 1))))))
          (even? 10))
        """
        seq, gnn, pyg = _all_eval(src)
        assert seq == pytest.approx(1.0)
        assert gnn == pytest.approx(1.0)
        assert pyg == pytest.approx(1.0)


class TestTCOGPU:
    """GPU tests for TCO'd programs."""

    @pytest.fixture
    def gpu_device(self):
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        pytest.skip("No GPU available")

    def test_factorial_tco_gpu(self, gpu_device):
        src = "(letrec ((f (lambda (n acc) (if (= n 0) acc (f (- n 1) (* acc n)))))) (f 5 1))"
        graph = compile_scheme(src)
        model = DirectModule(graph).to(gpu_device)
        assert model({}).item() == 120.0

    def test_tco_with_input_gpu(self, gpu_device):
        src = "(letrec ((f (lambda (n acc) (if (= n 0) acc (f (- n 1) (* acc n)))))) (f x 1))"
        graph = compile_scheme(src, inputs={"x": None})
        model = DirectModule(graph).to(gpu_device)
        result = model({"x": torch.tensor(6.0, device=gpu_device)})
        assert result.item() == 720.0
