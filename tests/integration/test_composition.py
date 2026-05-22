############################################################
#
# Neural Compiler: Compiling scheme into composable and
#                  differentiable neural network representations
#
# test_composition.py: Integration tests for nested composition patterns (loops, letrec, nested loops)
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Integration tests for nested loops / recursion composition (v0.4.0).

Tests all composition patterns across all three evaluators:
  - Loop inside function body (letrec with loop in lambda)
  - Letrec inside loop body (loop body defines recursive functions)
  - Nested loops (loop inside loop)
  - Letrec calling loop-containing functions
  - Loop calling letrec-defined functions
"""

import pytest
import torch
from neural_compiler.compiler import compile_scheme, run_scheme
from neural_compiler.evaluator import evaluate, SchemeGNN, DirectModule


def _all_eval(source, inputs=None):
    """Evaluate with all three evaluators and return (seq, gnn, pyg)."""
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


class TestLoopInsideFunction:
    """Letrec where the function body contains a loop."""

    def test_factorial_with_internal_loop(self):
        src = """
        (letrec ((factorial (lambda (n)
          (loop ((i n) (acc 1))
            (if (= i 0) acc (recur (- i 1) (* acc i)))))))
          (factorial 10))
        """
        seq, gnn, pyg = _all_eval(src)
        assert seq == pytest.approx(3628800.0)
        assert gnn == pytest.approx(3628800.0)
        assert pyg == pytest.approx(3628800.0)

    def test_sum_with_internal_loop(self):
        src = """
        (letrec ((sum-to (lambda (n)
          (loop ((i n) (acc 0))
            (if (= i 0) acc (recur (- i 1) (+ acc i)))))))
          (sum-to 100))
        """
        seq, gnn, pyg = _all_eval(src)
        assert seq == pytest.approx(5050.0)
        assert gnn == pytest.approx(5050.0)
        assert pyg == pytest.approx(5050.0)

    def test_gcd_with_internal_loop(self):
        src = """
        (letrec ((gcd (lambda (a b)
          (loop ((x a) (y b))
            (if (= y 0) x (recur y (modulo x y)))))))
          (gcd 48 18))
        """
        seq, gnn, pyg = _all_eval(src)
        assert seq == pytest.approx(6.0)
        assert gnn == pytest.approx(6.0)
        assert pyg == pytest.approx(6.0)

    def test_function_with_loop_and_input(self):
        src = """
        (letrec ((power (lambda (base exp)
          (loop ((e exp) (acc 1))
            (if (= e 0) acc (recur (- e 1) (* acc base)))))))
          (power x n))
        """
        seq, gnn, pyg = _all_eval(src, {"x": 2.0, "n": 10.0})
        assert seq == pytest.approx(1024.0)
        assert gnn == pytest.approx(1024.0)
        assert pyg == pytest.approx(1024.0)

    def test_graph_structure_loop_in_function(self):
        """Non-tail-recursive function with internal loop keeps letrec structure."""
        src = """
        (letrec ((f (lambda (n)
          (if (= n 0) 0
            (+ (f (- n 1))
               (loop ((i n) (acc 0))
                 (if (= i 0) acc (recur (- i 1) (+ acc 1)))))))))
          (f 3))
        """
        graph = compile_scheme(src)
        assert graph.has_functions
        func_body = graph.functions["f"]
        assert func_body.body_graph.has_loops
        seq, gnn, pyg = _all_eval(src)
        assert seq == pytest.approx(6.0)
        assert gnn == pytest.approx(6.0)
        assert pyg == pytest.approx(6.0)


class TestLetrecInsideLoop:
    """Loop body that defines and uses recursive functions via letrec."""

    def test_loop_with_recursive_function(self):
        src = """
        (loop ((i 5) (sum 0))
          (if (= i 0) sum
            (recur (- i 1)
              (+ sum (letrec ((factorial (lambda (n)
                        (if (= n 0) 1 (* n (factorial (- n 1)))))))
                       (factorial i))))))
        """
        seq, gnn, pyg = _all_eval(src)
        expected = 1.0 + 2.0 + 6.0 + 24.0 + 120.0  # 1!+2!+3!+4!+5! = 153
        assert seq == pytest.approx(expected)
        assert gnn == pytest.approx(expected)
        assert pyg == pytest.approx(expected)

    def test_loop_body_with_tree_recursion(self):
        src = """
        (loop ((i 5) (sum 0))
          (if (= i 0) sum
            (recur (- i 1)
              (+ sum (letrec ((fib (lambda (n)
                        (if (< n 2) n (+ (fib (- n 1)) (fib (- n 2)))))))
                       (fib i))))))
        """
        seq, gnn, pyg = _all_eval(src)
        expected = 1.0 + 1.0 + 2.0 + 3.0 + 5.0  # fib(1)+fib(2)+fib(3)+fib(4)+fib(5)
        assert seq == pytest.approx(expected)
        assert gnn == pytest.approx(expected)
        assert pyg == pytest.approx(expected)

    def test_graph_structure_letrec_in_loop(self):
        src = """
        (loop ((i 3) (sum 0))
          (if (= i 0) sum
            (recur (- i 1)
              (+ sum (letrec ((f (lambda (n) (if (= n 0) 1 (* n (f (- n 1)))))))
                       (f i))))))
        """
        graph = compile_scheme(src)
        assert graph.has_loops
        loop_nid = next(nid for nid, n in graph.nodes.items() if n.op_type == "loop")
        loop_body = graph.loops[loop_nid]
        assert loop_body.body_graph.has_functions


class TestNestedLoops:
    """Loop inside another loop body."""

    def test_nested_sum(self):
        """Sum of sums: sum_{i=1}^{3} sum_{j=1}^{i} j"""
        src = """
        (loop ((i 3) (total 0))
          (if (= i 0) total
            (recur (- i 1)
              (+ total (loop ((j i) (acc 0))
                         (if (= j 0) acc (recur (- j 1) (+ acc j))))))))
        """
        seq, gnn, pyg = _all_eval(src)
        expected = (1.0) + (1.0 + 2.0) + (1.0 + 2.0 + 3.0)  # 1 + 3 + 6 = 10
        assert seq == pytest.approx(expected)
        assert gnn == pytest.approx(expected)
        assert pyg == pytest.approx(expected)

    def test_multiplication_table_sum(self):
        """Sum of all products i*j for i=1..3, j=1..3."""
        src = """
        (loop ((i 3) (total 0))
          (if (= i 0) total
            (recur (- i 1)
              (+ total (loop ((j 3) (row 0))
                         (if (= j 0) row (recur (- j 1) (+ row (* i j)))))))))
        """
        seq, gnn, pyg = _all_eval(src)
        expected = sum(i * j for i in range(1, 4) for j in range(1, 4))
        assert seq == pytest.approx(float(expected))
        assert gnn == pytest.approx(float(expected))
        assert pyg == pytest.approx(float(expected))

    def test_nested_loop_with_input(self):
        src = """
        (loop ((i N) (total 0))
          (if (= i 0) total
            (recur (- i 1)
              (+ total (loop ((j i) (acc 0))
                         (if (= j 0) acc (recur (- j 1) (+ acc 1))))))))
        """
        seq, gnn, pyg = _all_eval(src, {"N": 4.0})
        expected = 1.0 + 2.0 + 3.0 + 4.0  # triangular number T(4) = 10
        assert seq == pytest.approx(expected)
        assert gnn == pytest.approx(expected)
        assert pyg == pytest.approx(expected)

    def test_graph_structure_nested_loops(self):
        src = """
        (loop ((i 3) (total 0))
          (if (= i 0) total
            (recur (- i 1)
              (+ total (loop ((j i) (acc 0))
                         (if (= j 0) acc (recur (- j 1) (+ acc j))))))))
        """
        graph = compile_scheme(src)
        assert graph.has_loops
        loop_nid = next(nid for nid, n in graph.nodes.items() if n.op_type == "loop")
        loop_body = graph.loops[loop_nid]
        assert loop_body.body_graph.has_loops


class TestLetrecWithLoopCallingFunction:
    """Letrec where the loop body calls a sibling function."""

    def test_loop_calling_function(self):
        """A function defines a helper and uses it inside a loop."""
        src = """
        (letrec ((square (lambda (x) (* x x))))
          (loop ((i 4) (sum 0))
            (if (= i 0) sum (recur (- i 1) (+ sum (square i))))))
        """
        seq, gnn, pyg = _all_eval(src)
        expected = 1.0 + 4.0 + 9.0 + 16.0  # 1^2 + 2^2 + 3^2 + 4^2 = 30
        assert seq == pytest.approx(expected)
        assert gnn == pytest.approx(expected)
        assert pyg == pytest.approx(expected)

    def test_loop_calling_recursive_function(self):
        """Loop body calls a recursive function defined in enclosing letrec."""
        src = """
        (letrec ((factorial (lambda (n) (if (= n 0) 1 (* n (factorial (- n 1)))))))
          (loop ((i 5) (sum 0))
            (if (= i 0) sum (recur (- i 1) (+ sum (factorial i))))))
        """
        seq, gnn, pyg = _all_eval(src)
        expected = 1.0 + 2.0 + 6.0 + 24.0 + 120.0  # sum of factorials
        assert seq == pytest.approx(expected)
        assert gnn == pytest.approx(expected)
        assert pyg == pytest.approx(expected)

    def test_two_functions_loop_calls_both(self):
        src = """
        (letrec ((square (lambda (x) (* x x)))
                 (cube (lambda (x) (* x (* x x)))))
          (loop ((i 3) (sum 0))
            (if (= i 0) sum (recur (- i 1) (+ sum (+ (square i) (cube i)))))))
        """
        seq, gnn, pyg = _all_eval(src)
        expected = sum(i**2 + i**3 for i in range(1, 4))
        assert seq == pytest.approx(float(expected))
        assert gnn == pytest.approx(float(expected))
        assert pyg == pytest.approx(float(expected))


class TestFunctionWithInternalLoopCalledFromLoop:
    """A function uses a loop internally, and is called from another loop."""

    def test_loop_calls_loop_function(self):
        src = """
        (letrec ((sum-to (lambda (n)
          (loop ((i n) (acc 0))
            (if (= i 0) acc (recur (- i 1) (+ acc i)))))))
          (loop ((k 4) (total 0))
            (if (= k 0) total (recur (- k 1) (+ total (sum-to k))))))
        """
        seq, gnn, pyg = _all_eval(src)
        expected = sum(sum(range(1, k + 1)) for k in range(1, 5))
        assert seq == pytest.approx(float(expected))
        assert gnn == pytest.approx(float(expected))
        assert pyg == pytest.approx(float(expected))


class TestCompositionWithTCO:
    """TCO'd functions still compose correctly with loops."""

    def test_tco_function_in_loop(self):
        """A tail-recursive function (TCO'd to loop) called from an outer loop."""
        src = """
        (letrec ((sum-to (lambda (n acc)
          (if (= n 0) acc (sum-to (- n 1) (+ acc n))))))
          (loop ((k 4) (total 0))
            (if (= k 0) total (recur (- k 1) (+ total (sum-to k 0))))))
        """
        seq, gnn, pyg = _all_eval(src)
        expected = sum(sum(range(1, k + 1)) for k in range(1, 5))
        assert seq == pytest.approx(float(expected))
        assert gnn == pytest.approx(float(expected))
        assert pyg == pytest.approx(float(expected))

    def test_tco_factorial_used_in_let(self):
        src = """
        (let ((result (letrec ((f (lambda (n acc) (if (= n 0) acc (f (- n 1) (* acc n))))))
                        (f 10 1))))
          (+ result 1))
        """
        seq, gnn, pyg = _all_eval(src)
        assert seq == pytest.approx(3628801.0)
        assert gnn == pytest.approx(3628801.0)
        assert pyg == pytest.approx(3628801.0)


class TestCompositionGPU:
    """GPU tests for composition patterns."""

    @pytest.fixture
    def gpu_device(self):
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        pytest.skip("No GPU available")

    def test_loop_in_function_gpu(self, gpu_device):
        src = """
        (letrec ((sum-to (lambda (n)
          (loop ((i n) (acc 0))
            (if (= i 0) acc (recur (- i 1) (+ acc i)))))))
          (sum-to 10))
        """
        graph = compile_scheme(src)
        model = DirectModule(graph).to(gpu_device)
        assert model({}).item() == pytest.approx(55.0)

    def test_letrec_in_loop_gpu(self, gpu_device):
        src = """
        (loop ((i 4) (sum 0))
          (if (= i 0) sum
            (recur (- i 1)
              (+ sum (letrec ((f (lambda (n) (if (= n 0) 1 (* n (f (- n 1)))))))
                       (f i))))))
        """
        graph = compile_scheme(src)
        model = DirectModule(graph).to(gpu_device)
        expected = 1.0 + 2.0 + 6.0 + 24.0
        assert model({}).item() == pytest.approx(expected)

    def test_nested_loops_gpu(self, gpu_device):
        src = """
        (loop ((i 3) (total 0))
          (if (= i 0) total
            (recur (- i 1)
              (+ total (loop ((j i) (acc 0))
                         (if (= j 0) acc (recur (- j 1) (+ acc j))))))))
        """
        graph = compile_scheme(src)
        model = DirectModule(graph).to(gpu_device)
        assert model({}).item() == pytest.approx(10.0)

    def test_loop_calling_function_gpu(self, gpu_device):
        src = """
        (letrec ((square (lambda (x) (* x x))))
          (loop ((i 4) (sum 0))
            (if (= i 0) sum (recur (- i 1) (+ sum (square i))))))
        """
        graph = compile_scheme(src)
        model = DirectModule(graph).to(gpu_device)
        assert model({}).item() == pytest.approx(30.0)

    def test_composition_with_input_gpu(self, gpu_device):
        src = """
        (letrec ((power (lambda (base exp)
          (loop ((e exp) (acc 1))
            (if (= e 0) acc (recur (- e 1) (* acc base)))))))
          (power x n))
        """
        graph = compile_scheme(src, inputs={"x": None, "n": None})
        model = DirectModule(graph).to(gpu_device)
        result = model({
            "x": torch.tensor(3.0, device=gpu_device),
            "n": torch.tensor(4.0, device=gpu_device),
        })
        assert result.item() == pytest.approx(81.0)
