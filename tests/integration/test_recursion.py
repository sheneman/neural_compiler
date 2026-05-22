############################################################
#
# Neural Compiler: Compiling scheme into composable and
#                  differentiable neural network representations
#
# test_recursion.py: Integration tests for letrec named recursive functions
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Integration tests for letrec: named recursive functions with all evaluators."""

import pytest
import torch
from neural_compiler.compiler import compile_scheme, run_scheme
from neural_compiler.evaluator import evaluate, SchemeGNN, DirectModule


def _all_eval(source, inputs=None):
    """Evaluate with sequential, SchemeGNN, and DirectModule. Return all three results."""
    inputs = inputs or {}
    input_decl = {k: None for k in inputs}
    graph = compile_scheme(source, inputs=input_decl)

    seq_result = evaluate(graph, inputs)

    model_gnn = SchemeGNN(graph)
    tensor_inputs = {k: torch.tensor(v) for k, v in inputs.items()}
    gnn_result = model_gnn(tensor_inputs).item()

    model_pyg = DirectModule(graph)
    pyg_result = model_pyg(tensor_inputs).item()

    return seq_result, gnn_result, pyg_result


class TestFactorialRecursive:
    """Factorial via general recursion (not loop/recur)."""

    SRC = "(letrec ((f (lambda (n) (if (= n 0) 1 (* n (f (- n 1))))))) (f N))"

    @pytest.mark.parametrize("n,expected", [
        (0, 1.0),
        (1, 1.0),
        (2, 2.0),
        (3, 6.0),
        (5, 120.0),
        (10, 3628800.0),
    ])
    def test_factorial(self, n, expected):
        seq, gnn, pyg = _all_eval(self.SRC, {"N": float(n)})
        assert seq == pytest.approx(expected)
        assert gnn == pytest.approx(expected)
        assert pyg == pytest.approx(expected)


class TestFibonacciTree:
    """Fibonacci via tree recursion (exponential — keep n small)."""

    SRC = "(letrec ((fib (lambda (n) (if (< n 2) n (+ (fib (- n 1)) (fib (- n 2))))))) (fib N))"

    @pytest.mark.parametrize("n,expected", [
        (0, 0.0),
        (1, 1.0),
        (2, 1.0),
        (3, 2.0),
        (5, 5.0),
        (10, 55.0),
        (15, 610.0),
    ])
    def test_fibonacci(self, n, expected):
        seq, gnn, pyg = _all_eval(self.SRC, {"N": float(n)})
        assert seq == pytest.approx(expected)
        assert gnn == pytest.approx(expected)
        assert pyg == pytest.approx(expected)


class TestMutualRecursion:
    """Even/odd via mutual recursion."""

    EVEN_SRC = """
    (letrec ((even? (lambda (n) (if (= n 0) 1 (odd? (- n 1)))))
             (odd?  (lambda (n) (if (= n 0) 0 (even? (- n 1))))))
      (even? N))
    """
    ODD_SRC = """
    (letrec ((even? (lambda (n) (if (= n 0) 1 (odd? (- n 1)))))
             (odd?  (lambda (n) (if (= n 0) 0 (even? (- n 1))))))
      (odd? N))
    """

    @pytest.mark.parametrize("n,expected", [
        (0, 1.0),
        (1, 0.0),
        (2, 1.0),
        (7, 0.0),
        (10, 1.0),
        (20, 1.0),
    ])
    def test_even(self, n, expected):
        seq, gnn, pyg = _all_eval(self.EVEN_SRC, {"N": float(n)})
        assert seq == pytest.approx(expected)
        assert gnn == pytest.approx(expected)
        assert pyg == pytest.approx(expected)

    @pytest.mark.parametrize("n,expected", [
        (0, 0.0),
        (1, 1.0),
        (2, 0.0),
        (7, 1.0),
        (10, 0.0),
    ])
    def test_odd(self, n, expected):
        seq, gnn, pyg = _all_eval(self.ODD_SRC, {"N": float(n)})
        assert seq == pytest.approx(expected)
        assert gnn == pytest.approx(expected)
        assert pyg == pytest.approx(expected)


class TestSumRecursive:
    """Sum 1..n via general recursion (compare with loop/recur version)."""

    SRC = "(letrec ((sum (lambda (n) (if (= n 0) 0 (+ n (sum (- n 1))))))) (sum N))"

    @pytest.mark.parametrize("n,expected", [
        (0, 0.0),
        (1, 1.0),
        (5, 15.0),
        (10, 55.0),
        (100, 5050.0),
    ])
    def test_sum(self, n, expected):
        seq, gnn, pyg = _all_eval(self.SRC, {"N": float(n)})
        assert seq == pytest.approx(expected)
        assert gnn == pytest.approx(expected)
        assert pyg == pytest.approx(expected)


class TestGCDRecursive:
    """GCD via general recursion with two params."""

    SRC = "(letrec ((gcd (lambda (a b) (if (= b 0) a (gcd b (modulo a b)))))) (gcd A B))"

    @pytest.mark.parametrize("a,b,expected", [
        (48, 18, 6.0),
        (100, 75, 25.0),
        (17, 13, 1.0),
        (0, 5, 5.0),
        (7, 0, 7.0),
    ])
    def test_gcd(self, a, b, expected):
        seq, gnn, pyg = _all_eval(self.SRC, {"A": float(a), "B": float(b)})
        assert seq == pytest.approx(expected)
        assert gnn == pytest.approx(expected)
        assert pyg == pytest.approx(expected)


class TestPowerRecursive:
    """Exponentiation via recursion."""

    SRC = "(letrec ((pow (lambda (b e) (if (= e 0) 1 (* b (pow b (- e 1))))))) (pow BASE EXP))"

    @pytest.mark.parametrize("base,exp,expected", [
        (2, 0, 1.0),
        (2, 10, 1024.0),
        (3, 4, 81.0),
    ])
    def test_power(self, base, exp, expected):
        seq, gnn, pyg = _all_eval(self.SRC, {"BASE": float(base), "EXP": float(exp)})
        assert seq == pytest.approx(expected)
        assert gnn == pytest.approx(expected)
        assert pyg == pytest.approx(expected)


class TestBaseCase:
    """Immediate base case — no recursive calls made."""

    def test_immediate_return(self):
        src = "(letrec ((f (lambda (n) (if (= n 0) 42 (f (- n 1)))))) (f 0))"
        seq, gnn, pyg = _all_eval(src)
        assert seq == 42.0
        assert gnn == 42.0
        assert pyg == 42.0


class TestLetrecWithInput:
    """Letrec using external inputs."""

    def test_factorial_with_input(self):
        src = "(letrec ((f (lambda (n) (if (= n 0) 1 (* n (f (- n 1))))))) (f x))"
        seq, gnn, pyg = _all_eval(src, {"x": 7.0})
        assert seq == pytest.approx(5040.0)
        assert gnn == pytest.approx(5040.0)
        assert pyg == pytest.approx(5040.0)


class TestDepthLimit:
    """Recursion that exceeds depth limit should raise RuntimeError.

    Uses a non-tail-recursive function so TCO doesn't convert to loop/recur.
    """

    NON_TAIL_SRC = "(letrec ((f (lambda (n) (+ 1 (f (+ n 1)))))) (f 0))"

    def test_depth_exceeded_sequential(self):
        graph = compile_scheme(self.NON_TAIL_SRC)
        with pytest.raises(RuntimeError, match="Recursion depth exceeded"):
            evaluate(graph, {}, max_depth=50)

    def test_depth_exceeded_gnn(self):
        graph = compile_scheme(self.NON_TAIL_SRC)
        model = SchemeGNN(graph, max_depth=50)
        with pytest.raises(RuntimeError, match="Recursion depth exceeded"):
            model({})

    def test_depth_exceeded_pyg(self):
        graph = compile_scheme(self.NON_TAIL_SRC)
        model = DirectModule(graph, max_depth=50)
        with pytest.raises(RuntimeError, match="Recursion depth exceeded"):
            model({})


class TestRecursionGPU:
    """GPU tests for recursive functions."""

    @pytest.fixture
    def gpu_device(self):
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        pytest.skip("No GPU available")

    def test_factorial_gpu(self, gpu_device):
        src = "(letrec ((f (lambda (n) (if (= n 0) 1 (* n (f (- n 1))))))) (f 5))"
        graph = compile_scheme(src)
        model = DirectModule(graph).to(gpu_device)
        result = model({})
        assert result.item() == 120.0

    def test_fibonacci_gpu(self, gpu_device):
        src = "(letrec ((fib (lambda (n) (if (< n 2) n (+ (fib (- n 1)) (fib (- n 2))))))) (fib 10))"
        graph = compile_scheme(src)
        model = DirectModule(graph).to(gpu_device)
        result = model({})
        assert result.item() == 55.0

    def test_with_input_gpu(self, gpu_device):
        src = "(letrec ((f (lambda (n) (if (= n 0) 1 (* n (f (- n 1))))))) (f x))"
        graph = compile_scheme(src, inputs={"x": None})
        model = DirectModule(graph).to(gpu_device)
        result = model({"x": torch.tensor(6.0, device=gpu_device)})
        assert result.item() == 720.0


class TestSchemeGNNModule:
    """Test SchemeGNN (sequential module) with recursion."""

    def test_factorial_module(self):
        src = "(letrec ((f (lambda (n) (if (= n 0) 1 (* n (f (- n 1))))))) (f 5))"
        graph = compile_scheme(src)
        model = SchemeGNN(graph)
        assert model({}).item() == 120.0

    def test_repeated_calls(self):
        src = "(letrec ((f (lambda (n) (if (= n 0) 1 (* n (f (- n 1))))))) (f x))"
        graph = compile_scheme(src, inputs={"x": None})
        model = SchemeGNN(graph)
        assert model({"x": torch.tensor(3.0)}).item() == 6.0
        assert model({"x": torch.tensor(5.0)}).item() == 120.0
        assert model({"x": torch.tensor(0.0)}).item() == 1.0
