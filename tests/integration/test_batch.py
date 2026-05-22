############################################################
#
# Neural Compiler: Compiling scheme into composable and
#                  differentiable neural network representations
#
# test_batch.py: Integration tests for batched execution with DirectModule
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Integration tests for batched execution (v0.5.0).

Tests that forward_batch() on DirectModule produces correct results
for batches of inputs, matching individual forward() calls.
"""

import pytest
import torch
from neural_compiler.compiler import compile_scheme
from neural_compiler.evaluator import DirectModule


class TestBatchBasic:
    """Basic batched execution on DAG programs."""

    def test_add_two_inputs(self):
        graph = compile_scheme("(+ x y)", inputs={"x": None, "y": None})
        model = DirectModule(graph)
        result = model.forward_batch({
            "x": torch.tensor([1.0, 2.0, 3.0]),
            "y": torch.tensor([10.0, 20.0, 30.0]),
        })
        assert result.shape == (3,)
        expected = torch.tensor([11.0, 22.0, 33.0])
        assert torch.allclose(result, expected)

    def test_multiply(self):
        graph = compile_scheme("(* x y)", inputs={"x": None, "y": None})
        model = DirectModule(graph)
        result = model.forward_batch({
            "x": torch.tensor([2.0, 3.0, 4.0]),
            "y": torch.tensor([5.0, 6.0, 7.0]),
        })
        expected = torch.tensor([10.0, 18.0, 28.0])
        assert torch.allclose(result, expected)

    def test_single_input(self):
        graph = compile_scheme("(* x x)", inputs={"x": None})
        model = DirectModule(graph)
        result = model.forward_batch({
            "x": torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]),
        })
        expected = torch.tensor([1.0, 4.0, 9.0, 16.0, 25.0])
        assert torch.allclose(result, expected)

    def test_batch_size_one(self):
        graph = compile_scheme("(+ x 1)", inputs={"x": None})
        model = DirectModule(graph)
        result = model.forward_batch({
            "x": torch.tensor([42.0]),
        })
        assert result.shape == (1,)
        assert result[0].item() == pytest.approx(43.0)

    def test_four_inputs(self):
        graph = compile_scheme(
            "(+ (* a b) (- c d))",
            inputs={"a": None, "b": None, "c": None, "d": None},
        )
        model = DirectModule(graph)
        result = model.forward_batch({
            "a": torch.tensor([3.0, 1.0, 5.0]),
            "b": torch.tensor([4.0, 2.0, 12.0]),
            "c": torch.tensor([10.0, 5.0, 100.0]),
            "d": torch.tensor([3.0, 1.0, 40.0]),
        })
        expected = torch.tensor([19.0, 6.0, 120.0])
        assert torch.allclose(result, expected)


class TestBatchWithConstants:
    """Batched execution on programs mixing constants and inputs."""

    def test_add_constant(self):
        graph = compile_scheme("(+ x 10)", inputs={"x": None})
        model = DirectModule(graph)
        result = model.forward_batch({
            "x": torch.tensor([1.0, 2.0, 3.0]),
        })
        expected = torch.tensor([11.0, 12.0, 13.0])
        assert torch.allclose(result, expected)

    def test_polynomial(self):
        """ax^2 + bx + c with fixed a=1, c=1, variable x and b."""
        graph = compile_scheme(
            "(+ (+ (* (* x x) 1) (* b x)) 1)",
            inputs={"x": None, "b": None},
        )
        model = DirectModule(graph)
        result = model.forward_batch({
            "x": torch.tensor([0.0, 1.0, 2.0, 3.0]),
            "b": torch.tensor([0.0, 0.0, 0.0, 0.0]),
        })
        expected = torch.tensor([1.0, 2.0, 5.0, 10.0])
        assert torch.allclose(result, expected)

    def test_nested_arithmetic(self):
        graph = compile_scheme(
            "(* (+ x 1) (- x 1))",
            inputs={"x": None},
        )
        model = DirectModule(graph)
        result = model.forward_batch({
            "x": torch.tensor([2.0, 3.0, 5.0, 10.0]),
        })
        expected = torch.tensor([3.0, 8.0, 24.0, 99.0])
        assert torch.allclose(result, expected)


class TestBatchConditionals:
    """Batched execution with MUX-style if (both branches evaluated)."""

    def test_absolute_value(self):
        graph = compile_scheme(
            "(if (> x 0) x (- 0 x))",
            inputs={"x": None},
        )
        model = DirectModule(graph)
        result = model.forward_batch({
            "x": torch.tensor([-5.0, -1.0, 0.0, 3.0, 7.0]),
        })
        expected = torch.tensor([5.0, 1.0, 0.0, 3.0, 7.0])
        assert torch.allclose(result, expected)

    def test_max_via_if(self):
        graph = compile_scheme(
            "(if (> x y) x y)",
            inputs={"x": None, "y": None},
        )
        model = DirectModule(graph)
        result = model.forward_batch({
            "x": torch.tensor([1.0, 5.0, 3.0]),
            "y": torch.tensor([4.0, 2.0, 3.0]),
        })
        expected = torch.tensor([4.0, 5.0, 3.0])
        assert torch.allclose(result, expected)

    def test_clamp(self):
        """Clamp x to [0, 10]."""
        graph = compile_scheme(
            "(if (< x 0) 0 (if (> x 10) 10 x))",
            inputs={"x": None},
        )
        model = DirectModule(graph)
        result = model.forward_batch({
            "x": torch.tensor([-5.0, 0.0, 5.0, 10.0, 15.0]),
        })
        expected = torch.tensor([0.0, 0.0, 5.0, 10.0, 10.0])
        assert torch.allclose(result, expected)


class TestBatchLetBindings:
    """Batched execution with let bindings."""

    def test_let_square(self):
        graph = compile_scheme(
            "(let ((sq (* x x))) (+ sq 1))",
            inputs={"x": None},
        )
        model = DirectModule(graph)
        result = model.forward_batch({
            "x": torch.tensor([0.0, 1.0, 2.0, 3.0]),
        })
        expected = torch.tensor([1.0, 2.0, 5.0, 10.0])
        assert torch.allclose(result, expected)

    def test_discriminant(self):
        graph = compile_scheme(
            "(let ((d (- (* b b) (* 4 (* a c))))) (if (>= d 0) 1 0))",
            inputs={"a": None, "b": None, "c": None},
        )
        model = DirectModule(graph)
        result = model.forward_batch({
            "a": torch.tensor([1.0, 1.0, 1.0]),
            "b": torch.tensor([5.0, 2.0, 0.0]),
            "c": torch.tensor([6.0, 1.0, 1.0]),
        })
        expected = torch.tensor([1.0, 1.0, 0.0])
        assert torch.allclose(result, expected)


class TestBatchTranscendental:
    """Batched execution with transcendental operations."""

    def test_sin_batch(self):
        graph = compile_scheme("(sin x)", inputs={"x": None})
        model = DirectModule(graph)
        x = torch.tensor([0.0, 1.0, 2.0, 3.14159])
        result = model.forward_batch({"x": x})
        expected = torch.sin(x)
        assert torch.allclose(result, expected, atol=1e-4)

    def test_cos_batch(self):
        graph = compile_scheme("(cos x)", inputs={"x": None})
        model = DirectModule(graph)
        x = torch.tensor([0.0, 1.0, 2.0, 3.14159])
        result = model.forward_batch({"x": x})
        expected = torch.cos(x)
        assert torch.allclose(result, expected, atol=1e-4)

    def test_exp_batch(self):
        graph = compile_scheme("(exp x)", inputs={"x": None})
        model = DirectModule(graph)
        x = torch.tensor([0.0, 1.0, 2.0, -1.0])
        result = model.forward_batch({"x": x})
        expected = torch.exp(x)
        assert torch.allclose(result, expected, atol=1e-4)

    def test_sqrt_batch(self):
        graph = compile_scheme("(sqrt x)", inputs={"x": None})
        model = DirectModule(graph)
        x = torch.tensor([1.0, 4.0, 9.0, 16.0])
        result = model.forward_batch({"x": x})
        expected = torch.sqrt(x)
        assert torch.allclose(result, expected, atol=1e-4)

    def test_log_batch(self):
        graph = compile_scheme("(log x)", inputs={"x": None})
        model = DirectModule(graph)
        x = torch.tensor([1.0, 2.718282, 10.0, 100.0])
        result = model.forward_batch({"x": x})
        expected = torch.log(x)
        assert torch.allclose(result, expected, atol=1e-4)

    def test_pow_batch(self):
        graph = compile_scheme("(pow x 2)", inputs={"x": None})
        model = DirectModule(graph)
        x = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        result = model.forward_batch({"x": x})
        expected = x ** 2
        assert torch.allclose(result, expected, atol=1e-4)

    def test_kinetic_energy_batch(self):
        graph = compile_scheme("(* 0.5 (* m (pow v 2)))", inputs={"m": None, "v": None})
        model = DirectModule(graph)
        m = torch.tensor([1.0, 2.0, 3.0])
        v = torch.tensor([2.0, 3.0, 4.0])
        result = model.forward_batch({"m": m, "v": v})
        expected = 0.5 * m * v ** 2
        assert torch.allclose(result, expected, atol=1e-4)

    def test_sin_cos_identity_batch(self):
        graph = compile_scheme(
            "(+ (pow (sin x) 2) (pow (cos x) 2))",
            inputs={"x": None},
        )
        model = DirectModule(graph)
        x = torch.linspace(0, 6.28, 100)
        result = model.forward_batch({"x": x})
        expected = torch.ones(100)
        assert torch.allclose(result, expected, atol=1e-4)


class TestBatchMatchesSingle:
    """Verify batched results match individual forward() calls."""

    @pytest.mark.parametrize("source,input_names", [
        ("(+ x y)", ["x", "y"]),
        ("(* (+ x 1) (- y 2))", ["x", "y"]),
        ("(if (> x 0) (* x 2) (- 0 x))", ["x"]),
        ("(let ((s (+ a b))) (* s s))", ["a", "b"]),
        ("(+ (* a b) (* c d))", ["a", "b", "c", "d"]),
        ("(if (= x y) 1 0)", ["x", "y"]),
        ("(abs (- x y))", ["x", "y"]),
        ("(min x y)", ["x", "y"]),
        ("(max (+ x 1) (* y 2))", ["x", "y"]),
    ])
    def test_batch_matches_individual(self, source, input_names):
        input_decl = {n: None for n in input_names}
        graph = compile_scheme(source, inputs=input_decl)
        model = DirectModule(graph)

        torch.manual_seed(42)
        batch_size = 8
        batch_inputs = {n: torch.randn(batch_size) for n in input_names}

        batch_result = model.forward_batch(batch_inputs)

        for i in range(batch_size):
            single_inputs = {n: batch_inputs[n][i] for n in input_names}
            single_result = model(single_inputs)
            assert batch_result[i].item() == pytest.approx(
                single_result.item(), abs=1e-5
            ), f"Mismatch at index {i} for {source}"


class TestBatchLargeBatch:
    """Test with larger batch sizes."""

    def test_batch_1000(self):
        graph = compile_scheme("(+ (* x x) y)", inputs={"x": None, "y": None})
        model = DirectModule(graph)

        x = torch.arange(1000, dtype=torch.float32)
        y = torch.ones(1000)
        result = model.forward_batch({"x": x, "y": y})

        expected = x * x + y
        assert torch.allclose(result, expected)

    def test_batch_10000(self):
        graph = compile_scheme("(* (+ x 1) (- x 1))", inputs={"x": None})
        model = DirectModule(graph)

        x = torch.arange(1, 10001, dtype=torch.float32)
        result = model.forward_batch({"x": x})

        expected = (x + 1) * (x - 1)
        assert torch.allclose(result, expected)


class TestBatchErrors:
    """Error handling for batched execution."""

    def test_loop_batch(self):
        graph = compile_scheme(
            "(loop ((n x) (acc 1)) (if (= n 0) acc (recur (- n 1) (* acc n))))",
            inputs={"x": None},
        )
        model = DirectModule(graph)
        result = model.forward_batch({"x": torch.tensor([0.0, 1.0, 2.0, 3.0, 5.0, 7.0])})
        expected = torch.tensor([1.0, 1.0, 2.0, 6.0, 120.0, 5040.0])
        assert torch.allclose(result, expected), f"{result} != {expected}"

    def test_recursion_raises(self):
        graph = compile_scheme(
            "(letrec ((f (lambda (n) (if (= n 0) 1 (* n (f (- n 1))))))) (f 5))"
        )
        model = DirectModule(graph)
        with pytest.raises(NotImplementedError, match="general recursion"):
            model.forward_batch({"dummy": torch.tensor([1.0])})

    def test_empty_inputs_raises(self):
        graph = compile_scheme("(+ x 1)", inputs={"x": None})
        model = DirectModule(graph)
        with pytest.raises(ValueError, match="at least one input"):
            model.forward_batch({})


class TestBatchGPU:
    """GPU tests for batched execution."""

    @pytest.fixture
    def gpu_device(self):
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        pytest.skip("No GPU available")

    def test_batch_gpu(self, gpu_device):
        graph = compile_scheme("(+ (* x x) y)", inputs={"x": None, "y": None})
        model = DirectModule(graph).to(gpu_device)
        result = model.forward_batch({
            "x": torch.tensor([1.0, 2.0, 3.0], device=gpu_device),
            "y": torch.tensor([10.0, 20.0, 30.0], device=gpu_device),
        })
        expected = torch.tensor([11.0, 24.0, 39.0], device=gpu_device)
        assert torch.allclose(result, expected)

    def test_batch_gpu_large(self, gpu_device):
        graph = compile_scheme(
            "(if (> x 0) (* x 2) (- 0 x))",
            inputs={"x": None},
        )
        model = DirectModule(graph).to(gpu_device)
        x = torch.randn(10000, device=gpu_device)
        result = model.forward_batch({"x": x})

        expected = torch.where(x > 0, x * 2, -x)
        assert torch.allclose(result, expected)

    def test_batch_gpu_matches_cpu(self, gpu_device):
        graph = compile_scheme(
            "(+ (* a b) (- c d))",
            inputs={"a": None, "b": None, "c": None, "d": None},
        )
        model_cpu = DirectModule(graph)
        model_gpu = DirectModule(graph).to(gpu_device)

        torch.manual_seed(0)
        inputs_cpu = {n: torch.randn(100) for n in ["a", "b", "c", "d"]}
        inputs_gpu = {n: v.to(gpu_device) for n, v in inputs_cpu.items()}

        result_cpu = model_cpu.forward_batch(inputs_cpu)
        result_gpu = model_gpu.forward_batch(inputs_gpu)

        assert torch.allclose(result_cpu, result_gpu.cpu(), atol=1e-5)
