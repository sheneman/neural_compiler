############################################################
#
# Neural Compiler: Compiling scheme into composable and
#                  differentiable neural network representations
#
# test_parser.py: Unit tests for tokenizer and parser on S-expressions and Scheme constructs
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Unit tests for the Scheme parser."""

import pytest
from neural_compiler.parser import parse, Const, Var, If, Lambda, Let, App
from neural_compiler.parser.scheme_parser import tokenize


class TestTokenizer:
    def test_simple_expression(self):
        assert tokenize("(+ 1 2)") == ["(", "+", "1", "2", ")"]

    def test_nested(self):
        tokens = tokenize("(+ (* 3 x) 1)")
        assert tokens == ["(", "+", "(", "*", "3", "x", ")", "1", ")"]

    def test_booleans(self):
        assert tokenize("#t") == ["#t"]
        assert tokenize("#f") == ["#f"]

    def test_whitespace_variants(self):
        assert tokenize("( +  1\t2\n)") == ["(", "+", "1", "2", ")"]

    def test_comments(self):
        tokens = tokenize("(+ 1 ; comment\n2)")
        assert tokens == ["(", "+", "1", "2", ")"]

    def test_empty(self):
        assert tokenize("") == []
        assert tokenize("   ") == []

    def test_negative_numbers(self):
        tokens = tokenize("(- 0 -3)")
        assert tokens == ["(", "-", "0", "-3", ")"]

    def test_float_literal(self):
        tokens = tokenize("3.14")
        assert tokens == ["3.14"]

    def test_invalid_hash(self):
        with pytest.raises(SyntaxError):
            tokenize("#x")


class TestParseAtoms:
    def test_integer(self):
        assert parse("42") == Const(42)

    def test_negative_integer(self):
        assert parse("-7") == Const(-7)

    def test_float(self):
        assert parse("3.14") == Const(3.14)

    def test_boolean_true(self):
        assert parse("#t") == Const(True)

    def test_boolean_false(self):
        assert parse("#f") == Const(False)

    def test_variable(self):
        assert parse("x") == Var("x")

    def test_variable_with_dashes(self):
        assert parse("my-var") == Var("my-var")


class TestParseExpressions:
    def test_simple_application(self):
        result = parse("(+ 1 2)")
        assert result == App(func=Var("+"), args=(Const(1), Const(2)))

    def test_nested_application(self):
        result = parse("(+ (* 3 x) 1)")
        expected = App(
            func=Var("+"),
            args=(
                App(func=Var("*"), args=(Const(3), Var("x"))),
                Const(1),
            ),
        )
        assert result == expected

    def test_if_expression(self):
        result = parse("(if #t 1 0)")
        assert result == If(test=Const(True), then_=Const(1), else_=Const(0))

    def test_if_nested(self):
        result = parse("(if (> x 0) x (- 0 x))")
        expected = If(
            test=App(func=Var(">"), args=(Var("x"), Const(0))),
            then_=Var("x"),
            else_=App(func=Var("-"), args=(Const(0), Var("x"))),
        )
        assert result == expected

    def test_lambda(self):
        result = parse("(lambda (x y) (+ x y))")
        expected = Lambda(
            params=("x", "y"),
            body=App(func=Var("+"), args=(Var("x"), Var("y"))),
        )
        assert result == expected

    def test_let(self):
        result = parse("(let ((a 1) (b 2)) (+ a b))")
        expected = Let(
            bindings=(("a", Const(1)), ("b", Const(2))),
            body=App(func=Var("+"), args=(Var("a"), Var("b"))),
        )
        assert result == expected

    def test_let_nested(self):
        result = parse("(let ((x (+ 1 2))) (* x x))")
        expected = Let(
            bindings=(("x", App(func=Var("+"), args=(Const(1), Const(2)))),),
            body=App(func=Var("*"), args=(Var("x"), Var("x"))),
        )
        assert result == expected

    def test_deeply_nested(self):
        result = parse("(+ (+ (+ 1 2) 3) 4)")
        assert isinstance(result, App)
        assert isinstance(result.args[0], App)
        assert isinstance(result.args[0].args[0], App)


class TestParseErrors:
    def test_empty_input(self):
        with pytest.raises(SyntaxError):
            parse("")

    def test_unmatched_open(self):
        with pytest.raises(SyntaxError):
            parse("(+ 1 2")

    def test_unmatched_close(self):
        with pytest.raises(SyntaxError):
            parse(")")

    def test_extra_tokens(self):
        with pytest.raises(SyntaxError):
            parse("(+ 1 2) extra")

    def test_if_wrong_arity(self):
        with pytest.raises(SyntaxError):
            parse("(if #t 1)")

    def test_lambda_wrong_form(self):
        with pytest.raises(SyntaxError):
            parse("(lambda x (+ x 1))")

    def test_let_bad_bindings(self):
        with pytest.raises(SyntaxError):
            parse("(let (a 1) (+ a 1))")

    def test_empty_list(self):
        with pytest.raises(SyntaxError):
            parse("()")
