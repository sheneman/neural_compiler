############################################################
#
# Neural Compiler: Compiling scheme into composable and
#                  differentiable neural network representations
#
# scheme_parser.py: Tokenizer and recursive-descent parser for Scheme S-expressions
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""Tokenizer and recursive-descent parser for the Scheme subset."""

from __future__ import annotations

from neural_compiler.parser.ast_nodes import (
    ASTNode,
    App,
    Const,
    If,
    Lambda,
    Let,
    Letrec,
    Loop,
    Recur,
    Var,
    PRIMITIVES,
)


def tokenize(source: str) -> list[str]:
    """Split Scheme source into tokens (strings, parens, atoms)."""
    tokens: list[str] = []
    i = 0
    while i < len(source):
        c = source[i]
        if c.isspace():
            i += 1
        elif c == ";":
            while i < len(source) and source[i] != "\n":
                i += 1
        elif c in ("(", ")", "[", "]"):
            tokens.append(c)
            i += 1
        elif c == "#":
            if i + 1 < len(source) and source[i + 1] in ("t", "f"):
                tokens.append(source[i : i + 2])
                i += 2
            else:
                raise SyntaxError(f"Unexpected '#' at position {i}")
        else:
            j = i
            while j < len(source) and source[j] not in ("(", ")", "[", "]", " ", "\t", "\n", "\r", ";"):
                j += 1
            tokens.append(source[i:j])
            i = j
    return tokens


def _parse_sexpr(tokens: list[str], pos: int) -> tuple[object, int]:
    """Parse one S-expression, returning (sexpr, new_pos).

    An S-expression is either an atom (str) or a list of S-expressions.
    """
    if pos >= len(tokens):
        raise SyntaxError("Unexpected end of input")

    token = tokens[pos]

    if token == "(":
        pos += 1
        items = []
        while pos < len(tokens) and tokens[pos] != ")":
            item, pos = _parse_sexpr(tokens, pos)
            items.append(item)
        if pos >= len(tokens):
            raise SyntaxError("Missing closing parenthesis")
        pos += 1  # skip ')'
        return items, pos

    if token == "[":
        pos += 1
        items = ["vec"]
        while pos < len(tokens) and tokens[pos] != "]":
            item, pos = _parse_sexpr(tokens, pos)
            items.append(item)
        if pos >= len(tokens):
            raise SyntaxError("Missing closing bracket")
        pos += 1  # skip ']'
        return items, pos

    if token in (")", "]"):
        raise SyntaxError(f"Unexpected '{token}' at token position {pos}")

    return token, pos + 1


def _atom_to_ast(atom: str) -> ASTNode:
    """Convert an atom string to a Const or Var node."""
    if atom == "#t":
        return Const(True)
    if atom == "#f":
        return Const(False)
    try:
        return Const(int(atom))
    except ValueError:
        pass
    try:
        return Const(float(atom))
    except ValueError:
        pass
    return Var(atom)


def _sexpr_to_ast(sexpr: object) -> ASTNode:
    """Convert a nested S-expression (lists and strings) to an AST."""
    if isinstance(sexpr, str):
        return _atom_to_ast(sexpr)

    if not isinstance(sexpr, list) or len(sexpr) == 0:
        raise SyntaxError(f"Empty application: {sexpr}")

    head = sexpr[0]

    if head == "if":
        if len(sexpr) != 4:
            raise SyntaxError(f"'if' requires 3 arguments, got {len(sexpr) - 1}")
        return If(
            test=_sexpr_to_ast(sexpr[1]),
            then_=_sexpr_to_ast(sexpr[2]),
            else_=_sexpr_to_ast(sexpr[3]),
        )

    if head == "lambda":
        if len(sexpr) != 3:
            raise SyntaxError(f"'lambda' requires params and body, got {len(sexpr) - 1} parts")
        params = sexpr[1]
        if not isinstance(params, list) or not all(isinstance(p, str) for p in params):
            raise SyntaxError(f"'lambda' params must be a list of identifiers: {params}")
        return Lambda(
            params=tuple(params),
            body=_sexpr_to_ast(sexpr[2]),
        )

    if head == "let":
        if len(sexpr) != 3:
            raise SyntaxError(f"'let' requires bindings and body, got {len(sexpr) - 1} parts")
        raw_bindings = sexpr[1]
        if not isinstance(raw_bindings, list):
            raise SyntaxError(f"'let' bindings must be a list: {raw_bindings}")
        bindings = []
        for b in raw_bindings:
            if not isinstance(b, list) or len(b) != 2 or not isinstance(b[0], str):
                raise SyntaxError(f"Invalid let binding: {b}")
            bindings.append((b[0], _sexpr_to_ast(b[1])))
        return Let(
            bindings=tuple(bindings),
            body=_sexpr_to_ast(sexpr[2]),
        )

    if head == "letrec":
        if len(sexpr) != 3:
            raise SyntaxError(f"'letrec' requires bindings and body, got {len(sexpr) - 1} parts")
        raw_bindings = sexpr[1]
        if not isinstance(raw_bindings, list):
            raise SyntaxError(f"'letrec' bindings must be a list: {raw_bindings}")
        bindings = []
        for b in raw_bindings:
            if not isinstance(b, list) or len(b) != 2 or not isinstance(b[0], str):
                raise SyntaxError(f"Invalid letrec binding: {b}")
            rhs = _sexpr_to_ast(b[1])
            if not isinstance(rhs, Lambda):
                raise SyntaxError(f"letrec binding '{b[0]}' must be a lambda expression")
            bindings.append((b[0], rhs))
        return Letrec(
            bindings=tuple(bindings),
            body=_sexpr_to_ast(sexpr[2]),
        )

    if head == "loop":
        if len(sexpr) != 3:
            raise SyntaxError(f"'loop' requires bindings and body, got {len(sexpr) - 1} parts")
        raw_bindings = sexpr[1]
        if not isinstance(raw_bindings, list):
            raise SyntaxError(f"'loop' bindings must be a list: {raw_bindings}")
        bindings = []
        for b in raw_bindings:
            if not isinstance(b, list) or len(b) != 2 or not isinstance(b[0], str):
                raise SyntaxError(f"Invalid loop binding: {b}")
            bindings.append((b[0], _sexpr_to_ast(b[1])))
        return Loop(
            bindings=tuple(bindings),
            body=_sexpr_to_ast(sexpr[2]),
        )

    if head == "recur":
        if len(sexpr) < 2:
            raise SyntaxError("'recur' requires at least one argument")
        args = tuple(_sexpr_to_ast(a) for a in sexpr[1:])
        return Recur(args=args)

    func = _sexpr_to_ast(head)
    args = tuple(_sexpr_to_ast(a) for a in sexpr[1:])
    return App(func=func, args=args)


def parse(source: str) -> ASTNode:
    """Parse a Scheme source string into an AST."""
    tokens = tokenize(source)
    if not tokens:
        raise SyntaxError("Empty input")
    sexpr, pos = _parse_sexpr(tokens, 0)
    if pos < len(tokens):
        raise SyntaxError(f"Unexpected tokens after expression: {tokens[pos:]}")
    return _sexpr_to_ast(sexpr)
