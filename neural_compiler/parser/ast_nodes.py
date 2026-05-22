############################################################
#
# Neural Compiler: Compiling scheme into composable and
#                  differentiable neural network representations
#
# ast_nodes.py: AST node definitions for Scheme expressions (Const, Var, If, Lambda, Let, Letrec, App, Loop, Recur)
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""AST node types for the Scheme subset.

Node types:
  Const(value)           — numeric or boolean literal
  Var(name)              — variable reference
  If(test, then_, else_) — conditional
  Lambda(params, body)   — fixed-arity function abstraction
  Let(bindings, body)    — let bindings: [(name, expr), ...]
  App(func, args)        — function/primitive application
  Loop(bindings, body)   — tail-recursive loop: (loop ((var init) ...) body)
  Recur(args)            — tail call back to enclosing loop: (recur expr ...)
  Letrec(bindings, body) — recursive bindings: (letrec ((f (lambda ...)) ...) body)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Union


@dataclass(frozen=True)
class ASTNode:
    pass


@dataclass(frozen=True)
class Const(ASTNode):
    value: Union[int, float, bool]


@dataclass(frozen=True)
class Var(ASTNode):
    name: str


@dataclass(frozen=True)
class If(ASTNode):
    test: ASTNode
    then_: ASTNode
    else_: ASTNode


@dataclass(frozen=True)
class Lambda(ASTNode):
    params: tuple[str, ...]
    body: ASTNode


@dataclass(frozen=True)
class Let(ASTNode):
    bindings: tuple[tuple[str, ASTNode], ...]
    body: ASTNode


@dataclass(frozen=True)
class App(ASTNode):
    func: ASTNode
    args: tuple[ASTNode, ...]


@dataclass(frozen=True)
class Loop(ASTNode):
    bindings: tuple[tuple[str, ASTNode], ...]
    body: ASTNode


@dataclass(frozen=True)
class Recur(ASTNode):
    args: tuple[ASTNode, ...]


@dataclass(frozen=True)
class Letrec(ASTNode):
    bindings: tuple[tuple[str, ASTNode], ...]
    body: ASTNode


PRIMITIVES = {
    "+", "-", "*", "/",
    "=", "<", ">", "<=", ">=",
    "not", "and", "or",
    "min", "max", "abs",
    "modulo", "remainder",
    "sin", "cos", "exp", "sqrt", "log", "pow",
    # Vector operations
    "vec", "ref", "dot", "cross", "norm", "normalize", "vsum", "scale", "vlen",
    # Matrix operations
    "mat", "matmul", "matvec", "transpose", "trace", "det", "inv",
    "outer", "eye", "zeros", "ones",
}
