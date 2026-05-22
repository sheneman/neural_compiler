############################################################
#
# Neural Compiler: Compiling scheme into composable and
#                  differentiable neural network representations
#
# anf_nodes.py: ANF node definitions representing normalized form with trivial arguments
#
# Luke Sheneman
# Research Computing and Data Services (RCDS)
# Institute for Interdisciplinary Data Sciences (IIDS)
# University of Idaho
# sheneman@uidaho.edu
#
############################################################

"""ANF node types.

In ANF, all arguments to applications are trivial (Const or Var).
Compound expressions are let-bound to fresh temporaries.

ANFConst(value)                  — literal
ANFVar(name)                     — variable reference
ANFLet(name, rhs, body)          — let name = rhs in body
ANFIf(test, then_, else_)        — conditional (test is trivial)
ANFApp(func, args)               — application (all args trivial)
ANFLambda(params, body)          — function abstraction
ANFLoop(params, inits, body)     — tail-recursive loop with initial values
ANFRecur(args)                   — tail call back to enclosing loop (all args trivial)
ANFLetrec(bindings, body)        — recursive function bindings (each rhs is ANFLambda)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Union


@dataclass(frozen=True)
class ANFNode:
    pass


@dataclass(frozen=True)
class ANFConst(ANFNode):
    value: Union[int, float, bool]


@dataclass(frozen=True)
class ANFVar(ANFNode):
    name: str


@dataclass(frozen=True)
class ANFLet(ANFNode):
    name: str
    rhs: ANFNode
    body: ANFNode


@dataclass(frozen=True)
class ANFIf(ANFNode):
    test: ANFNode  # must be trivial (ANFConst or ANFVar)
    then_: ANFNode
    else_: ANFNode


@dataclass(frozen=True)
class ANFApp(ANFNode):
    func: ANFNode  # trivial
    args: tuple[ANFNode, ...]  # all trivial


@dataclass(frozen=True)
class ANFLambda(ANFNode):
    params: tuple[str, ...]
    body: ANFNode


@dataclass(frozen=True)
class ANFLoop(ANFNode):
    params: tuple[str, ...]
    inits: tuple[ANFNode, ...]  # all trivial
    body: ANFNode


@dataclass(frozen=True)
class ANFRecur(ANFNode):
    args: tuple[ANFNode, ...]  # all trivial


@dataclass(frozen=True)
class ANFLetrec(ANFNode):
    bindings: tuple[tuple[str, ANFLambda], ...]
    body: ANFNode
