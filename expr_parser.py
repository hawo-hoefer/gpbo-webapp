from __future__ import annotations

import abc
import math
import typing
from dataclasses import dataclass, field
from typing import Literal, Mapping, Type, TypeVar, Union

import torch

# grammar for the expressions
# expression =
#     term
#     | term '+' expression
#     | term '-' expression
#     | - '(' expression ')'
#
# term =
#     factor
#     | factor '*' term
#     | factor '/' term
#
# factor =
#     exp_term
#     | exp_term '^' exp_term
#
# exp_term =
#     number
#     | builtin
#     | identifier
#     | '(' expression ')'
#
# neg = '-' factor
#
# builtin = builtin_name '(' expression ')'
# builtin_name =
#     "log"
#     | "exp"
#     | "ln"


@dataclass
class Token(abc.ABC):
    pos: int

    def name(self):
        return self.__str__()


class BuiltinName(Token):
    pass


class Log(BuiltinName):
    pass


class Exp(BuiltinName):
    pass


class Ln(BuiltinName):
    pass


Numeric = float | torch.Tensor
Ctx = Mapping[str, Numeric]


class Node(abc.ABC):
    pass

    @abc.abstractmethod
    def __call__(self, _: Ctx) -> Numeric: ...

    @abc.abstractmethod
    def normalized_repr(self) -> str: ...


@dataclass
class Ident(Token, Node):
    value: str

    def __call__(self, ctx: Ctx) -> Numeric:
        return ctx[self.value]

    def normalized_repr(self) -> str:
        return self.value


@dataclass
class Builtin(Node):
    name: BuiltinName
    arg: Node

    def __call__(self, ctx: Ctx) -> Numeric:
        arg = self.arg(ctx)

        log_minval = 1e-320

        if isinstance(arg, torch.Tensor):
            if isinstance(self.name, Log):
                arg = arg.clamp_min(log_minval)
                return arg.log10()
            elif isinstance(self.name, Exp):
                return arg.exp()
            elif isinstance(self.name, Ln):
                arg = arg.clamp_min(log_minval)
                return arg.log()
            assert False, f"invalid builtin {self.name}"
        else:
            if isinstance(self.name, Log):
                if 0.0 <= arg <= 1e-320:
                    arg = 1e-320
                return math.log10(arg)
            elif isinstance(self.name, Exp):
                return math.exp(arg)
            elif isinstance(self.name, Ln):
                if 0.0 <= arg <= 1e-320:
                    arg = 1e-320
                return math.log(arg)
            else:
                assert False, f"invalid builtin {self.name}"

    def normalized_repr(self) -> str:
        inner = self.arg.normalized_repr()
        if isinstance(self.name, Log):
            return f"log({inner})"
        elif isinstance(self.name, Exp):
            return f"exp({inner})"
        elif isinstance(self.name, Ln):
            return f"ln({inner})"
        else:
            assert False, f"invalid builtin {self.name}"


@dataclass
class Neg(Node):
    arg: Node

    def __call__(self, ctx: Ctx) -> Numeric:
        return -self.arg(ctx)

    def normalized_repr(self) -> str:
        arg = self.arg.normalized_repr()
        if (isinstance(self.arg, BinOp) and self.arg.op.precedence < 1) or isinstance(
            self.arg, Neg
        ):
            return f"-({arg})"

        return f"-{arg}"


@dataclass
class Num(Node, Token):
    value: float

    def __call__(self, _: Ctx) -> float:
        return self.value

    def normalized_repr(self) -> str:
        return f"{self.value}"


class Operator(Token):
    precedence: int
    pass


class Add(Operator):
    precedence: int = 0

    def __str__(self):
        return "+"


class Sub(Operator):
    precedence: int = 0

    def __str__(self):
        return "-"


class Mul(Operator):
    precedence: int = 1

    def __str__(self):
        return "*"


class Div(Operator):
    precedence: int = 1

    def __str__(self):
        return "/"


class ExpOp(Operator):
    precedence: int = 2

    def __str__(self):
        return "^"


@dataclass
class BinOp(Node):
    lhs: Node
    rhs: Node
    op: Operator

    def __call__(self, ctx: Ctx) -> Numeric:
        lhs = self.lhs(ctx)
        rhs = self.rhs(ctx)

        if isinstance(self.op, Add):
            return lhs + rhs
        elif isinstance(self.op, Sub):
            return lhs - rhs
        elif isinstance(self.op, Mul):
            return lhs * rhs
        elif isinstance(self.op, Div):
            return lhs / rhs
        elif isinstance(self.op, ExpOp):
            return lhs**rhs
        else:
            assert False, f"invalid op: {self.op}"

    def normalized_repr(self) -> str:
        lhs = self.lhs.normalized_repr()
        rhs = self.rhs.normalized_repr()

        def parens(
            term: str, precedence: int, own_precedence: int = self.op.precedence
        ) -> str:
            if precedence < own_precedence:
                return f"({term})"
            return term

        if isinstance(self.lhs, BinOp):
            lhs = parens(lhs, self.lhs.op.precedence)
        elif isinstance(self.lhs, Neg):
            lhs = parens(lhs, -1)

        if isinstance(self.rhs, BinOp):
            rhs = parens(rhs, self.rhs.op.precedence)
        elif isinstance(self.rhs, Neg):
            rhs = parens(rhs, -1)

        return f"{lhs} {self.op.name()} {rhs}"


OPERATORS = Literal["+", "-", "*", "/", "^"]
PARENS = Literal["(", ")"]
PUNCT = (*typing.get_args(OPERATORS), *typing.get_args(PARENS))


class Oparen(Token):
    def __repr__(self) -> str:
        return "Oparen"

    def __str__(self) -> str:
        return "Oparen"


class Cparen(Token):
    def __repr__(self) -> str:
        return "Cparen"

    def __str__(self) -> str:
        return "Cparen"


def ident_or_builtin_name(data: str, start: int) -> Ident | BuiltinName:
    if data == "log":
        return Log(start)
    elif data == "exp":
        return Exp(start)
    elif data == "ln":
        return Ln(start)
    else:
        return Ident(start, data)


@dataclass
class Lexer:
    data: str
    cur: int = 0

    def skip_whitespace(self):
        while self.cur < len(self.data) and self.data[self.cur].isspace():
            self.cur += 1

    def chop_identifier_or_builtin_name(self) -> Token | str:
        start = self.cur
        if self.data[start] == '"':
            self.cur += 1
            # TODO: support escaping at some point if someone complains
            while self.cur < len(self.data) and self.data[self.cur] != '"':
                self.cur += 1
            if self.cur >= len(self.data):
                return f"Error, unmatched '\"' in identifier name. Opening '\"' is at position {start}"
            self.cur += 1

            start += 1
            end = self.cur - 1

        else:
            while (
                self.cur < len(self.data)
                and not self.data[self.cur].isspace()
                and self.data[self.cur] not in {"+", "-", "*", "/", "^", "(", ")"}
            ):
                self.cur += 1
            end = self.cur

        return ident_or_builtin_name(self.data[start:end], start)

    def chop_number_or_weird_ident(self) -> Token:
        start = self.cur
        while self.cur < len(self.data) and (
            self.data[self.cur].isnumeric() or self.data[self.cur] == "."
        ):
            self.cur += 1

        maybe_num = self.data[start : self.cur]

        try:
            return Num(start, float(maybe_num))
        except ValueError:
            return ident_or_builtin_name(maybe_num, start)

    def next_token(self) -> Token | str:
        if self.data[self.cur].isnumeric():
            return self.chop_number_or_weird_ident()

        if self.data[self.cur] == "+":
            self.cur += 1
            return Add(self.cur - 1)
        elif self.data[self.cur] == "-":
            self.cur += 1
            return Sub(self.cur - 1)
        elif self.data[self.cur] == "*":
            self.cur += 1
            return Mul(self.cur - 1)
        elif self.data[self.cur] == "/":
            self.cur += 1
            return Div(self.cur - 1)
        elif self.data[self.cur] == "^":
            self.cur += 1
            return ExpOp(self.cur - 1)
        elif self.data[self.cur] == "(":
            self.cur += 1
            return Oparen(self.cur - 1)
        elif self.data[self.cur] == ")":
            self.cur += 1
            return Cparen(self.cur - 1)

        return self.chop_identifier_or_builtin_name()

    def tokens(self) -> list[Token] | str:
        tokens = []
        while True:
            self.skip_whitespace()
            if self.cur >= len(self.data):
                break
            tok = self.next_token()
            if isinstance(tok, str):
                return tok

            tokens.append(tok)
        return tokens


T = TypeVar("T")


@dataclass
class ParseError:
    msg: str
    pos: int
    context: list[str] = field(default_factory=list)

    def ctx(self, ctx: str):
        self.context.append(ctx)
        return self


def expect_token_type(
    tokens: list[Token], token: Type[T], pos_if_empty: int
) -> tuple[T, list[Token]] | ParseError:
    if len(tokens) == 0:
        return ParseError(
            f"expected {token.__name__}, but input is empty.", pos_if_empty
        )

    tok = tokens[0]
    if not isinstance(tok, token):
        return ParseError(f"expected {token.__name__}, got {tok.name()}", tok.pos)
    return tok, tokens[1:]


ParseResult = Union[tuple[Node, list[Token]], ParseError]


def parse_builtin(tokens: list[Token]) -> ParseResult:
    """parse a builtin expression

    from the grammar:
    # builtin = builtin_name '(' expression ')'
    # builtin_name =
    #     "log"
    #     | "exp"
    #     | "ln"

    Args:
        tokens: tokens to parse

    Returns:
        Builtin Node on success or string error message on failure
    """

    r0 = expect_token_type(tokens, BuiltinName, 0)
    if isinstance(r0, ParseError):
        return r0
    builtin_name, tokens = r0

    r1 = expect_token_type(tokens, Oparen, builtin_name.pos)
    if isinstance(r1, ParseError):
        return r1
    oparen, tokens = r1

    r2 = parse_expr(tokens)
    if isinstance(r2, ParseError):
        return r2
    arg, tokens = r2

    r3 = expect_token_type(tokens, Cparen, oparen.pos)
    if isinstance(r3, ParseError):
        return r3
    _, tokens = r3

    return Builtin(builtin_name, arg), tokens


def parse_neg(tokens: list[Token]) -> ParseResult:
    """parse a negation
    neg = '-' factor

    Args:
        tokens: tokens to parse

    Returns:
        Node on success, ParseError on failure
    """
    r = expect_token_type(tokens, Sub, tokens[0].pos)
    assert not isinstance(r, ParseError)
    sub, tokens = r

    r2 = parse_factor(tokens)
    if isinstance(r2, ParseError):
        return r2.ctx("parsing negation")
    expr, tokens = r2

    return Neg(expr), tokens


def parse_paren_expr(tokens: list[Token]) -> ParseResult:
    """parse '(' expression ')'"""
    # parse paren operation (.....)
    oparen = tokens[0]
    tokens = tokens[1:]

    r0 = parse_expr(tokens)
    if isinstance(r0, ParseError):
        return r0
    expr, tokens = r0

    r1 = expect_token_type(tokens, Cparen, oparen.pos)
    if isinstance(r1, ParseError):
        return r1
    _, tokens = r1

    return expr, tokens


def parse_exp_term(tokens: list[Token]) -> ParseResult:
    """Parse an exponential term

    from the grammar:
    # exp_term =
    #     number
    #     | builtin
    #     | identifier
    #     | '(' expression ')'
    #     | '-' expression

    Args:
        tokens: tokens to parse

    Returns:
        Node on success, string message on failure
    """
    if isinstance(tokens[0], Num):
        return tokens[0], tokens[1:]

    if isinstance(tokens[0], Ident):
        return tokens[0], tokens[1:]

    if isinstance(tokens[0], BuiltinName):
        return parse_builtin(tokens)

    if isinstance(tokens[0], Oparen):
        r = parse_paren_expr(tokens)
        if isinstance(r, ParseError):
            return r.ctx("parsing parenthesized expression")
        return r

    # if a sub is here, it has so be at the beginning, right?
    if isinstance(tokens[0], Sub):
        r0 = parse_neg(tokens)
        if isinstance(r0, ParseError):
            return r0
        expr, tokens = r0
        return expr, tokens

    assert False, ("unreachable", tokens)


def parse_factor(tokens: list[Token]) -> ParseResult:
    """parse a factor

    from the grammar:
    # factor =
    #     exp_term
    #     | exp_term '^' factor

    Args:
        tokens: tokens to parse

    Returns:
        factor node on success, string message on failure

    """
    r0 = parse_exp_term(tokens)
    if isinstance(r0, ParseError):
        return r0.ctx("parsing factor or lhs of factor")
    lhs, tokens = r0

    if len(tokens) == 0:  # reached end of input
        return lhs, tokens

    op = tokens[0]
    if isinstance(op, Cparen):
        return lhs, tokens

    if isinstance(op, Operator) and not isinstance(op, ExpOp):
        return lhs, tokens

    r1 = expect_token_type(tokens, ExpOp, 0)
    if isinstance(r1, ParseError):
        return r1.ctx("unreachable")
    op, tokens = r1

    r2 = parse_factor(tokens)
    if isinstance(r2, ParseError):
        return r2.ctx("parsing factor")
    rhs, tokens = r2

    return BinOp(lhs, rhs, op), tokens


def parse_term(tokens: list[Token]) -> ParseResult:
    """parse a term

    from the grammar
    # term =
    #     factor
    #     | factor '*' term
    #     | factor '/' term

    Args:
        tokens: tokens to parse

    Returns:
        Term

    """
    r0 = parse_factor(tokens)
    if isinstance(r0, ParseError):
        return r0.ctx("parsing term or lhs of term")
    lhs, tokens = r0
    if len(tokens) == 0:
        # reached end of input
        return lhs, tokens

    op = tokens[0]
    if isinstance(op, Cparen):
        return lhs, tokens

    if isinstance(op, Operator) and not isinstance(op, Mul | Div):
        return lhs, tokens

    if not isinstance(op, Mul | Div):
        return ParseError(
            f"expected '*' or '/' while parsing factor. got '{op}'", op.pos
        ).ctx("parsing term")
    tokens = tokens[1:]

    r1 = parse_term(tokens)
    if isinstance(r1, ParseError):
        return r1.ctx("parsing rhs of term")
    rhs, tokens = r1

    return BinOp(lhs, rhs, op), tokens


def parse_expr(tokens: list[Token]) -> ParseResult:
    """parse an expression

    from the grammar:
    # expression =
    #     term
    #     | term '+' expression
    #     | term '-' expression

    Args:
        tokens: tokens to parse

    Returns:
        expression node or string in case of error
    """
    r0 = parse_term(tokens)
    if isinstance(r0, ParseError):
        return r0
    lhs, tokens = r0

    if len(tokens) == 0:
        # reached end of input
        return lhs, tokens

    op = tokens[0]
    if isinstance(op, Cparen):
        return lhs, tokens

    if not isinstance(op, Add | Sub):
        return ParseError(f"expected '+' or '-' while parsing term. got '{op}'", op.pos)

    tokens = tokens[1:]

    r1 = parse_expr(tokens)
    if isinstance(r1, ParseError):
        return r1
    rhs, tokens = r1

    return BinOp(lhs, rhs, op), tokens


@dataclass
class NonlinearInequalityConstraint:
    lhs: Node
    rhs: float
    greater: bool
    names: list[str]

    def __call__(self, x: torch.Tensor | list[float]) -> Numeric:
        """evaluate the constraint for values x

        Args:
            x: input values

        Returns:
            the constraint value
        """
        # constraint(x) >= rhs
        # constraint(x) - rhs >= 0
        assert len(x) == len(
            self.names
        ), "Number of entries in input must be the same as length of names in inequality constraint"
        ctx = {name: value for name, value in zip(self.names, x)}
        lhs = self.lhs(ctx)

        if self.greater:
            return lhs - self.rhs
        else:
            return -lhs + self.rhs

    def normalized_repr(self) -> str:
        op = ">" if self.greater else "<"
        return f"{self.lhs.normalized_repr()} {op} {self.rhs}"


def parse_nonlinear_constraint(
    expr: str, variables: list[str]
) -> NonlinearInequalityConstraint | str:
    op = None
    if ">" in expr:
        op = ">"

    if "<" in expr:
        if op is not None:
            return (
                "found '>' and '<' in expression. expected only one equality operator"
            )
        op = "<"

    if op is None:
        return "expected either '>' or '<' in expression."

    try:
        lhs, rhs = expr.split(op)
    except ValueError:
        return f"Expected only one '{op}' in the expression"

    try:
        rhs_val = float(rhs.strip())
    except ValueError:
        return f"right hand side of expression must be a number. got '{rhs}'"

    tokens = Lexer(lhs).tokens()
    if isinstance(tokens, str):
        return f"Error during processing of expression: {tokens}"

    ret = parse_expr(tokens)
    if isinstance(ret, ParseError):
        return f"error parsing left hand side of nonlinear inequality constraint: {ret.msg} at position {ret.pos}. Context: {expr[ret.pos:]}. Context: {ret.context}"
    lhs_node, tokens = ret
    if len(tokens) != 0:
        return f"error parsing left hand side of nonlinear inequality constraint: additional tokens: {tokens}"

    return NonlinearInequalityConstraint(lhs_node, rhs_val, op == ">", variables)
