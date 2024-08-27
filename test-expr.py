import expr_parser as ep


def test_lexer_add():
    tokens = ep.Lexer("1 + 2").tokens()
    assert tokens == [ep.Num(0, 1.0), ep.Add(2), ep.Num(4, 2.0)]


def test_lexer_log():
    tokens = ep.Lexer("1 + log(10)").tokens()
    assert tokens == [
        ep.Num(0, 1.0),
        ep.Add(2),
        ep.Log(4),
        ep.Oparen(7),
        ep.Num(8, 10.0),
        ep.Cparen(10),
    ]


def test_lexer_exp():
    tokens = ep.Lexer("2 + exp(10)").tokens()
    assert tokens == [
        ep.Num(0, 2.0),
        ep.Add(2),
        ep.Exp(4),
        ep.Oparen(7),
        ep.Num(8, 10.0),
        ep.Cparen(10),
    ]


def test_lexer_ln():
    tokens = ep.Lexer("2 + ln(10)").tokens()
    assert tokens == [
        ep.Num(0, 2.0),
        ep.Add(2),
        ep.Ln(4),
        ep.Oparen(6),
        ep.Num(7, 10.0),
        ep.Cparen(9),
    ]


def test_lexer_quoted_indent_only():
    tokens = ep.Lexer('"foo "').tokens()
    assert tokens == [
        ep.Ident(1, "foo "),
    ]


def test_lexer_ident():
    tokens = ep.Lexer("2 + foo * bar").tokens()
    assert tokens == [
        ep.Num(0, 2.0),
        ep.Add(2),
        ep.Ident(4, "foo"),
        ep.Mul(8),
        ep.Ident(10, "bar"),
    ]


def test_lexer_ident_quotes():
    tokens = ep.Lexer('"foo " * " bar"').tokens()
    assert tokens == [
        ep.Ident(1, "foo "),
        ep.Mul(7),
        ep.Ident(10, " bar"),
    ]


def test_lexer_ident_quotes_unfinished():
    tokens = ep.Lexer('"foo * 1.2').tokens()
    assert isinstance(tokens, str)


def test_lexer_punct():
    tokens = ep.Lexer("+-*/^()").tokens()
    assert tokens == [
        ep.Add(0),
        ep.Sub(1),
        ep.Mul(2),
        ep.Div(3),
        ep.ExpOp(4),
        ep.Oparen(5),
        ep.Cparen(6),
    ]


def test_lexer_negative_number():
    tokens = ep.Lexer("-1.0").tokens()
    assert tokens == [
        ep.Sub(0),
        ep.Num(1, 1.0),
    ]


def test_parser_add():
    tokens = ep.Lexer("1 + 1").tokens()
    assert not isinstance(tokens, str)
    ret = ep.parse_expr(tokens)
    assert not isinstance(ret, ep.ParseError)
    node, rest = ret

    assert rest == []
    assert node == ep.BinOp(ep.Num(0, 1.0), ep.Num(4, 1.0), ep.Add(2))


def test_parser_mul():
    tokens = ep.Lexer("2 * foo").tokens()
    assert not isinstance(tokens, str)
    ret = ep.parse_expr(tokens)
    assert not isinstance(ret, ep.ParseError)
    node, rest = ret

    assert rest == []
    assert node == ep.BinOp(ep.Num(0, 2.0), ep.Ident(4, "foo"), ep.Mul(2))


def test_parser_exp():
    tokens = ep.Lexer("2^foo").tokens()
    assert not isinstance(tokens, str)
    ret = ep.parse_expr(tokens)
    assert not isinstance(ret, ep.ParseError)
    node, rest = ret

    assert rest == []
    assert node == ep.BinOp(ep.Num(0, 2.0), ep.Ident(2, "foo"), ep.ExpOp(1))


def test_parser_neg():
    tokens = ep.Lexer("-1.0").tokens()
    assert not isinstance(tokens, str)
    ret = ep.parse_expr(tokens)
    assert not isinstance(ret, ep.ParseError)
    node, rest = ret

    assert rest == []
    assert node == ep.Neg(ep.Num(1, 1.0))


def test_parser_neg_mul():
    tokens = ep.Lexer("(-(1.0)) * a").tokens()
    assert not isinstance(tokens, str)
    ret = ep.parse_expr(tokens)
    assert not isinstance(ret, ep.ParseError)
    node, rest = ret

    assert rest == []
    expected = ep.BinOp(
        ep.Neg(ep.Num(3, 1.0)),
        ep.Ident(11, "a"),
        ep.Mul(9),
    )
    assert node == expected


def test_parser_neg_mul_no_parens():
    tokens = ep.Lexer("-1.0 * a").tokens()
    assert not isinstance(tokens, str)
    ret = ep.parse_expr(tokens)
    assert not isinstance(ret, ep.ParseError)
    node, rest = ret

    assert rest == []
    expected = ep.BinOp(
        ep.Neg(ep.Num(1, 1.0)),
        ep.Ident(7, "a"),
        ep.Mul(5),
    )
    assert node == expected


def test_parser_neg_mul_no_parens_end():
    tokens = ep.Lexer("a * -1.0").tokens()
    assert not isinstance(tokens, str)
    ret = ep.parse_expr(tokens)
    assert not isinstance(ret, ep.ParseError)
    node, rest = ret

    assert rest == []
    expected = ep.BinOp(
        ep.Ident(0, "a"),
        ep.Neg(ep.Num(5, 1.0)),
        ep.Mul(2),
    )
    assert node == expected


def test_parser_neg_mul_no_parens_end_add():
    tokens = ep.Lexer("a * -1.0 + 2").tokens()
    assert not isinstance(tokens, str)
    ret = ep.parse_expr(tokens)
    assert not isinstance(ret, ep.ParseError)
    node, rest = ret

    assert rest == []
    expected = ep.BinOp(
        ep.BinOp(
            ep.Ident(0, "a"),
            ep.Neg(ep.Num(5, 1.0)),
            ep.Mul(2),
        ),
        ep.Num(11, 2.0),
        ep.Add(9),
    )
    assert node == expected


def test_parser_builtin_ln():
    tokens = ep.Lexer("ln(foo + bar)").tokens()
    assert not isinstance(tokens, str)
    ret = ep.parse_expr(tokens)
    assert not isinstance(ret, ep.ParseError)
    node, rest = ret

    assert rest == []
    assert node == ep.Builtin(
        ep.Ln(0), ep.BinOp(ep.Ident(3, "foo"), ep.Ident(9, "bar"), ep.Add(7))
    )


def test_parser_add_mul_precedence():
    tokens = ep.Lexer("a + b * c").tokens()
    assert not isinstance(tokens, str)
    ret = ep.parse_expr(tokens)
    assert not isinstance(ret, ep.ParseError)
    node, rest = ret

    assert rest == []
    assert node == ep.BinOp(
        ep.Ident(0, "a"),
        ep.BinOp(ep.Ident(4, "b"), ep.Ident(8, "c"), ep.Mul(6)),
        ep.Add(2),
    )


def test_parser_mul_builtin():
    tokens = ep.Lexer("a * ln(a)").tokens()
    assert not isinstance(tokens, str)
    ret = ep.parse_expr(tokens)
    assert not isinstance(ret, ep.ParseError)
    node, rest = ret

    assert rest == []
    assert node == ep.BinOp(
        ep.Ident(0, "a"), ep.Builtin(ep.Ln(4), ep.Ident(7, "a")), ep.Mul(2)
    )


def test_triple_mul():
    tokens = ep.Lexer("a * b * c").tokens()
    assert not isinstance(tokens, str)
    ret = ep.parse_expr(tokens)
    assert not isinstance(ret, ep.ParseError)
    node, rest = ret

    assert rest == []
    assert node == ep.BinOp(
        ep.Ident(0, "a"),
        ep.BinOp(
            ep.Ident(4, "b"),
            ep.Ident(8, "c"),
            ep.Mul(6),
        ),
        ep.Mul(2),
    )


def test_triple_add():
    tokens = ep.Lexer("a + b + c").tokens()
    assert not isinstance(tokens, str)
    ret = ep.parse_expr(tokens)
    assert not isinstance(ret, ep.ParseError)
    node, rest = ret

    assert rest == []
    assert node == ep.BinOp(
        ep.Ident(0, "a"),
        ep.BinOp(
            ep.Ident(4, "b"),
            ep.Ident(8, "c"),
            ep.Add(6),
        ),
        ep.Add(2),
    )


def test_triple_exp():
    tokens = ep.Lexer("a ^ b ^ c").tokens()
    assert not isinstance(tokens, str)
    ret = ep.parse_expr(tokens)
    assert not isinstance(ret, ep.ParseError)
    node, rest = ret

    assert rest == []
    assert node == ep.BinOp(
        ep.Ident(0, "a"),
        ep.BinOp(
            ep.Ident(4, "b"),
            ep.Ident(8, "c"),
            ep.ExpOp(6),
        ),
        ep.ExpOp(2),
    )


def test_operator_precedence():
    tokens = ep.Lexer("a + b * c ^ ln(d)").tokens()
    assert not isinstance(tokens, str)
    ret = ep.parse_expr(tokens)
    assert not isinstance(ret, ep.ParseError)
    node, rest = ret

    assert rest == []
    assert node == ep.BinOp(
        ep.Ident(0, "a"),
        ep.BinOp(
            ep.Ident(4, "b"),
            ep.BinOp(
                ep.Ident(8, "c"),
                ep.Builtin(ep.Ln(12), ep.Ident(15, "d")),
                ep.ExpOp(10),
            ),
            ep.Mul(6),
        ),
        ep.Add(2),
    )


def test_real_examle():
    expr = "1.0 * log(Fe) * Fe + log(Mn) * Mn"
    tokens = ep.Lexer(expr).tokens()
    assert not isinstance(tokens, str)
    ret = ep.parse_expr(tokens)

    assert not isinstance(ret, ep.ParseError)
    node, rest = ret

    assert rest == []
    assert node == ep.BinOp(
        ep.BinOp(
            ep.Num(0, 1.0),
            ep.BinOp(
                ep.Builtin(ep.Log(6), ep.Ident(10, "Fe")),
                ep.Ident(16, "Fe"),
                ep.Mul(14),
            ),
            ep.Mul(4),
        ),
        ep.BinOp(
            ep.Builtin(ep.Log(21), ep.Ident(25, "Mn")),
            ep.Ident(31, "Mn"),
            ep.Mul(29),
        ),
        ep.Add(19),
    )


def test_constraint_ok():
    ret = ep.parse_nonlinear_constraint("foo + a < 12", ["foo", "a"])
    assert not isinstance(ret, str)
    constraint = ret
    x = [10.0, 0.0]
    assert constraint(x) >= 0


def test_constraint_gt():
    ret = ep.parse_nonlinear_constraint("foo > 12", ["foo"])
    assert not isinstance(ret, str)
    constraint = ret
    x = [10.0]
    assert not (constraint(x) >= 0)


def test_constraint_normalization():
    ret = ep.parse_nonlinear_constraint(
        "-1.0 * (x * ln(x) + y * ln(y)) > 0.1", ["x", "y"]
    )
    assert not isinstance(ret, str)
    constraint = ret

    assert ret.normalized_repr() == "(-1.0) * (x * ln(x) + y * ln(y)) > 0.1"
