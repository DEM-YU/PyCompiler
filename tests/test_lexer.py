import pytest

from error import CompilerError
from lexer import Lexer, Token, TokenType


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def lex_all(source: str) -> list[Token]:
    """Tokenize source and return every token except the final EOF."""
    lexer = Lexer(source)
    tokens: list[Token] = []
    while True:
        token = lexer.get_next_token()
        if token.type == TokenType.EOF:
            break
        tokens.append(token)
    return tokens


# ---------------------------------------------------------------------------
# Arithmetic expressions
# ---------------------------------------------------------------------------

def test_arithmetic_integer_operators() -> None:
    tokens = lex_all("1 + 2 * 3 - 4 / 2 % 5")
    types = [t.type for t in tokens]
    assert types == [
        TokenType.INTEGER, TokenType.PLUS,    TokenType.INTEGER,
        TokenType.STAR,    TokenType.INTEGER,  TokenType.MINUS,
        TokenType.INTEGER, TokenType.SLASH,    TokenType.INTEGER,
        TokenType.PERCENT, TokenType.INTEGER,
    ]
    assert tokens[0].value == 1
    assert tokens[2].value == 2


def test_arithmetic_float_literal() -> None:
    tokens = lex_all("x + 3.14")
    assert tokens[0].type == TokenType.ID
    assert tokens[1].type == TokenType.PLUS
    assert tokens[2].type == TokenType.FLOAT_LITERAL
    assert tokens[2].value == pytest.approx(3.14)


# ---------------------------------------------------------------------------
# Variable declaration  (type keyword + identifier, with colon separator)
# ---------------------------------------------------------------------------

def test_variable_type_keywords() -> None:
    tokens = lex_all("int count float speed bool active string name")
    types = [t.type for t in tokens]
    assert types == [
        TokenType.INT,    TokenType.ID,
        TokenType.FLOAT,  TokenType.ID,
        TokenType.BOOL,   TokenType.ID,
        TokenType.STRING, TokenType.ID,
    ]
    assert tokens[1].value == "count"
    assert tokens[3].value == "speed"


def test_str_lexes_as_identifier() -> None:
    tokens = lex_all("str")
    assert tokens[0].type == TokenType.ID
    assert tokens[0].value == "str"


def test_typed_identifier_syntax() -> None:
    # Covers `name: type` patterns used in declarations and function parameters.
    tokens = lex_all("x: int")
    assert [t.type for t in tokens] == [TokenType.ID, TokenType.COLON, TokenType.INT]
    assert tokens[0].value == "x"


# ---------------------------------------------------------------------------
# Comparison and logical operators
# ---------------------------------------------------------------------------

def test_comparison_operators() -> None:
    tokens = lex_all("a == b != c < d > e <= f >= g")
    types = [t.type for t in tokens]
    assert types == [
        TokenType.ID, TokenType.EQ_EQ,   TokenType.ID,
        TokenType.BANG_EQ, TokenType.ID, TokenType.LT,
        TokenType.ID, TokenType.GT,      TokenType.ID,
        TokenType.LT_EQ,  TokenType.ID,  TokenType.GT_EQ,
        TokenType.ID,
    ]


def test_logical_operators() -> None:
    tokens = lex_all("a && b || !c")
    types = [t.type for t in tokens]
    assert types == [
        TokenType.ID, TokenType.AMP_AMP, TokenType.ID,
        TokenType.PIPE_PIPE, TokenType.BANG, TokenType.ID,
    ]


# ---------------------------------------------------------------------------
# if / else control flow
# ---------------------------------------------------------------------------

def test_if_else_control_flow() -> None:
    source = "if x >= 0 { return true; } else { return false; }"
    tokens = lex_all(source)
    types = [t.type for t in tokens]

    assert types[0] == TokenType.IF
    assert TokenType.GT_EQ   in types
    assert TokenType.RETURN  in types
    assert TokenType.TRUE    in types
    assert TokenType.ELSE    in types
    assert TokenType.FALSE   in types
    assert types[-1] == TokenType.RBRACE


def test_if_else_boolean_values() -> None:
    tokens = lex_all("true false")
    assert tokens[0].type == TokenType.TRUE
    assert tokens[0].value == "true"
    assert tokens[1].type == TokenType.FALSE


# ---------------------------------------------------------------------------
# fn function definition and return
# ---------------------------------------------------------------------------

def test_fn_definition_tokens() -> None:
    source = "fn add(a: int, b: int) -> int { return a + b; }"
    tokens = lex_all(source)
    types = [t.type for t in tokens]

    assert types[0] == TokenType.FN
    assert types[1] == TokenType.ID       # add
    assert types[2] == TokenType.LPAREN
    assert TokenType.ARROW  in types
    assert TokenType.RETURN in types
    assert TokenType.PLUS   in types
    assert types[-1] == TokenType.RBRACE


def test_fn_arrow_token() -> None:
    tokens = lex_all("-> int")
    assert tokens[0].type == TokenType.ARROW
    assert tokens[0].value == "->"
    assert tokens[1].type == TokenType.INT


# ---------------------------------------------------------------------------
# String literals
# ---------------------------------------------------------------------------

def test_string_literal_value() -> None:
    tokens = lex_all('"hello, world"')
    assert len(tokens) == 1
    assert tokens[0].type == TokenType.STRING_LITERAL
    assert tokens[0].value == "hello, world"


def test_empty_string_literal() -> None:
    tokens = lex_all('""')
    assert tokens[0].type == TokenType.STRING_LITERAL
    assert tokens[0].value == ""


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_empty_source() -> None:
    assert lex_all("") == []


def test_only_whitespace() -> None:
    assert lex_all("   \t\n  ") == []


def test_only_comments() -> None:
    assert lex_all("# first comment\n# second comment\n") == []


def test_inline_comment_ignored() -> None:
    tokens = lex_all("42  # the answer")
    assert len(tokens) == 1
    assert tokens[0].value == 42


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------

def test_unclosed_string_raises_error() -> None:
    with pytest.raises(CompilerError) as exc_info:
        lex_all('"unclosed')
    err = exc_info.value
    assert "unterminated" in err.message
    assert err.line == 1
    assert err.col == 1


def test_string_with_newline_raises_error() -> None:
    with pytest.raises(CompilerError) as exc_info:
        lex_all('"hello\nworld"')
    assert "unterminated" in exc_info.value.message


def test_unknown_character_raises_error() -> None:
    with pytest.raises(CompilerError) as exc_info:
        lex_all("@")
    err = exc_info.value
    assert "'@'" in err.message
    assert err.line == 1
    assert err.col == 1


def test_unknown_character_reports_correct_position() -> None:
    with pytest.raises(CompilerError) as exc_info:
        lex_all("fn @")
    err = exc_info.value
    assert err.col == 4


# ---------------------------------------------------------------------------
# Line and column tracking
# ---------------------------------------------------------------------------

def test_line_tracking_across_newlines() -> None:
    tokens = lex_all("fn\n  return")
    assert tokens[0].line == 1 and tokens[0].col == 1
    assert tokens[1].line == 2 and tokens[1].col == 3


def test_col_tracking_within_line() -> None:
    tokens = lex_all("fn add")
    assert tokens[0].col == 1   # fn
    assert tokens[1].col == 4   # add
