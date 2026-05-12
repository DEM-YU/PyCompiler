from dataclasses import dataclass
from enum import Enum
from typing import Any

from error import CompilerError


class TokenType(Enum):
    # Keywords — variable declaration
    LET = "let"

    # Keywords — control flow
    FN     = "fn"
    IF     = "if"
    ELSE   = "else"
    WHILE  = "while"
    FOR    = "for"
    RETURN = "return"

    # Keywords — types
    INT    = "int"
    FLOAT  = "float"
    BOOL   = "bool"
    STRING = "string"

    # Keywords — boolean values
    TRUE  = "true"
    FALSE = "false"

    # Operators
    PLUS      = "+"
    MINUS     = "-"
    STAR      = "*"
    SLASH     = "/"
    PERCENT   = "%"
    ASSIGN    = "="
    EQ_EQ     = "=="
    BANG_EQ   = "!="
    LT        = "<"
    GT        = ">"
    LT_EQ     = "<="
    GT_EQ     = ">="
    AMP_AMP   = "&&"
    PIPE_PIPE = "||"
    BANG      = "!"

    # Punctuation
    LPAREN    = "("
    RPAREN    = ")"
    LBRACE    = "{"
    RBRACE    = "}"
    LBRACKET  = "["
    RBRACKET  = "]"
    COMMA     = ","
    COLON     = ":"
    ARROW     = "->"
    SEMICOLON = ";"

    # Literals and identifiers
    ID             = "ID"
    INTEGER        = "INTEGER"
    FLOAT_LITERAL  = "FLOAT_LITERAL"
    STRING_LITERAL = "STRING_LITERAL"

    # End of file
    EOF = "EOF"


# Used by the lexer to resolve whether an identifier is a reserved keyword.
KEYWORDS: dict[str, TokenType] = {
    "let":    TokenType.LET,
    "fn":     TokenType.FN,
    "if":     TokenType.IF,
    "else":   TokenType.ELSE,
    "while":  TokenType.WHILE,
    "for":    TokenType.FOR,
    "return": TokenType.RETURN,
    "int":    TokenType.INT,
    "float":  TokenType.FLOAT,
    "bool":   TokenType.BOOL,
    "string": TokenType.STRING,
    "true":   TokenType.TRUE,
    "false":  TokenType.FALSE,
}


# Operator lookup tables used by _read_symbol; kept at module level to avoid
# rebuilding them on every token.
_TWO_CHAR_OPS: dict[str, TokenType] = {
    "==": TokenType.EQ_EQ,
    "!=": TokenType.BANG_EQ,
    "<=": TokenType.LT_EQ,
    ">=": TokenType.GT_EQ,
    "->": TokenType.ARROW,
    "&&": TokenType.AMP_AMP,
    "||": TokenType.PIPE_PIPE,
}

_SINGLE_CHAR_TOKENS: dict[str, TokenType] = {
    "+": TokenType.PLUS,
    "-": TokenType.MINUS,
    "*": TokenType.STAR,
    "/": TokenType.SLASH,
    "%": TokenType.PERCENT,
    "=": TokenType.ASSIGN,
    "<": TokenType.LT,
    ">": TokenType.GT,
    "!": TokenType.BANG,
    "(": TokenType.LPAREN,
    ")": TokenType.RPAREN,
    "{": TokenType.LBRACE,
    "}": TokenType.RBRACE,
    "[": TokenType.LBRACKET,
    "]": TokenType.RBRACKET,
    ",": TokenType.COMMA,
    ":": TokenType.COLON,
    ";": TokenType.SEMICOLON,
}


@dataclass
class Token:
    type: TokenType
    value: Any
    line: int
    col: int


class Lexer:
    def __init__(self, source_code: str) -> None:
        self._source = source_code
        self.pos = 0
        self.line = 1
        self.col = 1
        self.current_char: str | None = source_code[0] if source_code else None

    def _advance(self) -> None:
        if self.current_char == "\n":
            self.line += 1
            self.col = 1
        else:
            self.col += 1
        self.pos += 1
        if self.pos < len(self._source):
            self.current_char = self._source[self.pos]
        else:
            self.current_char = None

    def _peek(self) -> str | None:
        next_pos = self.pos + 1
        if next_pos < len(self._source):
            return self._source[next_pos]
        return None

    def _skip_whitespace_and_comments(self) -> None:
        while self.current_char is not None:
            if self.current_char == "#":
                # Consume the rest of the line; the \n itself is consumed on
                # the next iteration so line/col accounting stays correct.
                while self.current_char is not None and self.current_char != "\n":
                    self._advance()
            elif self.current_char in " \t\r\n":
                self._advance()
            else:
                break

    def _read_identifier(self) -> Token:
        start_line = self.line
        start_col = self.col
        chars: list[str] = []
        while self.current_char is not None and (self.current_char.isalnum() or self.current_char == "_"):
            chars.append(self.current_char)
            self._advance()
        word = "".join(chars)
        token_type = KEYWORDS.get(word, TokenType.ID)
        return Token(type=token_type, value=word, line=start_line, col=start_col)

    def _read_number(self) -> Token:
        start_line = self.line
        start_col = self.col
        digits: list[str] = []
        while self.current_char is not None and self.current_char.isdigit():
            digits.append(self.current_char)
            self._advance()
        if self.current_char == ".":
            next_char = self._peek()
            if next_char is not None and next_char.isdigit():
                digits.append(".")
                self._advance()
                while self.current_char is not None and self.current_char.isdigit():
                    digits.append(self.current_char)
                    self._advance()
                return Token(type=TokenType.FLOAT_LITERAL, value=float("".join(digits)), line=start_line, col=start_col)
            raise CompilerError(
                message=f"invalid number literal: trailing '.'",
                line=start_line,
                col=start_col,
            )
        return Token(type=TokenType.INTEGER, value=int("".join(digits)), line=start_line, col=start_col)

    def _read_string(self) -> Token:
        start_line = self.line
        start_col = self.col
        self._advance()  # consume opening "
        chars: list[str] = []
        while self.current_char is not None and self.current_char != '"':
            if self.current_char == "\n":
                raise CompilerError(
                    message="unterminated string literal",
                    line=start_line,
                    col=start_col,
                )
            chars.append(self.current_char)
            self._advance()
        if self.current_char is None:
            raise CompilerError(
                message="unterminated string literal",
                line=start_line,
                col=start_col,
            )
        self._advance()  # consume closing "
        return Token(type=TokenType.STRING_LITERAL, value="".join(chars), line=start_line, col=start_col)

    def _read_symbol(self) -> Token:
        line = self.line
        col = self.col
        char = self.current_char or ""
        two_char = char + (self._peek() or "")
        if two_char in _TWO_CHAR_OPS:
            self._advance()
            self._advance()
            return Token(type=_TWO_CHAR_OPS[two_char], value=two_char, line=line, col=col)
        if char in _SINGLE_CHAR_TOKENS:
            self._advance()
            return Token(type=_SINGLE_CHAR_TOKENS[char], value=char, line=line, col=col)
        raise CompilerError(
            message=f"unexpected character '{char}'",
            line=line,
            col=col,
        )

    def get_next_token(self) -> Token:
        self._skip_whitespace_and_comments()
        if self.current_char is None:
            return Token(type=TokenType.EOF, value=None, line=self.line, col=self.col)
        if self.current_char.isalpha() or self.current_char == "_":
            return self._read_identifier()
        if self.current_char.isdigit():
            return self._read_number()
        if self.current_char == '"':
            return self._read_string()
        return self._read_symbol()


# ---------------------------------------------------------------------------
# CLI entry point — python lexer.py <file>
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("usage: python lexer.py <source_file>", file=sys.stderr)
        sys.exit(1)

    try:
        with open(sys.argv[1]) as f:
            source = f.read()
    except FileNotFoundError:
        print(f"error: file not found: {sys.argv[1]}", file=sys.stderr)
        sys.exit(1)

    lexer = Lexer(source)
    try:
        while True:
            token = lexer.get_next_token()
            print(f"[{token.line}:{token.col:<3}] {token.type.name:<16} {token.value!r}")
            if token.type == TokenType.EOF:
                break
    except CompilerError as e:
        print(e.format(), file=sys.stderr)
        sys.exit(1)
