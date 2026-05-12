"""
End-to-end compilation tests (frontend + IR pipeline).

Covers: Lexer → Parser → SemanticAnalyzer → IRGenerator → Optimizer

Each test verifies that the compiler either accepts a valid program without
error, or rejects an invalid one with the correct CompilerError message.
The optimizer integration tests additionally inspect the optimised TAC to
confirm that constant folding and propagation produce the expected IR.
"""
from __future__ import annotations

import pytest

from error import CompilerError
from ir_gen import IRGenerator, TACInstruction
from lexer import Lexer, TokenType
from optimizer import Optimizer
from parser import Parser
from semantic import SemanticAnalyzer


# ---------------------------------------------------------------------------
# Pipeline helper
# ---------------------------------------------------------------------------

def _compile(source: str) -> list[TACInstruction]:
    """Lex → parse → semantic analysis → IR → optimise.  Returns optimised TAC."""
    lexer = Lexer(source)
    tokens = []
    while True:
        tok = lexer.get_next_token()
        tokens.append(tok)
        if tok.type == TokenType.EOF:
            break
    program = Parser(tokens).parse()
    SemanticAnalyzer().analyze(program)
    tac = IRGenerator().generate(program)
    return Optimizer().optimize(tac)


# ---------------------------------------------------------------------------
# Nova programs
# ---------------------------------------------------------------------------

_FIBONACCI = """
fn fib(n: int) -> int {
    if n <= 1 { return n; }
    return fib(n - 1) + fib(n - 2);
}

fn main() -> int {
    print(fib(10));
    return 0;
}
"""

_FACTORIAL = """
fn fact(n: int) -> int {
    if n <= 1 { return 1; }
    return n * fact(n - 1);
}

fn main() -> int {
    print(fact(5));
    print(fact(0));
    return 0;
}
"""

_MUTUAL_RECURSION = """
fn is_even(n: int) -> bool {
    if n == 0 { return true; }
    return is_odd(n - 1);
}

fn is_odd(n: int) -> bool {
    if n == 0 { return false; }
    return is_even(n - 1);
}

fn main() -> int {
    if is_even(4) { print(1); }
    if is_odd(3)  { print(1); }
    return 0;
}
"""

_WHILE_COUNT = """
fn main() -> int {
    int i = 1;
    while i <= 5 {
        print(i);
        i = i + 1;
    }
    return 0;
}
"""

_FOR_SUM = """
fn main() -> int {
    int sum = 0;
    for i = 1; i <= 5; i = i + 1 {
        sum = sum + i;
    }
    return sum;
}
"""

_MULTI_FUNC = """
fn square(x: int) -> int {
    return x * x;
}

fn sum_of_squares(a: int, b: int) -> int {
    return square(a) + square(b);
}

fn main() -> int {
    print(sum_of_squares(3, 4));
    return 0;
}
"""

_ABS_VAL = """
fn abs_val(n: int) -> int {
    if n < 0 { return 0 - n; }
    return n;
}

fn main() -> int {
    print(abs_val(-7));
    print(abs_val(3));
    return 0;
}
"""

_IF_ELSE_CHAIN = """
fn classify(n: int) -> int {
    if n < 0  { return -1; }
    if n == 0 { return 0; }
    return 1;
}

fn main() -> int {
    print(classify(-5));
    print(classify(0));
    print(classify(9));
    return 0;
}
"""

_CONSTANT_FOLDING = """
fn main() -> int {
    int x = 10 * 2;
    int y = x + 5;
    return y;
}
"""


# ---------------------------------------------------------------------------
# Recursion
# ---------------------------------------------------------------------------

class TestRecursion:
    def test_fibonacci_compiles(self):
        _compile(_FIBONACCI)

    def test_factorial_compiles(self):
        _compile(_FACTORIAL)

    def test_mutual_recursion_compiles(self):
        _compile(_MUTUAL_RECURSION)


# ---------------------------------------------------------------------------
# Iteration
# ---------------------------------------------------------------------------

class TestIteration:
    def test_while_loop_compiles(self):
        _compile(_WHILE_COUNT)

    def test_for_loop_compiles(self):
        _compile(_FOR_SUM)


# ---------------------------------------------------------------------------
# Multiple functions
# ---------------------------------------------------------------------------

class TestMultipleFunctions:
    def test_sum_of_squares_compiles(self):
        _compile(_MULTI_FUNC)

    def test_abs_val_compiles(self):
        _compile(_ABS_VAL)

    def test_if_else_chain_compiles(self):
        _compile(_IF_ELSE_CHAIN)


# ---------------------------------------------------------------------------
# Optimizer integration
# ---------------------------------------------------------------------------

class TestOptimizerIntegration:
    def test_constant_folding_reaches_25(self):
        # 10 * 2 = 20 → constant-folded (no MUL in output);
        # x + 5 = 25 → propagated (a COPY with arg1 == "25" appears).
        tac = _compile(_CONSTANT_FOLDING)
        assert not any(i.op == "*" for i in tac), "MUL should have been folded away"
        assert any(i.op == "COPY" and str(i.arg1) == "25" for i in tac)

    def test_optimiser_preserves_recursive_control_flow(self):
        # The optimiser must not crash or corrupt fib's branching structure.
        tac = _compile(_FIBONACCI)
        func_names = [i.result for i in tac if i.op == "FUNC"]
        assert "fib" in func_names
        assert "main" in func_names


# ---------------------------------------------------------------------------
# Compiler error cases — caught at semantic analysis, before IR
# ---------------------------------------------------------------------------

class TestCompilerErrors:
    def test_undefined_variable(self):
        src = "fn main() -> int { return z; }"
        with pytest.raises(CompilerError, match="undefined"):
            _compile(src)

    def test_undefined_function(self):
        src = "fn main() -> int { return ghost(); }"
        with pytest.raises(CompilerError, match="undefined"):
            _compile(src)

    def test_type_mismatch_in_assignment(self):
        src = "fn main() -> int { int x = true; return x; }"
        with pytest.raises(CompilerError, match="type mismatch"):
            _compile(src)

    def test_return_type_mismatch(self):
        src = "fn main() -> int { return true; }"
        with pytest.raises(CompilerError, match="return type mismatch"):
            _compile(src)

    def test_duplicate_variable_in_same_scope(self):
        src = "fn main() -> int { int x = 1; int x = 2; return x; }"
        with pytest.raises(CompilerError, match="already defined"):
            _compile(src)

    def test_wrong_condition_type_in_while(self):
        src = "fn main() -> int { while 1 { return 0; } return 0; }"
        with pytest.raises(CompilerError, match="type mismatch"):
            _compile(src)
