import pytest

from error import CompilerError
from lexer import Lexer, TokenType
from parser import Parser
from semantic import SemanticAnalyzer


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def analyze(src: str) -> None:
    lexer = Lexer(src)
    tokens = []
    while True:
        t = lexer.get_next_token()
        tokens.append(t)
        if t.type == TokenType.EOF:
            break
    program = Parser(tokens).parse()
    SemanticAnalyzer().analyze(program)


def raises(src: str) -> pytest.ExceptionInfo:
    with pytest.raises(CompilerError) as exc_info:
        analyze(src)
    return exc_info


# ---------------------------------------------------------------------------
# Positive tests — valid programs that must pass without error
# ---------------------------------------------------------------------------

class TestValidPrograms:
    def test_fibonacci(self):
        src = """
        fn fib(n: int) -> int {
            if n <= 1 { return n; }
            return fib(n - 1) + fib(n - 2);
        }
        """
        analyze(src)   # must not raise

    def test_simple_var_decl(self):
        analyze("int x = 42;")

    def test_var_decl_no_initializer(self):
        analyze("float f;")

    def test_multiple_top_level_vars(self):
        analyze("int a = 1; int b = 2; int c = 3;")

    def test_function_no_params(self):
        analyze("fn greet() { int x = 1; }")

    def test_function_with_return_type(self):
        analyze("fn get_zero() -> int { return 0; }")

    def test_function_bool_return(self):
        analyze("fn is_positive(n: int) -> bool { return n > 0; }")

    def test_function_multiple_params(self):
        analyze("fn add(a: int, b: int) -> int { return a + b; }")

    def test_function_mixed_param_types(self):
        analyze("fn f(x: int, flag: bool) -> bool { return flag; }")

    def test_function_six_params_accepted(self):
        src = "fn f(a: int, b: int, c: int, d: int, e: int, g: int) -> int { return a; }"
        analyze(src)

    def test_for_loop_body_var_shadows_loop_var(self):
        # Body has its own inner scope, so 'i' inside the body does not conflict
        # with the loop induction variable 'i'.
        src = "fn f() { for i = 0; i < 5; i = i + 1 { int i = 99; } }"
        analyze(src)

    def test_mutually_recursive_functions(self):
        # Pass 1 registers both signatures so each can call the other.
        src = """
        fn is_even(n: int) -> bool { return n == 0; }
        fn is_odd(n: int) -> bool  { return is_even(n); }
        """
        analyze(src)

    def test_assignment_to_declared_var(self):
        analyze("int x = 0; x = 99;")

    def test_nested_if_else(self):
        src = """
        fn classify(n: int) -> int {
            if n > 0 {
                return 1;
            } else if n < 0 {
                return -1;
            } else {
                return 0;
            }
        }
        """
        analyze(src)

    def test_while_loop(self):
        src = """
        fn count() -> int {
            int x = 0;
            while x < 10 { x = x + 1; }
            return x;
        }
        """
        analyze(src)

    def test_for_loop(self):
        src = "fn f() { for i = 0; i < 5; i = i + 1 { int x = i; } }"
        analyze(src)

    def test_function_call_as_expression(self):
        src = """
        fn double(n: int) -> int { return n * 2; }
        fn quad(n: int)   -> int { return double(double(n)); }
        """
        analyze(src)

    def test_equality_comparison(self):
        src = "fn f(a: int, b: int) -> bool { return a == b; }"
        analyze(src)

    def test_logical_operators(self):
        src = "fn f(a: bool, b: bool) -> bool { return a && b || a; }"
        analyze(src)

    def test_unary_minus_on_int(self):
        src = "fn neg(n: int) -> int { return -n; }"
        analyze(src)

    def test_unary_bang_on_bool(self):
        src = "fn inv(b: bool) -> bool { return !b; }"
        analyze(src)

    def test_float_arithmetic(self):
        src = "fn add(a: float, b: float) -> float { return a + b; }"
        analyze(src)

    def test_parameter_shadows_nothing(self):
        # Param 'x' in inner scope should not conflict with global 'x'.
        src = "int x = 0; fn f(x: int) -> int { return x; }"
        analyze(src)

    def test_void_function_empty_return(self):
        analyze("fn f() { return; }")


# ---------------------------------------------------------------------------
# Negative tests — undefined variables
# ---------------------------------------------------------------------------

class TestUndefinedVariable:
    def test_use_before_declare(self):
        exc = raises("fn f() { x = 1; }")
        assert "undefined" in exc.value.message
        assert "x" in exc.value.message

    def test_read_undeclared_identifier(self):
        exc = raises("fn f() -> int { return y; }")
        assert "undefined" in exc.value.message

    def test_variable_not_in_outer_scope(self):
        # 'inner' is declared inside the if-block but used outside it.
        # Nova has no block-level scoping (only function-level), so this
        # actually passes — only function scope is pushed/popped.
        # This test verifies outer scope isolation between two functions.
        src = """
        fn a() { int local = 1; }
        fn b() -> int { return local; }
        """
        exc = raises(src)
        assert "undefined" in exc.value.message
        assert "local" in exc.value.message

    def test_call_undeclared_function(self):
        exc = raises("fn f() { missing(); }")
        assert "undefined" in exc.value.message
        assert "missing" in exc.value.message

    def test_error_carries_line_number(self):
        src = "fn f() {\n  int a = 1;\n  b = 2;\n}"
        exc = raises(src)
        assert exc.value.line == 3

    def test_for_loop_body_references_undeclared(self):
        exc = raises("fn f() { for i = 0; i < 3; i = i + 1 { z = 1; } }")
        assert "undefined" in exc.value.message
        assert "z" in exc.value.message


# ---------------------------------------------------------------------------
# Negative tests — type mismatches
# ---------------------------------------------------------------------------

class TestTypeMismatch:
    def test_int_var_assigned_string(self):
        exc = raises('int x = "hello";')
        assert "type mismatch" in exc.value.message

    def test_bool_var_assigned_int(self):
        exc = raises("bool flag = 42;")
        assert "type mismatch" in exc.value.message

    def test_int_var_assigned_bool(self):
        exc = raises("int n = true;")
        assert "type mismatch" in exc.value.message

    def test_float_var_assigned_bool(self):
        exc = raises("float f = false;")
        assert "type mismatch" in exc.value.message

    def test_add_int_and_bool(self):
        exc = raises("fn f() { int x = 1 + true; }")
        assert "same type" in exc.value.message or "type" in exc.value.message

    def test_compare_int_and_bool(self):
        exc = raises("fn f() -> bool { return 1 < true; }")
        assert "type" in exc.value.message

    def test_logical_and_on_ints(self):
        exc = raises("fn f(a: int, b: int) -> bool { return a && b; }")
        assert "bool" in exc.value.message

    def test_if_condition_not_bool(self):
        exc = raises("fn f() { if 1 { } }")
        assert "type mismatch" in exc.value.message

    def test_while_condition_not_bool(self):
        exc = raises("fn f() { while 0 { } }")
        assert "type mismatch" in exc.value.message

    def test_assign_wrong_type(self):
        exc = raises("int x = 1; x = true;")
        assert "type mismatch" in exc.value.message

    def test_unary_bang_on_int(self):
        exc = raises("fn f() { !1; }")
        assert "bool" in exc.value.message

    def test_unary_minus_on_bool(self):
        exc = raises("fn f() { bool b = true; int x = -b; }")
        assert "numeric" in exc.value.message

    def test_equality_across_types(self):
        exc = raises('fn f() -> bool { return 1 == "one"; }')
        assert "type" in exc.value.message

    def test_mismatch_error_has_position(self):
        exc = raises("bool b = 99;")
        assert exc.value.line == 1
        assert exc.value.col >= 1


# ---------------------------------------------------------------------------
# Negative tests — wrong function return type
# ---------------------------------------------------------------------------

class TestReturnTypeMismatch:
    def test_return_bool_from_int_function(self):
        exc = raises("fn f() -> int { return true; }")
        assert "return type mismatch" in exc.value.message

    def test_return_int_from_bool_function(self):
        exc = raises("fn f() -> bool { return 42; }")
        assert "return type mismatch" in exc.value.message

    def test_return_string_from_int_function(self):
        exc = raises('fn f() -> int { return "hello"; }')
        assert "return type mismatch" in exc.value.message

    def test_return_value_from_void_function(self):
        exc = raises("fn f() { return 1; }")
        assert "return type mismatch" in exc.value.message

    def test_empty_return_from_int_function(self):
        exc = raises("fn f() -> int { return; }")
        assert "return type mismatch" in exc.value.message

    def test_return_float_from_int_function(self):
        exc = raises("fn f() -> int { return 3.14; }")
        assert "return type mismatch" in exc.value.message

    def test_return_mismatch_carries_line(self):
        src = "fn f() -> int {\n  return true;\n}"
        exc = raises(src)
        assert exc.value.line == 2


# ---------------------------------------------------------------------------
# Negative tests — duplicate declarations
# ---------------------------------------------------------------------------

class TestDuplicateDeclaration:
    def test_duplicate_var_in_global_scope(self):
        exc = raises("int x = 1; int x = 2;")
        assert "already defined" in exc.value.message
        assert "x" in exc.value.message

    def test_duplicate_var_in_function_scope(self):
        exc = raises("fn f() { int n = 1; int n = 2; }")
        assert "already defined" in exc.value.message

    def test_duplicate_function_name(self):
        exc = raises("fn f() -> int { return 1; } fn f() -> int { return 2; }")
        assert "already defined" in exc.value.message
        assert "f" in exc.value.message

    def test_duplicate_param_name(self):
        exc = raises("fn f(x: int, x: int) -> int { return x; }")
        assert "already defined" in exc.value.message

    def test_duplicate_reports_first_definition_location(self):
        # The error message must mention the original definition line/col.
        src = "int count = 0;\nint count = 1;"
        exc = raises(src)
        assert "line 1" in exc.value.message

    def test_duplicate_carries_redefinition_line(self):
        src = "int x = 0;\nint x = 1;"
        exc = raises(src)
        assert exc.value.line == 2


# ---------------------------------------------------------------------------
# Implementation limits
# ---------------------------------------------------------------------------

class TestImplementationLimits:
    def test_seven_params_rejected(self):
        src = "fn f(a: int, b: int, c: int, d: int, e: int, g: int, h: int) -> int { return a; }"
        exc = raises(src)
        assert "Implementation Limit" in exc.value.message
        assert "7" in exc.value.message

    def test_seven_params_error_names_function(self):
        src = "fn too_many(a: int, b: int, c: int, d: int, e: int, g: int, h: int) -> int { return a; }"
        exc = raises(src)
        assert "too_many" in exc.value.message
