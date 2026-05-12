import pytest

from ast_nodes import (
    ArrayType, AssignStmt, BinaryOp, Block, ForStmt, FuncCall,
    FunctionDecl, Identifier, IfStmt, IndexAssignStmt, IndexExpr,
    Literal, Program, ReturnStmt, StringLiteral, UnaryOp, VarDecl, WhileStmt,
)
from error import CompilerError
from lexer import Lexer, TokenType
from parser import Parser


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def parse(src: str) -> Program:
    lexer = Lexer(src)
    tokens = []
    while True:
        t = lexer.get_next_token()
        tokens.append(t)
        if t.type == TokenType.EOF:
            break
    return Parser(tokens).parse()


def first_stmt(src: str):
    return parse(src).body[0]


# ---------------------------------------------------------------------------
# Operator precedence
# ---------------------------------------------------------------------------

class TestOperatorPrecedence:
    def test_mul_before_add(self):
        # 1 + 2 * 3  should parse as  1 + (2 * 3)
        node = first_stmt("1 + 2 * 3;")
        assert isinstance(node, BinaryOp)
        assert node.op == "+"
        assert isinstance(node.left, Literal)
        assert node.left.value == 1
        right = node.right
        assert isinstance(right, BinaryOp)
        assert right.op == "*"
        assert right.left.value == 2
        assert right.right.value == 3

    def test_add_left_associative(self):
        # 1 + 2 + 3  should parse as  (1 + 2) + 3
        node = first_stmt("1 + 2 + 3;")
        assert isinstance(node, BinaryOp)
        assert node.op == "+"
        left = node.left
        assert isinstance(left, BinaryOp)
        assert left.op == "+"
        assert left.left.value == 1
        assert left.right.value == 2
        assert node.right.value == 3

    def test_and_before_or(self):
        # a || b && c  should parse as  a || (b && c)
        node = first_stmt("a || b && c;")
        assert isinstance(node, BinaryOp)
        assert node.op == "||"
        assert isinstance(node.left, Identifier)
        assert node.left.name == "a"
        right = node.right
        assert isinstance(right, BinaryOp)
        assert right.op == "&&"

    def test_comparison_before_equality(self):
        # a == b < c  should parse as  a == (b < c)
        node = first_stmt("a == b < c;")
        assert isinstance(node, BinaryOp)
        assert node.op == "=="
        assert node.right.op == "<"

    def test_parens_override_precedence(self):
        # (1 + 2) * 3  should parse as  (1 + 2) * 3
        node = first_stmt("(1 + 2) * 3;")
        assert isinstance(node, BinaryOp)
        assert node.op == "*"
        left = node.left
        assert isinstance(left, BinaryOp)
        assert left.op == "+"

    def test_full_precedence_chain(self):
        # true || false && 1 == 2 + 3 * 4
        # parses as: true || (false && (1 == (2 + (3 * 4))))
        node = first_stmt("true || false && 1 == 2 + 3 * 4;")
        assert node.op == "||"
        and_node = node.right
        assert and_node.op == "&&"
        eq_node = and_node.right
        assert eq_node.op == "=="
        add_node = eq_node.right
        assert add_node.op == "+"
        mul_node = add_node.right
        assert mul_node.op == "*"

    def test_unary_minus(self):
        node = first_stmt("-x;")
        assert isinstance(node, UnaryOp)
        assert node.op == "-"
        assert isinstance(node.operand, Identifier)
        assert node.operand.name == "x"

    def test_unary_bang(self):
        node = first_stmt("!true;")
        assert isinstance(node, UnaryOp)
        assert node.op == "!"
        assert isinstance(node.operand, Literal)

    def test_unary_chaining(self):
        # !!x  parses as  !(!x)  — right-associative via recursion
        node = first_stmt("!!x;")
        assert isinstance(node, UnaryOp)
        assert node.op == "!"
        inner = node.operand
        assert isinstance(inner, UnaryOp)
        assert inner.op == "!"

    def test_modulo_precedence(self):
        # 10 % 3 + 1  parses as  (10 % 3) + 1
        node = first_stmt("10 % 3 + 1;")
        assert node.op == "+"
        assert node.left.op == "%"


# ---------------------------------------------------------------------------
# Literals and identifiers
# ---------------------------------------------------------------------------

class TestLiteralsAndIdentifiers:
    def test_integer_literal(self):
        node = first_stmt("42;")
        assert isinstance(node, Literal)
        assert node.value == 42
        assert node.value_type == "int"

    def test_float_literal(self):
        node = first_stmt("3.14;")
        assert isinstance(node, Literal)
        assert node.value == pytest.approx(3.14)
        assert node.value_type == "float"

    def test_string_literal(self):
        node = first_stmt('"hello";')
        assert isinstance(node, StringLiteral)
        assert node.value == "hello"

    def test_true_literal(self):
        node = first_stmt("true;")
        assert isinstance(node, Literal)
        assert node.value is True
        assert node.value_type == "bool"

    def test_false_literal(self):
        node = first_stmt("false;")
        assert isinstance(node, Literal)
        assert node.value is False

    def test_identifier(self):
        node = first_stmt("myVar;")
        assert isinstance(node, Identifier)
        assert node.name == "myVar"


# ---------------------------------------------------------------------------
# Variable declarations and assignments
# ---------------------------------------------------------------------------

class TestVarDeclAndAssign:
    def test_var_decl_with_init(self):
        node = first_stmt("int x = 5;")
        assert isinstance(node, VarDecl)
        assert node.var_type == "int"
        assert node.name == "x"
        assert isinstance(node.initializer, Literal)
        assert node.initializer.value == 5

    def test_var_decl_without_init(self):
        node = first_stmt("float y;")
        assert isinstance(node, VarDecl)
        assert node.var_type == "float"
        assert node.name == "y"
        assert node.initializer is None

    def test_bool_var_decl(self):
        node = first_stmt("bool flag = true;")
        assert isinstance(node, VarDecl)
        assert node.var_type == "bool"
        assert node.initializer.value is True

    def test_assign_stmt(self):
        node = first_stmt("x = 10;")
        assert isinstance(node, AssignStmt)
        assert node.name == "x"
        assert isinstance(node.value, Literal)
        assert node.value.value == 10

    def test_assign_with_expression(self):
        node = first_stmt("result = a + b;")
        assert isinstance(node, AssignStmt)
        assert node.name == "result"
        assert isinstance(node.value, BinaryOp)
        assert node.value.op == "+"


# ---------------------------------------------------------------------------
# If / else-if / else
# ---------------------------------------------------------------------------

class TestIfStmt:
    def test_simple_if(self):
        node = first_stmt("if x > 0 { y = 1; }")
        assert isinstance(node, IfStmt)
        assert isinstance(node.condition, BinaryOp)
        assert node.condition.op == ">"
        assert len(node.else_ifs) == 0
        assert node.else_block is None

    def test_if_else(self):
        node = first_stmt("if x > 0 { y = 1; } else { y = 2; }")
        assert isinstance(node, IfStmt)
        assert node.else_block is not None
        assert len(node.else_block.statements) == 1

    def test_if_else_if(self):
        src = "if a { x = 1; } else if b { x = 2; }"
        node = first_stmt(src)
        assert isinstance(node, IfStmt)
        assert len(node.else_ifs) == 1
        cond, block = node.else_ifs[0]
        assert isinstance(cond, Identifier)
        assert cond.name == "b"

    def test_if_else_if_else(self):
        src = "if a { x = 1; } else if b { x = 2; } else { x = 3; }"
        node = first_stmt(src)
        assert len(node.else_ifs) == 1
        assert node.else_block is not None

    def test_multiple_else_if_chains(self):
        src = "if a { x = 1; } else if b { x = 2; } else if c { x = 3; } else { x = 4; }"
        node = first_stmt(src)
        assert len(node.else_ifs) == 2
        assert node.else_block is not None

    def test_nested_if_inside_if(self):
        src = "if x > 0 { if y > 0 { z = 1; } }"
        node = first_stmt(src)
        assert isinstance(node, IfStmt)
        inner = node.then_block.statements[0]
        assert isinstance(inner, IfStmt)


# ---------------------------------------------------------------------------
# While loop
# ---------------------------------------------------------------------------

class TestWhileStmt:
    def test_while_basic(self):
        node = first_stmt("while x > 0 { x = x - 1; }")
        assert isinstance(node, WhileStmt)
        assert isinstance(node.condition, BinaryOp)
        assert node.condition.op == ">"
        assert len(node.body.statements) == 1

    def test_while_with_complex_condition(self):
        node = first_stmt("while a && b { x = 1; }")
        assert isinstance(node, WhileStmt)
        assert node.condition.op == "&&"


# ---------------------------------------------------------------------------
# For loop
# ---------------------------------------------------------------------------

class TestForStmt:
    def test_for_basic(self):
        node = first_stmt("for i = 0; i < 10; i = i + 1 { x = i; }")
        assert isinstance(node, ForStmt)
        assert node.var_name == "i"
        assert isinstance(node.init_value, Literal)
        assert node.init_value.value == 0
        assert isinstance(node.condition, BinaryOp)
        assert node.condition.op == "<"
        assert node.step_var == "i"
        assert isinstance(node.step_value, BinaryOp)
        assert node.step_value.op == "+"


# ---------------------------------------------------------------------------
# Return statement
# ---------------------------------------------------------------------------

class TestReturnStmt:
    def test_return_with_value(self):
        src = "fn f() -> int { return 42; }"
        decl = first_stmt(src)
        ret = decl.body.statements[0]
        assert isinstance(ret, ReturnStmt)
        assert isinstance(ret.value, Literal)
        assert ret.value.value == 42

    def test_return_without_value(self):
        src = "fn f() { return; }"
        decl = first_stmt(src)
        ret = decl.body.statements[0]
        assert isinstance(ret, ReturnStmt)
        assert ret.value is None


# ---------------------------------------------------------------------------
# Function declarations and calls
# ---------------------------------------------------------------------------

class TestFunctionDecl:
    def test_fn_no_params_no_return(self):
        node = first_stmt("fn greet() { x = 1; }")
        assert isinstance(node, FunctionDecl)
        assert node.name == "greet"
        assert len(node.params) == 0
        assert node.return_type is None

    def test_fn_with_return_type(self):
        node = first_stmt("fn get_zero() -> int { return 0; }")
        assert isinstance(node, FunctionDecl)
        assert node.return_type == "int"

    def test_fn_one_param(self):
        node = first_stmt("fn double(n: int) -> int { return n; }")
        assert isinstance(node, FunctionDecl)
        assert len(node.params) == 1
        assert node.params[0].name == "n"
        assert node.params[0].param_type == "int"

    def test_fn_multiple_params(self):
        node = first_stmt("fn add(a: int, b: int) -> int { return a; }")
        assert isinstance(node, FunctionDecl)
        assert len(node.params) == 2
        assert node.params[0].name == "a"
        assert node.params[1].name == "b"

    def test_fn_mixed_param_types(self):
        node = first_stmt("fn f(x: int, flag: bool, name: string) -> float { return 0; }")
        assert len(node.params) == 3
        assert node.params[0].param_type == "int"
        assert node.params[1].param_type == "bool"
        assert node.params[2].param_type == "string"

    def test_fn_fibonacci(self):
        src = "fn fib(n: int) -> int { if n <= 1 { return n; } return fib(n - 1) + fib(n - 2); }"
        node = first_stmt(src)
        assert isinstance(node, FunctionDecl)
        assert node.name == "fib"
        assert node.return_type == "int"
        stmts = node.body.statements
        assert isinstance(stmts[0], IfStmt)
        assert isinstance(stmts[1], ReturnStmt)


class TestFuncCall:
    def test_call_no_args(self):
        node = first_stmt("greet();")
        assert isinstance(node, FuncCall)
        assert node.name == "greet"
        assert len(node.args) == 0

    def test_call_one_arg(self):
        node = first_stmt("print(42);")
        assert isinstance(node, FuncCall)
        assert len(node.args) == 1
        assert node.args[0].value == 42

    def test_call_multiple_args(self):
        node = first_stmt("add(1, 2, 3);")
        assert isinstance(node, FuncCall)
        assert len(node.args) == 3

    def test_call_with_expression_arg(self):
        node = first_stmt("f(a + b);")
        assert isinstance(node, FuncCall)
        assert isinstance(node.args[0], BinaryOp)

    def test_nested_call(self):
        node = first_stmt("f(g(x));")
        assert isinstance(node, FuncCall)
        assert isinstance(node.args[0], FuncCall)
        assert node.args[0].name == "g"


# ---------------------------------------------------------------------------
# Program-level (multiple declarations)
# ---------------------------------------------------------------------------

class TestProgram:
    def test_multiple_functions(self):
        src = "fn a() { x = 1; } fn b() { y = 2; }"
        program = parse(src)
        assert len(program.body) == 2
        assert program.body[0].name == "a"
        assert program.body[1].name == "b"

    def test_top_level_statements(self):
        src = "int x = 0; x = 1;"
        program = parse(src)
        assert isinstance(program.body[0], VarDecl)
        assert isinstance(program.body[1], AssignStmt)


# ---------------------------------------------------------------------------
# Source position tracking
# ---------------------------------------------------------------------------

class TestPositionTracking:
    def test_literal_position(self):
        node = first_stmt("42;")
        assert node.line == 1
        assert node.col == 1

    def test_operator_position(self):
        node = first_stmt("1 + 2;")
        assert isinstance(node, BinaryOp)
        assert node.line == 1
        assert node.col == 3   # "+" is at column 3

    def test_identifier_position(self):
        node = first_stmt("   x;")
        assert isinstance(node, Identifier)
        assert node.col == 4

    def test_error_carries_position(self):
        with pytest.raises(CompilerError) as exc_info:
            parse("int ;")
        err = exc_info.value
        assert err.line == 1
        assert err.col > 1


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------

class TestSyntaxErrors:
    def test_error_missing_semicolon(self):
        with pytest.raises(CompilerError):
            parse("int x = 5")

    def test_error_missing_closing_paren(self):
        with pytest.raises(CompilerError):
            parse("f(1, 2;")

    def test_error_missing_closing_brace(self):
        with pytest.raises(CompilerError):
            parse("fn f() { return 1; ")

    def test_error_bad_type_annotation(self):
        with pytest.raises(CompilerError):
            parse("fn f(x: blah) { }")

    def test_error_missing_fn_name(self):
        with pytest.raises(CompilerError):
            parse("fn () { }")

    def test_error_missing_colon_in_param(self):
        with pytest.raises(CompilerError):
            parse("fn f(x int) { }")

    def test_error_unexpected_token(self):
        with pytest.raises(CompilerError):
            parse("@invalid;")

    def test_error_missing_arrow_type(self):
        # "fn f() -> ;" — arrow present but no type
        with pytest.raises(CompilerError):
            parse("fn f() -> ;")

    def test_error_for_missing_semicolons(self):
        with pytest.raises(CompilerError):
            parse("for i = 0 i < 10 i = i + 1 { }")


# ---------------------------------------------------------------------------
# Array types, index expressions, and index assignment statements
# ---------------------------------------------------------------------------

class TestArrayParsing:

    # --- ArrayType ---

    def test_array_decl_produces_var_decl(self):
        node = first_stmt("[int; 10] arr;")
        assert isinstance(node, VarDecl)
        assert node.name == "arr"

    def test_array_type_element_type_int(self):
        node = first_stmt("[int; 10] arr;")
        assert isinstance(node.var_type, ArrayType)
        assert node.var_type.element_type == "int"

    def test_array_type_size(self):
        node = first_stmt("[int; 10] arr;")
        assert node.var_type.size == 10

    def test_array_type_float_element(self):
        node = first_stmt("[float; 5] data;")
        assert node.var_type.element_type == "float"
        assert node.var_type.size == 5

    def test_array_type_bool_element(self):
        node = first_stmt("[bool; 3] flags;")
        assert node.var_type.element_type == "bool"
        assert node.var_type.size == 3

    def test_array_decl_no_initializer(self):
        node = first_stmt("[int; 4] nums;")
        assert node.initializer is None

    def test_array_type_position(self):
        node = first_stmt("[int; 5] arr;")
        assert node.var_type.line == 1
        assert node.var_type.col == 1

    # --- IndexExpr ---

    def test_index_expr_in_var_decl_initializer(self):
        node = first_stmt("int x = arr[0];")
        assert isinstance(node, VarDecl)
        assert isinstance(node.initializer, IndexExpr)

    def test_index_expr_name(self):
        node = first_stmt("int x = arr[0];")
        assert node.initializer.name == "arr"

    def test_index_expr_literal_index(self):
        node = first_stmt("int x = arr[0];")
        ix = node.initializer
        assert isinstance(ix.index, Literal)
        assert ix.index.value == 0

    def test_index_expr_identifier_index(self):
        node = first_stmt("int x = arr[i];")
        ix = node.initializer
        assert isinstance(ix.index, Identifier)
        assert ix.index.name == "i"

    def test_index_expr_expression_index(self):
        node = first_stmt("int x = arr[i + 1];")
        ix = node.initializer
        assert isinstance(ix.index, BinaryOp)
        assert ix.index.op == "+"

    def test_index_expr_in_return_stmt(self):
        src = "fn f() -> int { return nums[2]; }"
        ret = first_stmt(src).body.statements[0]
        assert isinstance(ret.value, IndexExpr)
        assert ret.value.name == "nums"
        assert ret.value.index.value == 2

    def test_index_expr_in_binary_expression(self):
        # arr[0] + arr[1] — both sides parse as IndexExpr
        node = first_stmt("int x = arr[0] + arr[1];")
        expr = node.initializer
        assert isinstance(expr, BinaryOp)
        assert isinstance(expr.left, IndexExpr)
        assert isinstance(expr.right, IndexExpr)

    # --- IndexAssignStmt ---

    def test_index_assign_produces_correct_node(self):
        node = first_stmt("arr[0] = 5;")
        assert isinstance(node, IndexAssignStmt)

    def test_index_assign_name(self):
        node = first_stmt("arr[0] = 5;")
        assert node.name == "arr"

    def test_index_assign_literal_index(self):
        node = first_stmt("arr[0] = 5;")
        assert isinstance(node.index, Literal)
        assert node.index.value == 0

    def test_index_assign_identifier_index(self):
        node = first_stmt("arr[i] = 10;")
        assert isinstance(node.index, Identifier)
        assert node.index.name == "i"

    def test_index_assign_expression_index(self):
        node = first_stmt("arr[i + 1] = 42;")
        assert isinstance(node.index, BinaryOp)
        assert node.index.op == "+"

    def test_index_assign_literal_value(self):
        node = first_stmt("arr[0] = 5;")
        assert isinstance(node.value, Literal)
        assert node.value.value == 5

    def test_index_assign_expression_value(self):
        node = first_stmt("arr[0] = x + y;")
        assert isinstance(node.value, BinaryOp)
        assert node.value.op == "+"

    def test_index_assign_position(self):
        node = first_stmt("arr[0] = 5;")
        assert node.line == 1
        assert node.col == 1

    # --- Combined sequence ---

    def test_full_array_sequence_inside_function(self):
        src = """
        fn fill() {
            [int; 3] nums;
            nums[0] = 10;
            nums[1] = 20;
            nums[2] = 30;
        }
        """
        stmts = first_stmt(src).body.statements
        assert isinstance(stmts[0], VarDecl)
        assert isinstance(stmts[0].var_type, ArrayType)
        assert stmts[0].var_type.size == 3
        assert all(isinstance(s, IndexAssignStmt) for s in stmts[1:])
        assert stmts[1].index.value == 0
        assert stmts[3].index.value == 2

    def test_array_read_and_write_in_function(self):
        src = """
        fn copy_first(src: int) -> int {
            [int; 2] buf;
            buf[0] = src;
            return buf[0];
        }
        """
        stmts = first_stmt(src).body.statements
        assert isinstance(stmts[0], VarDecl)         # [int; 2] buf;
        assert isinstance(stmts[1], IndexAssignStmt)  # buf[0] = src;
        assert isinstance(stmts[2], ReturnStmt)
        assert isinstance(stmts[2].value, IndexExpr)  # return buf[0];

    # --- Error cases ---

    def test_error_non_integer_size(self):
        with pytest.raises(CompilerError):
            parse("[int; x] arr;")          # identifier is not an integer literal

    def test_error_float_size(self):
        with pytest.raises(CompilerError):
            parse("[int; 3.14] arr;")       # float literal is not allowed as size

    def test_error_missing_semicolon_in_type(self):
        with pytest.raises(CompilerError):
            parse("[int 10] arr;")          # missing ';' between element type and size

    def test_error_missing_closing_bracket_in_type(self):
        with pytest.raises(CompilerError):
            parse("[int; 5 arr;")           # missing ']' after size

    def test_error_missing_assign_in_index_stmt(self):
        with pytest.raises(CompilerError):
            parse("arr[0];")               # bare index expression requires '='
