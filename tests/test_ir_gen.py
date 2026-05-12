import pytest

from ir_gen import IRGenerator, TACInstruction
from lexer import Lexer, TokenType
from parser import Parser
from semantic import SemanticAnalyzer


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def ir(src: str) -> list[TACInstruction]:
    """Full pipeline: lex → parse → semantic check → IR generation."""
    lexer = Lexer(src)
    tokens = []
    while True:
        t = lexer.get_next_token()
        tokens.append(t)
        if t.type == TokenType.EOF:
            break
    program = Parser(tokens).parse()
    SemanticAnalyzer().analyze(program)
    return IRGenerator().generate(program)


def select(op: str, instrs: list[TACInstruction]) -> list[TACInstruction]:
    return [i for i in instrs if i.op == op]


def index_of(op: str, instrs: list[TACInstruction]) -> int:
    return next(i for i, instr in enumerate(instrs) if instr.op == op)


# ---------------------------------------------------------------------------
# Arithmetic expressions and operator precedence
# ---------------------------------------------------------------------------

class TestArithmeticExpressions:
    def test_mul_before_add(self):
        # 1 + 2 * 3: the * instruction must appear before the + instruction
        instrs = ir("int a = 1 + 2 * 3;")
        arith = [i for i in instrs if i.op in ("+", "*")]
        assert arith[0].op == "*"
        assert arith[1].op == "+"

    def test_mul_result_feeds_into_add(self):
        # The temp produced by * becomes the right operand of +
        instrs = ir("int a = 1 + 2 * 3;")
        mul = next(i for i in instrs if i.op == "*")
        add = next(i for i in instrs if i.op == "+")
        assert mul.result == add.arg2

    def test_parens_override_precedence(self):
        # (1 + 2) * 3: the + must appear before the *
        instrs = ir("int a = (1 + 2) * 3;")
        arith = [i for i in instrs if i.op in ("+", "*")]
        assert arith[0].op == "+"
        assert arith[1].op == "*"

    def test_parens_result_feeds_into_mul(self):
        instrs = ir("int a = (1 + 2) * 3;")
        add = next(i for i in instrs if i.op == "+")
        mul = next(i for i in instrs if i.op == "*")
        assert add.result == mul.arg1

    def test_left_associativity(self):
        # 10 - 3 - 2 = (10 - 3) - 2: second sub consumes temp from first
        instrs = ir("int a = 10 - 3 - 2;")
        subs = [i for i in instrs if i.op == "-"]
        assert len(subs) == 2
        assert subs[1].arg1 == subs[0].result

    def test_literal_init_emits_one_copy(self):
        instrs = ir("int x = 42;")
        assert len(instrs) == 1
        assert instrs[0].op == "COPY"
        assert instrs[0].arg1 == "42"
        assert instrs[0].result == "x"

    def test_uninitialised_var_emits_nothing(self):
        assert ir("int x;") == []

    def test_assign_stmt_emits_copy(self):
        instrs = ir("int x = 0; x = 99;")
        copies = select("COPY", instrs)
        assert copies[-1].result == "x"
        assert copies[-1].arg1 == "99"

    def test_unary_minus_no_arg2(self):
        instrs = ir("fn f(n: int) -> int { return -n; }")
        unary = next(i for i in instrs if i.op == "-" and i.arg2 is None)
        assert unary.arg1 == "n"
        assert unary.result is not None

    def test_unary_bang(self):
        instrs = ir("fn f(b: bool) -> bool { return !b; }")
        bang = next(i for i in instrs if i.op == "!")
        assert bang.arg1 == "b"
        assert bang.result is not None

    def test_bool_literal_lowercase(self):
        instrs = ir("bool f = true;")
        assert instrs[0].arg1 == "true"

    def test_chained_addition(self):
        # a + b + c produces two + instructions; temps chain correctly
        instrs = ir("fn f(a: int, b: int, c: int) -> int { return a + b + c; }")
        adds = [i for i in instrs if i.op == "+"]
        assert len(adds) == 2
        assert adds[1].arg1 == adds[0].result


# ---------------------------------------------------------------------------
# If statement — label and jump structure
# ---------------------------------------------------------------------------

class TestIfStatement:
    def test_simple_if_has_if_false(self):
        instrs = ir("fn f(b: bool) { int x = 0; if b { x = 1; } }")
        assert any(i.op == "IF_FALSE" for i in instrs)

    def test_if_false_condition_is_the_bool_var(self):
        instrs = ir("fn f(b: bool) { int x = 0; if b { x = 1; } }")
        if_false = next(i for i in instrs if i.op == "IF_FALSE")
        assert if_false.arg1 == "b"

    def test_if_false_target_is_a_known_label(self):
        instrs = ir("fn f(b: bool) { int x = 0; if b { x = 1; } }")
        if_false = next(i for i in instrs if i.op == "IF_FALSE")
        label_names = {i.result for i in instrs if i.op == "LABEL"}
        assert if_false.result in label_names

    def test_if_else_has_jmp_to_skip_else(self):
        instrs = ir("fn f(b: bool) { int x = 0; if b { x = 1; } else { x = 2; } }")
        assert any(i.op == "JMP" for i in instrs)

    def test_if_false_and_jmp_point_to_different_labels(self):
        # IF_FALSE → else_label, JMP → end_label; they must differ
        instrs = ir("fn f(b: bool) { int x = 0; if b { x = 1; } else { x = 2; } }")
        if_false = next(i for i in instrs if i.op == "IF_FALSE")
        jmp = next(i for i in instrs if i.op == "JMP")
        assert if_false.result != jmp.result

    def test_if_false_points_to_first_label(self):
        # First LABEL emitted is the else-label; IF_FALSE must jump there
        instrs = ir("fn f(b: bool) { int x = 0; if b { x = 1; } else { x = 2; } }")
        if_false = next(i for i in instrs if i.op == "IF_FALSE")
        labels = select("LABEL", instrs)
        assert if_false.result == labels[0].result

    def test_jmp_points_to_second_label(self):
        # Second LABEL emitted is the end-label; JMP must jump there
        instrs = ir("fn f(b: bool) { int x = 0; if b { x = 1; } else { x = 2; } }")
        jmp = next(i for i in instrs if i.op == "JMP")
        labels = select("LABEL", instrs)
        assert jmp.result == labels[1].result

    def test_then_body_between_if_false_and_jmp(self):
        # Structure: IF_FALSE … COPY(x=1) … JMP
        instrs = ir("fn f(b: bool) { int x = 0; if b { x = 1; } else { x = 2; } }")
        ops = [i.op for i in instrs]
        if_false_idx = ops.index("IF_FALSE")
        jmp_idx = ops.index("JMP")
        assert "COPY" in ops[if_false_idx:jmp_idx]

    def test_else_body_between_labels(self):
        # Structure: LABEL(else) … COPY(x=2) … LABEL(end)
        instrs = ir("fn f(b: bool) { int x = 0; if b { x = 1; } else { x = 2; } }")
        label_indices = [i for i, instr in enumerate(instrs) if instr.op == "LABEL"]
        between = instrs[label_indices[0] + 1 : label_indices[1]]
        assert any(i.op == "COPY" and i.arg1 == "2" for i in between)

    def test_if_false_precedes_jmp(self):
        instrs = ir("fn f(b: bool) { int x = 0; if b { x = 1; } else { x = 2; } }")
        ops = [i.op for i in instrs]
        assert ops.index("IF_FALSE") < ops.index("JMP")


# ---------------------------------------------------------------------------
# While loop — label and jump structure
# ---------------------------------------------------------------------------

class TestWhileStatement:
    def test_first_instr_after_func_is_start_label(self):
        instrs = ir("fn f(b: bool) { while b { int x = 1; } }")
        func_idx = index_of("FUNC", instrs)
        assert instrs[func_idx + 1].op == "LABEL"

    def test_start_label_before_if_false(self):
        instrs = ir("fn f(b: bool) { while b { int x = 1; } }")
        label_idx = index_of("LABEL", instrs)
        if_false_idx = index_of("IF_FALSE", instrs)
        assert label_idx < if_false_idx

    def test_jmp_goes_back_to_start_label(self):
        instrs = ir("fn f(b: bool) { while b { int x = 1; } }")
        start_label = instrs[index_of("LABEL", instrs)].result
        jmps = select("JMP", instrs)
        assert any(j.result == start_label for j in jmps)

    def test_if_false_exits_to_end_label(self):
        instrs = ir("fn f(b: bool) { while b { int x = 1; } }")
        if_false = next(i for i in instrs if i.op == "IF_FALSE")
        labels = select("LABEL", instrs)
        # End label is the last LABEL emitted
        assert if_false.result == labels[-1].result

    def test_loop_structure_order(self):
        # LABEL(start) must come before IF_FALSE, JMP, LABEL(end)
        instrs = ir("fn f(b: bool) { while b { int x = 1; } }")
        ops = [i.op for i in instrs]
        first_label_idx = ops.index("LABEL")
        if_false_idx = ops.index("IF_FALSE")
        jmp_idx = ops.index("JMP")
        last_label_idx = max(i for i, op in enumerate(ops) if op == "LABEL")
        assert first_label_idx < if_false_idx < jmp_idx < last_label_idx

    def test_exactly_two_labels(self):
        instrs = ir("fn f(b: bool) { while b { int x = 1; } }")
        assert len(select("LABEL", instrs)) == 2


# ---------------------------------------------------------------------------
# For loop — init, condition, step, jump structure
# ---------------------------------------------------------------------------

class TestForStatement:
    def test_loop_var_initialised_before_start_label(self):
        instrs = ir("fn f() { for i = 0; i < 3; i = i + 1 { } }")
        func_idx = index_of("FUNC", instrs)
        # Init COPY comes right after FUNC
        assert instrs[func_idx + 1].op == "COPY"
        assert instrs[func_idx + 1].result == "i"
        assert instrs[func_idx + 1].arg1 == "0"

    def test_start_label_follows_init(self):
        instrs = ir("fn f() { for i = 0; i < 3; i = i + 1 { } }")
        func_idx = index_of("FUNC", instrs)
        assert instrs[func_idx + 2].op == "LABEL"

    def test_jmp_back_to_start_label(self):
        instrs = ir("fn f() { for i = 0; i < 3; i = i + 1 { } }")
        labels = select("LABEL", instrs)
        start_label = labels[0].result
        jmps = select("JMP", instrs)
        assert any(j.result == start_label for j in jmps)

    def test_if_false_exits_to_end_label(self):
        instrs = ir("fn f() { for i = 0; i < 3; i = i + 1 { } }")
        if_false = next(i for i in instrs if i.op == "IF_FALSE")
        end_label = select("LABEL", instrs)[-1].result
        assert if_false.result == end_label

    def test_step_update_before_jmp_back(self):
        instrs = ir("fn f() { for i = 0; i < 3; i = i + 1 { } }")
        ops = [i.op for i in instrs]
        # The last COPY (step assignment to i) must appear before the last JMP
        copy_indices = [idx for idx, op in enumerate(ops) if op == "COPY"]
        jmp_idx = ops.index("JMP")
        assert copy_indices[-1] < jmp_idx


# ---------------------------------------------------------------------------
# Function declarations and calls
# ---------------------------------------------------------------------------

class TestFunctionDecl:
    def test_func_entry_op_and_name(self):
        instrs = ir("fn greet() { return; }")
        assert instrs[0].op == "FUNC"
        assert instrs[0].result == "greet"

    def test_param_used_by_name_no_copy(self):
        # Parameters flow directly as named values; no COPY instruction for them
        instrs = ir("fn double(n: int) -> int { return n; }")
        ret = next(i for i in instrs if i.op == "RETURN")
        assert ret.arg1 == "n"

    def test_multiple_functions_each_have_func_label(self):
        src = "fn a() { return; } fn b() { return; }"
        instrs = ir(src)
        funcs = select("FUNC", instrs)
        assert len(funcs) == 2
        assert funcs[0].result == "a"
        assert funcs[1].result == "b"

    def test_fibonacci_has_two_recursive_calls(self):
        src = """
        fn fib(n: int) -> int {
            if n <= 1 { return n; }
            return fib(n - 1) + fib(n - 2);
        }
        """
        instrs = ir(src)
        calls = [i for i in instrs if i.op == "CALL" and i.arg1 == "fib"]
        assert len(calls) == 2
        assert all(c.arg2 == 1 for c in calls)

    def test_fibonacci_each_call_preceded_by_param(self):
        src = """
        fn fib(n: int) -> int {
            if n <= 1 { return n; }
            return fib(n - 1) + fib(n - 2);
        }
        """
        instrs = ir(src)
        for idx, instr in enumerate(instrs):
            if instr.op == "CALL" and instr.arg1 == "fib":
                assert instrs[idx - 1].op == "PARAM"

    def test_fibonacci_results_added_together(self):
        src = """
        fn fib(n: int) -> int {
            if n <= 1 { return n; }
            return fib(n - 1) + fib(n - 2);
        }
        """
        instrs = ir(src)
        calls = [i for i in instrs if i.op == "CALL"]
        add = next(i for i in instrs if i.op == "+")
        # The add's operands are the two call results
        call_results = {c.result for c in calls}
        assert add.arg1 in call_results
        assert add.arg2 in call_results


class TestFuncCall:
    def test_param_immediately_before_call(self):
        src = "fn id(n: int) -> int { return n; } fn f() -> int { return id(5); }"
        instrs = ir(src)
        call_idx = next(i for i, instr in enumerate(instrs)
                        if instr.op == "CALL" and instr.arg1 == "id")
        assert instrs[call_idx - 1].op == "PARAM"

    def test_call_arg_count(self):
        src = ("fn add(a: int, b: int) -> int { return a; }"
               " fn f() -> int { return add(1, 2); }")
        instrs = ir(src)
        call = next(i for i in instrs if i.op == "CALL" and i.arg1 == "add")
        assert call.arg2 == 2

    def test_two_params_before_two_arg_call(self):
        src = ("fn add(a: int, b: int) -> int { return a; }"
               " fn f() -> int { return add(1, 2); }")
        instrs = ir(src)
        call_idx = next(i for i, instr in enumerate(instrs)
                        if instr.op == "CALL" and instr.arg1 == "add")
        assert instrs[call_idx - 1].op == "PARAM"
        assert instrs[call_idx - 2].op == "PARAM"

    def test_call_result_is_temp(self):
        src = "fn id(n: int) -> int { return n; } fn f() -> int { return id(5); }"
        instrs = ir(src)
        call = next(i for i in instrs if i.op == "CALL" and i.arg1 == "id")
        assert call.result is not None
        assert str(call.result).startswith("t")


# ---------------------------------------------------------------------------
# __str__ rendering of key instruction forms
# ---------------------------------------------------------------------------

class TestInstructionStr:
    def test_func_label(self):
        from ir_gen import TACInstruction
        assert str(TACInstruction(op="FUNC", result="fib")) == "FUNC fib:"

    def test_binary(self):
        from ir_gen import TACInstruction
        assert str(TACInstruction(op="+", arg1="t0", arg2="t1", result="t2")) == "    t2 = t0 + t1"

    def test_unary(self):
        from ir_gen import TACInstruction
        assert str(TACInstruction(op="-", arg1="x", result="t0")) == "    t0 = -x"

    def test_if_false(self):
        from ir_gen import TACInstruction
        assert str(TACInstruction(op="IF_FALSE", arg1="t0", result="L1")) == "    IF_FALSE t0 JMP L1"

    def test_jmp(self):
        from ir_gen import TACInstruction
        assert str(TACInstruction(op="JMP", result="L0")) == "    JMP L0"

    def test_return_with_value(self):
        from ir_gen import TACInstruction
        assert str(TACInstruction(op="RETURN", arg1="t0")) == "    RETURN t0"

    def test_return_void(self):
        from ir_gen import TACInstruction
        assert str(TACInstruction(op="RETURN")) == "    RETURN"

    def test_call_with_result(self):
        from ir_gen import TACInstruction
        assert str(TACInstruction(op="CALL", arg1="fib", arg2=1, result="t2")) == "    t2 = CALL fib 1"
