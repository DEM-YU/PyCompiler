import pytest

from ir_gen import TACInstruction
from optimizer import Optimizer


# ---------------------------------------------------------------------------
# Shorthand constructor — positional: op, arg1, arg2, result
# All keyword so the intent is unambiguous in each test.
# ---------------------------------------------------------------------------

def instr(op: str, arg1=None, arg2=None, result=None) -> TACInstruction:
    return TACInstruction(op=op, arg1=arg1, arg2=arg2, result=result)


# ---------------------------------------------------------------------------
# Constant Folding  (Pass 1 + re-run after propagation)
# ---------------------------------------------------------------------------

class TestConstantFolding:
    def test_multiply_two_literals(self):
        result = Optimizer().optimize([instr("*", "10", "2", "t0")])
        assert result[0].op == "COPY"
        assert result[0].arg1 == "20"
        assert result[0].result == "t0"

    def test_addition(self):
        result = Optimizer().optimize([instr("+", "3", "4", "t0")])
        assert result[0].arg1 == "7"

    def test_subtraction(self):
        result = Optimizer().optimize([instr("-", "10", "3", "t0")])
        assert result[0].arg1 == "7"

    def test_integer_division_floor(self):
        # 10 / 3 in Nova (int / int) = 3, not 3.333
        result = Optimizer().optimize([instr("/", "10", "3", "t0")])
        assert result[0].arg1 == "3"

    def test_modulo(self):
        result = Optimizer().optimize([instr("%", "10", "3", "t0")])
        assert result[0].arg1 == "1"

    def test_float_addition(self):
        result = Optimizer().optimize([instr("+", "1.5", "2.5", "t0")])
        assert result[0].arg1 == "4.0"

    def test_result_temp_preserved(self):
        result = Optimizer().optimize([instr("*", "6", "7", "t99")])
        assert result[0].result == "t99"

    def test_variable_operand_not_folded(self):
        instrs = [instr("+", "x", "5", "t0")]
        result = Optimizer().optimize(instrs)
        assert result[0].op == "+"

    def test_division_by_zero_not_folded(self):
        instrs = [instr("/", "10", "0", "t0")]
        result = Optimizer().optimize(instrs)
        assert result[0].op == "/"

    def test_non_arithmetic_op_unchanged(self):
        instrs = [instr("COPY", "5", result="t0")]
        result = Optimizer().optimize(instrs)
        assert result[0].op == "COPY"
        assert result[0].arg1 == "5"

    def test_comparison_not_folded(self):
        # Comparisons (<, >, ==, !=) are not in _FOLDABLE_OPS
        instrs = [instr("<", "3", "5", "t0")]
        result = Optimizer().optimize(instrs)
        assert result[0].op == "<"

    def test_unrelated_instructions_unchanged(self):
        instrs = [
            instr("FUNC", result="f"),
            instr("RETURN", arg1="t0"),
        ]
        result = Optimizer().optimize(instrs)
        assert result[0].op == "FUNC"
        assert result[1].op == "RETURN"


# ---------------------------------------------------------------------------
# Constant Propagation  (Pass 2)
# ---------------------------------------------------------------------------

class TestConstantPropagation:
    def test_propagate_literal_into_binary_op(self):
        # t0 = 20 (COPY); t1 = t0 + 5  →  t1 = 25
        instrs = [
            instr("COPY", "20", result="t0"),
            instr("+", "t0", "5", "t1"),
        ]
        result = Optimizer().optimize(instrs)
        t1 = next(i for i in result if i.result == "t1")
        assert t1.op == "COPY"
        assert t1.arg1 == "25"

    def test_fold_then_propagate_chain(self):
        # t0 = 10 * 2  →  (fold)  t0 = 20
        # t1 = t0 + 5  →  (prop + fold)  t1 = 25
        instrs = [
            instr("*", "10", "2", "t0"),
            instr("+", "t0", "5", "t1"),
        ]
        result = Optimizer().optimize(instrs)
        t0 = next(i for i in result if i.result == "t0")
        t1 = next(i for i in result if i.result == "t1")
        assert t0.op == "COPY" and t0.arg1 == "20"
        assert t1.op == "COPY" and t1.arg1 == "25"

    def test_chained_copy_propagation(self):
        # t0 = 5; t1 = t0; t2 = t1 + 1  →  t2 = 6
        instrs = [
            instr("COPY", "5", result="t0"),
            instr("COPY", "t0", result="t1"),
            instr("+", "t1", "1", "t2"),
        ]
        result = Optimizer().optimize(instrs)
        t2 = next(i for i in result if i.result == "t2")
        assert t2.op == "COPY"
        assert t2.arg1 == "6"

    def test_propagation_stops_at_label(self):
        # After a LABEL the temp's value is unknown (it's a join point).
        # The + must remain a runtime operation.
        instrs = [
            instr("COPY", "20", result="t0"),
            instr("LABEL", result="L0"),           # clears known constants
            instr("+", "t0", "5", "t1"),
        ]
        result = Optimizer().optimize(instrs)
        t1 = next(i for i in result if i.result == "t1")
        assert t1.op == "+"

    def test_non_constant_copy_not_propagated(self):
        # COPY x → t0: x is a named variable, not a literal
        instrs = [
            instr("COPY", "x", result="t0"),
            instr("+", "t0", "5", "t1"),
        ]
        result = Optimizer().optimize(instrs)
        t1 = next(i for i in result if i.result == "t1")
        assert t1.op == "+"

    def test_propagation_stops_at_func_entry(self):
        instrs = [
            instr("COPY", "20", result="t0"),
            instr("FUNC", result="g"),             # clears known constants
            instr("+", "t0", "3", "t1"),
        ]
        result = Optimizer().optimize(instrs)
        t1 = next(i for i in result if i.result == "t1")
        assert t1.op == "+"

    def test_original_instructions_not_mutated(self):
        original = [
            instr("*", "6", "7", "t0"),
            instr("+", "t0", "1", "t1"),
        ]
        before_ops = [i.op for i in original]
        Optimizer().optimize(original)
        assert [i.op for i in original] == before_ops


# ---------------------------------------------------------------------------
# Dead Code Elimination  (Pass 3)
# ---------------------------------------------------------------------------

class TestDeadCodeElimination:
    def test_instructions_after_return_removed(self):
        instrs = [
            instr("RETURN", arg1="t0"),
            instr("COPY", "5", result="x"),        # unreachable
            instr("LABEL", result="L0"),
        ]
        result = Optimizer().optimize(instrs)
        ops = [i.op for i in result]
        assert "RETURN" in ops
        assert "COPY" not in ops                   # removed: unreachable
        assert "LABEL" not in ops                  # removed: orphan (no JMP to L0)

    def test_instructions_after_jmp_removed(self):
        instrs = [
            instr("JMP", result="L0"),
            instr("COPY", "1", result="x"),        # unreachable
            instr("LABEL", result="L0"),
            instr("RETURN"),
        ]
        result = Optimizer().optimize(instrs)
        assert [i.op for i in result] == ["JMP", "LABEL", "RETURN"]

    def test_orphan_label_removed(self):
        # No jump references L99 — it should vanish entirely.
        result = Optimizer().optimize([instr("LABEL", result="L99")])
        assert result == []

    def test_referenced_label_kept(self):
        instrs = [
            instr("JMP", result="L0"),
            instr("LABEL", result="L0"),
        ]
        result = Optimizer().optimize(instrs)
        assert any(i.op == "LABEL" for i in result)

    def test_if_false_referenced_label_kept(self):
        instrs = [
            instr("IF_FALSE", arg1="t0", result="L1"),
            instr("RETURN", arg1="1"),
            instr("LABEL", result="L1"),
            instr("RETURN", arg1="2"),
        ]
        result = Optimizer().optimize(instrs)
        label_names = [i.result for i in result if i.op == "LABEL"]
        assert "L1" in label_names

    def test_func_resets_unreachable_region(self):
        # The second FUNC and its RETURN must survive even though the
        # first function ends with RETURN (setting skip=True).
        instrs = [
            instr("FUNC", result="a"),
            instr("RETURN", arg1="1"),
            instr("FUNC", result="b"),             # must NOT be treated as dead
            instr("RETURN", arg1="2"),
        ]
        result = Optimizer().optimize(instrs)
        funcs = [i for i in result if i.op == "FUNC"]
        assert len(funcs) == 2
        assert funcs[0].result == "a"
        assert funcs[1].result == "b"

    def test_while_loop_both_labels_kept(self):
        # Both the start label (back-edge target) and exit label must survive.
        instrs = [
            instr("LABEL", result="L0"),
            instr("IF_FALSE", arg1="b", result="L1"),
            instr("COPY", "1", result="x"),
            instr("JMP", result="L0"),
            instr("LABEL", result="L1"),
        ]
        result = Optimizer().optimize(instrs)
        label_names = [i.result for i in result if i.op == "LABEL"]
        assert "L0" in label_names
        assert "L1" in label_names

    def test_fibonacci_dead_jmp_and_orphan_label_removed(self):
        # After RETURN n, the JMP L0 is dead; L0 then has no referencing
        # jump and is also removed.
        instrs = [
            instr("IF_FALSE", arg1="t0", result="L1"),
            instr("RETURN", arg1="n"),
            instr("JMP", result="L0"),             # dead
            instr("LABEL", result="L1"),           # referenced by IF_FALSE → kept
            instr("LABEL", result="L0"),           # orphan after JMP removed → removed
            instr("RETURN", arg1="t5"),
        ]
        result = Optimizer().optimize(instrs)
        ops = [i.op for i in result]
        assert "JMP" not in ops
        label_names = [i.result for i in result if i.op == "LABEL"]
        assert "L1" in label_names
        assert "L0" not in label_names


# ---------------------------------------------------------------------------
# Combined / integration
# ---------------------------------------------------------------------------

class TestCombinedOptimization:
    def test_empty_list(self):
        assert Optimizer().optimize([]) == []

    def test_idempotent_on_already_optimised(self):
        # Running optimize twice on an already-optimised list returns the same ops.
        instrs = [
            instr("COPY", "20", result="t0"),
            instr("COPY", "25", result="t1"),
        ]
        first = Optimizer().optimize(instrs)
        second = Optimizer().optimize(first)
        assert [i.op for i in first] == [i.op for i in second]

    def test_fold_and_dce_interact(self):
        # Constant folding turns the arithmetic into COPY; if the folded
        # constant is used only in unreachable code, DCE can eliminate both.
        instrs = [
            instr("RETURN"),
            instr("*", "3", "4", "t0"),            # dead after RETURN
        ]
        result = Optimizer().optimize(instrs)
        assert len(result) == 1
        assert result[0].op == "RETURN"

    def test_original_list_not_mutated(self):
        original = [instr("*", "5", "6", "t0")]
        Optimizer().optimize(original)
        assert original[0].op == "*"               # must not be overwritten

    def test_control_flow_only_unchanged(self):
        # An if/else with no constants: nothing to fold, no dead code.
        instrs = [
            instr("FUNC", result="f"),
            instr("IF_FALSE", arg1="b", result="L1"),
            instr("JMP", result="L0"),
            instr("LABEL", result="L1"),
            instr("LABEL", result="L0"),
            instr("RETURN"),
        ]
        result = Optimizer().optimize(instrs)
        label_names = {i.result for i in result if i.op == "LABEL"}
        assert "L0" in label_names
        assert "L1" in label_names


# ---------------------------------------------------------------------------
# Side-effect ops — MALLOC_STR and ALLOC_ARR survive DCE
# ---------------------------------------------------------------------------

class TestSideEffectOps:
    def test_malloc_str_survives_after_return(self):
        instrs = [
            instr("FUNC", result="f"),
            instr("RETURN"),
            instr("MALLOC_STR", arg1="x", arg2="hello"),
        ]
        result = Optimizer().optimize(instrs)
        assert any(i.op == "MALLOC_STR" for i in result)

    def test_alloc_arr_survives_after_return(self):
        instrs = [
            instr("FUNC", result="f"),
            instr("RETURN"),
            instr("ALLOC_ARR", arg1="arr", arg2=5),
        ]
        result = Optimizer().optimize(instrs)
        assert any(i.op == "ALLOC_ARR" for i in result)

    def test_plain_copy_still_removed_after_return(self):
        instrs = [
            instr("FUNC", result="f"),
            instr("RETURN"),
            instr("COPY", "5", result="x"),
        ]
        result = Optimizer().optimize(instrs)
        copies = [i for i in result if i.op == "COPY"]
        assert not copies

    def test_result_type_preserved_through_folding(self):
        from ir_gen import TACInstruction as TAC
        original = TAC(op="+", arg1="3", arg2="4", result="t0", result_type="float")
        result = Optimizer().optimize([original])
        assert result[0].result_type == "float"

    def test_result_type_preserved_through_propagation(self):
        from ir_gen import TACInstruction as TAC
        instrs = [
            TAC(op="MALLOC_STR", arg1="s", arg2="hi", result_type="string"),
            TAC(op="COPY", arg1="s", result="s2", result_type="string"),
        ]
        result = Optimizer().optimize(instrs)
        copy = next(i for i in result if i.op == "COPY")
        assert copy.result_type == "string"
