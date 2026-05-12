from __future__ import annotations

from typing import Any

from ir_gen import TACInstruction


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

_FOLDABLE_OPS: frozenset[str] = frozenset({"+", "-", "*", "/", "%"})

# Ops that unconditionally end the current execution path.
_TERMINAL_OPS: frozenset[str] = frozenset({"JMP", "RETURN"})

# Ops whose result field names a jump target label.
_JUMP_OPS: frozenset[str] = frozenset({"JMP", "IF_FALSE", "IF_TRUE"})


def _parse_number(val: Any) -> int | float | None:
    """Return val parsed as int or float, or None if it is not a numeric literal."""
    if not isinstance(val, str):
        return None
    try:
        return int(val)
    except ValueError:
        pass
    try:
        return float(val)
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Optimizer
# ---------------------------------------------------------------------------

class Optimizer:
    """
    Applies a sequence of optimisation passes to a flat TAC instruction list.
    Each pass receives a list and returns a *new* list — no in-place mutation.
    """

    def optimize(self, instructions: list[TACInstruction]) -> list[TACInstruction]:
        # Fold first so propagation sees COPY-of-literal, not raw arithmetic ops.
        instructions = self._constant_folding(instructions)
        # Propagation may expose new foldable pairs, so fold a second time.
        instructions = self._constant_propagation(instructions)
        instructions = self._constant_folding(instructions)
        instructions = self._dead_code_elimination(instructions)
        return instructions

    # ------------------------------------------------------------------
    # Pass 1 — Constant Folding
    # ------------------------------------------------------------------

    def _constant_folding(self, instructions: list[TACInstruction]) -> list[TACInstruction]:
        result: list[TACInstruction] = []
        for instr in instructions:
            folded = self._try_fold(instr)
            result.append(folded if folded is not None else instr)
        return result

    def _try_fold(self, instr: TACInstruction) -> TACInstruction | None:
        """
        If instr is a foldable binary op with two numeric literals, return a
        COPY instruction carrying the pre-computed result.  Otherwise None.
        """
        if instr.op not in _FOLDABLE_OPS:
            return None
        left = _parse_number(instr.arg1)
        right = _parse_number(instr.arg2)
        if left is None or right is None:
            return None
        val = self._apply_op(instr.op, left, right)
        if val is None:
            return None
        return TACInstruction(op="COPY", arg1=str(val), result=instr.result)

    def _apply_op(self, op: str, left: int | float, right: int | float) -> int | float | None:
        """Evaluate op(left, right) at compile time; return None when undefined."""
        try:
            if op == "+":
                return left + right
            if op == "-":
                return left - right
            if op == "*":
                return left * right
            if op == "/":
                if right == 0:
                    return None
                # Preserve integer semantics when both operands are int.
                if isinstance(left, int) and isinstance(right, int):
                    return left // right
                return left / right
            if op == "%":
                if right == 0:
                    return None
                return left % right
        except OverflowError:
            return None
        return None

    # ------------------------------------------------------------------
    # Pass 2 — Constant Propagation
    # ------------------------------------------------------------------

    def _constant_propagation(self, instructions: list[TACInstruction]) -> list[TACInstruction]:
        """Replace uses of temps that hold a known constant with that constant."""
        known: dict[str, str] = {}   # temp/var name → constant value string
        result: list[TACInstruction] = []
        for instr in instructions:
            # At a label or function entry we may be arriving from multiple paths;
            # facts proved before the jump are no longer reliable.
            if instr.op in ("LABEL", "FUNC"):
                known.clear()
            a1 = known.get(str(instr.arg1), instr.arg1) if instr.arg1 is not None else None
            a2 = known.get(str(instr.arg2), instr.arg2) if instr.arg2 is not None else None
            new_instr = TACInstruction(op=instr.op, arg1=a1, arg2=a2, result=instr.result)
            result.append(new_instr)
            # Record that result now holds a constant (or forget a stale binding).
            if new_instr.op == "COPY" and new_instr.result is not None:
                if _parse_number(str(new_instr.arg1)) is not None:
                    known[str(new_instr.result)] = str(new_instr.arg1)
                else:
                    known.pop(str(new_instr.result), None)
        return result

    # ------------------------------------------------------------------
    # Pass 3 — Dead Code Elimination
    # ------------------------------------------------------------------

    def _dead_code_elimination(self, instructions: list[TACInstruction]) -> list[TACInstruction]:
        after_unreachable = self._remove_unreachable(instructions)
        return self._remove_unused_labels(after_unreachable)

    def _remove_unreachable(self, instructions: list[TACInstruction]) -> list[TACInstruction]:
        """Drop instructions between a terminal (JMP/RETURN) and the next label."""
        result: list[TACInstruction] = []
        skip = False
        for instr in instructions:
            # A label or function entry ends any unreachable region.
            if instr.op in ("LABEL", "FUNC"):
                skip = False
            if not skip:
                result.append(instr)
            if instr.op in _TERMINAL_OPS:
                skip = True
        return result

    def _remove_unused_labels(self, instructions: list[TACInstruction]) -> list[TACInstruction]:
        """Drop LABEL instructions that no jump instruction references."""
        referenced: set[str] = set()
        for instr in instructions:
            if instr.op in _JUMP_OPS and instr.result is not None:
                referenced.add(str(instr.result))
        result: list[TACInstruction] = []
        for instr in instructions:
            if instr.op == "LABEL" and instr.result not in referenced:
                continue
            result.append(instr)
        return result
