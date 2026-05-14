from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ast_nodes import (
    ASTVisitor, ArrayType, Block, BinaryOp, FuncCall, FunctionDecl,
    Identifier, IfStmt, IndexAssignStmt, IndexExpr, Literal, Program,
    ReturnStmt, StringLiteral, UnaryOp, VarDecl, AssignStmt, WhileStmt, ForStmt,
)


# ---------------------------------------------------------------------------
# Three-Address Code instruction
#
# One flat dataclass covers every instruction form.  Which fields are
# meaningful depends on op:
#
#   Binary op   op='+' arg1='a'  arg2='b'   result='t1'   →  t1 = a + b
#   Unary op    op='-' arg1='a'  arg2=None  result='t1'   →  t1 = -a
#   Copy        op='COPY'        arg1='a'   result='t1'   →  t1 = a
#   Func entry  op='FUNC'                   result='fib'  →  FUNC fib:
#   Label       op='LABEL'                  result='L1'   →  L1:
#   Jump        op='JMP'                    result='L1'   →  JMP L1
#   Cond jump   op='IF_FALSE'    arg1='t1'  result='L2'   →  IF_FALSE t1 JMP L2
#   Param       op='PARAM'       arg1='t1'                →  PARAM t1
#   Call        op='CALL'        arg1='f'   arg2=N  result='t1' or None
#   Return      op='RETURN'      arg1='t1' (or None)      →  RETURN t1
# ---------------------------------------------------------------------------

@dataclass
class TACInstruction:
    op: str
    arg1: Any = None        # first operand or function name for CALL
    arg2: Any = None        # second operand or argument count for CALL; None for unary
    result: Any = None      # destination temp or label name; None for PARAM/void RETURN
    result_type: str = "int"  # scalar type of the result value

    def __str__(self) -> str:
        if self.op == "FUNC":
            return f"FUNC {self.result}:"
        if self.op == "LABEL":
            return f"{self.result}:"
        if self.op == "JMP":
            return f"    JMP {self.result}"
        if self.op in ("IF_FALSE", "IF_TRUE"):
            return f"    {self.op} {self.arg1} JMP {self.result}"
        if self.op == "PARAM":
            return f"    PARAM {self.arg1}"
        if self.op == "RETURN":
            return f"    RETURN {self.arg1}" if self.arg1 is not None else "    RETURN"
        if self.op == "CALL":
            call = f"CALL {self.arg1} {self.arg2}"
            return f"    {self.result} = {call}" if self.result is not None else f"    {call}"
        if self.op == "COPY":
            return f"    {self.result} = {self.arg1}"
        if self.op == "ALLOC_ARR":
            return f"    ALLOC_ARR {self.arg1}[{self.arg2}]"
        if self.op == "LOAD_INDEX":
            return f"    {self.result} = {self.arg1}[{self.arg2}]"
        if self.op == "STORE_INDEX":
            return f"    {self.arg1}[{self.arg2}] = {self.result}"
        if self.op == "MALLOC_STR":
            return f"    MALLOC_STR {self.arg1}, \"{self.arg2}\""
        if self.op == "FREE":
            return f"    FREE {self.arg1}"
        if self.op == "CONCAT_STR":
            return f"    {self.result} = CONCAT_STR {self.arg1}, {self.arg2}"
        if self.arg2 is None:
            # Unary: op is the symbol itself ('-' or '!')
            return f"    {self.result} = {self.op}{self.arg1}"
        # Binary
        return f"    {self.result} = {self.arg1} {self.op} {self.arg2}"


# ---------------------------------------------------------------------------
# IR Generator
# ---------------------------------------------------------------------------

class IRGenerator(ASTVisitor):
    """Walks a typed AST and emits a flat list of TACInstructions."""

    def __init__(self) -> None:
        self.instructions: list[TACInstruction] = []
        self.temp_count: int = 0
        self.label_count: int = 0

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def generate(self, program: Program) -> list[TACInstruction]:
        program.accept(self)
        return self.instructions

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def new_temp(self) -> str:
        name = f"t{self.temp_count}"
        self.temp_count += 1
        return name

    def new_label(self) -> str:
        name = f"L{self.label_count}"
        self.label_count += 1
        return name

    def _emit(self, instr: TACInstruction) -> None:
        self.instructions.append(instr)

    def _expr_type(self, node: ASTNode) -> str:
        if isinstance(node, StringLiteral):
            return "string"
        if isinstance(node, Literal):
            return node.value_type
        return getattr(node, "eval_type", "int")

    # ------------------------------------------------------------------
    # Top-level
    # ------------------------------------------------------------------

    def visit_program(self, node: Program) -> None:
        for decl in node.body:
            decl.accept(self)

    def visit_block(self, node: Block) -> None:
        for stmt in node.statements:
            stmt.accept(self)

    # ------------------------------------------------------------------
    # Declarations
    # ------------------------------------------------------------------

    def visit_function_decl(self, node: FunctionDecl) -> None:
        self._emit(TACInstruction(op="FUNC", result=node.name))
        node.body.accept(self)

    def visit_var_decl(self, node: VarDecl) -> None:
        if isinstance(node.var_type, ArrayType):
            self._emit(TACInstruction(op="ALLOC_ARR", arg1=node.name, arg2=node.var_type.size))
            return
        if node.var_type == "string" and isinstance(node.initializer, StringLiteral):
            self._emit(TACInstruction(op="MALLOC_STR", arg1=node.name, arg2=node.initializer.value, result_type="string"))
            return
        if node.initializer is not None:
            val = node.initializer.accept(self)
            self._emit(TACInstruction(op="COPY", arg1=val, result=node.name, result_type=str(node.var_type)))

    # ------------------------------------------------------------------
    # Statements
    # ------------------------------------------------------------------

    def visit_assign_stmt(self, node: AssignStmt) -> None:
        val = node.value.accept(self)
        self._emit(TACInstruction(op="COPY", arg1=val, result=node.name, result_type=self._expr_type(node.value)))

    def visit_return_stmt(self, node: ReturnStmt) -> None:
        if node.value is not None:
            val = node.value.accept(self)
            self._emit(TACInstruction(op="RETURN", arg1=val))
        else:
            self._emit(TACInstruction(op="RETURN"))

    def visit_if_stmt(self, node: IfStmt) -> None:
        end_label = self.new_label()
        cond = node.condition.accept(self)
        else_label = self.new_label()
        self._emit(TACInstruction(op="IF_FALSE", arg1=cond, result=else_label))
        node.then_block.accept(self)
        self._emit(TACInstruction(op="JMP", result=end_label))
        self._emit(TACInstruction(op="LABEL", result=else_label))
        for ei_cond, ei_block in node.else_ifs:
            ei_val = ei_cond.accept(self)
            next_label = self.new_label()
            self._emit(TACInstruction(op="IF_FALSE", arg1=ei_val, result=next_label))
            ei_block.accept(self)
            self._emit(TACInstruction(op="JMP", result=end_label))
            self._emit(TACInstruction(op="LABEL", result=next_label))
        if node.else_block is not None:
            node.else_block.accept(self)
        self._emit(TACInstruction(op="LABEL", result=end_label))

    def visit_while_stmt(self, node: WhileStmt) -> None:
        start_label = self.new_label()
        end_label = self.new_label()
        self._emit(TACInstruction(op="LABEL", result=start_label))
        cond = node.condition.accept(self)
        self._emit(TACInstruction(op="IF_FALSE", arg1=cond, result=end_label))
        node.body.accept(self)
        self._emit(TACInstruction(op="JMP", result=start_label))
        self._emit(TACInstruction(op="LABEL", result=end_label))

    def visit_for_stmt(self, node: ForStmt) -> None:
        init_val = node.init_value.accept(self)
        self._emit(TACInstruction(op="COPY", arg1=init_val, result=node.var_name, result_type="int"))
        start_label = self.new_label()
        end_label = self.new_label()
        self._emit(TACInstruction(op="LABEL", result=start_label))
        cond = node.condition.accept(self)
        self._emit(TACInstruction(op="IF_FALSE", arg1=cond, result=end_label))
        node.body.accept(self)
        step_val = node.step_value.accept(self)
        self._emit(TACInstruction(op="COPY", arg1=step_val, result=node.step_var, result_type="int"))
        self._emit(TACInstruction(op="JMP", result=start_label))
        self._emit(TACInstruction(op="LABEL", result=end_label))

    # ------------------------------------------------------------------
    # Expressions  (each returns the name of the temp holding the value)
    # ------------------------------------------------------------------

    def visit_literal(self, node: Literal) -> str:
        if isinstance(node.value, bool):
            return "true" if node.value else "false"
        return str(node.value)

    def visit_identifier(self, node: Identifier) -> str:
        return node.name

    def visit_binary_op(self, node: BinaryOp) -> str:
        left = node.left.accept(self)
        right = node.right.accept(self)
        if node.op == "+" and node.eval_type == "string":
            t = self.new_temp()
            self._emit(TACInstruction(op="CONCAT_STR", arg1=left, arg2=right, result=t, result_type="string"))
            return t
        t = self.new_temp()
        self._emit(TACInstruction(op=node.op, arg1=left, arg2=right, result=t, result_type=node.eval_type))
        return t

    def visit_unary_op(self, node: UnaryOp) -> str:
        operand = node.operand.accept(self)
        t = self.new_temp()
        self._emit(TACInstruction(op=node.op, arg1=operand, result=t, result_type=node.eval_type))
        return t

    def visit_func_call(self, node: FuncCall) -> str:
        if node.name == "free" and len(node.args) == 1:
            target = node.args[0].accept(self)
            self._emit(TACInstruction(op="FREE", arg1=target))
            return self.new_temp()   # dummy temp; free is void
        arg_vals = [arg.accept(self) for arg in node.args]
        for val, arg in zip(arg_vals, node.args):
            self._emit(TACInstruction(op="PARAM", arg1=val, result_type=self._expr_type(arg)))
        t = self.new_temp()
        self._emit(TACInstruction(op="CALL", arg1=node.name, arg2=len(node.args), result=t, result_type=node.eval_type))
        return t

    def visit_string_literal(self, node: StringLiteral) -> str:
        tmp = self.new_temp()
        self._emit(TACInstruction(op="MALLOC_STR", arg1=tmp, arg2=node.value, result_type="string"))
        return tmp

    def visit_index_expr(self, node: IndexExpr) -> str:
        index_temp = node.index.accept(self)
        result_temp = self.new_temp()
        self._emit(TACInstruction(op="LOAD_INDEX", arg1=node.name, arg2=index_temp, result=result_temp, result_type=node.eval_type))
        return result_temp

    def visit_index_assign_stmt(self, node: IndexAssignStmt) -> None:
        index_temp = node.index.accept(self)
        value_temp = node.value.accept(self)
        self._emit(TACInstruction(op="STORE_INDEX", arg1=node.name, arg2=index_temp, result=value_temp))


# ---------------------------------------------------------------------------
# CLI entry point — python ir_gen.py <file.nv>
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from error import CompilerError
    from lexer import Lexer, TokenType
    from parser import Parser
    from semantic import SemanticAnalyzer

    if len(sys.argv) != 2:
        print("usage: python ir_gen.py <source.nv>", file=sys.stderr)
        sys.exit(1)

    try:
        with open(sys.argv[1]) as f:
            source = f.read()
    except FileNotFoundError:
        print(f"error: file not found: {sys.argv[1]}", file=sys.stderr)
        sys.exit(1)

    try:
        from optimizer import Optimizer

        lexer = Lexer(source)
        tokens = []
        while True:
            tok = lexer.get_next_token()
            tokens.append(tok)
            if tok.type == TokenType.EOF:
                break
        program = Parser(tokens).parse()
        SemanticAnalyzer().analyze(program)
        before = IRGenerator().generate(program)
        after = Optimizer().optimize(before)

        print("; ── before optimisation ─────────────────────")
        for instr in before:
            print(instr)
        print()
        print("; ── after optimisation ──────────────────────")
        for instr in after:
            print(instr)
    except CompilerError as e:
        print(e.format(), file=sys.stderr)
        sys.exit(1)
