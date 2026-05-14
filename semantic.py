from __future__ import annotations

from dataclasses import dataclass, field

from ast_nodes import (
    ASTNode, ASTVisitor, ArrayType, Block, BinaryOp, FuncCall, FunctionDecl,
    Identifier, IfStmt, IndexAssignStmt, IndexExpr, Literal, Program,
    ReturnStmt, StringLiteral, UnaryOp, VarDecl, AssignStmt, WhileStmt, ForStmt,
)
from error import CompilerError


# ---------------------------------------------------------------------------
# Symbol
# ---------------------------------------------------------------------------

@dataclass
class Symbol:
    name: str
    type: str        # "int" | "float" | "bool" | "str" | "void" for func returns
    category: str    # "var" | "func"
    line: int
    col: int
    param_types: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# SymbolTable
# ---------------------------------------------------------------------------

class SymbolTable:
    def __init__(self, parent: SymbolTable | None = None) -> None:
        self.symbols: dict[str, Symbol] = {}
        self.parent: SymbolTable | None = parent

    def define(self, symbol: Symbol) -> None:
        if symbol.name in self.symbols:
            existing = self.symbols[symbol.name]
            raise CompilerError(
                message=f"'{symbol.name}' is already defined in this scope "
                        f"(first defined at line {existing.line}, col {existing.col})",
                line=symbol.line,
                col=symbol.col,
            )
        self.symbols[symbol.name] = symbol

    def lookup(self, name: str) -> Symbol | None:
        if name in self.symbols:
            return self.symbols[name]
        if self.parent is not None:
            return self.parent.lookup(name)
        return None


# ---------------------------------------------------------------------------
# Operator classification  (module-level so they are built once)
# ---------------------------------------------------------------------------

_ARITHMETIC_OPS: frozenset[str] = frozenset({"+", "-", "*", "/", "%"})
_COMPARISON_OPS: frozenset[str] = frozenset({"<", ">", "<=", ">="})
_EQUALITY_OPS:   frozenset[str] = frozenset({"==", "!="})
_LOGICAL_OPS:    frozenset[str] = frozenset({"&&", "||"})
_NUMERIC_TYPES:  frozenset[str] = frozenset({"int", "float"})


# ---------------------------------------------------------------------------
# SemanticAnalyzer
# ---------------------------------------------------------------------------

class SemanticAnalyzer(ASTVisitor):
    """
    Two-pass semantic checker.
      Pass 1 — analyze_signatures: registers every function name in global scope
               so that mutually-recursive calls resolve correctly.
      Pass 2 — program.accept(self): full type-checking walk.
    """

    def __init__(self) -> None:
        self.global_scope: SymbolTable = SymbolTable()
        self.current_scope: SymbolTable = self.global_scope
        self._current_return_type: str | None = None

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def analyze(self, program: Program) -> None:
        self.analyze_signatures(program)
        program.accept(self)

    # ------------------------------------------------------------------
    # Pass 1 — function signature registration
    # ------------------------------------------------------------------

    def analyze_signatures(self, program: Program) -> None:
        # Built-in functions available in all Nova programs.
        self.global_scope.define(Symbol(name="print", type="void",
                                        category="func", line=0, col=0))
        self.global_scope.define(Symbol(name="free", type="void",
                                        category="func", line=0, col=0))
        for node in program.body:
            if isinstance(node, FunctionDecl):
                ret_type = node.return_type if node.return_type is not None else "void"
                sym = Symbol(
                    name=node.name,
                    type=ret_type,
                    category="func",
                    line=node.line,
                    col=node.col,
                    param_types=[p.param_type for p in node.params],
                )
                self.global_scope.define(sym)

    # ------------------------------------------------------------------
    # Scope helpers
    # ------------------------------------------------------------------

    def _push_scope(self) -> None:
        self.current_scope = SymbolTable(parent=self.current_scope)

    def _pop_scope(self) -> None:
        assert self.current_scope.parent is not None, "cannot pop global scope"
        self.current_scope = self.current_scope.parent

    # ------------------------------------------------------------------
    # Pass 2 — visitor methods
    # ------------------------------------------------------------------

    def visit_program(self, node: Program) -> None:
        for decl in node.body:
            decl.accept(self)

    # --- declarations -------------------------------------------------

    def visit_function_decl(self, node: FunctionDecl) -> None:
        if len(node.params) > 6:
            raise CompilerError(
                message=(
                    f"Implementation Limit: Function parameters exceeding 6 are not yet supported "
                    f"(X86-64 ABI stack spilling not implemented). "
                    f"'{node.name}' declares {len(node.params)}."
                ),
                line=node.line,
                col=node.col,
            )
        saved_return_type = self._current_return_type
        self._current_return_type = node.return_type if node.return_type is not None else "void"
        self._push_scope()
        for param in node.params:
            sym = Symbol(name=param.name, type=param.param_type,
                         category="var", line=param.line, col=param.col)
            self.current_scope.define(sym)
        node.body.accept(self)
        self._pop_scope()
        self._current_return_type = saved_return_type

    def visit_var_decl(self, node: VarDecl) -> None:
        if isinstance(node.var_type, ArrayType):
            type_str = f"[{node.var_type.element_type}; {node.var_type.size}]"
        else:
            type_str = node.var_type
            if node.initializer is not None:
                init_type = node.initializer.accept(self)
                self._assert_type(type_str, init_type, node.line, node.col,
                                  context=f"initializer for '{node.name}'")
        sym = Symbol(name=node.name, type=type_str,
                     category="var", line=node.line, col=node.col)
        self.current_scope.define(sym)

    # --- statements ---------------------------------------------------

    def visit_block(self, node: Block) -> None:
        for stmt in node.statements:
            stmt.accept(self)

    def visit_assign_stmt(self, node: AssignStmt) -> None:
        sym = self.current_scope.lookup(node.name)
        if sym is None:
            raise CompilerError(
                message=f"undefined variable '{node.name}'",
                line=node.line,
                col=node.col,
            )
        value_type = node.value.accept(self)
        self._assert_type(sym.type, value_type, node.line, node.col,
                          context=f"assignment to '{node.name}'")

    def visit_if_stmt(self, node: IfStmt) -> None:
        cond_type = node.condition.accept(self)
        self._assert_type("bool", cond_type, node.line, node.col, context="if condition")
        node.then_block.accept(self)
        for ei_cond, ei_block in node.else_ifs:
            ei_type = ei_cond.accept(self)
            self._assert_type("bool", ei_type, ei_cond.line, ei_cond.col,
                              context="else if condition")
            ei_block.accept(self)
        if node.else_block is not None:
            node.else_block.accept(self)

    def visit_while_stmt(self, node: WhileStmt) -> None:
        cond_type = node.condition.accept(self)
        self._assert_type("bool", cond_type, node.line, node.col, context="while condition")
        node.body.accept(self)

    def visit_for_stmt(self, node: ForStmt) -> None:
        init_type = node.init_value.accept(self)
        self._assert_type("int", init_type, node.line, node.col, context="for-loop initializer")
        # Loop scope: holds only the induction variable. The step expression is
        # evaluated here so it can see the counter but not body-local variables.
        self._push_scope()
        loop_sym = Symbol(name=node.var_name, type="int",
                          category="var", line=node.line, col=node.col)
        self.current_scope.define(loop_sym)
        cond_type = node.condition.accept(self)
        self._assert_type("bool", cond_type, node.line, node.col, context="for-loop condition")
        step_type = node.step_value.accept(self)
        self._assert_type("int", step_type, node.line, node.col, context="for-loop step")
        # Body scope: isolated from the step so body-local declarations cannot
        # collide with or shadow the induction variable at the step level.
        self._push_scope()
        node.body.accept(self)
        self._pop_scope()
        self._pop_scope()

    def visit_return_stmt(self, node: ReturnStmt) -> None:
        actual = "void" if node.value is None else node.value.accept(self)
        expected = self._current_return_type
        if actual != expected:
            raise CompilerError(
                message=f"return type mismatch: expected '{expected}' but got '{actual}'",
                line=node.line,
                col=node.col,
            )

    # --- expressions --------------------------------------------------

    def visit_literal(self, node: Literal) -> str:
        return node.value_type

    def visit_string_literal(self, node: StringLiteral) -> str:
        return "string"

    def visit_identifier(self, node: Identifier) -> str:
        sym = self.current_scope.lookup(node.name)
        if sym is None:
            raise CompilerError(
                message=f"undefined name '{node.name}'",
                line=node.line,
                col=node.col,
            )
        node.eval_type = sym.type
        return sym.type

    def visit_unary_op(self, node: UnaryOp) -> str:
        operand_type = node.operand.accept(self)
        if node.op == "!" and operand_type != "bool":
            raise CompilerError(
                message=f"operator '!' requires a 'bool' operand, got '{operand_type}'",
                line=node.line,
                col=node.col,
            )
        if node.op == "-" and operand_type not in _NUMERIC_TYPES:
            raise CompilerError(
                message=f"operator '-' requires a numeric operand, got '{operand_type}'",
                line=node.line,
                col=node.col,
            )
        node.eval_type = operand_type
        return operand_type

    def visit_binary_op(self, node: BinaryOp) -> str:
        left_type = node.left.accept(self)
        right_type = node.right.accept(self)
        op = node.op
        if op in _ARITHMETIC_OPS:
            result = self._check_arithmetic(op, left_type, right_type, node.line, node.col)
        elif op in _COMPARISON_OPS:
            result = self._check_comparison(op, left_type, right_type, node.line, node.col)
        elif op in _EQUALITY_OPS:
            result = self._check_equality(op, left_type, right_type, node.line, node.col)
        elif op in _LOGICAL_OPS:
            result = self._check_logical(op, left_type, right_type, node.line, node.col)
        else:
            raise CompilerError(message=f"unknown operator '{op}'", line=node.line, col=node.col)
        node.eval_type = result
        return result

    def visit_func_call(self, node: FuncCall) -> str:
        sym = self.current_scope.lookup(node.name)
        if sym is None:
            raise CompilerError(
                message=f"undefined function '{node.name}'",
                line=node.line,
                col=node.col,
            )
        if sym.category != "func":
            raise CompilerError(
                message=f"'{node.name}' is a variable, not a function",
                line=node.line,
                col=node.col,
            )
        if sym.param_types:
            if len(node.args) != len(sym.param_types):
                raise CompilerError(
                    message=(
                        f"'{node.name}' expects {len(sym.param_types)} argument(s) "
                        f"but got {len(node.args)}"
                    ),
                    line=node.line,
                    col=node.col,
                )
            for i, (arg, expected) in enumerate(zip(node.args, sym.param_types)):
                actual = arg.accept(self)
                self._assert_type(
                    expected, actual, arg.line, arg.col,
                    context=f"argument {i + 1} of '{node.name}'",
                )
        else:
            for arg in node.args:
                arg.accept(self)
        node.eval_type = sym.type
        return sym.type

    def visit_index_expr(self, node: IndexExpr) -> str:
        sym = self.current_scope.lookup(node.name)
        if sym is None:
            raise CompilerError(
                message=f"undefined variable '{node.name}'",
                line=node.line,
                col=node.col,
            )
        self._require_int_index(node.index)
        if sym.type == "string":
            result = "int"
        elif not sym.type.startswith("["):
            raise CompilerError(
                message=f"'{node.name}' is not indexable",
                line=node.line,
                col=node.col,
            )
        else:
            result = self._array_element_type(sym.type)
        node.eval_type = result
        return result

    def visit_index_assign_stmt(self, node: IndexAssignStmt) -> None:
        sym = self.current_scope.lookup(node.name)
        if sym is None:
            raise CompilerError(
                message=f"undefined variable '{node.name}'",
                line=node.line,
                col=node.col,
            )
        self._require_int_index(node.index)
        if sym.type == "string":
            element_type = "int"
        elif sym.type.startswith("["):
            element_type = self._array_element_type(sym.type)
        else:
            raise CompilerError(
                message=f"'{node.name}' is not indexable",
                line=node.line,
                col=node.col,
            )
        value_type = node.value.accept(self)
        self._assert_type(element_type, value_type, node.line, node.col,
                          context=f"assignment to '{node.name}[...]'")

    # ------------------------------------------------------------------
    # Type-checking helpers
    # ------------------------------------------------------------------

    def _array_element_type(self, type_str: str) -> str:
        # Parses "[int; 5]" → "int"
        return type_str[1:type_str.index(";")].strip()

    def _require_array(self, name: str, line: int, col: int) -> Symbol:
        sym = self.current_scope.lookup(name)
        if sym is None:
            raise CompilerError(
                message=f"undefined variable '{name}'",
                line=line,
                col=col,
            )
        if not sym.type.startswith("["):
            raise CompilerError(
                message=f"'{name}' is not an array",
                line=line,
                col=col,
            )
        return sym

    def _require_int_index(self, index_expr: ASTNode) -> None:
        index_type = index_expr.accept(self)
        if index_type != "int":
            raise CompilerError(
                message=f"array index must be 'int', got '{index_type}'",
                line=index_expr.line,
                col=index_expr.col,
            )

    def _assert_type(self, expected: str, actual: str | None,
                     line: int, col: int, context: str) -> None:
        if actual is None or actual == expected:
            return
        raise CompilerError(
            message=f"type mismatch in {context}: expected '{expected}' but got '{actual}'",
            line=line,
            col=col,
        )

    def _require_numeric(self, type_name: str, op: str, line: int, col: int) -> None:
        if type_name not in _NUMERIC_TYPES:
            raise CompilerError(
                message=f"operator '{op}' requires numeric operands, got '{type_name}'",
                line=line,
                col=col,
            )

    def _require_matching(self, left: str, right: str, op: str, line: int, col: int) -> None:
        if left != right:
            raise CompilerError(
                message=f"operator '{op}' operands must be the same type, "
                        f"got '{left}' and '{right}'",
                line=line,
                col=col,
            )

    def _check_arithmetic(self, op: str, left: str, right: str, line: int, col: int) -> str:
        if op == "+" and left == "string" and right == "string":
            return "string"
        self._require_numeric(left, op, line, col)
        self._require_matching(left, right, op, line, col)
        return left

    def _check_comparison(self, op: str, left: str, right: str, line: int, col: int) -> str:
        self._require_numeric(left, op, line, col)
        self._require_matching(left, right, op, line, col)
        return "bool"

    def _check_equality(self, op: str, left: str, right: str, line: int, col: int) -> str:
        self._require_matching(left, right, op, line, col)
        return "bool"

    def _check_logical(self, op: str, left: str, right: str, line: int, col: int) -> str:
        if left != "bool":
            raise CompilerError(
                message=f"operator '{op}' requires 'bool' operands, got '{left}'",
                line=line,
                col=col,
            )
        if right != "bool":
            raise CompilerError(
                message=f"operator '{op}' requires 'bool' operands, got '{right}'",
                line=line,
                col=col,
            )
        return "bool"
