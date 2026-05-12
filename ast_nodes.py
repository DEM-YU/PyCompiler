from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Base node
# ---------------------------------------------------------------------------

@dataclass
class ASTNode:
    line: int
    col: int

    def accept(self, visitor: ASTVisitor) -> Any:
        raise NotImplementedError(f"{type(self).__name__} must implement accept()")


# ---------------------------------------------------------------------------
# Helper: function parameter  (not a full AST node — not visited directly)
# ---------------------------------------------------------------------------

@dataclass
class Param:
    name: str
    param_type: str  # "int" | "float" | "bool" | "str"
    line: int
    col: int


# ---------------------------------------------------------------------------
# Type annotation node
# ---------------------------------------------------------------------------

@dataclass
class ArrayType(ASTNode):
    # grammar: type "[" INTEGER "]"   e.g.  int[10]
    element_type: str   # "int" | "float" | "bool" | "str"
    size: int

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_array_type(self)


# ---------------------------------------------------------------------------
# Expression nodes
# ---------------------------------------------------------------------------

@dataclass
class Literal(ASTNode):
    # grammar: INTEGER | FLOAT_LITERAL | STRING_LITERAL | "true" | "false"
    value: int | float | bool | str
    value_type: str  # "int" | "float" | "bool" | "str"

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_literal(self)


@dataclass
class StringLiteral(ASTNode):
    # grammar: STRING_LITERAL  (heap-managed "string" type)
    value: str

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_string_literal(self)


@dataclass
class Identifier(ASTNode):
    # grammar: ID
    name: str
    eval_type: str = "int"

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_identifier(self)


@dataclass
class BinaryOp(ASTNode):
    # grammar: expr op expr
    op: str   # "+" | "-" | "*" | "/" | "%" | "==" | "!=" | "<" | ">" | "<=" | ">=" | "&&" | "||"
    left: ASTNode
    right: ASTNode
    eval_type: str = "int"

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_binary_op(self)


@dataclass
class UnaryOp(ASTNode):
    # grammar: ("-" | "!") unary_expr
    op: str   # "-" | "!"
    operand: ASTNode
    eval_type: str = "int"

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_unary_op(self)


@dataclass
class FuncCall(ASTNode):
    # grammar: ID "(" arg_list? ")"
    name: str
    args: list[ASTNode]
    eval_type: str = "int"

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_func_call(self)


@dataclass
class IndexExpr(ASTNode):
    # grammar: ID "[" expression "]"   e.g.  arr[i]
    name: str
    index: ASTNode
    eval_type: str = "int"

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_index_expr(self)


# ---------------------------------------------------------------------------
# Statement nodes
# ---------------------------------------------------------------------------

@dataclass
class Block(ASTNode):
    # grammar: "{" statement* "}"
    statements: list[ASTNode]

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_block(self)


@dataclass
class VarDecl(ASTNode):
    # grammar: type ID ("=" expression)? ";"
    var_type: str | ArrayType   # scalar: "int" | "float" | "bool" | "str"; or ArrayType
    name: str
    initializer: ASTNode | None = None

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_var_decl(self)


@dataclass
class AssignStmt(ASTNode):
    # grammar: ID "=" expression ";"
    name: str
    value: ASTNode

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_assign_stmt(self)


@dataclass
class IndexAssignStmt(ASTNode):
    # grammar: ID "[" expression "]" "=" expression ";"   e.g.  arr[i] = 10;
    name: str
    index: ASTNode
    value: ASTNode

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_index_assign_stmt(self)


@dataclass
class IfStmt(ASTNode):
    # grammar: "if" expr block ("else" "if" expr block)* ("else" block)?
    condition: ASTNode
    then_block: Block
    else_ifs: list[tuple[ASTNode, Block]] = field(default_factory=list)
    else_block: Block | None = None

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_if_stmt(self)


@dataclass
class WhileStmt(ASTNode):
    # grammar: "while" expression block
    condition: ASTNode
    body: Block

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_while_stmt(self)


@dataclass
class ForStmt(ASTNode):
    # grammar: "for" ID "=" expr ";" expr ";" ID "=" expr block
    var_name: str
    init_value: ASTNode
    condition: ASTNode
    step_var: str
    step_value: ASTNode
    body: Block

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_for_stmt(self)


@dataclass
class ReturnStmt(ASTNode):
    # grammar: "return" expression? ";"
    value: ASTNode | None = None

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_return_stmt(self)


# ---------------------------------------------------------------------------
# Top-level nodes
# ---------------------------------------------------------------------------

@dataclass
class FunctionDecl(ASTNode):
    # grammar: "fn" ID "(" param_list? ")" ("->" type)? block
    name: str
    params: list[Param]
    return_type: str | None   # None when no "->" annotation is present
    body: Block

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_function_decl(self)


@dataclass
class Program(ASTNode):
    # grammar: top_decl*
    body: list[ASTNode]   # FunctionDecl or statement nodes

    def accept(self, visitor: ASTVisitor) -> Any:
        return visitor.visit_program(self)


# ---------------------------------------------------------------------------
# Visitor base class
# ---------------------------------------------------------------------------

class ASTVisitor:
    """
    Override only the visit_* methods you need.
    Unoverridden methods return None by default.
    """

    def visit_literal(self, node: Literal) -> Any:
        pass

    def visit_string_literal(self, node: StringLiteral) -> Any:
        pass

    def visit_identifier(self, node: Identifier) -> Any:
        pass

    def visit_binary_op(self, node: BinaryOp) -> Any:
        pass

    def visit_unary_op(self, node: UnaryOp) -> Any:
        pass

    def visit_func_call(self, node: FuncCall) -> Any:
        pass

    def visit_array_type(self, node: ArrayType) -> Any:
        pass

    def visit_index_expr(self, node: IndexExpr) -> Any:
        pass

    def visit_block(self, node: Block) -> Any:
        pass

    def visit_var_decl(self, node: VarDecl) -> Any:
        pass

    def visit_assign_stmt(self, node: AssignStmt) -> Any:
        pass

    def visit_index_assign_stmt(self, node: IndexAssignStmt) -> Any:
        pass

    def visit_if_stmt(self, node: IfStmt) -> Any:
        pass

    def visit_while_stmt(self, node: WhileStmt) -> Any:
        pass

    def visit_for_stmt(self, node: ForStmt) -> Any:
        pass

    def visit_return_stmt(self, node: ReturnStmt) -> Any:
        pass

    def visit_function_decl(self, node: FunctionDecl) -> Any:
        pass

    def visit_program(self, node: Program) -> Any:
        pass


# ---------------------------------------------------------------------------
# AST printer  (concrete ASTVisitor for debugging)
# ---------------------------------------------------------------------------

class ASTPrinter(ASTVisitor):
    """Prints the AST as an indented tree to stdout."""

    def __init__(self) -> None:
        self._depth = 0

    def _print(self, text: str) -> None:
        print("  " * self._depth + text)

    def _child(self, node: ASTNode) -> None:
        self._depth += 1
        node.accept(self)
        self._depth -= 1

    # Expression nodes

    def visit_literal(self, node: Literal) -> None:
        self._print(f"Literal({node.value_type}) {node.value!r}")

    def visit_string_literal(self, node: StringLiteral) -> None:
        self._print(f"StringLiteral {node.value!r}")

    def visit_identifier(self, node: Identifier) -> None:
        self._print(f"Identifier {node.name}")

    def visit_binary_op(self, node: BinaryOp) -> None:
        self._print(f"BinaryOp {node.op!r}")
        self._child(node.left)
        self._child(node.right)

    def visit_unary_op(self, node: UnaryOp) -> None:
        self._print(f"UnaryOp {node.op!r}")
        self._child(node.operand)

    def visit_func_call(self, node: FuncCall) -> None:
        self._print(f"FuncCall {node.name}")
        for arg in node.args:
            self._child(arg)

    def visit_array_type(self, node: ArrayType) -> None:
        self._print(f"ArrayType {node.element_type}[{node.size}]")

    def visit_index_expr(self, node: IndexExpr) -> None:
        self._print(f"IndexExpr {node.name}")
        self._child(node.index)

    # Statement nodes

    def visit_block(self, node: Block) -> None:
        self._print("Block")
        for stmt in node.statements:
            self._child(stmt)

    def visit_var_decl(self, node: VarDecl) -> None:
        self._print(f"VarDecl {node.var_type} {node.name}")
        if node.initializer is not None:
            self._child(node.initializer)

    def visit_assign_stmt(self, node: AssignStmt) -> None:
        self._print(f"AssignStmt {node.name} =")
        self._child(node.value)

    def visit_index_assign_stmt(self, node: IndexAssignStmt) -> None:
        self._print(f"IndexAssignStmt {node.name}[...] =")
        self._depth += 1
        self._print("index:")
        self._child(node.index)
        self._print("value:")
        self._child(node.value)
        self._depth -= 1

    def visit_if_stmt(self, node: IfStmt) -> None:
        self._print("IfStmt")
        self._depth += 1
        self._print("condition:")
        self._child(node.condition)
        self._print("then:")
        self._child(node.then_block)
        for cond, block in node.else_ifs:
            self._print("else if:")
            self._child(cond)
            self._child(block)
        if node.else_block is not None:
            self._print("else:")
            self._child(node.else_block)
        self._depth -= 1

    def visit_while_stmt(self, node: WhileStmt) -> None:
        self._print("WhileStmt")
        self._depth += 1
        self._print("condition:")
        self._child(node.condition)
        self._print("body:")
        self._child(node.body)
        self._depth -= 1

    def visit_for_stmt(self, node: ForStmt) -> None:
        self._print(f"ForStmt {node.var_name}")
        self._depth += 1
        self._print("init:")
        self._child(node.init_value)
        self._print("condition:")
        self._child(node.condition)
        self._print(f"step {node.step_var} =")
        self._child(node.step_value)
        self._print("body:")
        self._child(node.body)
        self._depth -= 1

    def visit_return_stmt(self, node: ReturnStmt) -> None:
        self._print("ReturnStmt")
        if node.value is not None:
            self._child(node.value)

    # Top-level nodes

    def visit_function_decl(self, node: FunctionDecl) -> None:
        ret = node.return_type or "void"
        self._print(f"FunctionDecl {node.name} -> {ret}")
        self._depth += 1
        for param in node.params:
            self._print(f"Param {param.name}: {param.param_type}")
        self._depth -= 1
        self._child(node.body)

    def visit_program(self, node: Program) -> None:
        self._print("Program")
        for decl in node.body:
            self._child(decl)
