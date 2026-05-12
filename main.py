from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

from ast_nodes import ASTPrinter, FunctionDecl
from codegen_x86 import X86CodeGenerator
from error import CompilerError
from ir_gen import IRGenerator, TACInstruction
from lexer import Lexer, TokenType
from optimizer import Optimizer
from parser import Parser
from semantic import SemanticAnalyzer


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

def _read_source(path: str) -> str:
    try:
        with open(path) as f:
            return f.read()
    except FileNotFoundError:
        print(f"error: file not found: {path}", file=sys.stderr)
        sys.exit(1)


def _lex(source: str) -> list:
    lexer = Lexer(source)
    tokens = []
    while True:
        tok = lexer.get_next_token()
        tokens.append(tok)
        if tok.type == TokenType.EOF:
            break
    return tokens


def _extract_func_params(program: Any) -> dict[str, list[str]]:
    params: dict[str, list[str]] = {}
    for node in program.body:
        if isinstance(node, FunctionDecl):
            params[node.name] = [p.name for p in node.params]
    return params


def _compile_to_ir(source: str) -> tuple:
    """Lex → parse → analyze → IR → optimize."""
    tokens = _lex(source)
    program = Parser(tokens).parse()
    SemanticAnalyzer().analyze(program)
    raw_tac = IRGenerator().generate(program)
    opt_tac = Optimizer().optimize(raw_tac)
    return program, raw_tac, opt_tac


# ---------------------------------------------------------------------------
# Debug output helpers
# ---------------------------------------------------------------------------

def _print_section(title: str) -> None:
    bar = "─" * max(4, 48 - len(title))
    print(f"; ── {title} {bar}")


def _print_tac(title: str, instructions: list[TACInstruction]) -> None:
    _print_section(title)
    for instr in instructions:
        print(instr)
    print()


# ---------------------------------------------------------------------------
# Command handler
# ---------------------------------------------------------------------------

def _run(args: argparse.Namespace) -> None:
    source = _read_source(args.file)
    program, raw_tac, opt_tac = _compile_to_ir(source)
    if args.dump_ast:
        _print_section("AST")
        ASTPrinter().visit_program(program)
        print()
    if args.dump_ir:
        _print_tac("IR (before optimisation)", raw_tac)
        _print_tac("IR (after optimisation)", opt_tac)
    func_params = _extract_func_params(program)
    asm = X86CodeGenerator().generate(opt_tac, func_params=func_params)
    asm_path = Path(args.file).with_suffix(".asm")
    asm_path.write_text(asm)
    print(f"Assembly generated: {asm_path}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(
        prog="nova",
        description="Nova language compiler — emits x86-64 NASM assembly",
    )
    sub = p.add_subparsers(dest="command", required=True)

    run_p = sub.add_parser("run", help="compile a .nv source file to x86-64 assembly")
    run_p.add_argument("file", help="path to the .nv source file")
    run_p.add_argument(
        "--entry-point", default="main", metavar="FUNC",
        help="function to invoke first (default: main)",
    )
    run_p.add_argument("--dump-ast", action="store_true",
                       help="print the AST before semantic analysis")
    run_p.add_argument("--dump-ir", action="store_true",
                       help="print TAC before and after optimisation")

    args = p.parse_args()
    try:
        _run(args)
    except CompilerError as e:
        print(e.format(), file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
