# PyCompiler — A Native x86-64 Compiler for the Nova Language

```
  Nova Source (.nv)
       │
       ▼
  ┌─────────┐     hand-written        ┌──────────────┐
  │  Lexer  │ ──── recursive ──────►  │    Parser    │
  └─────────┘     descent             └──────────────┘
       │                                     │
   Token stream                        AST  │  @dataclass nodes
                                            ▼
                                  ┌──────────────────┐
                                  │ SemanticAnalyzer │  two-pass: signatures
                                  │  (type checker)  │  then full walk
                                  └──────────────────┘
                                            │
                                    typed AST (no changes)
                                            ▼
                                  ┌──────────────────┐
                                  │   IRGenerator    │  Three-Address Code
                                  └──────────────────┘
                                            │
                                     flat TAC list
                                            ▼
                                  ┌──────────────────┐
                                  │    Optimizer     │  constant folding
                                  │                  │  copy propagation
                                  │                  │  dead-code removal
                                  └──────────────────┘
                                            │
                                   optimised TAC
                                            ▼
                                  ┌──────────────────┐
                                  │ X86CodeGenerator │  two-pass:
                                  │  (native backend)│  prescan → emit
                                  └──────────────────┘
                                            │
                                     NASM .asm file
                                            │
                              nasm + gcc/ld │
                                            ▼
                                   native ELF / Mach-O
```

A compiler built from scratch — no ANTLR, no PLY, no LLVM.  
Every token is lexed by hand. Every register argument is placed by code you can read.

> **Integer and String focused.** Nova's type system is designed around 64-bit integers and heap-managed strings. Floating-point and array support are present but secondary; the core semantics, optimizer, and x86-64 backend are optimized for `int` and `string` workloads.

---

## What This Is

PyCompiler translates **Nova**, a statically-typed systems language, all the way down to **native x86-64 machine code** via NASM assembly. The compiler is written entirely in Python with zero compilation-framework dependencies. Optimized for integer-based systems tasks and strict register pressure management on x86-64.

This project answers the question: *what actually happens between `source code` and `./program`?*

---

## Core Features

### Strict Static Typing
An integer-focused type system (with string support) designed to eliminate runtime type overhead and provide deterministic register allocation.

### Systems-Level Memory Control
Manual heap management via `malloc` and `free` integration, allowing for high-performance dynamic memory patterns.

### Static Type System
Nova is statically typed with compile-time type checking at every expression boundary.

| Type     | Description                                      |
|----------|--------------------------------------------------|
| `int`    | 64-bit signed integer                            |
| `float`  | 64-bit IEEE 754 double                           |
| `bool`   | Boolean (`true` / `false`)                       |
| `string` | Heap-allocated, null-terminated byte string      |
| `T[N]`   | Fixed-size stack array (e.g. `[int; 8]`)         |

The analyzer runs in **two passes**: pass 1 registers all function signatures globally (enabling mutual recursion), pass 2 type-checks the full AST.

### Physical x86-64 Backend
The code generator targets **x86-64 NASM** assembly, fully compliant with the **System V AMD64 ABI**:

- Integer parameters passed in `rdi`, `rsi`, `rdx`, `rcx`, `r8`, `r9`
- Return values in `rax`
- Callee-saved registers preserved across calls
- Stack frame aligned to 16 bytes before every `call` instruction
- Two-pass generation: prescan allocates stack slots (`[rbp-N]`), then instruction translation fills the body — no backpatching

Parameters are spilled to stack-frame slots immediately after the prologue. Every operation follows the strict load-store pattern: load operands into registers, compute, write result back to its frame slot.

### Heap Memory Management
Nova programs can allocate and free heap memory via the C runtime:

```nova
let greeting: string = "Hello, World!";
print(greeting);
greeting[0] = 74;          // mutate in place — 'H' → 'J'
print(greeting);

let s1: string = "foo";
let s2: string = "bar";
let s3: string = s1 + s2;  // malloc + strlen + rep movsb
print(s3);

free(greeting);
free(s1);
free(s2);
free(s3);
```

String allocation calls `_malloc` with `strlen + 1` bytes, copies the literal using `rep movsb`, and stores the heap pointer in the variable's stack slot. The `+` operator on two strings calls `_strlen` on each operand, allocates the concatenated buffer, and fills it with two `rep movsb` passes.

### Optimizer
A single-pass optimizer runs on the Three-Address Code before code generation:

| Pass               | Example                                              |
|--------------------|------------------------------------------------------|
| Constant folding   | `t0 = 10 * 2` → `t0 = 20`                          |
| Copy propagation   | `x = 20; t1 = x + 5` → `t1 = 25`                   |
| Dead code removal  | `t0 = 20` with no live uses → instruction removed   |

The optimizer makes a single forward sweep over the TAC list — simple enough to reason about formally, effective enough to eliminate all redundant computation in constant-initialised programs.

---

## The Nova Language

Nova is a small systems language with a clean, readable syntax.

### Recursive Fibonacci

```nova
fn fib(n: int) -> int {
    if n <= 1 { return n; }
    return fib(n - 1) + fib(n - 2);
}

fn main() {
    let result: int = fib(10);
    print(result);
}
```

### Bubble Sort with Stack Arrays

```nova
fn main() {
    let arr: [int; 5];
    arr[0] = 64;
    arr[1] = 34;
    arr[2] = 25;
    arr[3] = 12;
    arr[4] = 22;

    let n: int = 5;
    let i: int = 0;

    while i < n - 1 {
        let j: int = 0;
        while j < n - i - 1 {
            if arr[j] > arr[j + 1] {
                let temp: int = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
            j = j + 1;
        }
        i = i + 1;
    }

    let k: int = 0;
    while k < n {
        print(arr[k]);
        k = k + 1;
    }
}
```

### String Manipulation

```nova
fn main() {
    let msg: string = "Hello, System-Level Programming!";
    let suffix: string = " — PyCompiler x86-64";
    let full: string = msg + suffix;

    print(full);
    full[0] = 74;   // 'H' → 'J'
    print(full);

    free(msg);
    free(suffix);
    free(full);
}
```

---

## Compiler Pipeline in Detail

```
nova run examples/fibonacci.nv
```

| Stage               | Input           | Output           | File           |
|---------------------|-----------------|------------------|----------------|
| Lexer               | Source text      | Token stream     | `lexer.py`     |
| Parser              | Tokens          | AST              | `parser.py`    |
| Semantic Analyzer   | AST             | Validated AST    | `semantic.py`  |
| IR Generator        | AST             | TAC instructions | `ir_gen.py`    |
| Optimizer           | Raw TAC         | Optimised TAC    | `optimizer.py` |
| x86-64 Code Gen     | Optimised TAC   | NASM `.asm`      | `codegen_x86.py` |

---

## Building & Running

### Prerequisites

- Python 3.11+
- [NASM](https://nasm.us/) (`brew install nasm` on macOS)
- `gcc` or `clang` (ships with Xcode Command Line Tools on macOS)

### Step 1 — Compile Nova source to assembly

```bash
python main.py run examples/fibonacci.nv
# → fibonacci.asm
```

Use `--dump-ir` to inspect the Three-Address Code before and after optimisation:

```bash
python main.py run examples/fibonacci.nv --dump-ir
```

Use `--dump-ast` to inspect the Abstract Syntax Tree:

```bash
python main.py run examples/fibonacci.nv --dump-ast
```

### Step 2 — Assemble and link (Linux x86-64)

```bash
nasm -f elf64 examples/fibonacci.asm -o fibonacci.o
gcc fibonacci.o -o fibonacci
./fibonacci
# 55
```

### Step 2 — Assemble and link (macOS via Rosetta / x86-64)

> **Note:** The compiler targets the Linux System V ABI (no leading underscores on symbols). macOS Mach-O requires underscored symbols (`_main`, `_printf`, …). Running the generated `.asm` directly under macOS therefore requires either a Linux Docker container or a local symbol-rename step before assembling with `nasm -f macho64`.

```bash
# Recommended: use Docker
docker run --rm -v "$PWD":/src -w /src nasm/nasm:latest \
  nasm -f elf64 examples/fibonacci.asm -o fibonacci.o && \
  gcc fibonacci.o -o fibonacci && ./fibonacci
```

### Running Tests

```bash
pip install pytest
pytest tests/ -v
```

The test suite covers the lexer, parser, semantic analyzer, IR generator, and optimizer — 263 tests, all independent of platform-specific toolchains.

---

## Project Structure

```
PyCompiler/
├── lexer.py          # Hand-written lexer — no regex
├── parser.py         # Recursive-descent parser
├── ast_nodes.py      # @dataclass AST nodes + visitor base
├── semantic.py       # Two-pass type checker + symbol table
├── ir_gen.py         # AST → Three-Address Code (TAC)
├── optimizer.py      # Constant folding, copy propagation, DCE
├── codegen_x86.py    # TAC → NASM x86-64 assembly
├── error.py          # CompilerError with line/col tracking
├── main.py           # CLI entry point
├── grammar.bnf       # Formal grammar specification
├── examples/
│   ├── fibonacci.nv
│   ├── bubble_sort.nv
│   └── string_test.nv
└── tests/
    ├── test_lexer.py
    ├── test_parser.py
    ├── test_semantic.py
    ├── test_ir_gen.py
    ├── test_optimizer.py
    └── test_e2e.py
```

---

## What the Generated Assembly Looks Like

Nova source:
```nova
fn fib(n: int) -> int {
    if n <= 1 { return n; }
    return fib(n - 1) + fib(n - 2);
}
```

Generated x86-64 (excerpt, Linux ELF64 / System V ABI):
```nasm
fib:
    push rbp
    mov rbp, rsp
    sub rsp, 48
    mov qword [rbp-8], rdi      ; spill parameter n
    mov rax, [rbp-8]            ; load n
    mov rcx, 1
    cmp rax, rcx
    setle al
    movzx rax, al               ; n <= 1
    mov qword [rbp-16], rax
    mov rax, [rbp-16]
    test rax, rax
    jz L1                       ; if false, skip then-block
    mov rax, [rbp-8]
    leave
    ret                         ; return n
L1:
    mov rdi, [rbp-8]
    mov rcx, 1
    sub rdi, rcx                ; n - 1
    call fib
    mov qword [rbp-32], rax     ; fib(n-1)
    mov rdi, [rbp-8]
    mov rcx, 2
    sub rdi, rcx                ; n - 2
    call fib
    mov qword [rbp-40], rax     ; fib(n-2)
    mov rax, [rbp-32]
    add rax, [rbp-40]
    leave
    ret
```

No abstractions, no magic. Every line of assembly is a direct translation of a TAC instruction.

---

## Development Philosophy

The project prioritizes architectural transparency. The pipeline is designed to be highly traceable, ensuring each stage — from AST to TAC to x86-64 — can be verified and debugged at the instruction level.

**Concrete rules that shaped every module:**

- Functions stay under 30 lines. One function, one job.
- No metaclasses, no descriptor magic, no `*args/**kwargs` without cause.
- No regex for parsing — the lexer exists for a reason.
- No third-party compilation frameworks (no PLY, ANTLR, Lark). The parser is a 400-line recursive-descent function that any CS student can trace.
- Comments explain *why*, not *what*. `# Advance past the closing paren before parsing the body — otherwise the body parser will consume it as an expression.` Not `# advance token`.
- `@dataclass` for every AST node. Flat. No inheritance chains beyond the visitor base.

The goal was a codebase where any intermediate engineer can open a file, read it top-to-bottom, and know exactly what it does and why — not because it's trivial, but because every decision was made with explainability as the primary constraint.

---

## Roadmap

- [ ] **IEEE 754 Floating Point** — Integrating XMM registers and floating-point instruction set.
- [ ] **Self-Hosting** — Re-implementing the compiler in the Nova language itself.
- [ ] **ARM64 Backend** — Extending the codegen to support Apple Silicon and Graviton architectures.

---

## License

MIT. Build on it, learn from it, break it, fix it.

---

*Built to understand what computers actually do — one instruction at a time.*
