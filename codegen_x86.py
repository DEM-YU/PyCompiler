from __future__ import annotations

from ir_gen import TACInstruction

# System V AMD64 integer argument registers, in order.
_PARAM_REGS: list[str] = ["rdi", "rsi", "rdx", "rcx", "r8", "r9"]


class X86CodeGenerator:
    def __init__(self) -> None:
        self.var_offsets: dict[str, int] = {}
        self.current_offset: int = 0
        self._pending_params: list[tuple[str, str]] = []
        self._func_params: dict[str, list[str]] = {}
        self._str_vars: set[str] = set()
        self._str_pool: list[tuple[str, str]] = []
        self._str_index: dict[str, str] = {}

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def generate(
        self,
        instructions: list[TACInstruction],
        func_params: dict[str, list[str]] | None = None,
    ) -> str:
        self._func_params = func_params or {}
        self._str_pool = []
        self._str_index = {}
        for instr in instructions:
            if instr.op == "MALLOC_STR":
                self._get_str_label(str(instr.arg2))
        data_lines = ["section .data", 'fmt db "%lld", 10, 0', 'fmts db "%s", 10, 0']
        for label, value in self._str_pool:
            data_lines.append(f'{label} db "{value}", 0')
        header = data_lines + [
            "extern printf",
            "extern malloc",
            "extern free",
            "extern strlen",
            "section .text",
            "global main",
        ]
        body = self._generate_all_functions(instructions)
        return "\n".join(header + body)

    # ------------------------------------------------------------------
    # Function-level generation
    # ------------------------------------------------------------------

    def _generate_all_functions(
        self, instructions: list[TACInstruction]
    ) -> list[str]:
        # Split the flat TAC list into per-function chunks.
        funcs: list[tuple[str, list[TACInstruction]]] = []
        current_name: str | None = None
        current_instrs: list[TACInstruction] = []
        for instr in instructions:
            if instr.op == "FUNC":
                if current_name is not None:
                    funcs.append((current_name, current_instrs))
                current_name = str(instr.result)
                current_instrs = []
            else:
                current_instrs.append(instr)
        if current_name is not None:
            funcs.append((current_name, current_instrs))
        lines: list[str] = []
        for name, instrs in funcs:
            lines.extend(self._generate_function(name, instrs))
        return lines

    def _generate_function(
        self, func_name: str, instrs: list[TACInstruction]
    ) -> list[str]:
        # Reset per-function state.
        self.var_offsets = {}
        self.current_offset = 0
        self._pending_params = []
        self._str_vars = set()

        # Pass 1: allocate stack slots for params then all other names.
        params = self._func_params.get(func_name, [])
        for p in params:
            self._get_offset(p)
        for instr in instrs:
            self._prescan(instr)

        # Round up to the nearest 16-byte boundary (ABI requirement).
        stack_size = (self.current_offset + 15) // 16 * 16

        # Prologue.
        lines: list[str] = [f"{func_name}:", "push rbp", "mov rbp, rsp"]
        if stack_size > 0:
            lines.append(f"sub rsp, {stack_size}")

        # Spill register parameters to their stack slots immediately.
        for i, p in enumerate(params[:6]):
            offset = self._get_offset(p)
            lines.append(f"mov qword [rbp{offset}], {_PARAM_REGS[i]}")

        # Pass 2: translate body.
        for instr in instrs:
            lines.extend(self._translate_instruction(instr))

        # Guard: if the function never emitted a ret, add a safe default epilogue.
        if not lines or lines[-1] != "ret":
            lines += ["xor rax, rax", "mov rsp, rbp", "pop rbp", "ret"]
        return lines

    # ------------------------------------------------------------------
    # Pass 1 helpers: pre-scan to discover all variable names
    # ------------------------------------------------------------------

    def _prescan(self, instr: TACInstruction) -> None:
        op = instr.op
        if op in ("LABEL", "JMP", "FUNC", "ALLOC_ARR"):
            return
        if op in ("IF_FALSE", "IF_TRUE"):
            # result is a label, not a variable — skip it.
            self._maybe_allocate(instr.arg1)
            return
        if op == "CALL":
            # arg1 is the function name, arg2 is the arg count — skip both.
            self._maybe_allocate(instr.result)
            return
        if op == "MALLOC_STR":
            # arg1 = variable name (pointer), arg2 = raw string value (not a var).
            self._maybe_allocate(instr.arg1)
            self._str_vars.add(str(instr.arg1))
            return
        if op == "FREE":
            # arg1 was already allocated when MALLOC_STR was prescanned.
            return
        if op == "CONCAT_STR":
            self._maybe_allocate(instr.arg1)
            self._maybe_allocate(instr.arg2)
            self._maybe_allocate(instr.result)
            # Synthetic stack slots for the two strlen results.
            self._get_offset(f"{instr.result}_len1")
            self._get_offset(f"{instr.result}_len2")
            self._str_vars.add(str(instr.result))
            return
        if op in ("LOAD_INDEX", "STORE_INDEX"):
            self._maybe_allocate(instr.arg1)
            self._maybe_allocate(instr.arg2)
            self._maybe_allocate(instr.result)
            return
        if op == "COPY" and instr.result_type == "string" and instr.result is not None:
            self._maybe_allocate(instr.arg1)
            self._maybe_allocate(instr.result)
            self._str_vars.add(str(instr.result))
            return
        for val in (instr.result, instr.arg1, instr.arg2):
            self._maybe_allocate(val)

    def _maybe_allocate(self, val: object) -> None:
        if not isinstance(val, str):
            return
        if val in ("true", "false"):
            return
        if not self._is_numeric_literal(val):
            self._get_offset(val)

    # ------------------------------------------------------------------
    # Pass 2 helpers: instruction-level translation
    # ------------------------------------------------------------------

    def _translate_instruction(self, instr: TACInstruction) -> list[str]:
        op = instr.op
        if op == "LABEL":
            return [f"{instr.result}:"]
        if op == "JMP":
            return [f"jmp {instr.result}"]
        if op in ("IF_FALSE", "IF_TRUE"):
            return self._translate_cond_jump(instr)
        if op == "COPY":
            return self._translate_copy(instr)
        if op == "PARAM":
            self._pending_params.append((str(instr.arg1), instr.result_type))
            return []
        if op == "CALL":
            return self._translate_call(instr)
        if op == "RETURN":
            return self._translate_return(instr)
        if op == "ALLOC_ARR":
            return [f"; {instr}  ; TODO: stack array x86 support"]
        if op == "STORE_INDEX":
            return self._translate_store_index_x86(instr)
        if op == "LOAD_INDEX":
            return self._translate_load_index_x86(instr)
        if op == "MALLOC_STR":
            return self._translate_malloc_str(instr)
        if op == "FREE":
            return self._translate_free(instr)
        if op == "CONCAT_STR":
            return self._translate_concat_str(instr)
        if instr.arg2 is None:
            return self._translate_unary(instr)
        return self._translate_binary(instr)

    def _translate_copy(self, instr: TACInstruction) -> list[str]:
        src = str(instr.arg1)
        dst_name = str(instr.result)
        lines = self._load_operand(src, "rax")
        dst = self._get_offset(dst_name)
        lines.append(f"mov qword [rbp{dst}], rax")
        return lines

    def _translate_binary(self, instr: TACInstruction) -> list[str]:
        lines = self._load_operand(str(instr.arg1), "rax")
        lines += self._load_operand(str(instr.arg2), "rcx")
        lines += self._apply_binary_op(instr.op)
        dst = self._get_offset(str(instr.result))
        lines.append(f"mov qword [rbp{dst}], rax")
        return lines

    def _translate_unary(self, instr: TACInstruction) -> list[str]:
        lines = self._load_operand(str(instr.arg1), "rax")
        if instr.op == "-":
            lines.append("neg rax")
        elif instr.op == "!":
            lines += ["test rax, rax", "setz al", "movzx rax, al"]
        dst = self._get_offset(str(instr.result))
        lines.append(f"mov qword [rbp{dst}], rax")
        return lines

    def _translate_cond_jump(self, instr: TACInstruction) -> list[str]:
        lines = self._load_operand(str(instr.arg1), "rax")
        lines.append("test rax, rax")
        jmp_op = "jz" if instr.op == "IF_FALSE" else "jnz"
        lines.append(f"{jmp_op} {instr.result}")
        return lines

    def _translate_call(self, instr: TACInstruction) -> list[str]:
        if instr.arg1 == "print":
            return self._translate_print_call()
        # User-defined function: load args into registers, then call.
        lines: list[str] = []
        for i, (param, _) in enumerate(self._pending_params[:6]):
            lines.extend(self._load_operand(param, _PARAM_REGS[i]))
        self._pending_params.clear()
        lines.append(f"call {instr.arg1}")
        if instr.result is not None:
            dst = self._get_offset(str(instr.result))
            lines.append(f"mov qword [rbp{dst}], rax")
        return lines

    def _translate_return(self, instr: TACInstruction) -> list[str]:
        lines: list[str] = []
        if instr.arg1 is not None:
            lines = self._load_operand(str(instr.arg1), "rax")
        lines += ["leave", "ret"]
        return lines

    def _translate_print_call(self) -> list[str]:
        param, param_type = self._pending_params.pop(0) if self._pending_params else ("0", "int")
        lines = self._load_operand(param, "rsi")
        fmt_label = "fmts" if param_type == "string" else "fmt"
        lines += [
            f"lea rdi, [rel {fmt_label}]",
            "xor rax, rax",
            "mov rbx, rsp",
            "and rsp, -16",
            "call printf",
            "mov rsp, rbx",
        ]
        return lines

    def _get_str_label(self, value: str) -> str:
        if value not in self._str_index:
            label = f"str_{len(self._str_pool)}"
            self._str_pool.append((label, value))
            self._str_index[value] = label
        return self._str_index[value]

    def _translate_malloc_str(self, instr: TACInstruction) -> list[str]:
        name = str(instr.arg1)
        value = str(instr.arg2)
        length = len(value)
        str_label = self._get_str_label(value)
        dst = self._get_offset(name)
        lines = [
            f"mov rdi, {length + 1}",
            "mov rbx, rsp",
            "and rsp, -16",
            "call malloc",
            "mov rsp, rbx",
            f"mov qword [rbp{dst}], rax",
            f"lea rsi, [rel {str_label}]",
            "mov rdi, rax",
            f"mov rcx, {length + 1}",
            "rep movsb",
        ]
        return lines

    def _translate_free(self, instr: TACInstruction) -> list[str]:
        name = str(instr.arg1)
        src = self._get_offset(name)
        return [
            f"mov rdi, [rbp{src}]",
            "mov rbx, rsp",
            "and rsp, -16",
            "call free",
            "mov rsp, rbx",
        ]

    def _translate_load_index_x86(self, instr: TACInstruction) -> list[str]:
        name = str(instr.arg1)
        index = str(instr.arg2)
        result = str(instr.result)
        src_offset = self._get_offset(name)
        result_offset = self._get_offset(result)
        lines = self._load_operand(index, "rcx")
        if name in self._str_vars:
            lines += [
                f"mov rax, [rbp{src_offset}]",
                "movzx rax, byte [rax + rcx]",
            ]
        else:
            lines += [
                f"lea rax, [rbp{src_offset}]",
                "mov rax, qword [rax + rcx*8]",
            ]
        lines.append(f"mov qword [rbp{result_offset}], rax")
        return lines

    def _translate_store_index_x86(self, instr: TACInstruction) -> list[str]:
        name = str(instr.arg1)
        index = str(instr.arg2)
        value = str(instr.result)
        src_offset = self._get_offset(name)
        lines = self._load_operand(index, "rcx")
        lines += self._load_operand(value, "rdx")
        if name in self._str_vars:
            lines += [
                f"mov rax, [rbp{src_offset}]",
                "mov byte [rax + rcx], dl",
            ]
        else:
            lines += [
                f"lea rax, [rbp{src_offset}]",
                "mov qword [rax + rcx*8], rdx",
            ]
        return lines

    def _translate_concat_str(self, instr: TACInstruction) -> list[str]:
        s1 = str(instr.arg1)
        s2 = str(instr.arg2)
        result = str(instr.result)
        s1_off = self._get_offset(s1)
        s2_off = self._get_offset(s2)
        result_off = self._get_offset(result)
        len1_off = self._get_offset(f"{result}_len1")
        len2_off = self._get_offset(f"{result}_len2")
        return [
            # strlen(s1)
            f"mov rdi, [rbp{s1_off}]",
            "mov rbx, rsp",
            "and rsp, -16",
            "call strlen",
            "mov rsp, rbx",
            f"mov qword [rbp{len1_off}], rax",
            # strlen(s2)
            f"mov rdi, [rbp{s2_off}]",
            "mov rbx, rsp",
            "and rsp, -16",
            "call strlen",
            "mov rsp, rbx",
            f"mov qword [rbp{len2_off}], rax",
            # malloc(len1 + len2 + 1)
            f"mov rax, [rbp{len1_off}]",
            f"add rax, [rbp{len2_off}]",
            "inc rax",
            "mov rdi, rax",
            "mov rbx, rsp",
            "and rsp, -16",
            "call malloc",
            "mov rsp, rbx",
            f"mov qword [rbp{result_off}], rax",
            # copy s1 bytes into result
            "mov rdi, rax",
            f"mov rsi, [rbp{s1_off}]",
            f"mov rcx, [rbp{len1_off}]",
            "rep movsb",
            # copy s2 bytes (rdi already advanced past s1)
            f"mov rsi, [rbp{s2_off}]",
            f"mov rcx, [rbp{len2_off}]",
            "rep movsb",
            # null terminator at rdi (= result + len1 + len2)
            "mov byte [rdi], 0",
        ]

    # ------------------------------------------------------------------
    # Low-level helpers
    # ------------------------------------------------------------------

    def _load_operand(self, name: str, reg: str) -> list[str]:
        if name == "true":
            return [f"mov {reg}, 1"]
        if name == "false":
            return [f"mov {reg}, 0"]
        if self._is_numeric_literal(name):
            return [f"mov {reg}, {name}"]
        offset = self._get_offset(name)
        return [f"mov {reg}, [rbp{offset}]"]

    def _apply_binary_op(self, op: str) -> list[str]:
        if op == "+":  return ["add rax, rcx"]
        if op == "-":  return ["sub rax, rcx"]
        if op == "*":  return ["imul rax, rcx"]
        if op == "/":  return ["cqo", "idiv rcx"]
        if op == "%":  return ["cqo", "idiv rcx", "mov rax, rdx"]
        if op == "==": return ["cmp rax, rcx", "sete al",  "movzx rax, al"]
        if op == "!=": return ["cmp rax, rcx", "setne al", "movzx rax, al"]
        if op == "<":  return ["cmp rax, rcx", "setl al",  "movzx rax, al"]
        if op == ">":  return ["cmp rax, rcx", "setg al",  "movzx rax, al"]
        if op == "<=": return ["cmp rax, rcx", "setle al", "movzx rax, al"]
        if op == ">=": return ["cmp rax, rcx", "setge al", "movzx rax, al"]
        if op == "&&": return ["and rax, rcx"]
        if op == "||": return ["or rax, rcx"]
        return [f"; unknown op {op!r}"]

    def _is_numeric_literal(self, val: object) -> bool:
        try:
            int(str(val))
            return True
        except (ValueError, TypeError):
            pass
        try:
            float(str(val))
            return True
        except (ValueError, TypeError):
            return False

    def _get_offset(self, var_name: str) -> int:
        if var_name in self.var_offsets:
            return self.var_offsets[var_name]
        self.current_offset += 8
        self.var_offsets[var_name] = -self.current_offset
        return self.var_offsets[var_name]
