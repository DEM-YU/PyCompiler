from ast_nodes import (
    ASTNode, ArrayType, AssignStmt, BinaryOp, Block, ForStmt, FuncCall,
    FunctionDecl, Identifier, IfStmt, IndexAssignStmt, IndexExpr,
    Literal, Param, Program, ReturnStmt, StringLiteral, UnaryOp, VarDecl,
    WhileStmt,
)

from error import CompilerError
from lexer import Token, TokenType


class Parser:
    def __init__(self, tokens: list[Token]) -> None:
        self._tokens = tokens
        self.pos = 0
        self.current_token = tokens[0] if tokens else Token(TokenType.EOF, None, 1, 1)

    # ------------------------------------------------------------------
    # Core navigation
    # ------------------------------------------------------------------

    def _peek(self) -> Token:
        return self.current_token

    def _advance(self) -> Token:
        token = self.current_token
        if self.pos + 1 < len(self._tokens):
            self.pos += 1
            self.current_token = self._tokens[self.pos]
        return token

    # ------------------------------------------------------------------
    # Token inspection helpers
    # ------------------------------------------------------------------

    def _check(self, token_type: TokenType) -> bool:
        return self.current_token.type == token_type

    def _match(self, *types: TokenType) -> bool:
        for token_type in types:
            if self._check(token_type):
                self._advance()
                return True
        return False

    def _expect(self, token_type: TokenType, msg: str) -> Token:
        if self._check(token_type):
            return self._advance()
        raise CompilerError(
            message=msg,
            line=self.current_token.line,
            col=self.current_token.col,
        )

    # ------------------------------------------------------------------
    # Expression parsing  (precedence: lowest → highest)
    # Each method calls the one below it in the chain; the while-loop in
    # each level handles left-associativity without recursion.
    # ------------------------------------------------------------------

    def _parse_expression(self) -> ASTNode:
        # Level 1 — logical OR  (lowest precedence binary operator)
        left = self._parse_and()
        while self._check(TokenType.PIPE_PIPE):
            op_token = self._advance()
            right = self._parse_and()
            left = BinaryOp(op=op_token.value, left=left, right=right,
                            line=op_token.line, col=op_token.col)
        return left

    def _parse_and(self) -> ASTNode:
        # Level 2 — logical AND  (binds tighter than ||)
        left = self._parse_equality()
        while self._check(TokenType.AMP_AMP):
            op_token = self._advance()
            right = self._parse_equality()
            left = BinaryOp(op=op_token.value, left=left, right=right,
                            line=op_token.line, col=op_token.col)
        return left

    def _parse_equality(self) -> ASTNode:
        # Level 3 — == and !=
        left = self._parse_comparison()
        while self._check(TokenType.EQ_EQ) or self._check(TokenType.BANG_EQ):
            op_token = self._advance()
            right = self._parse_comparison()
            left = BinaryOp(op=op_token.value, left=left, right=right,
                            line=op_token.line, col=op_token.col)
        return left

    def _parse_comparison(self) -> ASTNode:
        # Level 4 — < > <= >=
        left = self._parse_term()
        while (self._check(TokenType.LT) or self._check(TokenType.GT) or
               self._check(TokenType.LT_EQ) or self._check(TokenType.GT_EQ)):
            op_token = self._advance()
            right = self._parse_term()
            left = BinaryOp(op=op_token.value, left=left, right=right,
                            line=op_token.line, col=op_token.col)
        return left

    def _parse_term(self) -> ASTNode:
        # Level 5 — + and -
        left = self._parse_factor()
        while self._check(TokenType.PLUS) or self._check(TokenType.MINUS):
            op_token = self._advance()
            right = self._parse_factor()
            left = BinaryOp(op=op_token.value, left=left, right=right,
                            line=op_token.line, col=op_token.col)
        return left

    def _parse_factor(self) -> ASTNode:
        # Level 6 — * / %
        left = self._parse_unary()
        while (self._check(TokenType.STAR) or self._check(TokenType.SLASH) or
               self._check(TokenType.PERCENT)):
            op_token = self._advance()
            right = self._parse_unary()
            left = BinaryOp(op=op_token.value, left=left, right=right,
                            line=op_token.line, col=op_token.col)
        return left

    def _parse_unary(self) -> ASTNode:
        # Level 7 — unary - and !  (right-associative via recursion)
        if self._check(TokenType.BANG) or self._check(TokenType.MINUS):
            op_token = self._advance()
            operand = self._parse_unary()
            return UnaryOp(op=op_token.value, operand=operand,
                           line=op_token.line, col=op_token.col)
        return self._parse_primary()

    def _parse_primary(self) -> ASTNode:
        # Level 8 — literals, identifiers, calls, parenthesised expressions
        token = self._peek()

        if self._match(TokenType.INTEGER):
            return Literal(value=token.value, value_type="int",
                           line=token.line, col=token.col)
        if self._match(TokenType.FLOAT_LITERAL):
            return Literal(value=token.value, value_type="float",
                           line=token.line, col=token.col)
        if self._match(TokenType.STRING_LITERAL):
            return StringLiteral(value=token.value,
                                 line=token.line, col=token.col)
        if self._match(TokenType.TRUE):
            return Literal(value=True, value_type="bool",
                           line=token.line, col=token.col)
        if self._match(TokenType.FALSE):
            return Literal(value=False, value_type="bool",
                           line=token.line, col=token.col)
        if self._check(TokenType.ID):
            return self._parse_identifier_or_call()
        if self._match(TokenType.LPAREN):
            expr = self._parse_expression()
            self._expect(TokenType.RPAREN, "expected ')' after expression")
            return expr
        raise CompilerError(
            message=f"unexpected token '{token.value}'",
            line=token.line,
            col=token.col,
        )

    def _parse_identifier_or_call(self) -> ASTNode:
        # Disambiguates ID, ID "(" args ")", and ID "[" expr "]" with 1-token lookahead.
        id_token = self._advance()
        if self._match(TokenType.LPAREN):
            args: list[ASTNode] = []
            if not self._check(TokenType.RPAREN):
                args.append(self._parse_expression())
                while self._match(TokenType.COMMA):
                    args.append(self._parse_expression())
            self._expect(TokenType.RPAREN, "expected ')' after argument list")
            return FuncCall(name=id_token.value, args=args,
                            line=id_token.line, col=id_token.col)
        if self._match(TokenType.LBRACKET):
            index = self._parse_expression()
            self._expect(TokenType.RBRACKET, "expected ']' after index")
            return IndexExpr(name=id_token.value, index=index,
                             line=id_token.line, col=id_token.col)
        return Identifier(name=id_token.value, line=id_token.line, col=id_token.col)

    # ------------------------------------------------------------------
    # Type and keyword helpers
    # ------------------------------------------------------------------

    def _check_type_keyword(self) -> bool:
        return self.current_token.type in (
            TokenType.INT, TokenType.FLOAT,
            TokenType.BOOL, TokenType.STRING,
        )

    def _parse_type(self) -> str:
        if self._check_type_keyword():
            return str(self._advance().value)
        raise CompilerError(
            message=f"expected type but got '{self.current_token.value}'",
            line=self.current_token.line,
            col=self.current_token.col,
        )

    def _parse_array_type(self) -> ArrayType:
        # grammar: "[" type ";" INTEGER "]"   e.g.  [int; 10]
        bracket = self._expect(TokenType.LBRACKET, "expected '['")
        elem_type = self._parse_type()
        self._expect(TokenType.SEMICOLON, "expected ';' after element type in array type")
        size_tok = self._expect(TokenType.INTEGER, "expected integer size in array type")
        self._expect(TokenType.RBRACKET, "expected ']' after array size")
        return ArrayType(element_type=elem_type, size=int(size_tok.value),
                         line=bracket.line, col=bracket.col)

    # ------------------------------------------------------------------
    # Block
    # ------------------------------------------------------------------

    def _parse_block(self) -> Block:
        brace = self._expect(TokenType.LBRACE, "expected '{'")
        statements: list[ASTNode] = []
        while not self._check(TokenType.RBRACE) and not self._check(TokenType.EOF):
            statements.append(self._parse_statement())
        self._expect(TokenType.RBRACE, "expected '}'")
        return Block(statements=statements, line=brace.line, col=brace.col)

    # ------------------------------------------------------------------
    # Statement dispatcher
    # ------------------------------------------------------------------

    def _parse_statement(self) -> ASTNode:
        if self._check(TokenType.LET):
            return self._parse_let_decl()
        if self._check_type_keyword():
            return self._parse_var_decl()
        if self._check(TokenType.LBRACKET):
            return self._parse_array_var_decl()
        if self._check(TokenType.IF):
            return self._parse_if_stmt()
        if self._check(TokenType.WHILE):
            return self._parse_while_stmt()
        if self._check(TokenType.FOR):
            return self._parse_for_stmt()
        if self._check(TokenType.RETURN):
            return self._parse_return_stmt()
        if self._check(TokenType.LBRACE):
            return self._parse_block()
        return self._parse_assign_or_expr_stmt()

    # ------------------------------------------------------------------
    # Individual statement parsers
    # ------------------------------------------------------------------

    def _parse_let_decl(self) -> VarDecl:
        # grammar: "let" ID ":" (array_type | scalar_type) ("=" expression)? ";"
        let_token = self._advance()   # consume "let"
        name_token = self._expect(TokenType.ID, "expected variable name after 'let'")
        self._expect(TokenType.COLON, "expected ':' after variable name")
        if self._check(TokenType.LBRACKET):
            var_type: str | ArrayType = self._parse_array_type()
        else:
            var_type = self._parse_type()
        initializer: ASTNode | None = None
        if self._match(TokenType.ASSIGN):
            initializer = self._parse_expression()
        self._expect(TokenType.SEMICOLON, "expected ';' after variable declaration")
        return VarDecl(var_type=var_type, name=name_token.value,
                       initializer=initializer,
                       line=let_token.line, col=let_token.col)

    def _parse_var_decl(self) -> VarDecl:
        # grammar: type ID ("=" expression)? ";"
        type_token = self._advance()
        name_token = self._expect(TokenType.ID, "expected variable name after type")
        initializer: ASTNode | None = None
        if self._match(TokenType.ASSIGN):
            initializer = self._parse_expression()
        self._expect(TokenType.SEMICOLON, "expected ';' after variable declaration")
        return VarDecl(var_type=str(type_token.value), name=name_token.value,
                       initializer=initializer,
                       line=type_token.line, col=type_token.col)

    def _parse_array_var_decl(self) -> VarDecl:
        # grammar: "[" type ";" INTEGER "]" ID ("=" expression)? ";"
        array_type = self._parse_array_type()
        name_token = self._expect(TokenType.ID, "expected variable name after array type")
        initializer: ASTNode | None = None
        if self._match(TokenType.ASSIGN):
            initializer = self._parse_expression()
        self._expect(TokenType.SEMICOLON, "expected ';' after array declaration")
        return VarDecl(var_type=array_type, name=name_token.value,
                       initializer=initializer,
                       line=array_type.line, col=array_type.col)

    def _parse_index_assign_stmt(self) -> IndexAssignStmt:
        # grammar: ID "[" expression "]" "=" expression ";"
        name_token = self._advance()   # consume ID
        self._expect(TokenType.LBRACKET, "expected '['")
        index = self._parse_expression()
        self._expect(TokenType.RBRACKET, "expected ']' after index")
        self._expect(TokenType.ASSIGN, "expected '=' in index assignment")
        value = self._parse_expression()
        self._expect(TokenType.SEMICOLON, "expected ';' after index assignment")
        return IndexAssignStmt(name=name_token.value, index=index, value=value,
                               line=name_token.line, col=name_token.col)

    def _parse_assign_or_expr_stmt(self) -> ASTNode:
        # 2-token lookahead to choose between plain assign, index assign, and expr.
        is_assign = (
            self._check(TokenType.ID)
            and self.pos + 1 < len(self._tokens)
            and self._tokens[self.pos + 1].type == TokenType.ASSIGN
        )
        if is_assign:
            name_token = self._advance()
            self._advance()   # consume "="
            value = self._parse_expression()
            self._expect(TokenType.SEMICOLON, "expected ';' after assignment")
            return AssignStmt(name=name_token.value, value=value,
                              line=name_token.line, col=name_token.col)
        is_index_assign = (
            self._check(TokenType.ID)
            and self.pos + 1 < len(self._tokens)
            and self._tokens[self.pos + 1].type == TokenType.LBRACKET
        )
        if is_index_assign:
            return self._parse_index_assign_stmt()
        expr = self._parse_expression()
        self._expect(TokenType.SEMICOLON, "expected ';' after expression")
        return expr

    def _parse_if_stmt(self) -> IfStmt:
        # grammar: "if" expr block ("else" "if" expr block)* ("else" block)?
        if_token = self._expect(TokenType.IF, "expected 'if'")
        condition = self._parse_expression()
        then_block = self._parse_block()
        else_ifs: list[tuple[ASTNode, Block]] = []
        else_block: Block | None = None
        while self._match(TokenType.ELSE):
            if self._check(TokenType.IF):
                self._advance()   # consume "if"
                ei_cond = self._parse_expression()
                ei_block = self._parse_block()
                else_ifs.append((ei_cond, ei_block))
            else:
                else_block = self._parse_block()
                break
        return IfStmt(condition=condition, then_block=then_block,
                      else_ifs=else_ifs, else_block=else_block,
                      line=if_token.line, col=if_token.col)

    def _parse_while_stmt(self) -> WhileStmt:
        # grammar: "while" expression block
        while_token = self._expect(TokenType.WHILE, "expected 'while'")
        condition = self._parse_expression()
        body = self._parse_block()
        return WhileStmt(condition=condition, body=body,
                         line=while_token.line, col=while_token.col)

    def _parse_for_stmt(self) -> ForStmt:
        # grammar: "for" ID "=" expr ";" expr ";" ID "=" expr block
        for_token = self._expect(TokenType.FOR, "expected 'for'")
        var_token = self._expect(TokenType.ID, "expected loop variable after 'for'")
        self._expect(TokenType.ASSIGN, "expected '=' in for-loop initializer")
        init_value = self._parse_expression()
        self._expect(TokenType.SEMICOLON, "expected ';' after initializer")
        condition = self._parse_expression()
        self._expect(TokenType.SEMICOLON, "expected ';' after condition")
        step_var = self._expect(TokenType.ID, "expected variable name in step")
        self._expect(TokenType.ASSIGN, "expected '=' in for-loop step")
        step_value = self._parse_expression()
        body = self._parse_block()
        return ForStmt(var_name=var_token.value, init_value=init_value,
                       condition=condition, step_var=step_var.value,
                       step_value=step_value, body=body,
                       line=for_token.line, col=for_token.col)

    def _parse_return_stmt(self) -> ReturnStmt:
        # grammar: "return" expression? ";"
        ret_token = self._expect(TokenType.RETURN, "expected 'return'")
        value: ASTNode | None = None
        if not self._check(TokenType.SEMICOLON):
            value = self._parse_expression()
        self._expect(TokenType.SEMICOLON, "expected ';' after return")
        return ReturnStmt(value=value, line=ret_token.line, col=ret_token.col)

    # ------------------------------------------------------------------
    # Function declaration
    # ------------------------------------------------------------------

    def _parse_param(self) -> Param:
        # grammar: ID ":" type
        name_token = self._expect(TokenType.ID, "expected parameter name")
        self._expect(TokenType.COLON, "expected ':' after parameter name")
        param_type = self._parse_type()
        return Param(name=name_token.value, param_type=param_type,
                     line=name_token.line, col=name_token.col)

    def _parse_param_list(self) -> list[Param]:
        params: list[Param] = []
        if self._check(TokenType.RPAREN):
            return params
        params.append(self._parse_param())
        while self._match(TokenType.COMMA):
            params.append(self._parse_param())
        return params

    def _parse_func_decl(self) -> FunctionDecl:
        # grammar: "fn" ID "(" param_list? ")" ("->" type)? block
        fn_token = self._expect(TokenType.FN, "expected 'fn'")
        name_token = self._expect(TokenType.ID, "expected function name")
        self._expect(TokenType.LPAREN, "expected '(' after function name")
        params = self._parse_param_list()
        self._expect(TokenType.RPAREN, "expected ')' after parameters")
        return_type: str | None = None
        if self._match(TokenType.ARROW):
            return_type = self._parse_type()
        body = self._parse_block()
        return FunctionDecl(name=name_token.value, params=params,
                            return_type=return_type, body=body,
                            line=fn_token.line, col=fn_token.col)

    # ------------------------------------------------------------------
    # Program entry point
    # ------------------------------------------------------------------

    def _parse_declaration(self) -> ASTNode:
        # top_decl ::= func_decl | statement
        if self._check(TokenType.FN):
            return self._parse_func_decl()
        return self._parse_statement()

    def _parse_program(self) -> Program:
        first = self._peek()
        body: list[ASTNode] = []
        while not self._check(TokenType.EOF):
            body.append(self._parse_declaration())
        return Program(body=body, line=first.line, col=first.col)

    def parse(self) -> Program:
        return self._parse_program()


# ---------------------------------------------------------------------------
# CLI entry point — python parser.py <file.nv>
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from ast_nodes import ASTPrinter
    from lexer import Lexer

    if len(sys.argv) != 2:
        print("usage: python parser.py <source.nv>", file=sys.stderr)
        sys.exit(1)

    try:
        with open(sys.argv[1]) as f:
            source = f.read()
    except FileNotFoundError:
        print(f"error: file not found: {sys.argv[1]}", file=sys.stderr)
        sys.exit(1)

    try:
        lexer = Lexer(source)
        tokens = []
        while True:
            tok = lexer.get_next_token()
            tokens.append(tok)
            if tok.type == TokenType.EOF:
                break
        program = Parser(tokens).parse()
        program.accept(ASTPrinter())
    except CompilerError as e:
        print(e.format(), file=sys.stderr)
        sys.exit(1)
