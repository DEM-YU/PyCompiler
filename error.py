class CompilerError(Exception):
    def __init__(
        self,
        message: str,
        line: int,
        col: int,
        source_line: str | None = None,
        file_name: str | None = None,
    ) -> None:
        self.message = message
        self.line = line
        self.col = col
        self.source_line = source_line
        self.file_name = file_name
        super().__init__(message)

    def format(self) -> str:
        header = self._format_header()
        if self.source_line is None:
            return header
        pointer = self._format_pointer()
        return f"{header}\n    {self.source_line}\n    {pointer}"

    def _format_header(self) -> str:
        location = self.file_name if self.file_name else "<input>"
        return f"{location}:{self.line}:{self.col}: {self.message}"

    def _format_pointer(self) -> str:
        # col is 1-based; subtract 1 so the caret sits under the right character
        spaces = " " * (self.col - 1)
        return f"{spaces}^"
