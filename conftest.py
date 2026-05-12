# Empty — pytest finds this file and sets the project root as rootdir,
# which causes it to add this directory to sys.path automatically.
# That makes `from error import ...` and `from lexer import ...` work
# regardless of which directory you run pytest from.
