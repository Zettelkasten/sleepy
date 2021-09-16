from typing import Tuple, List

from sleepy.ast import AssignStatementAst, AbstractSyntaxTree
from sleepy.program_conversions.string_functions import replace, trim_whitespace
from sleepy.sleepy_parser import make_program_ast


# Note: Annotation are not included in the AssignStatementAst pos, so we do not need to handle them
def walk_ast(ast: AbstractSyntaxTree, input_program: str) -> List[Tuple[slice, str]]:
  if isinstance(ast, AssignStatementAst):
    if ast.declared_var_type is None:
      return []

    rhs = input_program[ast.var_val.pos.from_pos:ast.var_val.pos.to_pos]
    lhs = input_program[ast.var_target.pos.from_pos:ast.var_target.pos.to_pos-1]
    type_decl = input_program[ast.declared_var_type.pos.from_pos:ast.declared_var_type.pos.to_pos-1]

    at = trim_whitespace(input_program, slice(ast.pos.from_pos, ast.pos.to_pos))

    return [(at, f'{lhs}: {type_decl} = {rhs};')]
    pass
  else:
    return [el for child in ast.children() for el in walk_ast(child, input_program)]


def covert_program(input_program: str):
  ast = make_program_ast(input_program, False)
  replacements = walk_ast(ast, input_program)
  result = replace(input_program, replacements)
  print(result)


if __name__ == "__main__":
  with open("../../usage_examples/heat_simulation.slp") as program_file:
    covert_program(program_file.read())
