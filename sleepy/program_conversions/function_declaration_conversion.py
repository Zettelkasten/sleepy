from typing import List, Tuple

from sleepy.ast import AbstractSyntaxTree, FunctionDeclarationAst, AnnotationAst
from sleepy.program_conversions.string_functions import replace, trim_whitespace
from sleepy.sleepy_parser import make_program_ast


def covert_program(input_program: str):
  ast = make_program_ast(input_program, False)
  replacements = walk_ast(ast, input_program)
  result = replace(input_program, replacements)
  print(result)

def walk_ast(ast: AbstractSyntaxTree, input_program: str) -> List[Tuple[slice, str]]:
  if isinstance(ast, FunctionDeclarationAst):

    arg_list_str = ''
    for identifier, arg_type, annotations in zip(ast.arg_identifiers, ast.arg_types, ast.arg_annotations):
      arg_list_str += f'{annotations_to_string(annotations)}{identifier}: {input_program[arg_type.pos.from_pos:arg_type.pos.to_pos-1]}, '

    arg_list_str = arg_list_str[:-2] if len(ast.arg_identifiers) > 0 else arg_list_str

    body = input_program[ast.body_scope.pos.from_pos:ast.body_scope.pos.to_pos] if ast.body_scope is not None else ''
    keyword = 'extern_func' if ast.is_extern else 'func'
    signature: str = f'{keyword} {ast.identifier}({arg_list_str}) '
    if ast.return_type is not None:
      return_decl = f'-> {annotations_to_string(ast.return_annotation_list)} {input_program[ast.return_type.pos.from_pos:ast.return_type.pos.to_pos]}'
    else:
      return_decl = ''
    signature_end = ';' if ast.is_extern else ''


    at = trim_whitespace(input_program, slice(ast.pos.from_pos, ast.pos.to_pos))

    return [(at, signature + return_decl + signature_end + body)]
  else:
    return [el for child in ast.children() for el in walk_ast(child, input_program)]


def annotations_to_string(annotations: List[AnnotationAst]) -> str:
  return ' '.join([f'@{a.identifier}' for a in annotations]) + ' ' if len(annotations) > 0 else ''

if __name__ == "__main__":
  with open("../../usage_examples/opengl.slp") as program_file:
    program = program_file.read()
    covert_program(program)
