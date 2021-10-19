from pathlib import Path

from sleepy.ast import FileAst, TranslationUnitAst
from sleepy.sleepy_lexer import SLEEPY_LEXER
from sleepy.sleepy_parser import SLEEPY_ATTR_GRAMMAR, SLEEPY_PARSER


def make_program_ast(program: str) -> FileAst:
  tokens, tokens_pos = SLEEPY_LEXER.tokenize(program)
  _, root_eval = SLEEPY_PARSER.parse_syn_attr_analysis(SLEEPY_ATTR_GRAMMAR, program, tokens, tokens_pos)
  program_ast = root_eval['ast']
  assert isinstance(program_ast, FileAst)
  return program_ast

def make_preamble_ast() -> FileAst:
  preamble_path = Path(__file__).parent.joinpath("std/preamble.slp").resolve()
  with open(preamble_path) as preamble_file:
    preamble_program = preamble_file.read()
  file_ast = make_program_ast(preamble_program)
  file_ast.file_path = preamble_path
  return file_ast

def make_ast(program: str, add_preamble=True) -> TranslationUnitAst:
  file_asts = [make_preamble_ast()] if add_preamble else []
  file_asts.append(make_program_ast(program))

  return TranslationUnitAst.from_file_asts(file_asts)

