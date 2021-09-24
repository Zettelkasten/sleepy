from sleepy.ast import TopLevelAst, TranslationUnitAst
from sleepy.sleepy_lexer import SLEEPY_LEXER
from sleepy.sleepy_parser import SLEEPY_ATTR_GRAMMAR, add_preamble_to_ast, SLEEPY_PARSER


def parse_ast(program: str, add_preamble=True) -> TranslationUnitAst:
  print('---- input program:')
  print(program)
  tokens, tokens_pos = SLEEPY_LEXER.tokenize(program)
  analysis, program_eval = SLEEPY_PARSER.parse_syn_attr_analysis(SLEEPY_ATTR_GRAMMAR, program, tokens, tokens_pos)
  file_ast: TopLevelAst = program_eval['ast']
  assert isinstance(file_ast, TopLevelAst)

  if add_preamble:
    tu_ast: TranslationUnitAst = add_preamble_to_ast(file_ast)
  else:
    tu_ast: TranslationUnitAst = TranslationUnitAst(file_ast.pos, [file_ast])

  assert isinstance(tu_ast, TranslationUnitAst)
  return tu_ast
