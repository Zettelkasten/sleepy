from sleepy.ast import TopLevelAst
from sleepy.sleepy_lexer import SLEEPY_LEXER
from sleepy.sleepy_parser import SLEEPY_ATTR_GRAMMAR, add_preamble_to_ast, SLEEPY_PARSER


def parse_ast(program: str, add_preamble=True) -> TopLevelAst:
  print('---- input program:')
  print(program)
  tokens, tokens_pos = SLEEPY_LEXER.tokenize(program)
  analysis, eval = SLEEPY_PARSER.parse_syn_attr_analysis(SLEEPY_ATTR_GRAMMAR, program, tokens, tokens_pos)
  ast = eval['ast']
  assert isinstance(ast, TopLevelAst)
  if add_preamble:
    ast = add_preamble_to_ast(ast)
    assert isinstance(ast, TopLevelAst)
  return ast
