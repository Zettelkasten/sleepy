from sleepy.ast import SLEEPY_LEXER, SLEEPY_PARSER, SLEEPY_ATTR_GRAMMAR, TopLevelAst, add_preamble_to_ast


def parse_ast(program, add_preamble=True):
  """
  :param str program:
  :param bool add_preamble:
  :rtype: TopLevelAst
  """
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