import _setup_test_env  # noqa
import sys
import unittest
import better_exchook
from nose.tools import assert_equal, assert_raises, assert_equals

from sleepy.errors import ParseError
from sleepy.lexer import LexerGenerator
from sleepy.parser import ParserGenerator, make_first1_sets, get_first1_set_for_word
from sleepy.grammar import EPSILON, Production, Grammar, AttributeGrammar, SyntaxTree, \
  get_token_word_from_tokens_pos, TreePosition, get_line_col_from_pos
from sleepy.semantic import AttributeEvalGenerator


def test_Grammar():
  g = Grammar(
    Production('S2', 'S'),
    Production('S', 'A', 'b'),
    Production('A', 'a', 'a')
  )
  assert_equal(set(g.symbols), {'S2', 'S', 'A', 'a', 'b'})
  assert_equal(set(g.terminals), {'a', 'b'})
  assert_equal(set(g.non_terminals), {'S2', 'S', 'A'})
  assert_equal(set(g.get_prods_for('S')), {g.prods[1]})


def test_AttributeGrammar_syn():
  g = Grammar(
    Production('S', 'S', '+', 'S'),
    Production('S', 'zero'),
    Production('S', 'digit')
  )
  attr_g = AttributeGrammar(
    g,
    syn_attrs = {'res'},
    prod_attr_rules = [
      {'res.0': lambda res: res(1) + res(3)},
      {'res.0': lambda: 0},
      {'res.0': lambda res: res(1)}
    ],
    terminal_attr_rules = {
      'digit': {'res.0': lambda word: int(word)}
    }
  )
  assert_equal(attr_g.attrs, {'res'})
  assert_equal(attr_g.syn_attrs, {'res'})
  assert_equal(attr_g.inh_attrs, set())
  assert_equal(attr_g.get_terminal_syn_attr_eval('digit', 6), {'res': 6})
  assert_equal(attr_g.get_terminal_syn_attr_eval('zero', 0), {})
  assert_equal(attr_g.eval_prod_syn_attr(g.prods[0], {}, [{'res': 4}, {}, {'res': 7}]), {'res': 4 + 7})
  assert_equal(attr_g.eval_prod_syn_attr(g.prods[2], {}, [{'res': 8}]), {'res': 8})
  assert_equal(attr_g.eval_prod_syn_attr(g.prods[1], {}, [{}]), {'res': 0})


def test_get_line_col_from_pos():
  assert_equal(
    get_line_col_from_pos('oh, X marks the error.', error_pos=4, num_before_context_lines=1, num_after_context_lines=1),
    (1, 5, {1: 'oh, X marks the error.'}))

  word, pos = 'fine\nbad X bad\nfine2', 9
  line, col, context_lines = get_line_col_from_pos(
    word, error_pos=pos, num_before_context_lines=1, num_after_context_lines=1)
  error = ParseError(word, pos, 'Sample error message')
  print(error)
  assert_equal((line, col, context_lines), (2, 5, {1: 'fine', 2: 'bad X bad', 3: 'fine2'}))

  word, pos = '\n'.join(['line%s' % l for l in range(1, 13)]), 6 * 3 + 2
  line, col, context_lines = get_line_col_from_pos(
    word, error_pos=pos, num_before_context_lines=1, num_after_context_lines=1)
  error = ParseError(word, pos, 'Another sample error message')
  print(error)
  assert_equal((line, col, context_lines), (4, 3, {l: 'line%s' % l for l in range(3, 6)}))


def test_make_first1_sets():
  g = Grammar(
    Production('S', 'S', 'O', 'S'),
    Production('S', '(', 'S', ')'),
    Production('S', '0'),
    Production('S', '1'),
    Production('O', '+'),
    Production('O', '*'),
  )
  assert_equal(set(g.terminals), {'0', '1', '(', ')', '+', '*'})
  assert_equal(set(g.non_terminals), {'S', 'O'})
  first1 = make_first1_sets(g)
  print('first1 sets:', first1)
  assert_equal(first1['S'], {'0', '1', '('})
  assert_equal(first1['O'], {'+', '*'})
  assert_equal(get_first1_set_for_word(first1, ('(', 'S')), {'('})


def test_make_first1_sets_epsilon():
  g = Grammar(
    Production('S', 'A', 'B'),
    Production('A', 'a'),
    Production('A'),
    Production('B', 'A'),
    Production('B', 'b', 'c')
  )
  first1 = make_first1_sets(g)
  print('first1 sets:', first1)
  assert_equal(first1['A'], {EPSILON, 'a'})
  assert_equal(first1['B'], {EPSILON, 'a', 'b'})
  assert_equal(first1['S'], {EPSILON, 'a', 'b'})
  assert_equal(get_first1_set_for_word(first1, ('B', 'c')), {'a', 'b', 'c'})


def test_ParserGenerator_simple():
  g = Grammar(
    Production('S2', 'S'),
    Production('S', 'A', 'b'),
    Production('A', 'a', 'a')
  )
  parser = ParserGenerator(g)
  assert_equal(parser.parse_analysis('aab', ['a', 'a', 'b'], [0, 1, 2]), (g.prods[0], g.prods[1], g.prods[2]))
  with assert_raises(ParseError):
    parser.parse_analysis('aaab', ['a', 'a', 'a', 'b'], [0, 1, 2, 3])
  with assert_raises(ParseError):
    parser.parse_analysis('aabb', ['a', 'a', 'b', 'b'], [0, 1, 2, 3])
  with assert_raises(ParseError):
    parser.parse_analysis('aa', ['a', 'a'], [0, 1])


def test_ParserGenerator_simple_left_recursive():
  g = Grammar(
    Production('S2', 'S'),
    Production('S', 'S', 'a'),
    Production('S', 'a')
  )
  parser = ParserGenerator(g)
  for count in [1, 2, 3, 20, 100, 10000]:
    assert_equal(
      parser.parse_analysis(
        'a' * count, ['a'] * count, list(range(count))), (g.prods[0],) + (count-1) * (g.prods[1],) + (g.prods[2],))
  with assert_raises(ParseError):
    parser.parse_analysis('', [], [])


def test_ParserGenerator_simple_left_recursive_epsilon():
  g = Grammar(
    Production('S2', 'S'),
    Production('S', 'S', 'a'),
    Production('S')
  )
  parser = ParserGenerator(g)
  for count in [0, 1, 2, 3, 20, 100, 10000]:
    assert_equal(
      parser.parse_analysis(
        'a' * count, ['a'] * count, list(range(count))), (g.prods[0],) + count * (g.prods[1],) + (g.prods[2],))


def test_ParserGenerator_simple_lookahead():
  g = Grammar(
    Production('S2', 'S'),
    Production('S', 'b', 'b'),
    Production('S', 'a', 'b')
  )
  parser = ParserGenerator(g)
  assert_equal(parser.parse_analysis('bb', ['b', 'b'], [0, 1]), (g.prods[0], g.prods[1]))
  assert_equal(parser.parse_analysis('ab', ['a', 'b'], [0, 1]), (g.prods[0], g.prods[2]))
  with assert_raises(ParseError):
    parser.parse_analysis('ba', ['b', 'a'], [0, 1])


def test_ParserGenerator_cfg():
  g = Grammar(
    Production('Grammar', 'Decl'),
    Production('Decl', 'Prod', ';', 'Decl'),
    Production('Decl', 'Prod', '\n', 'Decl'),
    Production('Decl', 'Prod'),
    Production('Prod', 'Symb', '->', 'Right'),
    Production('Right', 'Symbs', '|', 'Symbs'),
    Production('Right', 'Symbs'),
    Production('Symbs', 'Symb', 'Symbs'),
    Production('Symbs'),
    Production('Symb', 'A'),
    Production('Symb', 'B'),
    Production('Symb', 'C'),
    Production('Symb', 'a'),
    Production('Symb', 'b'),
    Production('Symb', 'c')
  )
  parser = ParserGenerator(g)
  tokens = ['A', '->', 'B', 'c', '|', 'B', ';', 'B', '->', 'c', 'a']
  print('tokenized word:', tokens)
  analysis = parser.parse_analysis(''.join(tokens), tokens, list(range(len(tokens))))
  print('right-most analysis:', analysis)


def test_ParserGenerator_cfg_with_lexer():
  g = Grammar(
    Production('Grammar', 'Decl'),
    Production('Decl', 'Prod', ';', 'Decl'),
    Production('Decl', 'Prod'),
    Production('Prod', 'Symb', '->', 'Right'),
    Production('Right', 'Symbs', '|', 'Symbs'),
    Production('Right', 'Symbs'),
    Production('Symbs', 'Symb', 'Symbs'),
    Production('Symbs'),
  )
  lexer = LexerGenerator([';', '->', '|', 'Symb'], ['; *|[\n\r ]+', ' *\\-> *', ' *\\| *', '([a-z]|[A-Z])+'])
  parser = ParserGenerator(g)
  word = 'A->Bc|B; B->ca'
  print('input word:', word)
  tokens, tokens_pos = lexer.tokenize(word)
  print('tokenized word:', tokens, 'with decomposition', tokens_pos)
  analysis = parser.parse_analysis(word, tokens, tokens_pos)
  print('right-most analysis:', analysis)


def test_ParserGenerator_arithmetic():
  # left associative, with operator precedence
  op_plus = Production('Sum', 'Sum', '+', 'Prod')
  op_minus = Production('Sum', 'Sum', '-', 'Prod')
  op_mult = Production('Prod', 'Prod', '*', 'Term')
  op_div = Production('Prod', 'Prod', '/', 'Term')
  const_terms = {Production('Term', str(i)): i for i in range(11)}  # we can clean this up once we have proper tokens
  g = Grammar(*[
    Production('Expr', 'Sum'),
    op_plus, op_minus,
    Production('Sum', 'Prod'),
    op_mult, op_div,
    Production('Prod', 'Term'),
    Production('Term', '(', 'Sum', ')')]
    + list(const_terms.keys())
  )
  parser = ParserGenerator(g)

  def evaluate(word):
    """
    :param str word:
    """
    assert isinstance(word, str)
    tokens = list(word)
    tokens_pos = list(range(len(tokens)))
    print('input word:', word)
    analysis = parser.parse_analysis(word, tokens, tokens_pos)
    print('analysis:', analysis)

    stack = []
    for prod in reversed(analysis):
      if prod in const_terms:
        stack.append(const_terms[prod])
      elif prod == op_plus:
        b, a = stack.pop(-1), stack.pop(-1)
        stack.append(a + b)
      elif prod == op_minus:
        b, a = stack.pop(-1), stack.pop(-1)
        stack.append(a - b)
      elif prod == op_mult:
        b, a = stack.pop(-1), stack.pop(-1)
        stack.append(a * b)
      elif prod == op_div:
        b, a = stack.pop(-1), stack.pop(-1)
        stack.append(a / b)
    assert len(stack) == 1
    result_value = stack[0]
    print('result:', result_value)
    assert result_value == eval(word)

  evaluate('1*2+3')
  evaluate('1+2*3')
  evaluate('(1+2)*3')
  evaluate('1+2+3')
  evaluate('1+2-3')
  evaluate('1-2+3')
  evaluate('3-2-1')
  evaluate('1*2+3*4')
  evaluate('4/2-1')


def test_ParserGenerator_arithmetic_syn():
  import math
  lexer = LexerGenerator(
    token_names=['(', ')', '+', '-', '*', '**', '/', 'const', 'func'],
    token_regex_table=[
      '\\(', '\\)', '\\+', '\\-', '\\*', '\\*\\*', '/', '(0|[1-9][0-9]*)(\\.[0-9]+)?', '([a-z]|[A-Z])+']
  )
  # left associative (except ** that is right associative), with operator precedence
  g = Grammar(
    Production('Expr', 'Sum'),
    Production('Sum', 'Sum', '+', 'Prod'),
    Production('Sum', 'Sum', '-', 'Prod'),
    Production('Sum', 'Prod'),
    Production('Prod', 'Prod', '*', 'Pow'),
    Production('Prod', 'Prod', '/', 'Pow'),
    Production('Prod', 'Pow'),
    Production('Pow', 'Term', '**', 'Pow'),
    Production('Pow', 'Term'),
    Production('Term', '(', 'Sum', ')'),
    Production('Term', 'func', '(', 'Sum', ')'),
    Production('Term', 'const')
  )
  attr_g = AttributeGrammar(
    g,
    syn_attrs={'res', 'name'},
    prod_attr_rules=[
      {'res': lambda res: res(1)},
      {'res': lambda res: res(1) + res(3)},
      {'res': lambda res: res(1) - res(3)},
      {'res': lambda res: res(1)},
      {'res': lambda res: res(1) * res(3)},
      {'res': lambda res: res(1) / res(3)},
      {'res': lambda res: res(1)},
      {'res': lambda res: res(1) ** res(3)},
      {'res': lambda res: res(1)},
      {'res': lambda res: res(2)},
      {'res': lambda res, name: getattr(math, name(1))(res(3))},
      {'res': lambda res: res(1)},
    ],
    terminal_attr_rules={
      'const': {'res': lambda lit: float(lit)},
      'func': {'name': lambda lit: lit}
    }
  )
  parser = ParserGenerator(g)

  def evaluate(word):
    print('----')
    print('input word:', word)
    tokens, tokens_pos = lexer.tokenize(word)
    print('tokens:', tokens, 'with decomposition', tokens_pos)
    analysis, result = parser.parse_syn_attr_analysis(attr_g, word, tokens, tokens_pos)
    print('result:', result['res'])
    # import common operator names for python eval()
    sin, cos, tan, exp, sqrt = math.sin, math.cos, math.tan, math.exp, math.sqrt  # noqa
    assert_equal(result['res'], eval(word))

  for word in [
    '1*2+3', '5-3', '(1+2)*3', '1+2+3', '1+2-3', '1-2+3', '3-2-1', '1*2+3*4', '4/2-1', 'sin(3.1415)', '2**2**3',
    '(2**2)**3', '(3**2+4**2)**0.5']:
    evaluate(word)


def test_ParserGenerator_regex():
  from sleepy.regex import REGEX_PARSER, tokenize_regex, REGEX_LIT_OP, REGEX_LIT_ANY_OP, REGEX_CHOICE_OP, \
    REGEX_LITS_SINGLE_OP, REGEX_LITS_MULTIPLE_OP, REGEX_CONCAT_OP, REGEX_OPTIONAL_OP, REGEX_RANGE_OP, \
    REGEX_RANGE_LITS_OP, REGEX_REPEAT_OP, REGEX_REPEAT_EXISTS_OP, REGEX_INV_RANGE_OP, REGEX_INV_RANGE_LITS_OP, \
    REGEX_LIT_TOKEN
  parser = REGEX_PARSER

  def evaluate(word, target_value):
    """
    :param str word:
    :param set[str] target_value:
    """
    assert isinstance(word, str)
    tokens, tokens_pos = tokenize_regex(word)
    print('word:', word)
    print('tokens:', tokens)
    analysis = parser.parse_analysis(word, tokens, tokens_pos)
    print('right-most analysis:', analysis)

    stack = []
    pos = 0

    def next_literal_name():
      nonlocal pos
      while tokens[pos] != REGEX_LIT_TOKEN:
        pos += 1
      name = get_token_word_from_tokens_pos(word, tokens_pos, pos)
      pos += 1
      return name

    for prod in reversed(analysis):
      if prod in {REGEX_LIT_OP, REGEX_LITS_SINGLE_OP}:
        stack.append({next_literal_name()})
      elif prod == REGEX_CHOICE_OP:
        b, a = stack.pop(-1), stack.pop(-1)
        stack.append(a | b)
      elif prod == REGEX_LITS_MULTIPLE_OP:
        b, a = {next_literal_name()}, stack.pop()
        stack.append(a | b)
      elif prod == REGEX_CONCAT_OP:
        b, a = stack.pop(-1), stack.pop(-1)
        stack.append(set(i + j for i in a for j in b))
      elif prod == REGEX_OPTIONAL_OP:
        a = stack.pop(-1)
        stack.append(a | {''})
      elif prod == REGEX_RANGE_OP:
        a, b = next_literal_name(), next_literal_name()
        stack.append({chr(c) for c in range(ord(a), ord(b) + 1)})
      elif prod == REGEX_RANGE_LITS_OP:
        pass  # already handled by REGEX_LITS_OP
      elif prod in {
        REGEX_REPEAT_OP, REGEX_REPEAT_EXISTS_OP, REGEX_INV_RANGE_OP, REGEX_INV_RANGE_LITS_OP, REGEX_LIT_ANY_OP}:
        assert False, 'here not supported'
    assert len(stack) == 1
    result_value = stack[0]
    print('result:', sorted(result_value))
    assert_equal(result_value, target_value)

  evaluate('hello', {'hello'})
  evaluate('you|me', {'you', 'me'})
  evaluate('[aeiou]', {'a', 'e', 'i', 'o', 'u'})
  evaluate('regexp?', {'regex', 'regexp'})
  evaluate('hello (world|everybody|there)', {'hello world', 'hello everybody', 'hello there'})
  evaluate('([1-9][0-9]?|0)', {str(d) for d in range(0, 100)})
  evaluate('[0-9](\\.[1-9])?', {repr(d / 10) for d in range(0, 100) if not d % 10 == 0} | {str(d) for d in range(10)})
  evaluate('[4-2]', set())


def test_ParserGenerator_attr_syn():
  g = Grammar(
      Production('A', 'S'),
      Production('S', 'T', '+', 'S'),
      Production('S', 'T'),
      Production('T', 'zero'),
      Production('T', 'digit')
  )
  attr_g = AttributeGrammar(
    g,
    inh_attrs=set(),
    syn_attrs={'res'},
    prod_attr_rules=[
      {'res.0': lambda res: res(1)},
      {'res.0': lambda res: res(1) + res(3)},
      {'res.0': lambda res: res(1)},
      {'res.0': lambda res: 0},
      {'res.0': lambda res: res(1)}
    ],
    terminal_attr_rules={
      'digit': {'res.0': lambda word: int(word)}
    }
  )
  assert_equal(attr_g.attrs, {'res'})
  assert_equal(attr_g.syn_attrs, {'res'})
  assert_equal(attr_g.inh_attrs, set())

  word = '5+7'
  tokens = ['digit', '+', 'digit']
  tokens_pos = [0, 1, 2]
  print('tokens:', tokens, 'with decomposition', tokens_pos)
  parser = ParserGenerator(g)
  right_analysis, attr_eval = parser.parse_syn_attr_analysis(attr_g, word, tokens, tokens_pos)
  print('right analysis:', right_analysis)
  print('attribute eval (online):', attr_eval)
  assert_equal(right_analysis, (g.prods[0], g.prods[1], g.prods[2], g.prods[4], g.prods[4]))
  assert_equal(attr_eval, {'res': 5 + 7})
  tree = parser.parse_tree(word, tokens, tokens_pos)
  print('parse tree:', tree)
  assert_equal(
    tree, SyntaxTree(g.prods[0], SyntaxTree(
      g.prods[1], SyntaxTree(g.prods[4], None), None, SyntaxTree(g.prods[2], SyntaxTree(g.prods[4], None)))))
  attr_eval_gen = AttributeEvalGenerator(attr_g)
  tree_attr_eval = attr_eval_gen.eval_attrs(tree, word, tokens, tokens_pos)
  print('attribute eval (using tree):', tree_attr_eval)
  assert_equal(tree_attr_eval, {'res': 5 + 7})


def test_ParserGenerator_parse_tree_epsilon():
  g = Grammar(
    Production('S', 'A'),
    Production('A'),
    Production('A', 'B'),
    Production('B', 'a'),
    Production('B', 'a', 'B')
  )
  parser = ParserGenerator(g)
  word = 'aaa'
  tokens = ['a', 'a', 'a']
  tokens_pos = [0, 1, 2]
  assert_equal(
    parser.parse_analysis(word, tokens, tokens_pos), (g.prods[0], g.prods[2], g.prods[4], g.prods[4], g.prods[3]))
  assert_equal(
    parser.parse_tree(word, tokens, tokens_pos),
    SyntaxTree(g.prods[0], SyntaxTree(g.prods[2], SyntaxTree(g.prods[4], None, SyntaxTree(g.prods[4], None, SyntaxTree(
      g.prods[3], None))))))


def test_ParserGenerator_parse_tree_epsilon2():
  g = Grammar(
    Production('TopLevelExpr', 'ExprList'),
    Production('ExprList'),
    Production('ExprList', 'Expr', 'ExprList'),
    Production('Expr', 'Val', ';'),
    Production('Val', 'number')
  )
  parser = ParserGenerator(g)
  word = '42;'
  tokens = ['number', ';']
  tokens_pos = [0, 2]
  print(parser.parse_analysis(word, tokens, tokens_pos))
  print(parser.parse_tree(word, tokens, tokens_pos))


def test_ParserGenerator_simple_ast():
  class Ast:
    def __init__(self, pos):
      """
      :param TreePosition pos:
      """
      self.pos = pos

  class ConstantAst(Ast):
    def __init__(self, pos, num):
      super().__init__(pos)
      self.num = num

  class SumAst(Ast):
    def __init__(self, pos, ast, num):
      super().__init__(pos)
      self.ast = ast
      self.num = num

  g = Grammar(
    Production('A', 'B'),
    Production('B', 'num'),
    Production('B', 'B', '+', 'num')
  )
  attr_g = AttributeGrammar(
    g,
    inh_attrs=set(),
    syn_attrs={'ast', 'num'},
    prod_attr_rules=[
      {'ast': 'ast.1'},
      {'ast': lambda _pos, num: ConstantAst(_pos, num(1))},
      {'ast': lambda _pos, ast, num: SumAst(_pos, ast(1), num(3))}
    ],
    terminal_attr_rules={
      'num': {'num': lambda val: int(val)}
    }
  )
  assert_equal(set(g.terminals), {'num', '+'})
  assert_equal(attr_g.attrs, {'ast', 'num'})
  assert_equal(attr_g.syn_attrs, {'ast', 'num'})
  assert_equal(attr_g.inh_attrs, set())

  word = '1+2+3'
  tokens = ['num', '+', 'num', '+', 'num']
  tokens_pos = [0, 1, 2, 3, 4]
  print('tokens:', tokens, 'with decomposition', tokens_pos)
  parser = ParserGenerator(g)
  right_analysis, attr_eval = parser.parse_syn_attr_analysis(attr_g, word, tokens, tokens_pos)
  print('right analysis:', right_analysis)
  print('attribute eval:', attr_eval)
  assert 'ast' in attr_eval
  ast3 = attr_eval['ast']
  assert isinstance(ast3, SumAst)
  assert ast3.pos == TreePosition(word, 0, 5)
  assert ast3.num == 3
  ast2 = ast3.ast
  assert isinstance(ast2, SumAst)
  assert ast2.pos == TreePosition(word, 0, 3)
  assert ast2.num == 2
  ast1 = ast2.ast
  assert isinstance(ast1, ConstantAst)
  assert ast1.pos == TreePosition(word, 0, 1)
  assert ast1.num == 1



if __name__ == "__main__":
  try:
    better_exchook.install()
    if len(sys.argv) <= 1:
      for k, v in sorted(globals().items()):
        if k.startswith("test_"):
          print("-" * 40)
          print("Executing: %s" % k)
          try:
            v()
          except unittest.SkipTest as exc:
            print("SkipTest:", exc)
          print("-" * 40)
      print("Finished all tests.")
    else:
      assert len(sys.argv) >= 2
      for arg in sys.argv[1:]:
        print("Executing: %s" % arg)
        if arg in globals():
          globals()[arg]()  # assume function and execute
        else:
          eval(arg)  # assume Python code and execute
  finally:
    pass
