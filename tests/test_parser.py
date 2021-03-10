import sys
import unittest
import better_exchook
import nose.tools

from sleepy.parser import ParserGenerator, make_first1_sets, get_first1_set_for_word
from sleepy.grammar import EPSILON, Production, Grammar, ParseError


def test_Grammar():
  g = Grammar(
    Production('S2', 'S'),
    Production('S', 'A', 'b'),
    Production('A', 'a', 'a')
  )
  assert set(g.symbols) == set(['S2', 'S', 'A', 'a', 'b'])
  assert set(g.terminals) == set(['a', 'b'])
  assert set(g.non_terminals) == set(['S2', 'S', 'A'])
  assert set(g.get_prods_for('S')) == set([g.prods[1]])


def test_make_first1_sets():
  g = Grammar(
    Production('S', 'S', 'O', 'S'),
    Production('S', '(', 'S', ')'),
    Production('S', '0'),
    Production('S', '1'),
    Production('O', '+'),
    Production('O', '*'),
  )
  nose.tools.assert_equal(set(g.terminals), set(['0', '1', '(', ')', '+', '*']))
  nose.tools.assert_equal(set(g.non_terminals), set(['S', 'O']))
  first1 = make_first1_sets(g)
  print('first1 sets:', first1)
  nose.tools.assert_equal(first1['S'], set(['0', '1', '(']))
  nose.tools.assert_equal(first1['O'], set(['+', '*']))
  nose.tools.assert_equal(get_first1_set_for_word(first1, ('(', 'S')), set(['(']))


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
  nose.tools.assert_equal(first1['A'], set([EPSILON, 'a']))
  nose.tools.assert_equal(first1['B'], set([EPSILON, 'a', 'b']))
  nose.tools.assert_equal(first1['S'], set([EPSILON, 'a', 'b']))
  nose.tools.assert_equal(get_first1_set_for_word(first1, ('B', 'c')), set(['a', 'b', 'c']))


def test_ParserGenerator_simple():
  g = Grammar(
    Production('S2', 'S'),
    Production('S', 'A', 'b'),
    Production('A', 'a', 'a')
  )
  parser = ParserGenerator(g)
  nose.tools.assert_equal(parser.parse_analysis(['a', 'a', 'b']), [g.prods[0], g.prods[1], g.prods[2]])
  with nose.tools.assert_raises(ParseError):
    parser.parse_analysis(['a', 'a', 'a', 'b'])
  with nose.tools.assert_raises(ParseError):
    parser.parse_analysis(['a', 'a', 'b', 'b'])
  with nose.tools.assert_raises(ParseError):
    parser.parse_analysis(['a', 'a'])


def test_ParserGenerator_simple_left_recursive():
  g = Grammar(
    Production('S2', 'S'),
    Production('S', 'S', 'a'),
    Production('S', 'a')
  )
  parser = ParserGenerator(g)
  for count in [1, 2, 3, 20, 100, 10000]:
    nose.tools.assert_equal(
      parser.parse_analysis(['a'] * count), [g.prods[0]] + (count-1) * [g.prods[1]] + [g.prods[2]])
  with nose.tools.assert_raises(ParseError):
    parser.parse_analysis([])


def test_ParserGenerator_simple_left_recursive_epsilon():
  g = Grammar(
    Production('S2', 'S'),
    Production('S', 'S', 'a'),
    Production('S')
  )
  parser = ParserGenerator(g)
  for count in [0, 1, 2, 3, 20, 100, 10000]:
    nose.tools.assert_equal(
      parser.parse_analysis(['a'] * count), [g.prods[0]] + count * [g.prods[1]] + [g.prods[2]])


def test_ParserGenerator_simple_lookahead():
  g = Grammar(
    Production('S2', 'S'),
    Production('S', 'b', 'b'),
    Production('S', 'a', 'b')
  )
  parser = ParserGenerator(g)
  nose.tools.assert_equal(parser.parse_analysis(['b', 'b']), [g.prods[0], g.prods[1]])
  nose.tools.assert_equal(parser.parse_analysis(['a', 'b']), [g.prods[0], g.prods[2]])
  with nose.tools.assert_raises(ParseError):
    parser.parse_analysis(['b', 'a'])


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
  word = ['A', '->', 'B', 'c', '|', 'B', ';', 'B', '->', 'c', 'a']
  print('tokenized word:', word)
  analysis = parser.parse_analysis(word)
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

  def evaluate(raw_word):
    """
    :param str raw_word:
    """
    assert isinstance(raw_word, str)
    word = list(raw_word)
    print('input word:', raw_word)
    analysis = parser.parse_analysis(word)
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
    assert result_value == eval(raw_word)

  evaluate('1*2+3')
  evaluate('1+2*3')
  evaluate('(1+2)*3')
  evaluate('1+2+3')
  evaluate('1+2-3')
  evaluate('1-2+3')
  evaluate('3-2-1')
  evaluate('1*2+3*4')
  evaluate('4/2-1')


def test_ParserGenerator_regex():
  from sleepy.regex import REGEX_PARSER, tokenize_regex, REGEX_LIT_OP, REGEX_CHOICE_OP, REGEX_LITS_SINGLE_OP, \
    REGEX_LITS_MULTIPLE_OP, REGEX_CONCAT_OP, REGEX_OPTIONAL_OP, REGEX_RANGE_OP, REGEX_RANGE_LITS_OP, REGEX_REPEAT_OP, \
    REGEX_REPEAT_EXISTS_OP, REGEX_INV_RANGE_OP, REGEX_INV_RANGE_LITS_OP, REGEX_LIT_TOKEN
  parser = REGEX_PARSER

  def evaluate(word, target_value):
    """
    :param str word:
    :param set[str] target_value:
    """
    assert isinstance(word, str)
    tokens, token_attribute_table = tokenize_regex(word)
    print('word:', word)
    print('tokens:', tokens)
    analysis = parser.parse_analysis(tokens)
    print('right-most analysis:', analysis)

    stack = []
    pos = 0

    def next_literal_name():
      nonlocal pos
      while tokens[pos] != REGEX_LIT_TOKEN:
        pos += 1
      name = token_attribute_table[pos]
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
      elif prod in {REGEX_REPEAT_OP, REGEX_REPEAT_EXISTS_OP, REGEX_INV_RANGE_OP, REGEX_INV_RANGE_LITS_OP}:
        assert False, 'here not supported'
    assert len(stack) == 1
    result_value = stack[0]
    print('result:', sorted(result_value))
    nose.tools.assert_equal(result_value, target_value)

  evaluate('hello', {'hello'})
  evaluate('you|me', {'you', 'me'})
  evaluate('[aeiou]', {'a', 'e', 'i', 'o', 'u'})
  evaluate('regexp?', {'regex', 'regexp'})
  evaluate('hello (world|everybody|there)', {'hello world', 'hello everybody', 'hello there'})
  evaluate('([1-9][0-9]?|0)', {str(d) for d in range(0, 100)})
  evaluate('[0-9](.[1-9])?', {repr(d / 10) for d in range(0, 100) if not d % 10 == 0} | {str(d) for d in range(10)})
  evaluate('[4-2]', set())


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
