import sys
import unittest
from pprint import pprint
import better_exchook
import nose.tools

from parser import ParserGenerator, make_first1_sets, get_first1_set_for_word
from grammar import EPSILON, Production, Grammar, ParseError


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
  print('input word:', word)
  analysis = parser.parse_analysis(word)
  print('right-most analysis:', analysis)


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
