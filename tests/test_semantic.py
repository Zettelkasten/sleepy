import _setup_test_env  # noqa
import sys
import unittest
import better_exchook
from nose.tools import assert_equal, assert_raises, assert_equals

from sleepy.lexer import LexerGenerator
from sleepy.parser import ParserGenerator, make_first1_sets, get_first1_set_for_word
from sleepy.grammar import EPSILON, Production, Grammar, ParseError, AttributeGrammar, SyntaxTree
from sleepy.semantic import AttributeEvalGenerator


def test_AttributeEvalGenerator_check_declaredness():
  lexer = LexerGenerator(
    [';', '=', 'const', 'name'],
    [';', '=', '(0|[1-9])[0-9]*(\\.[0-9]*)?', '([a-z]|[A-Z])+']
  )
  g = Grammar(
    Production('Prog0', 'Decls'),
    Production('Decls', 'Decl'),
    Production('Decls', 'Decl', ';', 'Decls'),
    Production('Decl', 'name', '=', 'Var'),
    Production('Var', 'name'),
    Production('Var', 'const'),
  )
  parser = ParserGenerator(g)
  attr_g = AttributeGrammar(
    g,
    inh_attrs={'decl_i'},
    syn_attrs={'decl_s', 'name', 'ok'},
    prod_attr_rules=[
      {'decl_i.1': set(), 'decl_s.0': lambda decl_s: decl_s(1), 'ok': lambda ok: ok(1)},
      {'decl_i.1': lambda decl_i: decl_i(0), 'decl_s.0': lambda decl_s: decl_s(1), 'ok': lambda ok: ok(1)},
      {
        'decl_i.1': lambda decl_i: decl_i(0), 'decl_i.3': lambda decl_s: decl_s(1),
        'decl_s.0': lambda decl_s: decl_s(3), 'ok': lambda ok: ok(1) and ok(3)
      },
      {
        'decl_i.3': lambda decl_i: decl_i(0), 'decl_s.0': lambda decl_i, name: decl_i(0) | {name(1)},
        'ok.0': lambda ok: ok(3)
      },
      {'ok.0': lambda decl_i, name: name(1) in decl_i(0)},
      {'ok.0': True}
    ],
    terminal_attr_rules={
      'name': {'name.0': lambda value: value}
    }
  )
  attr_eval_gen = AttributeEvalGenerator(attr_g)

  def evaluate(word, target_eval):
    print('----')
    tokens, token_words = lexer.tokenize(word)
    print('tokens:', tokens, 'with decomposition', token_words)
    tree = parser.parse_tree(tokens, token_words)
    print('parsed tree:', tree)
    tree_attr_eval = attr_eval_gen.eval_attrs(tree, token_words)
    print('attribute eval:', tree_attr_eval)
    assert_equal(tree_attr_eval, target_eval)

  evaluate('alpha=5;beta=7;gamma=alpha', {'decl_s': {'alpha', 'beta', 'gamma'}, 'ok': True})
  evaluate('alpha=5;beta=alpha;gamma=alpha', {'decl_s': {'alpha', 'beta', 'gamma'}, 'ok': True})
  evaluate('alpha=1;beta=alpha;gamma=beta', {'decl_s': {'alpha', 'beta', 'gamma'}, 'ok': True})
  evaluate('alpha=alpha', {'decl_s': {'alpha'}, 'ok': False})
  evaluate('alpha=1.0;beta=gamma', {'decl_s': {'alpha', 'beta'}, 'ok': False})
  evaluate('alpha=alpha', {'decl_s': {'alpha'}, 'ok': False})
  evaluate('alpha=1.0;beta=gamma', {'decl_s': {'alpha', 'beta'}, 'ok': False})
  evaluate(
    'alpha=1.0;beta=4.0;gamma=beta;delta=alpha;alpha=alpha',
    {'decl_s': {'alpha', 'beta', 'gamma', 'delta'}, 'ok': True})


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
