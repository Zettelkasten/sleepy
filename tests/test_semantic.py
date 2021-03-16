import _setup_test_env  # noqa
import sys
import unittest
import better_exchook
from nose.tools import assert_equal, assert_raises, assert_equals, assert_almost_equal

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
      {'decl_i.1': set(), 'decl_s.0': 'decl_s.1', 'ok': 'ok.1'},
      {'decl_i.1': 'decl_i.0', 'decl_s.0': 'decl_s.1', 'ok': 'ok.1'},
      {'decl_i.1': 'decl_i.0', 'decl_i.3': 'decl_s.1', 'decl_s.0': 'decl_s.3', 'ok': lambda ok: ok(1) and ok(3)},
      {'decl_i.3': 'decl_i.0', 'decl_s.0': lambda decl_i, name: decl_i(0) | {name(1)}, 'ok.0': 'ok.3'},
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
    print('input word:', word)
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
  evaluate(
    'alpha=1.0;beta=4.0;gamma=beta;delta=alpha;alpha=alpha',
    {'decl_s': {'alpha', 'beta', 'gamma', 'delta'}, 'ok': True})


def test_AttributeEvalGenerator_typed_arithmetic():
  import math, numpy as np
  lexer = LexerGenerator(
    token_names=['(', ')', '+', '-', '*', '**', '/', '[', ']', ',', 'const', 'name', None],
    token_regex_table=[
      '\\(', '\\)', '\\+', '\\-', '\\*', '\\*\\*', '/', '\\[', '\\]', ',', '(0|[1-9][0-9]*)(\\.[0-9]+)?',
      '([a-z]|[A-Z])+', ' +']
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
    Production('Term', 'name', '(', 'Sum', ')'),
    Production('Term', 'const'),
    Production('Term', '[', 'ExprList', ']'),
    Production('ExprList', 'Sum'),
    Production('ExprList', 'ExprList', ',', 'Sum')
  )

  ERROR = None

  def op_plus_res(type, res):
    left, right, left_type, right_type = res(1), res(3), type(1), type(3)
    if left is ERROR or right is ERROR:
      return ERROR
    if left_type == right_type in {'num', 'vec', 'mat'}:
      if np.shape(left) != np.shape(right):
        return ERROR
      return left + right
    return ERROR

  def op_minus_res(type, res):
    left, right, left_type, right_type = res(1), res(3), type(1), type(3)
    if left is ERROR or right is ERROR:
      return ERROR
    if left_type == right_type in {'num', 'vec', 'mat'}:
      if np.shape(left) != np.shape(right):
        return ERROR
      return left - right
    return ERROR

  def op_times_type(type):
    left_type, right_type = type(1), type(3)
    if left_type == right_type == 'num':
      return 'num'
    if (left_type == 'vec' and right_type == 'num') or (left_type == 'num' and right_type == 'vec'):
      return 'vec'
    if left_type == 'mat' or right_type == 'mat':
      return 'mat'
    return ERROR

  def op_times_res(type, res):
    left, right, left_type, right_type = res(1), res(3), type(1), type(3)
    if left is ERROR or right is ERROR:
      return ERROR
    if left_type not in {'num', 'vec', 'mat'} or right_type not in {'num', 'vec', 'mat'}:
      return ERROR
    if left_type == 'num' or right_type == 'num':
      return left * right
    if left_type == 'mat' or right_type == 'mat':
      if left_type == 'vec':
        assert right_type == 'mat'
        if np.shape(left)[0] != np.shape(right)[0]:
          return ERROR
        return left * right
      if right_type == 'vec':
        assert left_type == 'mat'
        if np.shape(left)[1] != np.shape(right)[0]:
          return ERROR
        return left * right
      assert left_type == right_type == 'mat'
      if np.shape(left)[1] != np.shape(right)[0]:
        return ERROR
      return left * right
    return ERROR

  def op_divided_res(type, res):
    left, right, left_type, right_type = res(1), res(3), type(1), type(3)
    if left is ERROR or right is ERROR:
      return ERROR
    if left_type not in {'num', 'vec', 'mat'} or right_type not in {'num', 'vec', 'mat'}:
      return ERROR
    if right_type == 'num':
      return left / right
    return ERROR

  def op_pow_res(type, res):
    left, right, left_type, right_type = res(1), res(3), type(1), type(3)
    if left is ERROR or right is ERROR:
      return ERROR
    if left_type == right_type == 'num':
      assert isinstance(left, (float, int)) and isinstance(right, (float, int))
      return left ** right
    if left_type == 'mat' and right_type == 'num':
      if np.shape(left)[0] != np.shape(right)[0]:
        return ERROR
      return left ** right
    return ERROR

  def op_func_type(type, name):
    arg_type = type(3)
    if name(1) in {'ones', 'zeros'}:
      return 'vec'
    return arg_type

  def op_func_res(type, res, name):
    arg, arg_type = res(3), type(3)
    if arg is ERROR:
      return ERROR
    if name(1) in {'ones', 'zeros'}:
      if arg_type != 'num' or arg <= 1:
        return ERROR
      assert isinstance(arg, (float, int))
      return (np.ones if name(1) == 'ones' else np.zeros)(int(arg))
    if arg_type == 'num':
      if not hasattr(math, name(1)):
        return ERROR
      return getattr(math, name(1))(arg)
    return ERROR

  def op_const_vec_type(type_list, res_list):
    type_list_, res_list_ = type_list(2), res_list(2)
    assert len(type_list_) == len(res_list_)
    if len(type_list_) == 0:
      return ERROR
    type = type_list_[0]
    if not all(t == type for t in type_list_):
      return ERROR
    if type == 'num':
      return 'vec'
    if type == 'vec':
      return 'mat'
    return ERROR

  def op_const_vec_res(type_list, res_list):
    type_list_, res_list_ = type_list(2), res_list(2)
    assert len(type_list_) == len(res_list_)
    if len(type_list_) == 0:
      return ERROR
    type = type_list_[0]
    if not all(t == type for t in type_list_):
      return ERROR
    if type in {'num', 'vec'}:
      if type == 'vec':
        length = len(res_list_[0])
        if not all(len(vec) == length for vec in res_list_):
          return ERROR
      return np.array(res_list_)
    return ERROR

  attr_g = AttributeGrammar(
    g,
    syn_attrs={'type', 'res', 'name', 'type_list', 'res_list'},
    prod_attr_rules=[
      {'type': 'type.1', 'res': 'res.1'},
      {'type': 'type.1', 'res': op_plus_res},
      {'type': 'type.1', 'res': op_minus_res},
      {'type': 'type.1', 'res': 'res.1'},
      {'type': op_times_type, 'res': op_times_res},
      {'type': op_times_type, 'res': op_divided_res},
      {'type': 'type.1', 'res': 'res.1'},
      {'type': 'type.3', 'res': op_pow_res},
      {'type': 'type.1', 'res': 'res.1'},
      {'type': 'type.2', 'res': 'res.2'},
      {'type': op_func_type, 'res': op_func_res},
      {'type': 'type.1', 'res': 'res.1'},
      {'type': op_const_vec_type, 'res': op_const_vec_res},
      {'type_list': lambda type: [type(1)], 'res_list': lambda res: [res(1)]},
      {
        'type_list': lambda type, type_list: type_list(1) + [type(3)],
        'res_list': lambda res, res_list: res_list(1) + [res(3)]
      }
    ],
    terminal_attr_rules={
      'const': {'res': lambda lit: float(lit), 'type': lambda _: 'num'},
      'name': {'name': lambda lit: lit}
    }
  )
  parser = ParserGenerator(g)

  def evaluate(word, expected_result, expected_type=None):
    print('----')
    print('input word:', word)
    tokens, token_words = lexer.tokenize(word)
    print('tokens:', tokens, 'with decomposition', token_words)
    analysis, result = parser.parse_syn_attr_analysis(attr_g, tokens, token_words)
    print('result:')
    print(result['res'])
    if isinstance(expected_result, (float, int)):
      assert_almost_equal(result['res'], expected_result)
    else:
      np.testing.assert_equal(result['res'], expected_result)
    if result['res'] is not ERROR:
      print('result type:', result['type'])
      assert_equal(result['type'], expected_type)

  evaluate('1*2+3', 5, 'num')
  evaluate('5-3', 2, 'num')
  evaluate('(1+2)*3', 9, 'num')
  evaluate('1+2+3', 6, 'num')
  evaluate('1-2+3', 2, 'num')
  evaluate('3-2-1', 0, 'num')
  evaluate('1*2+3*4', 14, 'num')
  evaluate('4/2-1', 1, 'num')
  evaluate('sin(3.1415)', 9.26535897e-5, 'num')
  evaluate('2**2**3', 256, 'num')
  evaluate('(2**2)**3', 64, 'num')
  evaluate('(3**2+4**2)**0.5', 5, 'num')
  evaluate('ones(4)', [1,1,1,1], 'vec')
  evaluate('ones(4) * 5', [5,5,5,5], 'vec')
  evaluate('ones(2) - ones(2) * 0', [1,1], 'vec')
  evaluate('2 ** ones(2)', ERROR)
  evaluate('[1 , 2 , 3]', [1,2,3], 'vec')
  evaluate('[1, 2] + 2 * [3, 2]', [7,6], 'vec')
  evaluate('[[1,2],[3,4]]', [[1,2],[3,4]], 'mat')
  evaluate('[ones(2)]', [[1,1]], 'mat')
  evaluate('[ones(3), ones(4)]', ERROR)
  evaluate('zeros(0+2-2)', ERROR)
  evaluate('[[1,0,0],[0,1,0],[0,0,1]]', [[1,0,0],[0,1,0],[0,0,1]], 'mat')
  evaluate('[[1,0,0],[0,1,0],[0,0,1]]+[[0,1]]', ERROR)
  evaluate('[[1,0,0],[0,1,0],[0,0,1]]+[[1,0,0],[0,1,0],[0,0,1]]', [[2,0,0],[0,2,0],[0,0,2]], 'mat')
  evaluate('[[1,0,0],[0,1,0],[0,0,1]]+0*[[1,0,0],[0,1,0],[0,0,1]]', [[1,0,0],[0,1,0],[0,0,1]], 'mat')


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
