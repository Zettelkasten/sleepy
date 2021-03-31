import _setup_test_env  # noqa
from better_exchook import better_exchook
import sys
import unittest

from llvmlite import ir
from nose.tools import assert_equal, assert_almost_equal
from ctypes import CFUNCTYPE, c_double

from sleepy.ast import TopLevelExpressionAst, FunctionDeclarationAst, CallExpressionAst, ReturnExpressionAst, \
  IfExpressionAst, OperatorValueAst, ConstantValueAst, VariableValueAst, AssignExpressionAst, CallValueAst, \
  UnaryOperatorValueAst
from sleepy.grammar import Grammar, Production, AttributeGrammar
from sleepy.jit import make_execution_engine, compile_ir
from sleepy.lexer import LexerGenerator
from sleepy.parser import ParserGenerator

SLEEPY_LEXER = LexerGenerator(
  [
    'func', 'if', 'else', 'return', '{', '}', ';', ',', '(', ')', 'bool_op', 'sum_op', 'prod_op', '=', 'identifier',
    'number', None, None
  ], [
    'func', 'if', 'else', 'return', '{', '}', ';', ',', '\\(', '\\)', '==|!=|<=?|>=?', '\\+|\\-', '\\*|/', '=',
    '([A-Z]|[a-z]|_)([A-Z]|[a-z]|[0-9]|_)*', '(0|[1-9][0-9]*)(\\.[0-9]+)?', '#[^\n]*\n', '[ \n]+'
  ])

SLEEPY_GRAMMAR = Grammar(
  Production('TopLevelExpr', 'ExprList'),
  Production('ExprList'),
  Production('ExprList', 'Expr', 'ExprList'),
  Production('Expr', 'func', 'identifier', '(', 'IdentifierList', ')', '{', 'ExprList', '}'),
  Production('Expr', 'identifier', '(', 'ValList', ')', ';'),
  Production('Expr', 'return', 'Val', ';'),
  Production('Expr', 'identifier', '=', 'Val', ';'),
  Production('Expr', 'if', 'Val', '{', 'ExprList', '}'),
  Production('Expr', 'if', 'Val', '{', 'ExprList', '}', 'else', '{', 'ExprList', '}'),
  Production('Val', 'Val', 'bool_op', 'SumVal'),
  Production('Val', 'SumVal'),
  Production('SumVal', 'SumVal', 'sum_op', 'ProdVal'),
  Production('SumVal', 'ProdVal'),
  Production('ProdVal', 'ProdVal', 'prod_op', 'NegVal'),
  Production('ProdVal', 'NegVal'),
  Production('NegVal', 'sum_op', 'PrimaryVal'),
  Production('NegVal', 'PrimaryVal'),
  Production('PrimaryVal', 'number'),
  Production('PrimaryVal', 'identifier'),
  Production('PrimaryVal', 'identifier', '(', 'ValList', ')'),
  Production('PrimaryVal', '(', 'Val', ')'),
  Production('IdentifierList'),
  Production('IdentifierList', 'IdentifierList+'),
  Production('IdentifierList+', 'identifier'),
  Production('IdentifierList+', 'identifier', ',', 'IdentifierList+'),
  Production('ValList'),
  Production('ValList', 'ValList+'),
  Production('ValList+', 'Val'),
  Production('ValList+', 'Val', ',', 'ValList+')
)
SLEEPY_ATTR_GRAMMAR = AttributeGrammar(
  SLEEPY_GRAMMAR,
  syn_attrs={'ast', 'expr_list', 'identifier_list', 'val_list', 'identifier', 'op', 'number'},
  prod_attr_rules=[
    {'ast': lambda expr_list: TopLevelExpressionAst(expr_list(1))},
    {'expr_list': []},
    {'expr_list': lambda ast, expr_list: [ast(1)] + expr_list(2)},
    {'ast': lambda identifier, identifier_list, expr_list: (
      FunctionDeclarationAst(identifier(2), identifier_list(4), expr_list(7)))},
    {'ast': lambda identifier, val_list: CallExpressionAst(identifier(1), val_list(3))},
    {'ast': lambda ast: ReturnExpressionAst(ast(2))},
    {'ast': lambda identifier, ast: AssignExpressionAst(identifier(1), ast(3))},
    {'ast': lambda ast, expr_list: IfExpressionAst(ast(2), expr_list(4), [])},
    {'ast': lambda ast, expr_list: IfExpressionAst(ast(2), expr_list(4), expr_list(8))}] + [
    {'ast': lambda ast, op: OperatorValueAst(op(2), ast(1), ast(3))},
    {'ast': 'ast.1'}] * 3 + [
    {'ast': lambda ast, op: UnaryOperatorValueAst(op(1), ast(2))},
    {'ast': 'ast.1'},
    {'ast': lambda number: ConstantValueAst(number(1))},
    {'ast': lambda identifier: VariableValueAst(identifier(1))},
    {'ast': lambda identifier, val_list: CallValueAst(identifier(1), val_list(3))},
    {'ast': 'ast.2'},
    {'identifier_list': []},
    {'identifier_list': 'identifier_list.1'},
    {'identifier_list': lambda identifier: [identifier(1)]},
    {'identifier_list': lambda identifier, identifier_list: [identifier(1)] + identifier_list(3)},
    {'val_list': []},
    {'val_list': 'val_list.1'},
    {'val_list': lambda ast: [ast(1)]},
    {'val_list': lambda ast, val_list: [ast(1)] + val_list(3)}
  ],
  terminal_attr_rules={
    'bool_op': {'op': lambda value: value},
    'sum_op': {'op': lambda value: value},
    'prod_op': {'op': lambda value: value},
    'identifier': {'identifier': lambda value: value},
    'number': {'number': lambda value: float(value)}
  }
)

SLEEPY_PARSER = ParserGenerator(SLEEPY_GRAMMAR)


def _test_parse_ast(program):
  """
  :param str program:
  :rtype: TopLevelExpressionAst
  """
  print('---- input program:')
  print(program)
  tokens, token_words = SLEEPY_LEXER.tokenize(program)
  print('---- tokens:')
  print(tokens)
  analysis, eval = SLEEPY_PARSER.parse_syn_attr_analysis(SLEEPY_ATTR_GRAMMAR, tokens, token_words)
  ast = eval['ast']
  print('---- right-most analysis:')
  print(analysis)
  print('---- abstract syntax tree:')
  print(ast)
  assert isinstance(ast, TopLevelExpressionAst)
  return ast


def test_ast_parser():
  program1 = 'hello_world(123);'
  _test_parse_ast(program1)
  program2 = """# This function will just return 4.
func do_stuff(val) {
  return 4;
}
do_stuff(7.5);
"""
  _test_parse_ast(program2)
  program3 = """
  # Compute 0 + 1 + ... + n
  func sum_all(n) {
    if n <= 0 { return 0; }
    else { return sum_all(n-1) + n; }
  }
  
  sum_all(12);
  """
  _test_parse_ast(program3)


def _get_py_func_from_ast(engine, ast):
  """
  :param ExecutionEngine engine:
  :param FunctionDeclarationAst ast:
  :rtype: Callable
  """
  assert isinstance(ast, FunctionDeclarationAst)
  module = ir.Module(name='_test_last_declared_func')
  symbol_table = {}
  ast.build_expr_ir(module=module, builder=None, symbol_table=symbol_table)
  assert ast.identifier in symbol_table
  compile_ir(engine, module)
  func_ptr = engine.get_function_address(ast.identifier)
  return CFUNCTYPE(*((c_double,) + (c_double,) * len(ast.arg_identifiers)))(func_ptr)


def test_FunctionDeclarationAst_build_expr_ir():
  with make_execution_engine() as engine:
    ast1 = FunctionDeclarationAst(
      identifier='foo', arg_identifiers=[], expr_list=[ReturnExpressionAst(ConstantValueAst(42.0))])
    func1 = _get_py_func_from_ast(engine, ast1)
    assert_equal(func1(), 42.0)
  with make_execution_engine() as engine:
    ast2 = FunctionDeclarationAst(
      identifier='foo', arg_identifiers=[], expr_list=[
        ReturnExpressionAst(OperatorValueAst('+', ConstantValueAst(3.0), ConstantValueAst(5.0)))])
    func2 = _get_py_func_from_ast(engine, ast2)
    assert_equal(func2(), 8.0)
  with make_execution_engine() as engine:
    ast3 = FunctionDeclarationAst(
      identifier='sum', arg_identifiers=['a', 'b'], expr_list=[
        ReturnExpressionAst(OperatorValueAst('+', VariableValueAst('a'), VariableValueAst('b')))])
    func3 = _get_py_func_from_ast(engine, ast3)
    assert_equal(func3(7.0, 3.0), 10.0)


def _test_compile_program(engine, program, main_func_identifier='main', main_func_num_args=0):
  """
  :param ExecutionEngine engine:
  :param str program:
  :param str main_func_identifier:
  :rtype: Callable[[], float]
  """
  ast = _test_parse_ast(program)
  module_ir = ast.make_module_ir(module_name='test_parse_ast')
  print('---- module intermediate repr:')
  print(module_ir)
  compile_ir(engine, module_ir)
  main_func_ptr = engine.get_function_address(main_func_identifier)
  py_func = CFUNCTYPE(*((c_double,) + (c_double,) * main_func_num_args))(main_func_ptr)
  assert callable(py_func)
  return py_func


def test_simple_arithmetic():
  with make_execution_engine() as engine:
    program = """
    func main() {
      return 4.0 + 3.0;
    }
    """
    func = _test_compile_program(engine, program)
    assert_equal(func(), 4.0 + 3.0)
  with make_execution_engine() as engine:
    program = """
    func test() {
      return 2 * 4 - 3;
    }
    """
    func = _test_compile_program(engine, program, main_func_identifier='test')
    assert_equal(func(), 2.0 * 4.0 - 3.0)
  with make_execution_engine() as engine:
    program = """
    func sub(a, b) {
      return a - b;
    }
    """
    func = _test_compile_program(engine, program, main_func_identifier='sub', main_func_num_args=2)
    assert_equal(func(0.0, 1.0), 0.0 - 1.0)
    assert_equal(func(3.0, 5.0), 3.0 - 5.0)
    assert_equal(func(2.5, 2.5), 2.5 - 2.5)


def test_empty_func():
  with make_execution_engine() as engine:
    program = """
    func nothing() {
    }
    """
    nothing = _test_compile_program(engine, program, main_func_identifier='nothing')
    assert_equal(nothing(), 0.0)


def test_lerp():
  with make_execution_engine() as engine:
    program = """
    func lerp(x1, x2, time) {
      diff = x2 - x1;
      return x1 + diff * time;
    }
    """
    lerp = _test_compile_program(engine, program, main_func_identifier='lerp', main_func_num_args=3)
    assert_equal(lerp(0.0, 1.0, 0.3), 0.3)
    assert_equal(lerp(7.5, 3.2, 0.0), 7.5)
    assert_equal(lerp(7.5, 3.2, 1.0), 3.2)


def test_call_other_func():
  with make_execution_engine() as engine:
    program = """
    func square(x) {
      return x * x;
    }
    func dist_squared(x1, x2, y1, y2) {
      return square(x1 - y1) + square(x2 - y2);
    }
    """
    dist_squared = _test_compile_program(engine, program, main_func_identifier='dist_squared', main_func_num_args=4)
    assert_almost_equal(dist_squared(0.0, 0.0, 1.0, 0.0), 1.0)
    assert_almost_equal(dist_squared(3.0, 0.0, 0.0, 4.0), 25.0)
    assert_almost_equal(dist_squared(1.0, 2.0, 3.0, 4.0), (1.0 - 3.0)**2 + (2.0 - 4.0)**2)


@unittest.skip('global variables not yet implemented')
def test_global_var():
  with make_execution_engine() as engine:
    program = """
    PI = 3.1415;  # declare a global variable
    func cube(x) { return x * x * x; }
    func ball_volume(radius) {
      return 4/3 * PI * cube(radius);
    }
    """
    ball_volume = _test_compile_program(engine, program, main_func_identifier='ball_volume', main_func_num_args=1)
    for radius in [0.0, 2.0, 3.0, 124.343]:
      assert_almost_equal(ball_volume(radius), 4.0 / 3.0 * 3.1415 * radius ** 3.0)


def test_simple_mutable_assign():
  with make_execution_engine() as engine:
    program = """
    func main(x) {
      x = x + 1;
      x = x + 1;
      return x;
    }
    """
    main = _test_compile_program(engine, program, main_func_num_args=1)
    assert_equal(main(3), 3 + 2)


def test_nested_func_call():
  import numpy
  with make_execution_engine() as engine:
    program = """
    func ball_volume(radius) {
      func cube(x) { return x * x * x; }
      return 4/3 * 3.1415 * cube(radius);
    }
    # Compute relative volume difference of two balls.
    func main(radius1, radius2) {
      volume1 = ball_volume(radius1);
      volume2 = ball_volume(radius2);
      return volume1 / volume2;
    }
    """
    main = _test_compile_program(engine, program, main_func_num_args=2)
    for radius1 in [0.0, 2.0, 3.0, 124.343]:
      for radius2 in [0.0, 2.0, 3.0, 124.343]:
        volume1, volume2 = 4.0 / 3.0 * 3.1415 * radius1 ** 3.0, 4.0 / 3.0 * 3.1415 * radius2 ** 3.0
        numpy.testing.assert_almost_equal(main(radius1, radius2), numpy.divide(volume1, volume2))


def test_simple_if():
  with make_execution_engine() as engine:
    program = """
    func branch(cond, true_val, false_val) {
      if cond {
        return true_val;
      } else {
        return false_val;
      }
    }
    """
    branch = _test_compile_program(engine, program, main_func_identifier='branch', main_func_num_args=3)
    assert_equal(branch(0, 42, -13), -13)
    assert_equal(branch(1, 42, -13), 42)


def test_simple_if_max():
  with make_execution_engine() as engine:
    program = """
    func max(a, b) {
      if a < b {
        return b;
      } else {
        return a;
      }
    }
    """
    max_ = _test_compile_program(engine, program, main_func_identifier='max', main_func_num_args=2)
    assert_equal(max_(13, 18), 18)
    assert_equal(max_(-3, 4.23), 4.23)
    assert_equal(max_(0, 0), 0)
    assert_equal(max_(-4, -9), -4)


def test_simple_if_abs():
  with make_execution_engine() as engine:
    program = """ func abs(x) { if x < 0 { return -x; } else { return x; } } """
    abs_ = _test_compile_program(engine, program, main_func_identifier='abs', main_func_num_args=1)
    assert_equal(abs_(3.1415), 3.1415)
    assert_equal(abs_(0.0), 0.0)
    assert_equal(abs_(-5.1), 5.1)


def test_simple_simple_recursion_factorial():
  import math
  with make_execution_engine() as engine:
    program = """
    func fac(x) {
      if x <= 1 {
        return x;
      } else {
        return x * fac(x-1);
      }
    }
    """
    fac = _test_compile_program(engine, program, main_func_identifier='fac', main_func_num_args=1)
    assert_equal(fac(3), 3 * 2 * 1)
    assert_equal(fac(9), math.prod(range(1, 9 + 1)))
    assert_equal(fac(0), 0)


def test_simple_simple_recursion_fibonacci():
  with make_execution_engine() as engine:
    program = """
    func or(a, b) { if a { return a; } else { return b; } }
    func fibonacci(n) {
      # crashes if n <= 0 or if n is not integer :)
      if or(n == 1, n == 2) {
        return 1;
      } else {
        return fibonacci(n - 2) + fibonacci(n - 1);
      }
    }
    """

    def reference_fib(n):
      if n == 1 or n == 2:
        return 1
      else:
        return reference_fib(n - 2) + reference_fib(n - 1)

    fib = _test_compile_program(engine, program, main_func_identifier='fibonacci', main_func_num_args=1)
    for n in range(1, 15):
      assert_equal(fib(n), reference_fib(n))


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
