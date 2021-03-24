import _setup_test_env  # noqa
from better_exchook import better_exchook
import sys
import unittest

from llvmlite import ir
from nose.tools import assert_equal
from ctypes import CFUNCTYPE, c_double

from sleepy.ast import TopLevelExpressionAst, FunctionDeclarationAst, CallExpressionAst, ReturnExpressionAst, \
  IfExpressionAst, OperatorValueAst, ConstantValueAst, VariableValueAst, AssignExpressionAst
from sleepy.grammar import Grammar, Production, AttributeGrammar
from sleepy.jit import make_execution_engine, compile_ir
from sleepy.lexer import LexerGenerator
from sleepy.parser import ParserGenerator

SLEEPY_LEXER = LexerGenerator(
  [
    'func', 'if', 'else', 'return', '{', '}', ';', ',', '(', ')', 'bool_op', 'sum_op', 'prod_op', '=', 'identifier',
    'number', None, None
  ], [
    'func', 'if', 'else', 'return', '{', '}', ';', ',', '\\(', '\\)', '==|!=|<=?|>=?', '\\+|\\-', '\\*|\\\\', '=',
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
  Production('ProdVal', 'ProdVal', 'prod_op', 'PrimaryVal'),
  Production('ProdVal', 'PrimaryVal'),
  Production('PrimaryVal', 'number'),
  Production('PrimaryVal', 'identifier'),
  Production('PrimaryVal', 'identifier', '(', 'ValList', ')'),
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
    {'ast': lambda number: ConstantValueAst(number(1))},
    {'ast': lambda identifier: VariableValueAst(identifier(1))},
    {'ast': lambda identifier, val_list: CallExpressionAst(identifier(1), val_list(3))},
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


def test_simple_compile():
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
  with make_execution_engine() as engine:
    program = """
    func nothing() {
    }
    """
    nothing = _test_compile_program(engine, program, main_func_identifier='nothing')
    assert_equal(nothing(), 0.0)
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
