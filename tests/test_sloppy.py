import _setup_test_env  # noqa
from better_exchook import better_exchook
import sys
import unittest

from sleepy.ast import TopLevelExpressionAst, FunctionDeclarationAst, CallExpressionAst, ReturnExpressionAst, \
  IfExpressionAst, OperatorValueAst, ConstantValueAst, VariableValueAst
from sleepy.grammar import Grammar, Production, SyntaxTree, AttributeGrammar
from sleepy.lexer import LexerGenerator
from sleepy.parser import ParserGenerator

SLOPPY_LEXER = LexerGenerator(
  [
    'func', 'if', 'else', 'return', '{', '}', ';', ',', '(', ')', 'bool_op', 'sum_op', 'prod_op', 'identifier',
    'number', None, None
  ], [
    'func', 'if', 'else', 'return', '{', '}', ';', ',', '\\(', '\\)', '==|!=|<=?|>=?', '\\+|\\-', '\\*|\\\\',
    '([A-Z]|[a-z]|_)([A-Z]|[a-z]|[0-9]|_)*', '(0|[1-9][0-9]*)(\\.[0-9]+)?', '#[^\n]*\n', '[ \n]+'
  ])

SLOPPY_GRAMMAR = Grammar(
  Production('TopLevelExpr', 'ExprList'),
  Production('ExprList'),
  Production('ExprList', 'Expr', 'ExprList'),
  Production('Expr', 'func', 'identifier', '(', 'IdentifierList', ')', '{', 'ExprList', '}'),
  Production('Expr', 'identifier', '(', 'ValList', ')', ';'),
  Production('Expr', 'return', 'Val', ';'),
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
SLOPPY_ATTR_GRAMMAR = AttributeGrammar(
  SLOPPY_GRAMMAR,
  syn_attrs={'ast', 'expr_list', 'identifier_list', 'val_list', 'identifier', 'op', 'number'},
  prod_attr_rules=[
    {'ast': lambda expr_list: TopLevelExpressionAst(expr_list(1))},
    {'expr_list': []},
    {'expr_list': lambda ast, expr_list: [ast(1)] + expr_list(2)},
    {'ast': lambda identifier, identifier_list, expr_list: (
      FunctionDeclarationAst(identifier(2), identifier_list(4), expr_list(7)))},
    {'ast': lambda identifier, val_list: CallExpressionAst(identifier(1), val_list(3))},
    {'ast': lambda ast: ReturnExpressionAst(ast(2))},
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

SLOPPY_PARSER = ParserGenerator(SLOPPY_GRAMMAR)


def _test_sloppy_make_ast(program):
  """
  :param str program:
  """
  print('---- input program:')
  print(program)
  tokens, token_words = SLOPPY_LEXER.tokenize(program)
  print('---- tokens:')
  print(tokens)
  analysis, eval = SLOPPY_PARSER.parse_syn_attr_analysis(SLOPPY_ATTR_GRAMMAR, tokens, token_words)
  ast = eval['ast']
  print('---- right-most analysis:')
  print(analysis)
  print('---- abstract syntax tree:')
  print(ast)
  assert isinstance(ast, TopLevelExpressionAst)


def test_sloopy_parser():
  program1 = 'hello_world(123);'
  _test_sloppy_make_ast(program1)
  program2 = """# This function will just return 4.
func do_stuff(val) {
  return 4;
}
do_stuff(7.5);
"""
  _test_sloppy_make_ast(program2)
  program3 = """
  # Compute 0 + 1 + ... + n
  func sum_all(n) {
    if n <= 0 { return 0; }
    else { return sum_all(n-1) + n; }
  }
  
  sum_all(12);
  """
  _test_sloppy_make_ast(program3)


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
