import _setup_test_env  # noqa
from better_exchook import better_exchook
import sys
import unittest

from sleepy.grammar import Grammar, Production, SyntaxTree
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

# Operator precendence: * / stronger than + - stronger than == != < <= > >=
SLOPPY_GRAMMAR = Grammar(
  Production('TopLevelExpr', 'ExprList'),
  Production('ExprList'),
  Production('ExprList', 'Expr', 'ExprList'),
  Production('Expr', 'func', 'identifier', '(', 'IdentifierList', ')', '{', 'ExprList', '}'),
  Production('IdentifierList', ''),
  Production('IdentifierList', 'IdentifierList+'),
  Production('IdentifierList+', 'identifier'),
  Production('IdentifierList+', 'identifier', ',', 'IdentifierList+'),
  Production('Expr', 'Val', ';'),
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
  Production('ValList', ''),
  Production('ValList', 'ValList+'),
  Production('ValList+', 'Val'),
  Production('ValList+', 'Val', ',', 'ValList+'),
)

SLOPPY_PARSER = ParserGenerator(SLOPPY_GRAMMAR)


def _test_sloppy_make_ast(program):
  """
  :param str program:
  :rtype: SyntaxTree
  """
  print('---- input program:')
  print(program)
  tokens, token_words = SLOPPY_LEXER.tokenize(program)
  print('---- tokens:')
  print(tokens)
  tree = SLOPPY_PARSER.parse_tree(tokens, token_words)
  print('---- parse tree:')
  print(tree)
  return tree


def test_sloopy_parser():
  program1 = '1.3;'
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
