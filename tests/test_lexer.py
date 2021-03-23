import _setup_test_env  # noqa
import sys
import unittest
import better_exchook
from nose.tools import assert_equal, assert_raises

from sleepy.grammar import LexError
from sleepy.lexer import LexerGenerator


def test_LexerGenerator_simple():
  lexer = LexerGenerator(['1', '2', '3'], ['a+', 'a+b', 'b'])
  assert_equal(lexer.tokenize('aabab'), (('2', '2'), ('aab', 'ab')))
  assert_equal(lexer.tokenize('aabb'), (('2', '3'), ('aab', 'b')))
  assert_equal(lexer.tokenize('aa'), (('1',), ('aa',)))
  assert_equal(lexer.tokenize('baa'), (('3', '1'), ('b', 'aa')))
  with assert_raises(LexError):
    lexer.tokenize('abc')


def test_LexerGenerator_arithmetic():
  lexer = LexerGenerator(
    ['Func', 'Lit', 'Name', 'Op'], ['sin|cos|tan|exp', '[0-9]+', '([a-z]|[a-Z])([a-z]|[a-Z]|[0-9])*', '[\\+\\-\\*/]'])
  assert_equal(lexer.tokenize('sin+cos'), (('Func', 'Op', 'Func'), ('sin', '+', 'cos')))
  assert_equal(lexer.tokenize('a*b'), (('Name', 'Op', 'Name'), ('a', '*', 'b')))
  assert_equal(lexer.tokenize('si+3'), (('Name', 'Op', 'Lit'), ('si', '+', '3')))
  assert_equal(lexer.tokenize('index4/2'), (('Name', 'Op', 'Lit'), ('index4', '/', '2')))


def test_LexerGenerator_deleted_tokens():
  lexer = LexerGenerator(
    ['keyword', '=', 'op', ';', '(', ')', '{', '}', 'name', 'const', None], [
      'int|str|bool|if|for|while', '=', '<|>|==|<=|>=|!=', ';', '\\(', '\\)', '{', '}', '([a-z]|[A-Z]|_)+',
      '[1-9][0-9]*(\\.[0-9]*)?', ' +']
  )
  assert_equal(
    lexer.tokenize('int hello = 6; test = hello; while (hello >= 6) { print(hello); }'), ((
      'keyword', 'name', '=', 'const', ';', 'name', '=', 'name', ';', 'keyword', '(', 'name', 'op', 'const', ')', '{',
      'name', '(', 'name', ')', ';', '}'), (
      'int', 'hello', '=', '6', ';', 'test', '=', 'hello', ';', 'while', '(', 'hello', '>=', '6', ')', '{', 'print',
      '(', 'hello', ')', ';', '}')))
  assert_equal(
    lexer.tokenize('int my_int = 3;'), (('keyword', 'name', '=', 'const', ';'), ('int', 'my_int', '=', '3', ';')))


def test_LexerGenerator_comments():
  lexer = LexerGenerator(
    [None, 'keyword', '=', ';', 'name', 'const', None], [
      '(/\\*([^\\*]|\\*+[^/])*\\*+/)|(//[^\n]*\n?)', 'int', '=', ';', '([a-z]|[A-Z]|_)+', '[1-9][0-9]*(\\.[0-9]*)?',
      '[ \n]+']
  )
  assert_equal(lexer.tokenize('// hello world\n'), ((), ()))
  assert_equal(lexer.tokenize('// hello world\na = 6;'), (('name', '=', 'const', ';'), ('a', '=', '6', ';')))
  assert_equal(lexer.tokenize('/*one*/'), ((), ()))
  assert_equal(lexer.tokenize('/*one*//*two*/'), ((), ()))
  assert_equal(lexer.tokenize('/* one line \n two lines */'), ((), ()))
  assert_equal(lexer.tokenize('a = 5;\nb = 6;'), (
    ('name', '=', 'const', ';', 'name', '=', 'const', ';'), ('a', '=', '5', ';', 'b', '=', '6', ';')))
  assert_equal(lexer.tokenize('a = 5; /* set a to 5 */\na = 6;'),
    (('name', '=', 'const', ';', 'name', '=', 'const', ';'), ('a', '=', '5', ';', 'a', '=', '6', ';')))
  assert_equal(lexer.tokenize('a = 5; /* set a to 5 */\na = 6; /* and an extra comment */'),
    (('name', '=', 'const', ';', 'name', '=', 'const', ';'), ('a', '=', '5', ';', 'a', '=', '6', ';')))
  assert_equal(
    lexer.tokenize('a = 5; /* set a to 5 */\na = 6; // now set it to 6\n/* more\nand more */ b = 3;'),
    (('name', '=', 'const', ';', 'name', '=', 'const', ';', 'name', '=', 'const', ';'),
    ('a', '=', '5', ';', 'a', '=', '6', ';', 'b', '=', '3', ';'))
  )


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
