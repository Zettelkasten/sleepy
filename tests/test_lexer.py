import _setup_test_env  # noqa
import sys
import unittest
import better_exchook
from nose.tools import assert_equal, assert_raises

from sleepy.errors import LexError
from sleepy.grammar import IGNORED_TOKEN
from sleepy.lexer import LexerGenerator


def test_LexerGenerator_simple():
  lexer = LexerGenerator(['1', '2', '3'], ['a+', 'a+b', 'b'])
  assert_equal(lexer.tokenize('aabab'), (('2', '2'), (0, 3)))
  assert_equal(lexer.tokenize('aabb'), (('2', '3'), (0, 3)))
  assert_equal(lexer.tokenize('aa'), (('1',), (0,)))
  assert_equal(lexer.tokenize('baa'), (('3', '1'), (0, 1)))
  with assert_raises(LexError):
    lexer.tokenize('abc')


def test_LexerGenerator_arithmetic():
  lexer = LexerGenerator(
    ['Func', 'Lit', 'Name', 'Op'], ['sin|cos|tan|exp', '[0-9]+', '([a-z]|[a-Z])([a-z]|[a-Z]|[0-9])*', '[\\+\\-\\*/]'])
  assert_equal(lexer.tokenize('sin+cos'), (('Func', 'Op', 'Func'), (0, 3, 4)))
  assert_equal(lexer.tokenize('a*b'), (('Name', 'Op', 'Name'), (0, 1, 2)))
  assert_equal(lexer.tokenize('si+3'), (('Name', 'Op', 'Lit'), (0, 2, 3)))
  assert_equal(lexer.tokenize('index4/2'), (('Name', 'Op', 'Lit'), (0, 6, 7)))


def test_LexerGenerator_deleted_tokens():
  lexer = LexerGenerator(
    ['keyword', '=', 'op', ';', '(', ')', '{', '}', 'name', 'const', IGNORED_TOKEN], [
      'int|str|bool|if|for|while', '=', '<|>|==|<=|>=|!=', ';', '\\(', '\\)', '{', '}', '([a-z]|[A-Z]|_)+',
      '[1-9][0-9]*(\\.[0-9]*)?', ' +']
  )
  assert_equal(
    lexer.tokenize('int hello = 6; test = hello; while (hello >= 6) { print(hello); }'), (
      (
        'keyword', IGNORED_TOKEN, 'name', IGNORED_TOKEN, '=', IGNORED_TOKEN, 'const', ';', IGNORED_TOKEN, 'name',
        IGNORED_TOKEN, '=', IGNORED_TOKEN, 'name', ';', IGNORED_TOKEN, 'keyword', IGNORED_TOKEN, '(', 'name',
        IGNORED_TOKEN, 'op', IGNORED_TOKEN, 'const', ')', IGNORED_TOKEN, '{', IGNORED_TOKEN, 'name', '(', 'name', ')',
        ';', IGNORED_TOKEN, '}'), (
        0, 3, 4, 9, 10, 11, 12, 13, 14, 15, 19, 20, 21, 22, 27, 28, 29, 34, 35, 36, 41, 42, 44, 45, 46, 47, 48, 49, 50,
        55, 56, 61, 62, 63, 64
      )))
  assert_equal(
    lexer.tokenize('int my_int = 3;'), (
      ('keyword', IGNORED_TOKEN, 'name', IGNORED_TOKEN, '=', IGNORED_TOKEN, 'const', ';'),
      (0, 3, 4, 10, 11, 12, 13, 14)))


def test_LexerGenerator_comments():
  lexer = LexerGenerator(
    [IGNORED_TOKEN, 'keyword', '=', ';', 'name', 'const', IGNORED_TOKEN], [
      '(/\\*([^\\*]|\\*+[^/])*\\*+/)|(//[^\n]*\n?)', 'int', '=', ';', '([a-z]|[A-Z]|_)+', '[1-9][0-9]*(\\.[0-9]*)?',
      '[ \n]+']
  )
  assert_equal(lexer.tokenize('// hello world\n'), ((IGNORED_TOKEN,), (0,)))
  assert_equal(
    lexer.tokenize('// hello world\na = 6;'),
    ((IGNORED_TOKEN, 'name', IGNORED_TOKEN, '=', IGNORED_TOKEN, 'const', ';'), (0, 15, 16, 17, 18, 19, 20)))
  assert_equal(lexer.tokenize('/*one*/'), ((IGNORED_TOKEN,), (0,)))
  assert_equal(lexer.tokenize('/*one*//*two*/'), ((IGNORED_TOKEN, IGNORED_TOKEN), (0, 7)))
  assert_equal(lexer.tokenize('/*one*/    '), ((IGNORED_TOKEN, IGNORED_TOKEN), (0, 7)))
  assert_equal(lexer.tokenize(' /*one*/    '), ((IGNORED_TOKEN, IGNORED_TOKEN, IGNORED_TOKEN), (0, 1, 8)))
  assert_equal(lexer.tokenize('/* one line \n two lines */'), ((IGNORED_TOKEN,), (0,)))
  assert_equal(lexer.tokenize('a = 5;\nb = 6;'), (
    (
      'name', IGNORED_TOKEN, '=', IGNORED_TOKEN, 'const', ';', IGNORED_TOKEN, 'name', IGNORED_TOKEN, '=', IGNORED_TOKEN,
      'const', ';'),
    (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12)))
  assert_equal(
    lexer.tokenize('a = 5; /* set a to 5 */\na = 6;'),
    (
      (
        'name', IGNORED_TOKEN, '=', IGNORED_TOKEN, 'const', ';', IGNORED_TOKEN, IGNORED_TOKEN, IGNORED_TOKEN, 'name',
        IGNORED_TOKEN, '=', IGNORED_TOKEN, 'const', ';'),
      (0, 1, 2, 3, 4, 5, 6, 7, 23, 24, 25, 26, 27, 28, 29)))
  assert_equal(
    lexer.tokenize('a = 5; /* set a to 5 */\na = 6; /* and an extra comment */'),
    (
      (
        'name', IGNORED_TOKEN, '=', IGNORED_TOKEN, 'const', ';', IGNORED_TOKEN, IGNORED_TOKEN, IGNORED_TOKEN, 'name',
        IGNORED_TOKEN, '=', IGNORED_TOKEN, 'const', ';', IGNORED_TOKEN, IGNORED_TOKEN),
      (0, 1, 2, 3, 4, 5, 6, 7, 23, 24, 25, 26, 27, 28, 29, 30, 31)))
  print(lexer.tokenize('a = 5; /* set a to 5 */\na = 6; // now set it to 6\n/* more\nand more */ b = 3;'))
  assert_equal(
    lexer.tokenize('a = 5; /* set a to 5 */\na = 6; // now set it to 6\n/* more\nand more */ b = 3;'),
    (
      (
        'name', IGNORED_TOKEN, '=', IGNORED_TOKEN, 'const', ';', IGNORED_TOKEN, IGNORED_TOKEN, IGNORED_TOKEN, 'name',
        IGNORED_TOKEN, '=', IGNORED_TOKEN, 'const', ';', IGNORED_TOKEN, IGNORED_TOKEN, IGNORED_TOKEN, IGNORED_TOKEN,
        'name', IGNORED_TOKEN, '=', IGNORED_TOKEN, 'const', ';'),
      (0, 1, 2, 3, 4, 5, 6, 7, 23, 24, 25, 26, 27, 28, 29, 30, 31, 50, 69, 70, 71, 72, 73, 74, 75)))
