import sys
import unittest
import better_exchook
from nose.tools import assert_equal, assert_raises

from grammar import LexError


def test_tokenize_regex():
  from regex import tokenize_regex
  assert_equal(tokenize_regex('abcd'), (('a', 'a', 'a', 'a'), ('a', 'b', 'c', 'd')))
  assert_equal(tokenize_regex('[a-z]*'), (('[', 'a', '-', 'a', ']', '*'), (None, 'a', None, 'z', None, None)))
  assert_equal(tokenize_regex('\\\\\\?'), (('a', 'a'), ('\\', '?')))
  assert_equal(tokenize_regex('(b|c)+'), (('(', 'a', '|', 'a', ')', '+'), (None, 'b', None, 'c', None, None)))
  with assert_raises(LexError):
    tokenize_regex('never ending\\')
  with assert_raises(LexError):
    tokenize_regex('escape \\me')


def test_make_regex_nfa():
  from regex import make_regex_nfa
  nfa1 = make_regex_nfa('ab|c+')
  assert_equal(nfa1.accepts('ab'), True)
  assert_equal(nfa1.accepts('c'), True)
  assert_equal(nfa1.accepts('cccc'), True)
  assert_equal(nfa1.accepts('a'), False)
  assert_equal(nfa1.accepts('b'), False)
  assert_equal(nfa1.accepts(''), False)
  nfa2 = make_regex_nfa('1(0|1)*(.(0|1)+)?')
  assert_equal(nfa2.accepts('01'), False)
  assert_equal(nfa2.accepts('1'), True)
  assert_equal(nfa2.accepts('10000'), True)
  assert_equal(nfa2.accepts('10000.1011'), True)
  assert_equal(nfa2.accepts('1.'), False)
  assert_equal(nfa2.accepts('x'), False)


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
