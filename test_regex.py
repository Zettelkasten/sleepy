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


def test_make_regex_nfa_and_dfa():
  from regex import make_regex_nfa
  from automaton import make_dfa_from_nfa

  def test_nfa_dfa_equal(nfa, dfa, word, should_accept):
    """
    :param NonDeterministicAutomaton nfa:
    :param DeterministicAutomaton dfa:
    :param str word:
    :param bool should_accept:
    """
    nfa_accepts = nfa.accepts(word)
    assert_equal(nfa_accepts, should_accept)
    dfa_accepts = dfa.accepts(word)
    assert_equal(dfa_accepts, should_accept)

  nfa1 = make_regex_nfa('ab|c+')
  dfa1 = make_dfa_from_nfa(nfa1)
  test_nfa_dfa_equal(nfa1, dfa1, 'ab', True)
  test_nfa_dfa_equal(nfa1, dfa1, 'c', True)
  test_nfa_dfa_equal(nfa1, dfa1, 'cccc', True)
  test_nfa_dfa_equal(nfa1, dfa1, 'a', False)
  test_nfa_dfa_equal(nfa1, dfa1, 'b', False)
  test_nfa_dfa_equal(nfa1, dfa1, '', False)
  nfa2 = make_regex_nfa('1(0|1)*(.(0|1)+)?')
  dfa2 = make_dfa_from_nfa(nfa2)
  test_nfa_dfa_equal(nfa2, dfa2, '01', False)
  test_nfa_dfa_equal(nfa2, dfa2, '1', True)
  test_nfa_dfa_equal(nfa2, dfa2, '10000', True)
  test_nfa_dfa_equal(nfa2, dfa2, '1000011011101111111', True)
  test_nfa_dfa_equal(nfa2, dfa2, '10000.1011', True)
  test_nfa_dfa_equal(nfa2, dfa2, '1.', False)
  test_nfa_dfa_equal(nfa2, dfa2, 'x', False)
  nfa3 = make_regex_nfa('(file|https?)://(www.)?github.(com|de)?')
  dfa3 = make_dfa_from_nfa(nfa3)
  test_nfa_dfa_equal(nfa3, dfa3, 'file://www.github.com', True)
  test_nfa_dfa_equal(nfa3, dfa3, 'http://www.github.com', True)
  test_nfa_dfa_equal(nfa3, dfa3, 'https://github.de', True)
  test_nfa_dfa_equal(nfa3, dfa3, 'https://github', False)
  test_nfa_dfa_equal(nfa3, dfa3, 'files://github.com', False)


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
