import _setup_test_env  # noqa
from nose.tools import assert_equal, assert_raises

from sleepy.errors import LexError
from sleepy.grammar import IGNORED_TOKEN


def test_tokenize_regex():
  from sleepy.regex import tokenize_regex
  assert_equal(tokenize_regex('abcd'), (('a', 'a', 'a', 'a'), (0, 1, 2, 3)))
  assert_equal(tokenize_regex('[a-z]*'), (('[', 'a', '-', 'a', ']', '*'), (0, 1, 2, 3, 4, 5)))
  assert_equal(tokenize_regex('\\\\\\?'), ((IGNORED_TOKEN, 'a', IGNORED_TOKEN, 'a'), (0, 1, 2, 3)))
  assert_equal(tokenize_regex('(b|c)+'), (('(', 'a', '|', 'a', ')', '+'), (0, 1, 2, 3, 4, 5)))
  assert_equal(
    tokenize_regex('[^a-z]*'), (('[', '^', 'a', '-', 'a', ']', '*'), (0, 1, 2, 3, 4, 5, 6)))
  with assert_raises(LexError):
    tokenize_regex('never ending\\')
  with assert_raises(LexError):
    tokenize_regex('escape \\me')


def test_make_regex_nfa_and_dfa():
  from sleepy.regex import make_regex_nfa
  from sleepy.automaton import make_dfa_from_nfa

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
  nfa2 = make_regex_nfa('1(0|1)*(\\.(0|1)+)?')
  dfa2 = make_dfa_from_nfa(nfa2)
  test_nfa_dfa_equal(nfa2, dfa2, '01', False)
  test_nfa_dfa_equal(nfa2, dfa2, '1', True)
  test_nfa_dfa_equal(nfa2, dfa2, '10000', True)
  test_nfa_dfa_equal(nfa2, dfa2, '1000011011101111111', True)
  test_nfa_dfa_equal(nfa2, dfa2, '10000.1011', True)
  test_nfa_dfa_equal(nfa2, dfa2, '1.', False)
  test_nfa_dfa_equal(nfa2, dfa2, 'x', False)
  nfa3 = make_regex_nfa('(file|https?)://(www\\.)?github\\.(com|de)?')
  dfa3 = make_dfa_from_nfa(nfa3)
  test_nfa_dfa_equal(nfa3, dfa3, 'file://www.github.com', True)
  test_nfa_dfa_equal(nfa3, dfa3, 'http://www.github.com', True)  # noqa  # PyCharm complains about http here xD
  test_nfa_dfa_equal(nfa3, dfa3, 'https://github.de', True)
  test_nfa_dfa_equal(nfa3, dfa3, 'https://github', False)
  test_nfa_dfa_equal(nfa3, dfa3, 'files://github.com', False)
  nfa4 = make_regex_nfa('[A-Z][a-z]*')
  dfa4 = make_dfa_from_nfa(nfa4)
  test_nfa_dfa_equal(nfa4, dfa4, 'Hallloooo', True)
  test_nfa_dfa_equal(nfa4, dfa4, 'H', True)
  test_nfa_dfa_equal(nfa4, dfa4, 'HALLO', False)
  test_nfa_dfa_equal(nfa4, dfa4, 'hallo', False)
  nfa5 = make_regex_nfa('Hell[aoe] world!')
  dfa5 = make_dfa_from_nfa(nfa5)
  test_nfa_dfa_equal(nfa5, dfa5, 'Hello world!', True)
  test_nfa_dfa_equal(nfa5, dfa5, 'Helle world!', True)
  test_nfa_dfa_equal(nfa5, dfa5, 'Helloe world!', False)
  test_nfa_dfa_equal(nfa5, dfa5, 'Hell world!', False)
  nfa6 = make_regex_nfa('[^a-z]+[a-z]+')
  dfa6 = make_dfa_from_nfa(nfa6)
  test_nfa_dfa_equal(nfa6, dfa6, '12311baaa', True)
  test_nfa_dfa_equal(nfa6, dfa6, 'bushof', False)
  test_nfa_dfa_equal(nfa6, dfa6, '42', False)
  test_nfa_dfa_equal(nfa6, dfa6, '   test', True)
  nfa7 = make_regex_nfa('Char: .')
  dfa7 = make_dfa_from_nfa(nfa7)
  test_nfa_dfa_equal(nfa7, dfa7, 'Char: 1', True)
  test_nfa_dfa_equal(nfa7, dfa7, 'Char:  ', True)
  test_nfa_dfa_equal(nfa7, dfa7, 'Char: \\', True)
  test_nfa_dfa_equal(nfa7, dfa7, 'Char: ', False)
  nfa8 = make_regex_nfa('[^ ]+')
  dfa8 = make_dfa_from_nfa(nfa8)
  test_nfa_dfa_equal(nfa8, dfa8, 'hello', True)
  test_nfa_dfa_equal(nfa8, dfa8, 'a', True)
  test_nfa_dfa_equal(nfa8, dfa8, '-', True)
  test_nfa_dfa_equal(nfa8, dfa8, '', False)
  nfa9 = make_regex_nfa('(0|[1-9][0-9]*)(\\.[0-9]+)?')
  dfa9 = make_dfa_from_nfa(nfa9)
  test_nfa_dfa_equal(nfa9, dfa9, '1*2', False)
  test_nfa_dfa_equal(nfa9, dfa9, '0', True)
