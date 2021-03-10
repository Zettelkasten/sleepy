from typing import Dict, Tuple, Set, Union, Optional

from sleepy.automaton import ERROR_STATE
from sleepy.grammar import LexError
from sleepy.regex import make_regex_dfa


class LexerGenerator:
  """
  Implements a backtracking DFA.
  """

  def __init__(self, token_names, token_regex_table):
    """
    :param list[str] token_names: list of token names, sorted by priority.
    :param list[str] token_regex_table: corresponding regex's, not recognizing the empty word.
    """
    assert len(token_names) == len(token_regex_table)
    self.token_names = token_names
    self.token_regex_table = token_regex_table

    self._automatons = [make_regex_dfa(regex) for regex in self.token_regex_table]
    assert all(ERROR_STATE not in dfa.state_transition_table for dfa in self._automatons)
    assert all(dfa.initial_state not in dfa.final_states for dfa in self._automatons), (
      'all regex must not recognize the empty word')
    self._initial_state = tuple(dfa.initial_state for dfa in self._automatons)
    self._final_states = {}  # type: Dict[Tuple[Union[int,ERROR_STATE]],str]
    self._make_final_states()

  def _get_next_state(self, state, char):
    """
    :param tuple[int|None]|None state:
    :param str char:
    :returns: the next state, or `ERROR_STATE` if next state is not productive.
    :rtype: tuple[int|None]|None
    """
    if state is ERROR_STATE:
      return ERROR_STATE
    next_state = tuple(dfa.get_next_state(state[i], char) for i, dfa in enumerate(self._automatons))
    if all(dfa_state == ERROR_STATE for dfa_state in next_state):
      return ERROR_STATE
    else:
      return next_state

  def _make_final_states(self):
    """
    Compute and set `_final_states`.
    """
    self._final_states.clear()
    states_to_check = {self._initial_state}  # type: Set[Tuple[Union[int,LexerGenerator.ERROR_STATE]]]
    visited_states = set()  # type: Set[Tuple[Union[int,LexerGenerator.ERROR_STATE]]]
    while len(states_to_check) >= 1:
      state = states_to_check.pop()
      if state in visited_states:
        continue
      visited_states.add(state)
      final_state_of = [i for i, dfa in enumerate(self._automatons) if state[i] in dfa.final_states]
      if len(final_state_of) >= 1:
        self._final_states[state] = self.token_names[min(final_state_of)]
      possible_next_chars = {
        char for i, dfa in enumerate(self._automatons) if state[i] is not None
        for char in dfa.state_transition_table[state[i]]}
      states_to_check.update(self._get_next_state(state, char) for char in possible_next_chars)

  def tokenize(self, word):
    """
    Find first longest matching analysis.
    :param str word:
    :returns: token analysis + decomposition (i.e. token attribute table).
    :raises: LexError
    :rtype: tuple[tuple[str],tuple[str]]
    """
    pos = 0
    state = self._initial_state
    backtrack_mode = None  # type: Optional[str]
    backtrack_pos = None  # type: Optional[int]
    analysis = []  # type: List[str]
    decomposition = []  # type: List[str]
    token_begin_pos = 0

    def do_backtrack():
      """
      Do backtracking.
      """
      nonlocal pos, state, backtrack_mode, backtrack_pos, token_begin_pos
      assert backtrack_mode in self.token_names
      assert backtrack_pos is not None
      state = self._initial_state
      analysis.append(backtrack_mode)
      decomposition.append(word[token_begin_pos:pos])
      pos, token_begin_pos = backtrack_pos, backtrack_pos
      backtrack_mode, backtrack_pos = None, None
      assert state not in self._final_states

    while pos <= len(word):
      if pos == len(word):  # reached end-of-word
        if pos == token_begin_pos:
          break
        # unfinished word remaining
        if backtrack_mode is None:  # normal mode
          raise LexError(word, pos, 'Missing more characters, so far read %r / %r' % (analysis, decomposition))
        else:  # backtracking mode
          do_backtrack()
        continue
      assert pos < len(word)
      char = word[pos]
      state = self._get_next_state(state, char)
      if state is ERROR_STATE:
        if backtrack_mode is None:  # normal mode, no backtracking yet
          raise LexError(word, pos, 'Unrecognized pattern')
        else:  # backtracking mode, needs to backtrack now
          do_backtrack()
      else:
        pos += 1
      if state in self._final_states:
        backtrack_mode = self._final_states[state]
        backtrack_pos = pos

    return tuple(analysis), tuple(decomposition)
