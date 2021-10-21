from typing import Dict, Tuple, Set, Union, Optional, List

from sleepy.automaton import ERROR_STATE, OTHER_CHAR
from sleepy.errors import LexError
from sleepy.regex import make_regex_dfa, REGEX_RECOGNIZED_CHARS

State = Union[int, ERROR_STATE]
Comp_State = Tuple[State]


class LexerGenerator:
  """
  Implements a backtracking DFA.
  """

  NORMAL_MODE = object()

  def __init__(self, token_names: List[Optional[str]], token_regex_table: List[str]):
    """
    :param token_names: list of token names, sorted by priority.
      Tokens with name `IGNORE_TOKEN` will be ignored later (e.g. for whitespace, comments, etc.).
    :param token_regex_table: corresponding regex's, not recognizing the empty word.
    """
    self.cache_hits = 0
    self.cache_misses = 0
    self.recognized_chars = REGEX_RECOGNIZED_CHARS

    self.transition_table: Dict[Tuple[Comp_State, str], Tuple[Comp_State, bool]] = dict()

    assert len(token_names) == len(token_regex_table)
    self.token_names = token_names
    self.token_regex_table = token_regex_table

    self._automatons = [make_regex_dfa(regex) for regex in self.token_regex_table]
    assert all(ERROR_STATE not in dfa.state_transition_table for dfa in self._automatons)
    assert all(dfa.initial_state not in dfa.final_states for dfa in self._automatons), (
      'all regex must not recognize the empty word')
    self._initial_state = tuple(dfa.initial_state for dfa in self._automatons)
    self._final_states: Dict[Comp_State, str] = {}
    self._make_final_states()

  @classmethod
  def from_dict(cls, token_names_to_regex: Dict[str, str], ignore_token_regexes: List[str]):
    """
    :param token_names_to_regex: pairs token name -> regex. highest priority first
    :param ignore_token_regexes: regexes for tokens to ignore, will have lower priority than other tokens
    """
    assert None not in token_names_to_regex
    token_names = list(token_names_to_regex.keys()) + [None] * len(ignore_token_regexes)  # noqa
    token_regex_table = list(token_names_to_regex.values()) + ignore_token_regexes
    return LexerGenerator(token_names=token_names, token_regex_table=token_regex_table)

  def _get_next_state(self, state: Optional[Comp_State], char: str) -> Optional[Comp_State]:
    """
    :returns: the next state, or `ERROR_STATE` if next state is not productive.
    """
    if char not in self.recognized_chars:
      char = OTHER_CHAR
    if state is ERROR_STATE:
      return ERROR_STATE
    next_state, is_error = self.transition_table.get((state, char), (None, None))
    if next_state is None:
      self.cache_misses += 1
      next_state = tuple(
        dfa.get_next_state(state[i], char) for i, dfa in enumerate(self._automatons))  # type: Tuple[Optional[str]]
      is_error = all(dfa_state == ERROR_STATE for dfa_state in next_state)
      self.transition_table[(state, char)] = (next_state, is_error)
    else:
      self.cache_hits += 1

    return ERROR_STATE if is_error else next_state

  def _get_next_possible_chars(self, state):
    """
    :param tuple[int|None]|None state:
    :returns: all characters that transition to a productive state
    :rtype: set[str]
    """
    if state is ERROR_STATE:
      return set()
    return {char for i, dfa in enumerate(self._automatons) for char in dfa.get_next_possible_chars(state[i])}

  def _make_final_states(self):
    """
    Compute and set `_final_states`.
    """
    self._final_states.clear()
    states_to_check: Set[Tuple[Union[int, ERROR_STATE]]] = {self._initial_state}
    visited_states: Set[Tuple[Union[int, ERROR_STATE]]] = set()
    while len(states_to_check) >= 1:
      state = states_to_check.pop()
      if state in visited_states:
        continue
      if state is ERROR_STATE:
        continue
      visited_states.add(state)
      final_state_of = [i for i, dfa in enumerate(self._automatons) if state[i] in dfa.final_states]
      if len(final_state_of) >= 1:
        self._final_states[state] = self.token_names[min(final_state_of)]
      possible_next_chars = {
        char for i, dfa in enumerate(self._automatons) if state[i] is not None
        for char in dfa.state_transition_table[state[i]]}
      states_to_check.update(self._get_next_state(state, char) for char in possible_next_chars)

  def tokenize(self, word: str) -> Tuple[Tuple[str], Tuple[int]]:
    """
    Find first longest matching analysis.
    :returns: token analysis + decomposition (i.e. positions where tokens start).
    :raises: LexError
    """
    pos = 0
    state = self._initial_state
    backtrack_mode: Union[str, LexerGenerator.NORMAL_MODE] = self.NORMAL_MODE
    backtrack_pos: Optional[int] = None
    analysis: List[str] = []
    decomposition: List[int] = []
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
      decomposition.append(token_begin_pos)
      pos, token_begin_pos = backtrack_pos, backtrack_pos
      backtrack_mode, backtrack_pos = self.NORMAL_MODE, None
      assert state not in self._final_states

    word_length = len(word)
    while pos <= word_length:
      if pos == word_length:  # reached end-of-word
        if pos == token_begin_pos:
          break
        # unfinished word remaining
        if backtrack_mode is self.NORMAL_MODE:  # normal mode
          raise LexError(word, pos, 'Missing more characters, expected %r' % self._get_next_possible_chars(state))
        else:  # backtracking mode
          do_backtrack()
        continue
      assert pos < word_length
      char = word[pos]
      prev_state = state
      state = self._get_next_state(state, char)
      if state is ERROR_STATE:
        if backtrack_mode is self.NORMAL_MODE:  # normal mode, no backtracking yet
          raise LexError(word, pos, 'Unrecognized pattern to continue token %r, expected: %s' % (
            word[token_begin_pos:pos], ', '.join(['%r' % c for c in self._get_next_possible_chars(prev_state)])))
        else:  # backtracking mode, needs to backtrack now
          do_backtrack()
      else:
        pos += 1
      if state in self._final_states:
        backtrack_mode = self._final_states[state]
        backtrack_pos = pos

    return tuple(analysis), tuple(decomposition)