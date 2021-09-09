from sleepy.grammar import EPSILON
from typing import List, Dict, Set, FrozenSet


"""
General placeholder for all unproductive (error) states.
"""
ERROR_STATE = None


class NonDeterministicAutomaton:
  """
  A NFA with epsilon transitions.
  """

  def __init__(self, initial_state, final_states, state_transition_table):
    """
    :param int initial_state:
    :param set[int] final_states:
    :param list[dict[str, set[int]]] state_transition_table:
    """
    self.initial_state = initial_state
    self.final_states = final_states
    self.state_transition_table = state_transition_table

  def get_next_states(self, state, char):
    """
    :param int state:
    :param str|None char: char or epsilon
    :rtype: set[str]
    """
    return self.state_transition_table[state].get(char, set())

  def get_epsilon_closure(self, states):
    """
    Gets all states reachable from `states` only using epsilon transitions
    :param set[int]|list[int] states:
    :rtype: set[int]
    """
    state_set = set()
    to_add = list(states)
    while len(to_add) >= 1:
      state = to_add.pop()
      assert isinstance(state, int)
      if state in state_set:
        continue
      state_set.add(state)
      to_add.extend(self.get_next_states(state, EPSILON))
    return state_set

  def accepts(self, word):
    """
    :param str word:
    :rtype: bool
    """
    state_set = self.get_epsilon_closure({self.initial_state})
    for char in word:
      state_set = self.get_epsilon_closure({s for state in state_set for s in self.get_next_states(state, char)})
    return any(final_state in state_set for final_state in self.final_states)


class DeterministicAutomaton:
  """
  A DFA (without epsilon-transitions).
  """

  def __init__(self, initial_state, final_states, state_transition_table):
    """
    :param int initial_state:
    :param set[int] final_states:
    :param list[dict[str, int]] state_transition_table:
    """
    self.initial_state = initial_state
    self.final_states = final_states
    self.state_transition_table = state_transition_table

  def get_next_state(self, state, char):
    """
    :param int|None state:
    :param str|None char: char
    :returns: next state or `ERROR_STATE` if next state is not productive
    :rtype: str|None
    """
    if state is ERROR_STATE:
      return ERROR_STATE
    return self.state_transition_table[state].get(char, ERROR_STATE)

  def get_next_possible_chars(self, state):
    """
    :param int|None state:
    :returns: all characters that transition to a productive state
    :rtype: set[str]
    """
    if state is ERROR_STATE:
      return set()
    return set(self.state_transition_table[state].keys())

  def accepts(self, word):
    """
    :param str word:
    :rtype: bool
    """
    state = self.initial_state
    for char in word:
      if char not in self.state_transition_table[state]:
        return False
      state = self.state_transition_table[state][char]
    return state in self.final_states


def make_dfa_from_nfa(nfa):
  """
  Applies powerset construction.
  :param NonDeterministicAutomaton nfa:
  :rtype: DeterministicAutomaton
  """
  powerset_idx_transition_table: List[Dict[str, int]] = []
  final_powersets_idx: Set[int] = set()
  added_powersets: Dict[FrozenSet[int], int] = {}
  dirty_powersets: Set[FrozenSet[int]] = set()

  def add_next_powersets(powerset):
    """
    :param frozenset[int] powerset:
    """
    assert powerset in added_powersets
    powerset_idx = added_powersets[powerset]
    added_chars: Set[str] = set()
    for state in powerset:
      for char in nfa.state_transition_table[state].keys():
        if char in added_chars or char is EPSILON:
          continue
        next_powerset = frozenset(
          nfa.get_epsilon_closure({s for state in powerset for s in nfa.get_next_states(state, char)}))
        if next_powerset in added_powersets:
          next_powerset_idx = added_powersets[next_powerset]
        else:
          next_powerset_idx = len(added_powersets)
          powerset_idx_transition_table.insert(next_powerset_idx, {})
          if any(final_state in next_powerset for final_state in nfa.final_states):
            final_powersets_idx.add(next_powerset_idx)
          added_powersets[next_powerset] = next_powerset_idx
          dirty_powersets.add(next_powerset)
        added_chars.add(char)
        powerset_idx_transition_table[powerset_idx][char] = next_powerset_idx

  initial_powerset = frozenset(nfa.get_epsilon_closure({nfa.initial_state}))
  initial_powerset_idx = 0
  powerset_idx_transition_table.insert(initial_powerset_idx, {})
  if any(final_state in initial_powerset for final_state in nfa.final_states):
    final_powersets_idx.add(initial_powerset_idx)
  added_powersets[initial_powerset] = initial_powerset_idx
  dirty_powersets.add(initial_powerset)

  while len(dirty_powersets) >= 1:
    add_next_powersets(dirty_powersets.pop())

  return DeterministicAutomaton(initial_powerset_idx, final_powersets_idx, powerset_idx_transition_table)
