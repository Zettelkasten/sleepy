from grammar import EPSILON


class NonDeterministicAutomaton():
  """
  A NFA with epsilon transitions.
  """

  def __init__(self, initial_state, final_states, state_transition_table):
    """
    :param int initial_state:
    :param set[int] final_states:
    :param list[dict[str, str[int]]] state_transition_table:
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
