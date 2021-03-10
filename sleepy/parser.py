from sleepy.grammar import EPSILON, ParseError


def make_first1_sets(grammar):
  """
  :param grammar.Grammar grammar:
  :rtype: dict[str, set[str|None]]
  """
  # fi(x) = {x} for all terminals, compute non-terminals iteratively
  first1_sets = {
    symbol: set([symbol]) if symbol in grammar.terminals else set()
    for symbol in grammar.symbols}  # type: dict[str, set[str|None]]
  changes = True
  while changes:
    changes = False
    for symbol in grammar.non_terminals:
      first1 = set()
      for prod in grammar.get_prods_for(symbol):
        first1.update(get_first1_set_for_word(first1_sets, prod.right))
      if first1 != first1_sets[symbol]:
        first1_sets[symbol] = first1
        changes = True

  assert all(all(x in grammar.terminals or x is EPSILON for x in first1) for first1 in first1_sets.values())
  return first1_sets


def get_first1_set_for_word(first1_sets, word):
  """
  :param dict[str, str[str|None]] first1_sets:
  :param tuple[str] word:
  :rtype: set[str|None]
  """
  assert EPSILON not in word
  first1 = set()
  for pos, right_symbol in enumerate(word):
    # add all fi(X_{i+1}) to fi(A)
    first1.update(first1_sets.get(right_symbol, set()) - {EPSILON})
    if EPSILON not in first1_sets.get(right_symbol, set()):
      break
  # if A -> X1 ... Xn, and EPSILON in fi(X1),...,fi(Xn), then EPSILON in fi(A)
  if all(EPSILON in first1_sets.get(right, set()) for right in word):
    first1.add(EPSILON)
  return first1


class _Item:
  def __init__(self, prod, pointer, la):
    """
    :param grammar.Production prod:
    :param int pointer:
    :param str|None la: look-ahead, None iff epsilon
    """
    assert 0 <= pointer <= len(prod.right)
    self.prod = prod
    self.pointer = pointer
    self.la = la

  def __eq__(self, other):
    if not isinstance(other, _Item):
      return False
    return self.prod == other.prod and self.pointer == other.pointer and self.la == other.la

  def __repr__(self):
    return '_Item[%s -> %s, %r]' % (
      self.prod.left, ' '.join(self.prod.right[:self.pointer] + ('.',) + self.prod.right[self.pointer:]), self.la)

  def __hash__(self):
    return hash((self.prod, self.pointer, self.la))


class _Action:
  """
  Actions used by parser
  """


class _AcceptAction(_Action):
  def __repr__(self):
    return '_AcceptAction'


class _ReduceAction(_Action):
  def __init__(self, prod):
    """
    :param grammar.Production prod:
    """
    self.prod = prod

  def __repr__(self):
    return '_ReduceAction[%r]' % self.prod


class _ShiftAction(_Action):
  def __init__(self, symbol):
    assert symbol is not EPSILON
    self.symbol = symbol

  def __repr__(self):
    return '_ShiftAction[%r]' % self.symbol


class ParserGenerator:
  """
  A general LR(1) grammar parser generator.
  """

  def __init__(self, grammar):
    self.grammar = grammar
    assert self.grammar.is_start_separated()
    start_prods = self.grammar.get_prods_for(self.grammar.start)
    assert len(start_prods) == 1
    self._start_prod = start_prods[0]

    self._initial_state = 0
    self._num_states = None  # type: None|int
    self._state_action_table = None  # type: None|list[dict[str, _Action]]
    self._state_goto_table = None  # type: None|list[dict[str, int]]
    self._state_descr = None  # type: None|list[str]

    self._make()

  def _make_item_set(self, initial_items):
    """
    Makes the smallest state s at least containing `initial_items`,
    computes its action table act(s,x) -> action (where x is a symbol or EPSILON),
    as well as a set of terminals a s.t. goto(s,a) != sink.
    :param list[_LrItem] initial_items:
    :returns: Completed item set, action function, next symbols
    :rtype: tuple[frozenset[_LrItem], dict[str, _LrAction], list[str]]
    """
    first1_sets = make_first1_sets(self.grammar)
    add_item_queue = initial_items.copy()

    state = set()
    actions = {}  # type: dict[str, _Action]
    next_symbols = set()

    while len(add_item_queue) >= 1:
      item = add_item_queue.pop(-1)
      if item in state:
        continue
      state.add(item)
      if item.pointer == len(item.prod.right):
        # Reduce A -> alpha .
        assert item.la not in actions, 'Grammar not LR(1)! Partial item set %r: item %r action conflicts with %r' % (
          state, item, actions)
        if item.prod != self._start_prod:
          actions[item.la] = _ReduceAction(item.prod)
        else:
          actions[item.la] = _AcceptAction()
        continue
      next_symbol = item.prod.right[item.pointer]
      next_symbols.add(next_symbol)
      if next_symbol in self.grammar.non_terminals:
        # If A-> alpha . B beta in state, add all [B -> . gamma, fi(beta x)]
        word_after_next_symbol = item.prod.right[item.pointer+1:] + (() if item.la is EPSILON else (item.la,))
        add_item_queue.extend([_Item(p, 0, new_la)
          for p in self.grammar.get_prods_for(next_symbol)
          for new_la in get_first1_set_for_word(first1_sets, word_after_next_symbol)])
      else:
        assert next_symbol in self.grammar.terminals
        # Shift A -> . a beta
        assert (
            next_symbol not in actions or
            (isinstance(actions[next_symbol], _ShiftAction) and actions[next_symbol].symbol == next_symbol)), (
            'Grammar not LR(1)! Partial item set %r: item %r action conflicts with %r' % (state, item, actions))
        actions[next_symbol] = _ShiftAction(next_symbol)
    return frozenset(state), actions, sorted(next_symbols)

  def _make(self):
    """
    Construct the automaton.
    """
    states = {}  # type: dict[frozenset[_Item], int]
    state_action_table = []  # type: list[dict[str, _Action]]
    state_goto_table = []  # type: list[dict[str, int]]

    def add_next_state(from_state, symbol):
      """
      Recursively adds the states reachable from `from_state` when first processing `symbol`.
      Populates`states`, `state_action_table` and `state_goto_table` for all added states
      as well as `state_goto_table` of `from_state`.
      :param frozenset[_LrItem] from_state: reachable (non-sink) state
      :param str symbol: a non-terminal
      """
      assert len(from_state) > 0, 'from_state should be reachable, i.e. not the sink state'
      assert symbol is not EPSILON
      to_state, to_state_actions, next_symbols = self._make_item_set([
        _Item(item.prod, item.pointer + 1, item.la)
        for item in from_state if item.pointer < len(item.prod.right) and item.prod.right[item.pointer] == symbol])
      from_state_idx = states[from_state]
      to_state_idx = states.get(to_state, len(states))
      if symbol in state_goto_table[from_state_idx]:
        assert state_goto_table[from_state_idx][symbol] == to_state_idx
      else:
        state_goto_table[from_state_idx][symbol] = to_state_idx
      if to_state in states:
        return
      assert len(state_action_table) < to_state_idx + 1 and len(state_goto_table) < to_state_idx + 1
      states[to_state] = to_state_idx
      state_action_table.append(to_state_actions)
      state_goto_table.append({})
      for next_symbol in next_symbols:
        add_next_state(to_state, next_symbol)

    initial_state, initial_actions, initial_next_symbols = self._make_item_set([_Item(self._start_prod, 0, EPSILON)])
    states[initial_state] = 0
    state_action_table.append(initial_actions)
    state_goto_table.append({})
    for initial_symbol in initial_next_symbols:
      add_next_state(initial_state, initial_symbol)

    self._num_states = len(states)
    self._state_action_table = state_action_table
    self._state_goto_table = state_goto_table
    self._state_descr = ['{%s}' % ', '.join([repr(item) for item in state]) for state in states.keys()]

  def parse_analysis(self, word):
    """
    :param list[str] word:
    :rtype: list[Production]:
    :raises: ParseError
    :returns: a right-most analysis of `word` or raises ParseError
    """
    assert EPSILON not in word

    accepted = False
    pos = 0
    state_stack = [self._initial_state]
    rev_analysis = []  # type: list[Production]

    while not accepted:
      la = EPSILON if pos == len(word) else word[pos]
      state = state_stack[-1]
      action = self._state_action_table[state].get(la)
      if isinstance(action, _ShiftAction) and action.symbol == la:
        pos += 1
        state_stack.append(self._state_goto_table[state][action.symbol])
      elif isinstance(action, _ReduceAction):
        for i in range(len(state_stack) - 1, len(state_stack) - len(action.prod.right) - 1, -1):
          del state_stack[i]
        state_stack.append(self._state_goto_table[state_stack[-1]][action.prod.left])
        rev_analysis.append(action.prod)
      elif isinstance(action, _AcceptAction) and len(state_stack) == 2:
        assert state_stack[0] == self._initial_state
        state_stack.clear()
        rev_analysis.append(self._start_prod)
        accepted = True
      else:  # error
        raise ParseError(
          word, pos, 'No action in state %s (%s) with lookahead %r possible' % (state, self._state_descr[state], la))

    assert rev_analysis[-1] == self._start_prod
    return list(reversed(rev_analysis))