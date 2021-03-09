from automaton import NonDeterministicAutomaton
from grammar import Grammar, Production, LexError, EPSILON
from parser import ParserGenerator

REGEX_LIT_TOKEN = 'a'
REGEX_SPECIAL_TOKENS = frozenset({'(', ')', '\\', '-', '[', ']', '*', '+', '?', '|'})

REGEX_CHOICE_OP = Production('Choice', 'Choice', '|', 'Concat')
REGEX_CONCAT_OP = Production('Concat', 'Concat', 'Repeat')
REGEX_REPEAT_OP = Production('Repeat', 'Range', '*')
REGEX_REPEAT_EXISTS_OP = Production('Repeat', 'Range', '+')
REGEX_OPTIONAL_OP = Production('Repeat', 'Range', '?')
REGEX_RANGE_OP = Production('Range', '[', 'Lit', '-', 'Lit', ']')
REGEX_RANGE_LITS_OP = Production('Range', '[', 'Lits', ']')
REGEX_INV_RANGE_OP = Production('Range', '[', '^', 'Lit', '-', 'Lit', ']')
REGEX_INV_RANGE_LITS_OP = Production('Range', '[', '^', 'Lits', ']')
REGEX_LIT_OP = Production('Range', 'Lit')
REGEX_LITS_MULTIPLE_OP = Production('Lits', 'Lit', 'Lits')
REGEX_LITS_SINGLE_OP = Production('Lits', 'Lit')

REGEX_GRAMMAR = Grammar(
  Production('Regex', 'Choice'),
  REGEX_CHOICE_OP,
  Production('Choice', 'Concat'),
  REGEX_CONCAT_OP,
  Production('Concat', 'Repeat'),
  REGEX_REPEAT_OP, REGEX_REPEAT_EXISTS_OP, REGEX_OPTIONAL_OP,
  Production('Repeat', 'Range'),
  REGEX_RANGE_OP, REGEX_RANGE_LITS_OP, REGEX_INV_RANGE_OP, REGEX_INV_RANGE_LITS_OP,
  REGEX_LIT_OP,
  Production('Range', '(', 'Choice', ')'),
  REGEX_LITS_MULTIPLE_OP, REGEX_LITS_SINGLE_OP,
  Production('Lit', REGEX_LIT_TOKEN)
)
REGEX_PARSER = ParserGenerator(REGEX_GRAMMAR)


def tokenize_regex(word):
  """
  :param str word:
  :raises: LexError
  :returns: word tokens with attribute table
  :rtype: tuple[tuple[str], tuple[str]]
  """
  escape_next = False
  tokens, attribute_table = [], []
  for pos, c in enumerate(word):
    if escape_next:
      if c not in REGEX_SPECIAL_TOKENS:
        raise LexError(word, pos, 'Cannot escape character %r' % c)
      escape_next = False
      tokens.append(REGEX_LIT_TOKEN)
      attribute_table.append(c)
    elif c == '\\':
      escape_next = True
    elif c in REGEX_SPECIAL_TOKENS:
      tokens.append(c)
      attribute_table.append(None)
    else:  # default case
      tokens.append(REGEX_LIT_TOKEN)
      attribute_table.append(c)

  if escape_next:
    raise LexError(word, len(word), 'Cannot end word with escape character')
  assert len(tokens) == len(attribute_table) <= len(word)
  return tuple(tokens), tuple(attribute_table)


def make_regex_nfa(regex):
  """
  :param str regex:
  :rtype: NonDeterministicAutomaton
  """
  tokens, token_attribute_table = tokenize_regex(regex)
  analysis = REGEX_PARSER.parse_analysis(tokens)
  REGEX_PARSER.parse_analysis(tokens)

  num_states = 0
  state_transition_table = []  # type: list[dict[str|None, set[int]]]
  # TODO: replace this with attribute grammars when we have that
  # for each subregex, build a epsilon-NFA with start state in from_stack, and single final state in to_stack.
  # invariant: no ingoing transitions into start, no outgoing transitions from end
  from_stack = []  # type: list[int]
  to_stack = []  # type: list[int]

  pos = 0
  for prod in reversed(analysis):
    from_state, to_state = num_states, num_states + 1
    if prod == REGEX_LIT_OP:
      # assume that REGEX_LIT_OP is only production generating 'a'
      while tokens[pos] != REGEX_LIT_TOKEN:
        pos += 1
      a = token_attribute_table[pos]
      pos += 1
      # build DFA for a
      state_transition_table.append({a: {to_state}})
      state_transition_table.append({})
    elif prod == REGEX_CHOICE_OP:
      b_start, b_end, a_start, a_end = from_stack.pop(), to_stack.pop(), from_stack.pop(), to_stack.pop()
      # build DFA for a|b
      state_transition_table.append({EPSILON: {a_start, b_start}})
      assert EPSILON not in state_transition_table[a_end]
      state_transition_table[a_end][EPSILON] = {to_state}
      assert EPSILON not in state_transition_table[b_end]
      state_transition_table[b_end][EPSILON] = {to_state}
      state_transition_table.append({})
    elif prod == REGEX_CONCAT_OP:
      b_start, b_end = from_stack.pop(), to_stack.pop()
      a_start, a_end = from_stack.pop(), to_stack.pop()
      # build DFA for ab
      state_transition_table.append({EPSILON: {a_start}})
      assert EPSILON not in state_transition_table[a_end]
      state_transition_table[a_end][EPSILON] = {b_start}
      assert EPSILON not in state_transition_table[b_end]
      state_transition_table[b_end][EPSILON] = {to_state}
      state_transition_table.append({})
    elif prod == REGEX_OPTIONAL_OP:
      a_start, a_end = from_stack.pop(), to_stack.pop()
      # build DFA for a?
      state_transition_table.append({EPSILON: {a_start, to_state}})
      assert None not in state_transition_table[a_end]
      state_transition_table[a_end][EPSILON] = {to_state}
      state_transition_table.append({})
    elif prod == REGEX_REPEAT_OP:
      a_start, a_end = from_stack.pop(), to_stack.pop()
      # build DFA for a*
      state_transition_table.append({EPSILON: {a_start, to_state}})
      assert None not in state_transition_table[a_end]
      state_transition_table[a_end][EPSILON] = {a_start, to_state}
      state_transition_table.append({})
    elif prod == REGEX_REPEAT_EXISTS_OP:
      a_start, a_end = from_stack.pop(), to_stack.pop()
      # build DFA for a+
      state_transition_table.append({EPSILON: {a_start}})
      assert None not in state_transition_table[a_end]
      state_transition_table[a_end][EPSILON] = {a_start, to_state}
      state_transition_table.append({})
    elif prod in {
      REGEX_RANGE_OP, REGEX_RANGE_LITS_OP, REGEX_INV_RANGE_OP, REGEX_INV_RANGE_LITS_OP, REGEX_LITS_SINGLE_OP,
      REGEX_LITS_MULTIPLE_OP}:
      assert False, 'here not supported'
    else:
      # all other productions don't change anything. do not add any states.
      continue
    assert state_transition_table[to_state] == {}
    num_states += 2
    assert len(state_transition_table) == num_states
    from_stack.append(from_state)
    to_stack.append(to_state)

  assert len(from_stack) == len(to_stack) == 1
  initial_state, final_state = from_stack.pop(), to_stack.pop()
  return NonDeterministicAutomaton(initial_state, set([final_state]), state_transition_table)
