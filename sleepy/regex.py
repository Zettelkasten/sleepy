from sleepy.automaton import NonDeterministicAutomaton, make_dfa_from_nfa
from sleepy.grammar import Grammar, Production, LexError, EPSILON, AttributeGrammar
from sleepy.parser import ParserGenerator
from typing import List, Dict, Set, Optional

REGEX_LIT_TOKEN = 'a'
REGEX_SPECIAL_TOKENS = frozenset({'(', ')', '\\', '-', '[', ']', '*', '+', '?', '|', '^', '.'})
# Currently we only recognize 7-bit ASCII
REGEX_RECOGNIZED_CHARS = frozenset({chr(c) for c in range(32, 128)})

REGEX_DESCENT_START_OP = Production('Regex', 'Choice')
REGEX_CHOICE_OP = Production('Choice', 'Choice', '|', 'Concat')
REGEX_DESCENT_CHOICE_OP = Production('Choice', 'Concat')
REGEX_CONCAT_OP = Production('Concat', 'Concat', 'Repeat')
REGEX_DESCENT_CONCAT_OP = Production('Concat', 'Repeat')
REGEX_REPEAT_OP = Production('Repeat', 'Range', '*')
REGEX_REPEAT_EXISTS_OP = Production('Repeat', 'Range', '+')
REGEX_OPTIONAL_OP = Production('Repeat', 'Range', '?')
REGEX_DESCENT_REPEAT_OP = Production('Repeat', 'Range')
REGEX_RANGE_OP = Production('Range', '[', 'LitSets', ']')
REGEX_INV_RANGE_OP = Production('Range', '[', '^', 'LitSets', ']')
REGEX_DESCENT_RANGE_OP = Production('Range', 'Lit')
REGEX_BRACKETS_OP = Production('Range', '(', 'Choice', ')')
REGEX_LIT_OP = Production('Lit', REGEX_LIT_TOKEN)
REGEX_LIT_ANY_OP = Production('Lit', '.')
REGEX_LIT_SETS_MULTIPLE_OP = Production('LitSets', 'LitSet', 'LitSets')
REGEX_LIT_SETS_SINGLE_OP = Production('LitSets', 'LitSet')
REGEX_LIT_SET_RANGE_OP = Production('ListSet', 'a', '-', 'a')
REGEX_LIT_SET_SINGLE_OP = Production('LitSet', 'a')

REGEX_GRAMMAR = Grammar(
  REGEX_DESCENT_START_OP,
  REGEX_CHOICE_OP,
  REGEX_DESCENT_CHOICE_OP,
  REGEX_CONCAT_OP,
  REGEX_DESCENT_CONCAT_OP,
  REGEX_REPEAT_OP, REGEX_REPEAT_EXISTS_OP, REGEX_OPTIONAL_OP,
  REGEX_DESCENT_REPEAT_OP,
  REGEX_RANGE_OP, REGEX_INV_RANGE_OP,
  REGEX_DESCENT_RANGE_OP,
  REGEX_BRACKETS_OP,
  REGEX_LIT_OP, REGEX_LIT_ANY_OP,
  REGEX_LIT_SETS_MULTIPLE_OP, REGEX_LIT_SETS_SINGLE_OP,
  REGEX_LIT_SET_RANGE_OP, REGEX_LIT_SET_SINGLE_OP
)
REGEX_PARSER = ParserGenerator(REGEX_GRAMMAR)


def tokenize_regex(word):
  """
  :param str word:
  :raises: LexError
  :returns: tokens with decomposition
  :rtype: tuple[tuple[str], tuple[str]]
  """
  escape_next = False
  tokens, token_words = [], []
  for pos, c in enumerate(word):
    if escape_next:
      if c not in REGEX_SPECIAL_TOKENS:
        raise LexError(word, pos, 'Cannot escape character %r' % c)
      escape_next = False
      tokens.append(REGEX_LIT_TOKEN)
      token_words.append(c)
    elif c == '\\':
      escape_next = True
    elif c in REGEX_SPECIAL_TOKENS:
      tokens.append(c)
      token_words.append(None)
    else:  # default case
      tokens.append(REGEX_LIT_TOKEN)
      token_words.append(c)

  if escape_next:
    raise LexError(word, len(word), 'Cannot end word with escape character')
  assert len(tokens) == len(token_words) <= len(word)
  return tuple(tokens), tuple(token_words)


def make_regex_nfa(regex):
  """
  :param str regex:
  :rtype: NonDeterministicAutomaton
  """
  tokens, token_words = tokenize_regex(regex)

  # for each subregex, build a epsilon-NFA with a specified start and end state.
  # invariant: no ingoing transitions into start, no outgoing transitions from end
  num_states = 0
  state_transition_table = []  # type: List[Dict[Optional[str], Set[int]]]

  def add_state():
    """
    :rtype: int
    """
    nonlocal num_states
    new_state = num_states
    state_transition_table.insert(new_state, {})
    num_states += 1
    return new_state

  def add_choice_nfa(start, end):
    """
    :param Callable[[int],str] start:
    :param Callable[[int],str] end:
    """
    start_self, end_self, start_a, end_a, start_b, end_b = start(0), end(0), start(1), end(1), start(3), end(3)
    assert EPSILON not in {
      state_transition_table[start_self], state_transition_table[end_a], state_transition_table[end_b]}
    state_transition_table[start_self][EPSILON] = {start_a, start_b}
    state_transition_table[end_a][EPSILON] = {end_self}
    state_transition_table[end_b][EPSILON] = {end_self}

  def add_concat_nfa(start, end):
    """
    :param Callable[[int],str] start:
    :param Callable[[int],str] end:
    """
    start_self, end_self, start_a, end_a, start_b, end_b = start(0), end(0), start(1), end(1), start(3), end(3)
    assert EPSILON not in {
      state_transition_table[start_self], state_transition_table[end_a], state_transition_table[end_b]}
    state_transition_table[start_self][EPSILON] = {start_a}
    state_transition_table[end_a][EPSILON] = {start_a}
    state_transition_table[end_b][EPSILON] = {end_self}

  def add_repeat_nfa(must_exist, can_repeat, start, end):
    """
    :param bool must_exist: if True, repeat >= 1 times
    :param bool can_repeat: if False, repeat <= 1 times
    :param Callable[[int],str] start:
    :param Callable[[int],str] end:
    """
    assert not (must_exist and not can_repeat)
    start_self, end_self, start_a, end_a = start(0), end(0), start(1), end(1)
    assert EPSILON not in {state_transition_table[start_self], state_transition_table[end_a]}
    state_transition_table[start_self][EPSILON] = {start_a} if must_exist else {start_a, end_self}
    state_transition_table[end_a][EPSILON] = {start_a, end_self} if can_repeat else {end_self}

  def add_range_nfa(inverted, start, end, lit_set):
    """
    :param bool inverted:
    :param Callable[[int],str] start:
    :param Callable[[int],str] end:
    :param Callable[[int],set[str]] lit_set:
    """
    start_self, end_self, range_lit_set = start(0), end(0), lit_set(2)
    if inverted:
      range_lit_set = REGEX_RECOGNIZED_CHARS - range_lit_set
    assert EPSILON not in state_transition_table[start_self]
    state_transition_table[start_self][EPSILON] = {char: {end_self} for char in range_lit_set}

  def add_literal_nfa(start, end, lit):
    """
    :param Callable[[int],str] start:
    :param Callable[[int],str] end:
    :param Callable[[int],str] lit:
    """
    start_self, end_self, char = start(0), end(0), lit(1)
    assert EPSILON not in state_transition_table[start_self]
    state_transition_table[start_self][EPSILON] = {char: {end_self}}

  def add_any_nfa(start, end):
    """
    :param Callable[[int],str] start:
    :param Callable[[int],str] end:
    """
    start_self, end_self = start(0), end(0)
    assert EPSILON not in state_transition_table[start_self]
    state_transition_table[start_self][EPSILON] = {char: {end_self} for char in REGEX_RECOGNIZED_CHARS}

  from functools import partial
  attribute_grammar = AttributeGrammar(
    prods=REGEX_GRAMMAR.prods,
    syn_attrs={'start', 'end', 'lit_set', 'lit'},
    prod_attr_rules={
      REGEX_DESCENT_START_OP: {'start': lambda start: start(1), 'end': lambda end: end(1)},
      REGEX_CHOICE_OP: {'start': add_state, 'end': add_state, '_after': add_choice_nfa},
      REGEX_DESCENT_CHOICE_OP: {'start': lambda start: start(1), 'end': lambda end: end(1)},
      REGEX_CONCAT_OP: {'start': add_state, 'end': add_state, '_after': add_concat_nfa},
      REGEX_DESCENT_CONCAT_OP: {'start': lambda start: start(1), 'end': lambda end: end(1)},
      REGEX_REPEAT_OP: {
        'start': add_state, 'end': add_state, '_after': partial(add_repeat_nfa, must_exist=False, can_repeat=True)},
      REGEX_REPEAT_EXISTS_OP: {
        'start': add_state, 'end': add_state, '_after': partial(add_repeat_nfa, must_exist=True, can_repeat=True)},
      REGEX_OPTIONAL_OP: {
        'start': add_state, 'end': add_state, '_after': partial(add_repeat_nfa, must_exist=False, can_repeat=False)},
      REGEX_DESCENT_REPEAT_OP: {'start': lambda start: start(1), 'end': lambda end: end(1)},
      REGEX_RANGE_OP: {'start': add_state, 'end': add_state, '_after': partial(add_range_nfa, inverted=False)},
      REGEX_INV_RANGE_OP: {'start': add_state, 'end': add_state, '_after': partial(add_range_nfa, inverted=True)},
      REGEX_DESCENT_RANGE_OP: {'start': lambda start: start(1), 'end': lambda end: end(1)},
      REGEX_BRACKETS_OP: {'start': lambda start: start(2), 'end': lambda end: end(2)},
      REGEX_LIT_OP: {'start': add_state, 'end': add_state, '_after': add_literal_nfa},
      REGEX_LIT_ANY_OP: {'start': add_state, 'end': add_state, '_after': add_any_nfa},
      REGEX_LIT_SETS_MULTIPLE_OP: {'lit_set': lambda lit_set: lit_set(1) + lit_set(2)},
      REGEX_LIT_SETS_SINGLE_OP: {'lit_set': lambda lit_set: lit_set(1)},
      REGEX_LIT_SET_RANGE_OP: {'lit_set': lambda lit: {chr(c) for c in range(ord(lit(1)), ord(lit(2)) + 1)}},
      REGEX_LIT_SET_SINGLE_OP: {'lit_set': lambda lit: {lit(1)}}
    },
    terminal_attr_rules={
      'a': {'lit': lambda value: value}
    }
  )
  parser = ParserGenerator(attribute_grammar)
  _, output = parser.parse_attr_analysis(tokens, token_words)
  assert output.keys() == {'start', 'end'}
  initial_state, final_state = output['start'], output['end']

  return NonDeterministicAutomaton(initial_state, {final_state}, state_transition_table)


def make_regex_dfa(regex):
  """
  :param str regex:
  :rtype: DeterministicAutomaton
  """
  return make_dfa_from_nfa(make_regex_nfa(regex))
