from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Set, Optional, Tuple

from sleepy.automaton import NonDeterministicAutomaton, make_dfa_from_nfa, DeterministicAutomaton, OTHER_CHAR
from sleepy.errors import LexError
from sleepy.grammar import Grammar, Production, EPSILON, IGNORED_TOKEN, get_token_word_from_tokens_pos, DummyPath
from sleepy.parser import ParserGenerator

REGEX_LIT_TOKEN = 'a'
REGEX_SPECIAL_TOKENS = frozenset({'(', ')', '\\', '-', '[', ']', '*', '+', '?', '|', '^', '.'})
# Currently we only recognize 7-bit ASCII
REGEX_RECOGNIZED_CHARS = frozenset({'\n', OTHER_CHAR} | {chr(c) for c in range(32, 128)})

REGEX_CHOICE_OP = Production('Choice', 'Choice', '|', 'Concat')
REGEX_CONCAT_OP = Production('Concat', 'Concat', 'Repeat')
REGEX_REPEAT_OP = Production('Repeat', 'Range', '*')
REGEX_REPEAT_EXISTS_OP = Production('Repeat', 'Range', '+')
REGEX_OPTIONAL_OP = Production('Repeat', 'Range', '?')
REGEX_RANGE_OP = Production('Range', '[', 'a', '-', 'a', ']')
REGEX_RANGE_LITS_OP = Production('Range', '[', 'Lits', ']')
REGEX_INV_RANGE_OP = Production('Range', '[', '^', 'a', '-', 'a', ']')
REGEX_INV_RANGE_LITS_OP = Production('Range', '[', '^', 'Lits', ']')
REGEX_LIT_OP = Production('Lit', REGEX_LIT_TOKEN)
REGEX_LIT_ANY_OP = Production('Lit', '.')
REGEX_LITS_MULTIPLE_OP = Production('Lits', 'a', 'Lits')
REGEX_LITS_SINGLE_OP = Production('Lits', 'a')

REGEX_GRAMMAR = Grammar(
  Production('Regex', 'Choice'),
  REGEX_CHOICE_OP,
  Production('Choice', 'Concat'),
  REGEX_CONCAT_OP,
  Production('Concat', 'Repeat'),
  REGEX_REPEAT_OP, REGEX_REPEAT_EXISTS_OP, REGEX_OPTIONAL_OP,
  Production('Repeat', 'Range'),
  REGEX_RANGE_OP, REGEX_RANGE_LITS_OP, REGEX_INV_RANGE_OP, REGEX_INV_RANGE_LITS_OP,
  Production('Range', 'Lit'),
  Production('Range', '(', 'Choice', ')'),
  REGEX_LITS_MULTIPLE_OP, REGEX_LITS_SINGLE_OP,
  REGEX_LIT_OP, REGEX_LIT_ANY_OP
)
REGEX_PARSER = ParserGenerator(REGEX_GRAMMAR)


def tokenize_regex(word, file_path: Path | DummyPath = DummyPath("regex")) -> (Tuple[str], Tuple[int]):
  """
  :param str word:
  :param Union[Path, DummyPath] file_path:
  :raises: LexError
  :returns: tokens with decomposition
  :rtype: tuple[tuple[str], tuple[int]]
  """
  escape_next = False
  tokens, tokens_pos = [], []
  for pos, c in enumerate(word):
    if escape_next:
      if c not in REGEX_SPECIAL_TOKENS:
        raise LexError(file_path, word, pos, 'Cannot escape character %r' % c)
      escape_next = False
      tokens.append(REGEX_LIT_TOKEN)
    elif c == '\\':
      escape_next = True
      tokens.append(IGNORED_TOKEN)
    elif c in REGEX_SPECIAL_TOKENS:
      tokens.append(c)
    else:  # default case
      tokens.append(REGEX_LIT_TOKEN)
    tokens_pos.append(pos)

  if escape_next:
    raise LexError(file_path, word, len(word), 'Cannot end word with escape character')
  assert len(tokens) == len(tokens_pos) == len(word)
  return tuple(tokens), tuple(tokens_pos)


def make_regex_nfa(regex):
  """
  :param str regex:
  :rtype: NonDeterministicAutomaton
  """
  tokens, tokens_pos = tokenize_regex(regex)
  analysis = REGEX_PARSER.parse_analysis(regex, tokens, tokens_pos)

  num_states = 0
  state_transition_table: List[Dict[Optional[str], Set[int]]] = []
  # TODO: replace this with attribute grammars when we have that
  # for each subregex, build a epsilon-NFA with start state in from_stack, and single final state in to_stack.
  # invariant: no ingoing transitions into start, no outgoing transitions from end
  from_stack: List[int] = []
  to_stack: List[int] = []
  range_lits: Set[str] = set()
  pos = 0

  def next_literal_name():
    """
    :rtype: str
    """
    nonlocal pos
    while tokens[pos] != REGEX_LIT_TOKEN:
      pos += 1
    name = get_token_word_from_tokens_pos(regex, tokens_pos, pos)
    pos += 1
    return name

  for prod in reversed(analysis):
    from_state, to_state = num_states, num_states + 1
    if prod == REGEX_LIT_OP:
      a = next_literal_name()
      # build DFA for a
      state_transition_table.append({a: {to_state}})
      state_transition_table.append({})
    elif prod == REGEX_LIT_ANY_OP:
      # build DFA for .
      state_transition_table.append({char: {to_state} for char in REGEX_RECOGNIZED_CHARS})
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
    elif prod in {REGEX_LITS_SINGLE_OP, REGEX_LITS_MULTIPLE_OP}:
      range_lits.add(next_literal_name())
      continue  # does not add states to NFA
    elif prod in {REGEX_RANGE_OP, REGEX_INV_RANGE_OP}:
      a, b = next_literal_name(), next_literal_name()
      # build DFA for [a-b]
      recognized = {chr(symbol_ord) for symbol_ord in range(ord(a), ord(b) + 1)}
      if prod == REGEX_INV_RANGE_OP:
        recognized = REGEX_RECOGNIZED_CHARS - recognized
      state_transition_table.append({char: {to_state} for char in recognized})
      state_transition_table.append({})
    elif prod in {REGEX_RANGE_LITS_OP, REGEX_INV_RANGE_LITS_OP}:
      assert len(range_lits) >= 1
      # build DFA for [abcd..]
      recognized = range_lits
      if prod == REGEX_INV_RANGE_LITS_OP:
        recognized = REGEX_RECOGNIZED_CHARS - recognized
      state_transition_table.append({char: {to_state} for char in recognized})
      state_transition_table.append({})
      range_lits.clear()
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
  return NonDeterministicAutomaton(initial_state, {final_state}, state_transition_table)


def make_regex_dfa(regex: str) -> DeterministicAutomaton:
  return make_dfa_from_nfa(make_regex_nfa(regex))
