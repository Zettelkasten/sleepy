from grammar import Grammar, Production, LexError
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
REGEX_LITS_OP = Production('Lits', 'Lit', 'Lits')
REGEX_LIT_OP = Production('Lit', REGEX_LIT_TOKEN)

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
  REGEX_LITS_OP,
  Production('Lits', 'Lit'),
  REGEX_LIT_OP
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
