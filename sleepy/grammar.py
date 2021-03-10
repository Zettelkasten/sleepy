"""
The empty symbol (!= the empty word).
Use empty tuple as empty word.
"""
from typing import Tuple

EPSILON = None


class Production:
  """
  A production A -> X_1 X_2 ... X_n.
  """

  def __init__(self, left, *right):
    """
    :param str left: A
    :param list[str]|tuple[str] right: X_1 ... X_n
    """
    self.left = left
    if isinstance(right, list):
      right = tuple(right)
    self.right = right  # type: Tuple[str]

  def __repr__(self):
    return 'Production[%r -> %s]' % (self.left, ' '.join([repr(symbol) for symbol in self.right]))

  def __hash__(self):
    return hash((self.left, self.right))


class Grammar:
  """
  A context free grammar.
  """

  def __init__(self, *prods, start=None):
    """
    :param tuple[Production]|list[Production] prods:
    :param None|str start: start non-terminal, by default left of first production
    """
    if not isinstance(prods, tuple):
      assert isinstance(prods, list)
      prods = tuple(prods)
    self.prods = prods  # type: Tuple[Production]
    if start is None:
      start = self.prods[0].left
    self.start = start

    self.non_terminals = set(p.left for p in self.prods)
    self.terminals = set(x for p in self.prods for x in p.right if x not in self.non_terminals)
    self.non_terminals, self.terminals = tuple(sorted(self.non_terminals)), tuple(sorted(self.terminals))
    self.symbols = self.non_terminals + self.terminals
    self._prods_by_left = {left: tuple([p for p in self.prods if p.left == left]) for left in self.non_terminals}

  def get_prods_for(self, left):
    """
    :param str left: left-hand non-terminal of production
    :rtype: tuple[Production]
    """
    assert left in self.non_terminals
    return self._prods_by_left[left]

  def is_start_separated(self):
    """
    :rtype: bool
    :return:
    """
    return (
      all(len(p.right) == 1 for p in self.get_prods_for(self.start)) and
      all(self.start not in p.right for p in self.prods))


class AttributeGrammar(Grammar):
  """
  A context-free L-attributed grammar with synthesized and inherited attributes.
  Synthesized attributes are evaluated bottom-to-top, inherited top-to-bottom.

  For synthesized attributes:
  For each terminal x, give function `syn.0 = f(token_word, inh.0)`
  For each production A -> B1 B2 ... Bn, give functions

  ``inh.i = f(inh.0, inh.1, ..., inh.{i-1}, syn.1, ..., syn.{i-1})`` for i=1,..,n,

  as well as

  ``syn.0 = f(inh.0, inh.1, ..., inh.n,     syn.1, ..., syn.n)``
  """

  def __init__(self, prods, inh_attrs, syn_attrs, prod_attr_rules, terminal_attr_rules, start=None):
    """
    :param tuple[Production]|list[Production] prods:
    :param set[str] inh_attrs: names of inherited (top-down) attributes
    :param set[str] syn_attrs: names of synthesized (bottom-up) attributes
    :param list[dict[str, function]] prod_attr_rules: functions that evaluate attributes for productions.
      For each production `A_0 -> A_1 ... A_n`, dict with keys `attr.i`
      (where `attr` is a name of attribute, and `i` a position in the productions symbols).
      If `attr` is synthesized, then `i` must be `0`.
      If `attr` is inherited, then `i` must be `>= 1`.
      For each attribute `attr`, the function is called with a argument of the same name `attr(j)`,
      which is a function giving previously computed attributes of the given symbol in the production.
    :param dict[str, function] terminal_attr_rules: functions that evaluate attributes for terminals.
    :param None|str start: start non-terminal, by default left of first production
    """
    super().__init__(prods, start=start)
    assert inh_attrs & syn_attrs == {}, 'inherited and synthesized attributes must be disjoint'
    self.inh_attrs = inh_attrs
    self.syn_attrs = syn_attrs
    assert len(prod_attr_rules) == len(self.prods), 'need one rule set for each production'
    self.prod_attr_rules = prod_attr_rules
    self.terminal_attr_rules = terminal_attr_rules

  @property
  def attrs(self):
    """
    All (inherited + synthesized) attribute names.
    :rtype: set[str]
    """
    return self.inh_attrs | self.syn_attrs


class LexError(Exception):
  """
  A lexical error, when a word is not recognized (does not have a first-longest-match analysis).
  """

  def __init__(self, word, pos, message):
    super().__init__(
      '%s: %s' % (' '.join([repr(s) for s in word[:pos]] + ['!'] + [repr(s) for s in word[pos:]]), message))


class ParseError(Exception):
  """
  A parse error, when a word is not recognized by a context free grammar.
  """

  def __init__(self, word, pos, message):
    super().__init__(
      '%s: %s' % (' '.join([repr(s) for s in word[:pos]] + ['!'] + [repr(s) for s in word[pos:]]), message))