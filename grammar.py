"""
The empty symbol (!= the empty word).
Use empty tuple as empty word.
"""
EPSILON = None;


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
    self.right = right  # type: tuple[str]

  def __repr__(self):
    return 'Production[%s -> %s]' % (self.left, ' '.join(self.right))

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
      prods = tuple(prods)
    self.prods = prods
    if start is None:
      start = prods[0].left
    self.start = start

    self.non_terminals = set(p.left for p in prods)
    self.terminals = set(x for p in prods for x in p.right if x not in self.non_terminals)
    self.non_terminals, self.terminals = tuple(sorted(self.non_terminals)), tuple(sorted(self.terminals))
    self.symbols = self.non_terminals + self.terminals
    self._prods_by_left = {left: tuple([p for p in prods if p.left == left]) for left in self.non_terminals}

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


class ParseError(Exception):
  """
  Thrown when a word is not recognized.
  """

  def __init__(self, word, pos, message):
    super().__init__(
      '%s: %s' % (' '.join([repr(s) for s in word[:pos]] + ['!'] + [repr(s) for s in word[pos:]]), message))