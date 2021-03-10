"""
The empty symbol (!= the empty word).
Use empty tuple as empty word.
"""
from typing import Tuple, Any, Dict, Set, Callable

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
  A context-free attributed grammar with synthesized and inherited attributes.

  For each terminal x, give function `syn.0 = f(token_word, inh.0)`
  For each production A -> B1 B2 ... Bn, give functions

  ``inh.i = f(inh.0, inh.1, ..., inh.{i-1}, syn.1, ..., syn.{i-1})`` for i=1,..,n,

  as well as

  ``syn.0 = f(inh.0, inh.1, ..., inh.n,     syn.1, ..., syn.n)``
  """

  def __init__(self, prods, prod_attr_rules, terminal_attr_rules, inh_attrs=None, syn_attrs=None, start=None):
    """
    :param tuple[Production]|list[Production] prods:
    :param dict[Production, dict[str, function]]|list[dict[str, function]] prod_attr_rules:
      functions that evaluate attributes for productions.
      For each production `A_0 -> A_1 ... A_n`, dict with keys `attr.i`
      (where `attr` is a name of attribute, and `i` a position in the productions symbols).
      If `attr` is synthesized, then `i` must be `0`.
      If `attr` is inherited, then `i` must be `>= 1`.
      For each attribute `attr`, the function is called with a argument of the same name `attr(j)`,
      which is a function giving previously computed attributes of the given symbol in the production.
    :param dict[str, dict[str, function]] terminal_attr_rules: functions that evaluate attributes for terminals.
      Same format as `prod_attr_rules`, but with terminal name instead of productions.
    :param set[str]|None inh_attrs: names of inherited (top-down) attributes
    :param set[str]|None syn_attrs: names of synthesized (bottom-up) attributes
    :param None|str start: start non-terminal, by default left of first production
    """
    super().__init__(*prods, start=start)
    if inh_attrs is None:
      inh_attrs = set()
    self.inh_attrs = inh_attrs  # type: Set[str]
    if syn_attrs is None:
      syn_attrs = set()
    self.syn_attrs = syn_attrs  # type: Set[str]
    assert self.inh_attrs & self.syn_attrs == set(), 'inherited and synthesized attributes must be disjoint'
    if isinstance(prod_attr_rules, (list, tuple)):
      assert len(self.prods) == len(prod_attr_rules)
      prod_attr_rules = dict(zip(self.prods, prod_attr_rules))
    assert isinstance(prod_attr_rules, dict)
    self.prod_attr_rules = prod_attr_rules  # type: Dict[Production, Dict[str, Callable]]
    assert tuple(prod_attr_rules.keys()) == self.prods, 'need one rule set for each production'
    self.prod_attr_rules = prod_attr_rules
    self.terminal_attr_rules = terminal_attr_rules  # type: Dict[str, Dict[str, Callable]]
    self._sanity_check()

  def _split_attr_name_pos(self, attr_target):
    """
    :param str attr_target: attr.i
    :rtype: tuple[str, int]
    :returns: attribute name + attribute position, i.e. (attr, i)
    """
    if '.' not in attr_target:
      # assume pos=0
      return attr_target, 0
    assert '.' in attr_target
    attr_name, attr_pos = attr_target.split('.', 2)
    assert attr_name in self.attrs
    attr_pos = int(attr_pos)
    assert attr_pos >= 0
    return attr_name, attr_pos

  def _sanity_check(self):
    """
    Some asserts that attribute rules are well defined.
    """
    for prod, attr_rules in self.prod_attr_rules.items():
      assert isinstance(attr_rules, dict)
      for target, func in attr_rules.items():
        attr_name, attr_pos = self._split_attr_name_pos(target)
        assert 0 <= attr_pos <= len(prod.right)
        assert attr_name in self.attrs
        if attr_name in self.inh_attrs:
          assert attr_pos >= 1, '%r: rules for inherited attributes only allowed for right side of production' % target
        elif attr_name in self.syn_attrs:
          assert attr_pos == 0, '%r: rules for synthesized attributes only allowed for left side of production' % target
        else:
          assert False
        assert callable(func)
        func_arg_names = func.__code__.co_varnames
        assert all(from_attr_name in self.attrs for from_attr_name in func_arg_names), (
          '%r: function arguments must be attributes, got %r but only have %r' % (target, func_arg_names, self.attrs))
    for terminal, attr_rules in self.terminal_attr_rules.items():
      assert isinstance(attr_rules, dict)
      for target, func in attr_rules.items():
        attr_name, attr_pos = self._split_attr_name_pos(target)
        assert attr_pos == 0
        assert attr_name in self.syn_attrs
        assert callable(func)
        assert func.__code__.co_argcount == 1

  @property
  def attrs(self):
    """
    All (inherited + synthesized) attribute names.
    :rtype: set[str]
    """
    return self.inh_attrs | self.syn_attrs

  def is_s_attributed(self):
    """
    Check whether this grammar is s-attributed, i.e. only contains synthesized attributes.
    :rtype: bool
    """
    return len(self.inh_attrs) == 0

  def get_terminal_syn_attr_eval(self, terminal, word):
    """
    :param str terminal: terminal
    :param str word:
    :rtype: dict[str,Any]
    """
    assert terminal in self.terminals
    if terminal not in self.terminal_attr_rules:
      return {}

    attr_eval = {}  # type: Dict[str, Any]
    for attr_target, func in self.terminal_attr_rules[terminal].items():
      attr_name, attr_pos = self._split_attr_name_pos(attr_target)
      if attr_name not in self.syn_attrs:
        continue
      assert attr_pos == 0
      attr_eval[attr_name] = func(word)
    return attr_eval

  def get_prod_syn_attr_eval(self, prod, right_attr_evals):
    """
    :param Production prod:
    :param list[dict[str,Any]] right_attr_evals: evaluations of right side of production
    :rtype: dict[str,Any]
    """
    assert len(right_attr_evals) == len(prod.right)
    assert prod in self.prod_attr_rules

    def make_attr_getter(get_attr_name):
      """
      :param str get_attr_name:
      :rtype: function[int, Any]
      """

      def get(pos):
        """
        Receives attr.pos.
        :param int pos:
        :rtype: Any
        """
        assert 1 <= pos <= len(prod.right) + 1, '%s.%s: invalid for production %r' % (get_attr_name, pos, prod)
        assert get_attr_name in right_attr_evals[pos - 1], '%s.%s: evaluation not available, only have %r' % (
          get_attr_name, pos, right_attr_evals)
        return right_attr_evals[pos - 1][get_attr_name]

      return get

    attr_eval = {}  # type: Dict[str, Any]
    for attr_target, func in self.prod_attr_rules[prod].items():
      attr_name, attr_pos = self._split_attr_name_pos(attr_target)
      if attr_name not in self.syn_attrs:
        continue
      assert attr_pos == 0

      func_arg_names = func.__code__.co_varnames
      assert all(
        any(from_attr_name in right_eval for right_eval in right_attr_evals) for from_attr_name in func_arg_names)
      func_kwargs = {from_attr_name: make_attr_getter(from_attr_name) for from_attr_name in func_arg_names}
      attr_eval[attr_name] = func(**func_kwargs)
    return attr_eval


class LexError(Exception):
  """
  A lexical error, when a word is not recognized (does not have a first-longest-match analysis).
  """

  def __init__(self, word, pos, message):
    """
    :param str word:
    :param int pos:
    :param str message:
    """
    super().__init__(
      '%s: %s' % (' '.join([repr(s) for s in word[:pos]] + ['!'] + [repr(s) for s in word[pos:]]), message))


class ParseError(Exception):
  """
  A parse error, when a word is not recognized by a context free grammar.
  """

  def __init__(self, tokens, pos, message, token_words=None):
    """
    :param list[str] tokens:
    :param int pos: token position where error occurred
    :param str message:
    :param None|list[str] token_words: decomposition of entire word into tokens
    """
    output = token_words if token_words is not None else tokens
    super().__init__(
      '%s: %s' % (' '.join([repr(s) for s in output[:pos]] + ['!'] + [repr(s) for s in output[pos:]]), message))
