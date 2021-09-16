"""
The empty symbol (!= the empty word).
Use empty tuple as empty word.
"""
from typing import Tuple, Any, Dict, Set, Callable, Union, Optional

EPSILON = None

IGNORED_TOKEN = None


def get_token_word_from_tokens_pos(word, tokens_pos, pos):
  """
  :param str word:
  :param tuple[int]|list[int] tokens_pos:
  :param int pos: index of token
  :rtype: str
  """
  assert 0 <= pos < len(tokens_pos)
  from_pos = tokens_pos[pos]
  assert 0 <= from_pos < len(word)
  if pos == len(tokens_pos) - 1:
    return word[from_pos:]
  else:
    to_pos = tokens_pos[pos + 1]
    assert from_pos < to_pos < len(word)
    return word[from_pos:to_pos]


class Production:
  """
  A production A -> X_1 X_2 ... X_n.
  """

  def __init__(self, left, *right):
    """
    :param str left: A
    :param str right: X_1 ... X_n
    """
    self.left = left
    if isinstance(right, list):
      right = tuple(right)
    assert '' not in right
    self.right: Tuple[str] = right

  def __repr__(self):
    return 'Production[%r -> %s]' % (self.left, ' '.join([repr(symbol) for symbol in self.right]))

  def __hash__(self):
    return hash((self.left, self.right))

  def __eq__(self, other):
    if not isinstance(other, Production):
      return False
    return self.left == other.left and self.right == other.right


class Grammar:
  """
  A context free grammar.
  """

  def __init__(self, *prods, start=None):
    """
    :param Production prods:
    :param None|str start: start non-terminal, by default left of first production
    """
    if not isinstance(prods, tuple):
      assert isinstance(prods, list)
      prods = tuple(prods)
    self.prods: Tuple[Production] = prods
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

  def copy_with_start(self, start):
    """
    :param str start: new start non-terminal symbol
    :rtype: Grammar
    """
    if start == self.start:
      return self
    assert start in self.non_terminals
    return Grammar(*self.prods, start=start)


class AttributeGrammar:
  """
  A context-free attributed grammar with synthesized and inherited attributes.

  For each terminal x, give function `syn.0 = f(token_word, inh.0)`
  For each production A -> B1 B2 ... Bn, give functions

  ``inh.i = f(inh.0, inh.1, ..., inh.{i-1}, syn.1, ..., syn.{i-1})`` for i=1,..,n,

  as well as

  ``syn.0 = f(inh.0, inh.1, ..., inh.n,     syn.1, ..., syn.n)``
  """

  def __init__(self, grammar, prod_attr_rules, terminal_attr_rules, inh_attrs=None, syn_attrs=None):
    """
    :param Grammar grammar:
    :param dict[Production, dict[str, function|str|Any]]|list[dict[str, function|str|Any]] prod_attr_rules:
      functions that evaluate attributes for productions.
      For each production `A_0 -> A_1 ... A_n`, dict with keys `attr.i`
      (where `attr` is a name of attribute, and `i` a position in the productions symbols).
      If `attr` is synthesized, then `i` must be `0`.
      If `attr` is inherited, then `i` must be `>= 1`.
      For each attribute `attr`, the function is called with a argument of the same name `attr(j)`,
      which is a function giving previously computed attributes of the given symbol in the production.
    :param dict[str, dict[str, function|Any]] terminal_attr_rules: functions that evaluate attributes for terminals.
      Same format as `prod_attr_rules`, but with terminal name instead of productions.
    :param set[str]|None inh_attrs: names of inherited (top-down) attributes
    :param set[str]|None syn_attrs: names of synthesized (bottom-up) attributes
    """
    self.grammar = grammar
    if inh_attrs is None:
      inh_attrs = set()
    self.inh_attrs: Set[str] = inh_attrs
    if syn_attrs is None:
      syn_attrs = set()
    self.syn_attrs: Set[str] = syn_attrs
    assert self.inh_attrs & self.syn_attrs == set(), 'inherited and synthesized attributes must be disjoint'
    if isinstance(prod_attr_rules, (list, tuple)):
      assert len(self.grammar.prods) == len(prod_attr_rules)
      prod_attr_rules = dict(zip(self.grammar.prods, prod_attr_rules))
    assert isinstance(prod_attr_rules, dict)
    self.prod_attr_rules: Dict[Production, Dict[str, Union[Callable, str, Any]]] = prod_attr_rules
    assert tuple(prod_attr_rules.keys()) == self.grammar.prods, 'need one rule set for each production'
    self.prod_attr_rules = prod_attr_rules
    self.terminal_attr_rules: Dict[str, Dict[str, Union[Callable, str]]] = terminal_attr_rules
    self._sanity_check()

  @classmethod
  def from_dict(cls,
                prods_attr_rules: Dict[Production, Dict[str, Union[Callable, str, Any]]],
                terminal_attr_rules: Dict[str, Dict[str, Union[Callable, Any]]],
                inh_attrs: Optional[Set[str]] = None, syn_attrs: Optional[Set[str]] = None,
                start: Optional[str] = None):
    grammar = Grammar(*prods_attr_rules.keys(), start=start)
    return AttributeGrammar(
      grammar=grammar, prod_attr_rules=prods_attr_rules, terminal_attr_rules=terminal_attr_rules,
      inh_attrs=inh_attrs, syn_attrs=syn_attrs)

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

  def _get_attr_func_arg_names(self, func):
    """
    :param Callable[tuple[Any], Any] func:
    :rtype: list[str]
    """
    assert callable(func)
    if hasattr(func, '__code__'):
      return func.__code__.co_varnames[:func.__code__.co_argcount]
    if hasattr(func, 'func'):  # partial function
      from functools import partial
      assert isinstance(func, partial)
      orig_func = func.func
      return [
        arg_name for arg_name in self._get_attr_func_arg_names(orig_func)[len(func.args):]
        if arg_name not in func.keywords]
    assert False

  def _sanity_check(self):
    """
    Some asserts that attribute rules are well defined.
    """
    for prod, attr_rules in self.prod_attr_rules.items():
      assert isinstance(attr_rules, dict)
      for target, func in attr_rules.items():
        attr_name, attr_pos = self._split_attr_name_pos(target)
        assert attr_name in self.attrs
        assert 0 <= attr_pos <= len(prod.right)
        if attr_name in self.inh_attrs:
          assert attr_pos >= 1, (
              '%r for %r: rules for inherited attributes only allowed for right side of production' % (target, prod))
        elif attr_name in self.syn_attrs:
          assert attr_pos == 0, (
              '%r for %r: rules for synthesized attributes only allowed for left side of production' % (target, prod))
        else:
          assert False
        if callable(func):
          func_arg_names = self._get_attr_func_arg_names(func)
          assert all(
            from_attr_name in self.attrs or from_attr_name.startswith('_') for from_attr_name in func_arg_names), (
            '%r for %r: function arguments must be attributes, got %r but only have %r' %
            (target, prod, func_arg_names, self.attrs))
        elif isinstance(func, str):
          from_attr_name, from_attr_pos = self._split_attr_name_pos(func)
          assert from_attr_name in self.attrs
          assert 0 <= from_attr_pos <= len(prod.right)
          if from_attr_name in self.inh_attrs:
            assert from_attr_pos == 0, (
              '%r for %r: may not depend on %s.%s' % (target, prod, from_attr_name, from_attr_pos))
          elif from_attr_name in self.syn_attrs:
            assert from_attr_pos >= 1, (
              '%r for %r: may not depend on %s.%s' % (target, prod, from_attr_name, from_attr_pos))
          else:
            assert False
        else:  # func is a constant
          pass
    for terminal, attr_rules in self.terminal_attr_rules.items():
      assert isinstance(attr_rules, dict)
      for target, func in attr_rules.items():
        attr_name, attr_pos = self._split_attr_name_pos(target)
        assert attr_pos == 0
        assert attr_name in self.syn_attrs
        if callable(func):
          assert len(self._get_attr_func_arg_names(func)) == 1
        elif isinstance(func, str):
          assert False, '%r for %r: terminal rules must not depend on other attributes' % (target, terminal)
        else:  # func is a constant
          pass

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
    assert terminal in self.grammar.terminals
    if terminal not in self.terminal_attr_rules:
      return {}

    attr_eval: Dict[str, Any] = {}
    if terminal not in self.terminal_attr_rules:
      return attr_eval
    for attr_target, func in self.terminal_attr_rules[terminal].items():
      attr_name, attr_pos = self._split_attr_name_pos(attr_target)
      if attr_name not in self.syn_attrs:
        continue
      assert attr_pos == 0
      if callable(func):
        attr_eval[attr_name] = func(word)
      elif isinstance(func, str):
        assert False, 'terminal attribute rules may not depend on anything else'
      else:
        attr_eval[attr_name] = func
    return attr_eval

  def _eval_prod_attr(self, prod, eval_pos, eval_attrs, left_attr_eval, right_attr_evals, helper_values=None):
    """
    Evaluate attr.pos for all attributes attr.

    :param Production prod:
    :param int eval_pos:
    :param set[str] eval_attrs: which attributes to evaluate
    :param dict[str,Any] left_attr_eval: (partial) evaluation of production left
    :param list[dict[str,Any]] right_attr_evals: (partial) evaluations of production right sides
    :param dict[str,Any]|None helper_values: can be used as additional parameters for attr eval.
      Names should be prefixed with _.
    :type: dict[str,Any]
    """
    assert prod in self.prod_attr_rules
    assert 0 <= eval_pos <= len(prod.right)
    assert all(attr in self.attrs for attr in eval_attrs)
    assert len(right_attr_evals) == len(prod.right)
    if helper_values is None:
      helper_values = {}
    assert all(helper_name.startswith('_') and helper_name not in self.attrs for helper_name in helper_values.keys())

    def make_attr_getter(get_attr_name, rule_attr_target):
      """
      :param str get_attr_name:
      :param str rule_attr_target: caller
      :rtype: Callable[tuple[int], Any]
      """
      if get_attr_name in helper_values:
        return helper_values[get_attr_name]

      def get(pos):
        """
        Receives attr.pos.
        :param int pos:
        :rtype: Any
        """
        assert 0 <= pos <= len(prod.right), '%s.%s: invalid for production %r' % (get_attr_name, pos, prod)
        if pos == 0:
          assert get_attr_name in left_attr_eval, '%r: evaluation of %s.%s for rule %r not available, only have %r' % (
            prod, get_attr_name, pos, rule_attr_target,
            _format_available_prod_eval(prod, left_attr_eval=left_attr_eval, right_attr_evals=right_attr_evals))
          return left_attr_eval[get_attr_name]
        else:
          assert 0 < pos <= len(prod.right)
          assert get_attr_name in right_attr_evals[pos - 1], (
            '%r: evaluation of %s.%s for rule %s not available, only have:\n%s' % (
              prod, get_attr_name, pos, rule_attr_target,
              _format_available_prod_eval(prod, left_attr_eval=left_attr_eval, right_attr_evals=right_attr_evals)))
          return right_attr_evals[pos - 1][get_attr_name]

      return get

    attr_eval: Dict[str, Any] = {}
    if eval_pos == 0:
      attr_eval.update(left_attr_eval)
    else:
      assert 0 < eval_pos <= len(prod.right)
      attr_eval.update(right_attr_evals[eval_pos - 1])

    for attr_target, func in self.prod_attr_rules[prod].items():
      attr_name, attr_pos = self._split_attr_name_pos(attr_target)
      if attr_name not in eval_attrs:
        continue
      if attr_pos is not eval_pos:
        continue
      if attr_name in self.inh_attrs:
        assert 0 < attr_pos <= len(prod.right)
      else:
        assert attr_name in self.syn_attrs
        assert attr_pos == 0
      assert attr_name not in attr_eval, 'already evaluated'

      available_attr_names = helper_values.keys() | left_attr_eval.keys() | {
        attr_name for right_eval in right_attr_evals for attr_name in right_eval}
      if callable(func):
        func_arg_names = self._get_attr_func_arg_names(func)
        assert all(from_attr_name in available_attr_names for from_attr_name in func_arg_names), (
          '%r: evaluation for %r not available, only have:\n%s' % (
            prod, func_arg_names,
            _format_available_prod_eval(prod, left_attr_eval=left_attr_eval, right_attr_evals=right_attr_evals)))
        func_kwargs = {
          from_attr_name: make_attr_getter(from_attr_name, rule_attr_target=attr_target)
          for from_attr_name in func_arg_names}
        attr_eval[attr_name] = func(**func_kwargs)
      elif isinstance(func, str):
        from_attr_name, from_attr_pos = self._split_attr_name_pos(func)
        assert from_attr_name in available_attr_names, (
          '%r: evaluation for %s.%s not available, only have:\n%s' % (
            prod, from_attr_name, from_attr_pos,
            _format_available_prod_eval(prod, left_attr_eval=left_attr_eval, right_attr_evals=right_attr_evals)))
        attr_eval[attr_name] = make_attr_getter(from_attr_name, rule_attr_target=attr_target)(from_attr_pos)
      else:
        attr_eval[attr_name] = func
    return attr_eval

  def eval_prod_syn_attr(self, prod, left_attr_eval, right_attr_evals, helper_values=None):
    """
    Evaluates synthetic attributes bottom-up, i.e. computes value of syn.0.

    :param Production prod:
    :param dict[str,Any] left_attr_eval: (partial) evaluation of production left
    :param list[dict[str,Any]] right_attr_evals: evaluations of right side of production
    :param dict[str,Any]|None helper_values: can be used as additional parameters for attr eval.
    :rtype: dict[str,Any]
    """
    return self._eval_prod_attr(
      prod, 0, eval_attrs=self.syn_attrs, left_attr_eval=left_attr_eval, right_attr_evals=right_attr_evals,
      helper_values=helper_values)

  def eval_prod_inh_attr(self, prod, eval_pos, left_attr_eval, right_attr_evals, helper_values=None):
    """
    Evaluates inherited attributes top-down, i.e. computes value of inh.pos

    :param Production prod:
    :param int eval_pos:
    :param dict[str,Any] left_attr_eval: (partial) evaluation of production left
    :param list[dict[str,Any]] right_attr_evals: (partial) evaluations of right side of production
    :param dict[str,Any]|None helper_values: can be used as additional parameters for attr eval.
    :rtype: dict[str,Any]
    """
    return self._eval_prod_attr(
      prod, eval_pos, eval_attrs=self.inh_attrs, left_attr_eval=left_attr_eval, right_attr_evals=right_attr_evals,
      helper_values=helper_values)

  def copy_with_start(self, start):
    """
    :param str start: new start non-terminal symbol for the underlying Grammar
    :rtype: AttributeGrammar
    """
    if start == self.grammar.start:
      return self
    assert start in self.grammar.start
    new_grammar = self.grammar.copy_with_start(start)
    return AttributeGrammar(
      grammar=new_grammar,
      prod_attr_rules=self.prod_attr_rules.copy(),
      terminal_attr_rules=self.terminal_attr_rules.copy(),
      inh_attrs=self.inh_attrs.copy(), syn_attrs=self.syn_attrs.copy())


def _format_available_prod_eval(prod, left_attr_eval, right_attr_evals):
  """
  For error messages.

  :param Production prod:
  :param dict[str,Any] left_attr_eval: (partial) evaluation of production left
  :param list[dict[str,Any]] right_attr_evals: (partial) evaluations of production right sides
  """
  all_prod_symbols = (prod.left,) + prod.right
  all_attr_evals = [left_attr_eval] + right_attr_evals
  assert len(all_prod_symbols) == len(all_attr_evals)
  return '\n'.join(
    ' - pos %s (symbol %r): %r' % (pos, symbol, attrs)
    for pos, (attrs, symbol) in enumerate(zip(all_attr_evals, all_prod_symbols)))


class SyntaxTree:
  """
  Simple (non-abstract, i.e. for each Production right side one child) syntax tree.
  """
  def __init__(self, prod, *right):
    """
    :param Production prod: production
    :param SyntaxTree|None right: trees corresponding to prod.right, or None for non-terminals
    """
    assert len(prod.right) == len(right)
    assert all(subtree is None or subtree.left == symbol for subtree, symbol in zip(right, prod.right))
    self.prod = prod
    self.right: Tuple[Optional[SyntaxTree]] = right

  @property
  def left(self):
    """
    :rtype: str
    """
    return self.prod.left

  def __repr__(self):
    return 'SyntaxTree[%r -> %s]' % (
      self.left, ' '.join([
        repr(symbol if subtree is None else subtree) for symbol, subtree in zip(self.prod.right, self.right)]))

  def __hash__(self):
    return hash((self.left, self.right))

  def __eq__(self, other):
    if not isinstance(other, SyntaxTree):
      return False
    return self.left == other.left and self.right == other.right

  def get_left_analysis(self):
    """
    :rtype: tuple[Production]
    """
    return (self.prod,) + tuple(
      prod for subtree in self.right if subtree is not None for prod in subtree.get_left_analysis())

  def get_right_analysis(self):
    """
    :rtype: tuple[Production]
    """
    return (self.prod,) + tuple(
      prod for subtree in reversed(self.right) if subtree is not None for prod in subtree.get_right_analysis())


class TreePosition:
  """
  The boundaries a sub-tree of a parsed word have.
  """

  def __init__(self, word, from_pos, to_pos):
    """
    :param str word:
    :param int from_pos:
    :param int to_pos:
    """
    assert 0 <= from_pos <= to_pos <= len(word)
    self.word = word
    self.from_pos = from_pos
    self.to_pos = to_pos

  def __repr__(self):
    """
    :rtype: str
    """
    return 'TreePosition(from=%r, to=%r)' % (self.from_pos, self.to_pos)

  def __eq__(self, other):
    """
    :rtype: bool
    """
    if not isinstance(other, TreePosition):
      return False
    return self.word == other.word and self.from_pos == other.from_pos and self.to_pos == other.to_pos

  def get_from_line_col(self) -> (int, int):
    res = get_line_col_from_pos(
      word=self.word, error_pos=self.from_pos, num_before_context_lines=0, num_after_context_lines=0)
    return res[0], res[1]

  def get_from_line(self) -> int:
    return self.get_from_line_col()[0]

  def get_from_col(self) -> int:
    return self.get_from_line_col()[1]

  @classmethod
  def from_token_pos(cls, word, tokens_pos, from_token_pos, to_token_pos):
    """
    :param str word:
    :param list[int] tokens_pos:
    :param int from_token_pos:
    :param int to_token_pos:
    :rtype: TreePosition
    """
    assert 0 <= from_token_pos <= to_token_pos <= len(tokens_pos)
    from_pos = tokens_pos[from_token_pos] if from_token_pos < len(tokens_pos) else len(word)
    to_pos = tokens_pos[to_token_pos] if to_token_pos < len(tokens_pos) else len(word)
    return TreePosition(word, from_pos, to_pos)


def get_line_col_from_pos(word, error_pos, num_before_context_lines=1, num_after_context_lines=1):
  """
  :param str word:
  :param int error_pos:
  :param int num_before_context_lines:
  :param int num_after_context_lines:
  :return: line + column, both starting counting at 1, as well as dict with context lines
  :rtype: tuple[int,int,dict[int,str]]
  """
  assert 0 <= error_pos <= len(word)
  if len(word) == 0:
    return 0, 1, {0: '\n'}
  char_pos = 0
  word_lines = word.splitlines()
  assert len(word_lines) > 0
  for line_num, line in enumerate(word_lines):
    assert char_pos <= error_pos
    if error_pos <= char_pos + len(line):
      col_num = error_pos - char_pos
      assert 0 <= col_num <= len(line)
      context_lines = {
        context_line_num + 1: word_lines[context_line_num]
        for context_line_num in range(
          max(0, line_num - num_before_context_lines), min(len(word_lines), line_num + num_after_context_lines + 1))}
      return line_num + 1, col_num + 1, context_lines
    char_pos += len(line) + 1  # consider end-of-line symbol
  assert char_pos == len(word)
  context_lines = {
    context_line_num + 1: word_lines[context_line_num]
    for context_line_num in range(max(0, len(word_lines) - num_before_context_lines), len(word_lines))}
  return len(word_lines), len(word_lines[-1]) + 1, context_lines
