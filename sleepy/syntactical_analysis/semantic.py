from typing import List, Dict, Any, Optional, Union

from sleepy.syntactical_analysis.grammar import AttributeGrammar, SyntaxTree, get_token_word_from_tokens_pos


class AttributeEvalGenerator:
  """
  Evaluates l-attributed attributes of a parsed syntax tree:
  All inherited attributes inh.i must only depend on inh.0 or on synthesized syn.j (j < i) attributes on their left.
  """
  def __init__(self, attr_grammar: AttributeGrammar):
    self.attr_grammar = attr_grammar

  @property
  def grammar(self):
    """
    :rtype: Grammar
    """
    return self.attr_grammar.grammar

  def eval_attrs(self, root_tree: SyntaxTree,
                 word: str,
                 tokens: List[str],
                 tokens_pos: List[str]) -> Dict[str, Any]:
    """
    :param root_tree: parse tree of start symbol.
    :param tokens_pos: start index of word for each token
    :return: evaluated attributes in start symbol
    """
    assert root_tree.prod.left == self.grammar.start
    assert len(tokens) == len(tokens_pos)
    token_pos = 0
    tree_stack: List[SyntaxTree] = [root_tree]
    prod_pos_stack: List[int] = [0]
    left_attr_eval_stack: List[Dict[str, Any]] = [{}]
    right_attr_eval_stack: List[List[Dict[str, Any]]] = [[{} for _ in range(len(root_tree.right))]]
    root_eval: Optional[Dict[str, Any]] = None

    while len(tree_stack) >= 1:
      current_tree, current_prod_pos = tree_stack[-1], prod_pos_stack[-1]
      current_left_attr_eval, current_right_attr_eval = left_attr_eval_stack[-1], right_attr_eval_stack[-1]
      assert 0 <= current_prod_pos <= len(current_tree.right) == len(current_right_attr_eval)
      if current_prod_pos < len(current_tree.right):
        next_symbol = current_tree.prod.right[current_prod_pos]
        next_subtree = current_tree.right[current_prod_pos]
        if next_symbol in self.grammar.non_terminals:
          # descent into non-terminal subtree: evaluate inherited attrs
          assert next_subtree is not None
          subtree_attr_eval = self.attr_grammar.eval_prod_inh_attr(
            current_tree.prod, current_prod_pos + 1, current_left_attr_eval, current_right_attr_eval)
          tree_stack.append(next_subtree)
          prod_pos_stack.append(0)
          left_attr_eval_stack.append(subtree_attr_eval)
          right_attr_eval_stack.append([{} for _ in range(len(next_subtree.right))])
        else:
          # scan terminal symbol
          assert tokens[token_pos] is next_symbol in self.grammar.terminals
          assert next_subtree is None
          next_token_word = get_token_word_from_tokens_pos(word, tokens_pos, token_pos)
          symbol_attr_eval = self.attr_grammar.get_terminal_syn_attr_eval(next_symbol, next_token_word)
          token_pos += 1
          prod_pos_stack[-1] = current_prod_pos + 1
          current_right_attr_eval[current_prod_pos] = symbol_attr_eval
      else:
        assert current_prod_pos == len(current_tree.right)
        # ascent into non-terminal parent tree
        parent_tree_attr_eval = self.attr_grammar.eval_prod_syn_attr(
          current_tree.prod, current_left_attr_eval, current_right_attr_eval)
        tree_stack.pop()
        prod_pos_stack.pop()
        left_attr_eval_stack.pop()
        right_attr_eval_stack.pop()
        if len(tree_stack) >= 1:
          parent_prod_pos = prod_pos_stack[-1]
          assert parent_prod_pos + 1 <= len(tree_stack[-1].right) == len(right_attr_eval_stack[-1])
          right_attr_eval_stack[-1][parent_prod_pos] = parent_tree_attr_eval
          prod_pos_stack[-1] = parent_prod_pos + 1
        else:
          # done parsing
          root_eval = parent_tree_attr_eval

    assert token_pos == len(tokens)
    assert len(tree_stack) == len(left_attr_eval_stack) == len(right_attr_eval_stack) == 0
    assert root_eval is not None
    assert len(root_eval) >= 1
    return root_eval
