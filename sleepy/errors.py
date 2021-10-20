from typing import Optional

from sleepy.grammar import get_line_col_from_pos


class CompilerError(Exception):
  def __init__(self, message: str):
    super(CompilerError, self).__init__(message)


def make_error_message(word: str, from_pos: int, error_name: str, message: str, to_pos: Optional[int] = None) -> str:
  line_num_pad_size = 3
  line, from_col, context_lines = get_line_col_from_pos(
    word, from_pos, num_before_context_lines=3, num_after_context_lines=3)
  assert line in context_lines
  if to_pos is not None:
    assert from_pos <= to_pos
    to_line, to_col, _ = get_line_col_from_pos(word, to_pos, num_before_context_lines=0, num_after_context_lines=0)
    if line != to_line:  # multi-line errors will only show the first line.
      to_col = len(context_lines[line])
  else:
    to_col = from_col
  return '%s on line %s:%s\n\n' % (error_name, line, from_col) + '\n'.join([
    ('%0' + str(line_num_pad_size) + 'i: %s%s') % (
      context_line_num, context_line,
      ('\n' + (' ' * (from_col - 1 + line_num_pad_size + 2)) + '^' * max(1, to_col - from_col))
      if context_line_num == line else '')
    for context_line_num, context_line in context_lines.items()]
  ) + '\n\n' + message


class LexError(CompilerError):
  """
  A lexical error, when a word is not recognized (does not have a first-longest-match analysis).
  """

  def __init__(self, word, pos, message):
    """
    :param str word:
    :param int pos:
    :param str message:
    """
    super().__init__(make_error_message(word, pos, error_name='Lexical error', message=message))


class ParseError(CompilerError):
  """
  A parse error, when a word is not recognized by a context free grammar.
  """

  def __init__(self, word, pos, message):
    """
    :param str word:
    :param int pos: word position where error occurred
    :param str message:
    """
    super().__init__(make_error_message(word, pos, error_name='Parse error', message=message))


class SemanticError(CompilerError):
  """
  A semantic error, during code generation.
  """

  def __init__(self, word, from_pos, to_pos, message: str):
    """
    :param str word:
    :param int from_pos: word position where error occurred
    :param int to_pos: up to which position
    :param str message:
    """
    super().__init__(make_error_message(word, from_pos, error_name='Semantic error', message=message, to_pos=to_pos))
