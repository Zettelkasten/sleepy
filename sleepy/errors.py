class CompilerError(Exception):
  def __init__(self, message: str):
    super(CompilerError, self).__init__(message)


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


def make_error_message(word, from_pos, error_name, message, to_pos=None):
  """
  :param str word:
  :param int from_pos:
  :param str error_name:
  :param str message:
  :param int|None to_pos:
  :rtype: str
  """
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