from typing import List

from sleepy.errors import CompilerError


class InvalidLiteralError(CompilerError):
  """
  Error raised when a literal is invalid.
  """
  def __init__(self, literal, message):
    """
    :param str message:
    """
    super().__init__("Literal: " + literal + " invalid.\n" + message)


ESCAPE_CHARACTERS = {'n': '\n', 'r': '\r', 't': '\t', "'": "'", '"': '"', '0': '\0'}


def parse_long(value):
  """
  :param str value: e.g. 123l, ...
  :rtype: str
  """
  assert value[-1] in {'l', 'L'}
  try:
    return int(value[:-1])
  except ValueError as e:
    raise InvalidLiteralError(value, "Invalid long literal.") from e


def parse_float(value):
  """
  :param str value: e.g. 0.5f, ...
  :rtype: str
  """
  assert value[-1] in {'f', 'F'}
  try:
    return float(value[:-1])
  except ValueError as e:
    raise InvalidLiteralError(value, "Invalid float literal.") from e


def parse_double(value: str):
  """
  :param str value: e.g. 0.5, ...
  :rtype: str
  """
  try:
    return float(value)
  except ValueError as e:
    raise InvalidLiteralError(value, "Invalid double literal.") from e


def parse_char(value):
  """
  :param str value: e.g. 'a', '\n', ...
  :rtype: str
  """
  assert 3 <= len(value) <= 4
  assert value[0] == value[-1] == "'"
  value = value[1:-1]
  if len(value) == 1:
    return value
  assert value[0] == '\\'
  assert value[1] in ESCAPE_CHARACTERS, 'unknown escape character \\%s' % [value[1]]
  return ESCAPE_CHARACTERS[value[1]]


def parse_string(value):
  """
  :param str value: e.g. "abc", "", "cool \"stuff\""
  :rtype: str
  """
  assert len(value) >= 2
  assert value[0] == value[-1] == '"'
  value = value[1:-1]
  res: List[chr] = []
  pos = 0
  while pos < len(value):
    char = value[pos]
    if char == '\\':
      if not pos + 1 < len(value): raise InvalidLiteralError("\"%s\"" % value, "Escape sequence at end of string.")
      if not value[pos + 1] in ESCAPE_CHARACTERS: raise InvalidLiteralError("\"%s\"" % value, 'Unknown escape character.')
      res.append(ESCAPE_CHARACTERS[value[pos + 1]])
      pos += 2
    else:
      res.append(value[pos])
      pos += 1
  return ''.join(res)


def parse_hex_int(value):
  """
  :param str value: e.g. 0x0043fabc
  :rtype: int
  """
  return int(value, 0)


def parse_assign_op(value):
  """
  :param str value: e.g. +=
  :rtype: str
  """
  assert len(value) >= 2
  assert value[-1] == '='
  return value[:-1]