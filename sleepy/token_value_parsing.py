from typing import List

from sleepy.errors import CompilerError


class InvalidLiteralError(CompilerError):
  """
  Error raised when a literal is invalid.
  """

  def __init__(self, literal: str, message: str):
    super().__init__("Literal: " + literal + " invalid.\n" + message)


ESCAPE_CHARACTERS = {'n': '\n', 'r': '\r', 't': '\t', "'": "'", '"': '"', '0': '\0'}


def parse_long(value: str) -> int:
  """
  :param value: e.g. 123l, ...
  """
  assert value[-1] in {'l', 'L'}
  try:
    return int(value[:-1])
  except ValueError as e:
    raise InvalidLiteralError(value, "Invalid long literal.") from e


def parse_float(value: str) -> float:
  """
  :param value: e.g. 0.5f, ...
  """
  assert value[-1] in {'f', 'F'}
  try:
    return float(value[:-1])
  except ValueError as e:
    raise InvalidLiteralError(value, "Invalid float literal.") from e


def parse_double(value: str) -> float:
  """
  :param value: e.g. 0.5, ...
  """
  try:
    return float(value)
  except ValueError as e:
    raise InvalidLiteralError(value, "Invalid double literal.") from e


def parse_char(value: str) -> str:
  """
  :param value: e.g. 'a', '\n', ...
  """
  assert 3 <= len(value) <= 4
  assert value[0] == value[-1] == "'"
  value = value[1:-1]
  if len(value) == 1:
    return value
  assert value[0] == '\\'
  assert value[1] in ESCAPE_CHARACTERS, 'unknown escape character \\%s' % [value[1]]
  return ESCAPE_CHARACTERS[value[1]]


def parse_string(value: str) -> str:
  """
  :param value: e.g. "abc", "", "cool \"stuff\""
  """
  assert len(value) >= 2
  assert value[0] == value[-1] == '"'
  value = value[1:-1]
  res: List[chr] = []
  pos = 0
  while pos < len(value):
    char = value[pos]
    if char == '\\':
      if pos + 1 >= len(value):
        raise InvalidLiteralError("\"%s\"" % value, "Escape sequence at end of string.")
      if value[pos + 1] not in ESCAPE_CHARACTERS:
        raise InvalidLiteralError("\"%s\"" % value, 'Unknown escape character.')
      res.append(ESCAPE_CHARACTERS[value[pos + 1]])
      pos += 2
    else:
      res.append(value[pos])
      pos += 1
  return ''.join(res)


def parse_hex_int(value: str) -> int:
  """
  :param value: e.g. 0x0043fabc
  """
  return int(value, 0)


def parse_assign_op(value: str) -> str:
  """
  :param value: e.g. +=
  """
  assert len(value) >= 2
  assert value[-1] == '='
  return value[:-1]
