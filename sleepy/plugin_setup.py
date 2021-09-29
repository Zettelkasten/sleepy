from typing import Tuple, List

from sleepy.lexer import LexerGenerator
from sleepy.sleepy_lexer import WHITESPACE_REGEX, COMMENT_REGEX, TOKEN_INFO

PLUGIN_LEXER = LexerGenerator(
  ["ANNOTATION"] + [name for name, _, _ in TOKEN_INFO] + ["COMMENT", "WHITESPACE", "BAD_CHARACTER"],
  ["@([A-Z]|[a-z]|_)([A-Z]|[a-z]|[0-9]|_)*"] + [regex for _, regex, _ in TOKEN_INFO] + [COMMENT_REGEX, WHITESPACE_REGEX, "[^b-a]"]
)

TOKEN_NAME_TO_HIGHLIGHT = dict(
  ((name, highlight) for name, _, highlight in TOKEN_INFO)
)

def tokenize(word:str) -> Tuple[List[str], List[int]]:
  names, positions = PLUGIN_LEXER.tokenize(word)
  return [TOKEN_NAME_TO_HIGHLIGHT.get(name, name) for name in names], list(positions)

print("INITIALIZED CORRECTLY")
