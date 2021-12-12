from typing import Tuple, List

from sleepy.syntactical_analysis.lexer import LexerGenerator
from sleepy.lexer_definition import WHITESPACE_REGEX, COMMENT_REGEX, TOKEN_INFO

from sleepy.parser_definition import SLEEPY_PARSER

PLUGIN_LEXER = LexerGenerator(
  [name for name, _, _ in TOKEN_INFO] + ["COMMENT", "WHITESPACE", "BAD_CHARACTER"],
  [regex for _, regex, _ in TOKEN_INFO] + [COMMENT_REGEX, WHITESPACE_REGEX, "[^b-a]"]
)

TOKEN_NAME_TO_HIGHLIGHT = dict(
  ((name, highlight) for name, _, highlight in TOKEN_INFO)
)

def token_name_to_highlight(name: str) -> str:
  return TOKEN_NAME_TO_HIGHLIGHT.get(name, name)

def tokenize(word:str) -> Tuple[List[str], List[int]]:
  names, positions = PLUGIN_LEXER.tokenize(word)
  return list(names), list(positions)

def parse(token_stream):
  SLEEPY_PARSER.parse_stream(token_stream)


print("INITIALIZED CORRECTLY")
