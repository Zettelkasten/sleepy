from sleepy.syntactical_analysis.lexer import LexerGenerator

TOKEN_INFO = [
  ('func', 'func', 'KEYWORD'),
  ('extern_func', 'extern_func', "KEYWORD"),
  ('struct', 'struct', "KEYWORD"),
  ('if', 'if', "KEYWORD"),
  ('else', 'else', "KEYWORD"),
  ('return', 'return', "KEYWORD"),
  ('while', 'while', "KEYWORD"),
  ('mutates', 'mutates', "KEYWORD"),
  ('import', 'import', 'KEYWORD'),
  (':', ':', ''),
  ('{', '{', "BRACE"),
  ('}', '}', "BRACE"),
  (';', ';', "SEMICOLON"),
  (',', ',', "COMMA"),
  ('.', '\\.', "DOT"),
  ('(', '\\(', "PARENTHESES"),
  (')', '\\)', "PARENTHESES"),
  ('|', '\\|', "OPERATOR"),
  ('->', '\\->', ""),
  ('@', '@', ""),
  ('cmp_op', '==|!=|<=?|>=?|as', "OPERATOR"),
  ('is', 'is', "OPERATOR"),
  ('in', 'in', "OPERATOR"),
  ('sum_op', '\\+|\\-', "OPERATOR"),
  ('prod_op', '\\*|/', "OPERATOR"),
  ('not', 'not', "OPERATOR"),
  ('=', '=', "OPERATOR"),
  ('assign_op', '===|!==|<==|>==|\\+=|\\-=|\\*=|/=', "OPERATOR"),
  ('unbind_op', '!', "OPERATOR"),
  ('[', '\\[', "BRACKETS"),
  (']', '\\]', "BRACKETS"),
  ('identifier', '([A-Z]|[a-z]|_)([A-Z]|[a-z]|[0-9]|_)*', "IDENTIFIER"),
  ('int', '(0|[1-9][0-9]*)', "NUMBER"),
  ('long', '(0|[1-9][0-9]*)l', "NUMBER"),
  ('double', '(0|[1-9][0-9]*)\\.([0-9]?)+d?', "NUMBER"),
  ('float', '(0|[1-9][0-9]*)((\\.([0-9]?))?)+f', "NUMBER"),
  ('char', '\'([^\']|\\\\[0nrt\'"])\'', "STRING"),
  ('str', '"([^"]|\\\\[0nrt\'"])*"', "STRING"),
  ('hex_int', '0x([0-9]|[A-F]|[a-f])+', "NUMBER"),
  ('new_line', '\n', "WHITESPACE")
]

COMMENT_REGEX = '#[^\n]*'
WHITESPACE_REGEX = '[ \t]+'

SLEEPY_LEXER = LexerGenerator(
  [name for name, _, _ in TOKEN_INFO] + [None, None],
  [regex for _, regex, _ in TOKEN_INFO] + [COMMENT_REGEX, WHITESPACE_REGEX]
)
