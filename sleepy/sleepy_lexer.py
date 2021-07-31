from sleepy.lexer import LexerGenerator

SLEEPY_LEXER = LexerGenerator(
  [
    'func', 'extern_func', 'struct', 'if', 'else', 'return', 'while', '{', '}', ';', ',', '.', '(', ')', '|',
    '->', '@', 'cmp_op', 'sum_op', 'prod_op', '=', ':', 'assign_op', '[', ']',
    'identifier',
    'int', 'long', 'double', 'float',
    'char', 'str', 'hex_int',
    None, None
  ], [
    'func', 'extern_func', 'struct', 'if', 'else', 'return', 'while', '{', '}', ';', ',', '\\.', '\\(', '\\)', '\\|',
    '\\->', '@', '==|!=|<=?|>=?|is', '\\+|\\-', '\\*|/', '=', ':', '===|!==|<==|>==|\\+=|\\-=|\\*=|/=', '\\[', '\\]',
    '([A-Z]|[a-z]|_)([A-Z]|[a-z]|[0-9]|_)*',
    '(0|[1-9][0-9]*)', '(0|[1-9][0-9]*)l', '(0|[1-9][0-9]*)\\.([0-9]?)+d?', '(0|[1-9][0-9]*)((\\.([0-9]?))?)+f',
    "'([^\']|\\\\[0nrt'\"])'", '"([^\"]|\\\\[0nrt\'"])*"', '0x([0-9]|[A-F]|[a-f])+',
    '#[^\n]*\n', '[ \n\t]+'
  ])