import os

import _setup_test_env  # noqa
from better_exchook import better_exchook
import sys
import unittest

from ctypes import CFUNCTYPE, c_double

from sleepy.ast import TopLevelExpressionAst
from sleepy.jit import make_execution_engine, compile_ir, preamble
from tests.test_ast import SLEEPY_LEXER, SLEEPY_PARSER, SLEEPY_ATTR_GRAMMAR


def _test_compile_example(code_file_name):
  print('\nLoading example from %s.' % code_file_name)
  with make_execution_engine() as engine:
    with open(code_file_name, 'r') as file:
      program = file.read()
    program = preamble + program
    tokens, tokens_pos = SLEEPY_LEXER.tokenize(program)
    _, root_eval = SLEEPY_PARSER.parse_syn_attr_analysis(SLEEPY_ATTR_GRAMMAR, program, tokens, tokens_pos)
    ast = root_eval['ast']
    assert isinstance(ast, TopLevelExpressionAst)
    module_ir = ast.make_module_ir(module_name='test_parse_ast')
    compile_ir(engine, module_ir)
    main_func_ptr = engine.get_function_address('main')
    py_func = CFUNCTYPE(c_double)(main_func_ptr)
    assert callable(py_func)
    print('Now execution:')
    return_val = py_func()
    print('Returned value: %r' % return_val)


def test_compile_examples():
  code_file_root, _, code_file_names = next(os.walk('tests/examples'))
  for code_file_name in code_file_names:
    yield _test_compile_example, os.path.join(code_file_root, code_file_name)


if __name__ == "__main__":
  try:
    better_exchook.install()
    if len(sys.argv) <= 1:
      for k, v in sorted(globals().items()):
        if k.startswith("test_"):
          print("-" * 40)
          print("Executing: %s" % k)
          try:
            v()
          except unittest.SkipTest as exc:
            print("SkipTest:", exc)
          print("-" * 40)
      print("Finished all tests.")
    else:
      assert len(sys.argv) >= 2
      for arg in sys.argv[1:]:
        print("Executing: %s" % arg)
        if arg in globals():
          globals()[arg]()  # assume function and execute
        else:
          eval(arg)  # assume Python code and execute
  finally:
    pass
