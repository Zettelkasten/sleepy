import os

import _setup_test_env  # noqa
import better_exchook
import sys
import unittest

from sleepy.ast import make_program_ast
from sleepy.jit import make_execution_engine, compile_ir
from sleepy.symbols import FunctionSymbol


def _test_run_example(code_file_name):
  print('\nLoading example from %s.' % code_file_name)
  with make_execution_engine() as engine:
    with open(code_file_name, 'r') as file:
      program = file.read()
    ast = make_program_ast(program)
    module_ir, symbol_table = ast.make_module_ir_and_symbol_table(module_name='test_parse_ast')
    compile_ir(engine, module_ir)
    assert 'main' in symbol_table, 'Need to declare a main function'
    main_func_symbol = symbol_table['main']
    assert isinstance(main_func_symbol, FunctionSymbol), 'main needs to be a function'
    assert len(main_func_symbol.concrete_funcs) == 1, 'need to declare exactly one main function'
    concrete_main_func = main_func_symbol.get_single_concrete_func()
    py_func = concrete_main_func.make_py_func(engine)
    print('Now executing:')
    return_val = py_func()
    print('Returned value: %r of type %r' % (return_val, concrete_main_func.return_type))


def test_run_examples():
  examples_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'examples')
  code_file_root, _, code_file_names = next(os.walk(examples_path))
  assert len(code_file_names) >= 1
  for code_file_name in code_file_names:
    yield _test_run_example, os.path.join(code_file_root, code_file_name)


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
