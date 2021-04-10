#!/usr/bin/env python3

"""
Main entry point: Load a sleepy program and execute its main function.
"""
import argparse
from ctypes import CFUNCTYPE
import better_exchook

import _setup_sleepy_env  # noqa
from sleepy.ast import make_program_ast
from sleepy.jit import make_execution_engine, compile_ir
from sleepy.symbols import FunctionSymbol


def main():
  """
  Main entry point.
  """
  better_exchook.install()
  parser = argparse.ArgumentParser(description='Run a Sleepy Script program.')
  parser.add_argument('program', help='Path to source code')
  args = parser.parse_args()

  main_func_identifier = 'main'
  with open(args.program) as program_file:
    program = program_file.read()
  ast = make_program_ast(program)
  module_ir, symbol_table = ast.make_module_ir_and_symbol_table(module_name='default_module')
  if main_func_identifier not in symbol_table:
    print('Error: Entry point function %r not found')
    exit()
  main_func_symbol = symbol_table[main_func_identifier]
  if not isinstance(main_func_symbol, FunctionSymbol):
    print('Error: Entry point %r must be a function')
    exit()

  with make_execution_engine() as engine:
    compile_ir(engine, module_ir)
  main_func_ptr = engine.get_function_address(main_func_identifier)
  py_func = CFUNCTYPE(
    main_func_symbol.return_type.c_type, *[arg_type.c_type for arg_type in main_func_symbol.arg_types])(main_func_ptr)
  assert callable(py_func)
  return_val = py_func()
  print('\nExited with return value %r of type %r' % (return_val, main_func_symbol.return_type))


if __name__ == '__main__':
  main()