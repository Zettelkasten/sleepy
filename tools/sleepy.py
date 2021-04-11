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
    print('Error: Entry point function %r not found' % main_func_identifier)
    exit()
  main_func_symbol = symbol_table[main_func_identifier]
  if not isinstance(main_func_symbol, FunctionSymbol):
    print('Error: Entry point %r must be a function' % main_func_identifier)
    exit()
  if len(main_func_symbol.concrete_funcs) != 1:
    print('Error: Must declare exactly one entry point function %r' % main_func_identifier)
    exit()
  concrete_main_func = main_func_symbol.get_single_concrete_func()
  with make_execution_engine() as engine:
    compile_ir(engine, module_ir)
  py_func = concrete_main_func.make_py_func(engine)
  return_val = py_func()
  print('\nExited with return value %r of type %r' % (return_val, concrete_main_func.return_type))


if __name__ == '__main__':
  main()