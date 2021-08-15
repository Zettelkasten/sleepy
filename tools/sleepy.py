#!/usr/bin/env python3

"""
Main entry point: Compile a sleepy program to object code.
"""
import argparse
import better_exchook

import _setup_sleepy_env  # noqa
from sleepy.sleepy_parser import make_program_ast
from sleepy.errors import CompilerError
from sleepy.jit import make_execution_engine, compile_ir, LIB_PATH
from sleepy.symbols import FunctionSymbol
import llvmlite.binding as llvm


def _make_file_name(source_file_name, file_ending, allow_exist=False):
  """
  :param str source_file_name:
  :param str file_ending: with leading . if applicable
  :param bool allow_exist:
  :rtype str:
  """
  if source_file_name.endswith('.slp'):
    base_file_name = source_file_name[:-len('.slp')]
  else:
    base_file_name = source_file_name
    if file_ending == '':
      base_file_name += '.out'
  import os.path
  if allow_exist or not os.path.isfile('%s%s' % (base_file_name, file_ending)):
    return '%s%s' % (base_file_name, file_ending)
  file_num = 0
  while os.path.isfile('%s.%s%s' % (base_file_name, file_num, file_ending)):
    file_num += 1
  return '%s.%s%s' % (base_file_name, file_num, file_ending)


def main():
  """
  Main entry point.
  """
  better_exchook.install()
  parser = argparse.ArgumentParser(description='Compile a Sleepy program to object code.')
  parser.add_argument('program', help='Path to source code')
  parser.add_argument(
    '--execute', dest='execute', default=False, action='store_true', help='Run program after compilation using JIT.')
  parser.add_argument(
    '--emit-ir', '-ir', dest='emit_ir', action='store_true', help='Emit LLVM intermediate representation.')
  parser.add_argument(
    '--emit-object', '-c', dest='emit_object', action='store_true', help='Emit object code, but do not link.')
  parser.add_argument('--compile-libs', '-libs', nargs='*', help='External libraries to link with', default=['m'])
  parser.add_argument(
    '--debug', dest='debug', action='store_true', help='Print full stacktrace for all compiler errors.')
  parser.add_argument(
    '--Optimization', '-O', dest='opt', action='store', type=int, default=0, help='Optimize code.')
  parser.add_argument('--no-preamble', default=False, action='store_true', help='Do not add preamble to source code.')

  args = parser.parse_args()

  main_func_identifier = 'main'
  source_file_name: str = args.program
  with open(source_file_name) as program_file:
    program = program_file.read()
  try:
    ast = make_program_ast(program, add_preamble=not args.no_preamble)
    module_ir, symbol_table = ast.make_module_ir_and_symbol_table(module_name='default_module')
    if main_func_identifier not in symbol_table:
      raise CompilerError('Error: Entry point function %r not found' % main_func_identifier)
    main_func_symbol = symbol_table[main_func_identifier]
    if not isinstance(main_func_symbol, FunctionSymbol):
      raise CompilerError('Error: Entry point %r must be a function' % main_func_identifier)
    if len(main_func_symbol.signatures) != 1:
      raise CompilerError('Error: Must declare exactly one entry point function %r' % main_func_identifier)
  except CompilerError as ce:
    if args.debug:
      raise ce
    else:
      print(str(ce))
      exit(1)
      return

  if args.execute:
    # Execute directly using JIT compilation.
    concrete_main_func = main_func_symbol.get_single_concrete_func()
    with make_execution_engine() as engine:
      compile_ir(engine, module_ir)
      py_func = concrete_main_func.make_py_func(engine)
      return_val = py_func()
    print('\nExited with return value %r of type %r' % (return_val, concrete_main_func.return_type))
    return

  object_file_name = _make_file_name(source_file_name, '.o', allow_exist=True)
  module_ref = llvm.parse_assembly(str(module_ir))

  print(f'Opt: {args.opt}')
  if args.opt != 0:
    # run optimizations on module, optimizations during emit_object are different and less powerful
    module_passes = llvm.ModulePassManager()
    builder = llvm.PassManagerBuilder()
    builder.opt_level = args.opt
    builder.inlining_threshold = 250
    builder.populate(module_passes)
    module_passes.run(module_ref)

  if args.emit_ir:
    ir_file_name = _make_file_name(source_file_name, '.ll', allow_exist=True)
    with open(ir_file_name, 'w') as file:
      file.write(str(module_ir))
    return

  target = llvm.Target.from_default_triple()
  machine = target.create_target_machine(opt=args.opt)
  with open(object_file_name, 'wb') as file:
    file.write(machine.emit_object(module_ref))
  if args.emit_object:
    return

  exec_file_name = _make_file_name(source_file_name, '', allow_exist=True)
  import subprocess
  subprocess.run(
    ['gcc', '-o', exec_file_name, object_file_name, LIB_PATH + '_static.a']
    + ['-l%s' % lib_name for lib_name in args.compile_libs])


if __name__ == '__main__':
  main()