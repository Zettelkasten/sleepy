#!/usr/bin/env python3

"""
Main entry point: Compile a sleepy program to object code.
"""
import argparse
from pathlib import Path

import better_exchook
import networkx.algorithms.dag

import _setup_sleepy_env  # noqa
# noinspection PyUnresolvedReferences
from sleepy.ast import FileAst, TranslationUnitAst
from sleepy.errors import CompilerError
# noinspection PyUnresolvedReferences
from sleepy.jit import make_execution_engine, compile_ir, PREAMBLE_BINARIES_PATH
# noinspection PyUnresolvedReferences
from sleepy.ast_generation import make_file_ast, make_translation_unit_ast
from sleepy.types import OverloadSet
import llvmlite.binding as llvm


def _make_file_name(source_path: Path, file_ending: str, allow_exist=False) -> Path:
  """
   :param file_ending: with leading . if applicable
  """
  if source_path.suffix != ".slp":
    file_ending = ".out" if file_ending == "" else file_ending

  source_path = source_path.with_suffix(file_ending)

  if allow_exist or not source_path.exists():
    return source_path
  file_num = 0
  source_path = source_path.with_stem(source_path.stem + str(file_num))
  while source_path.exists():
    file_num += 1
    source_path = source_path.with_stem(source_path.stem + str(file_num))
  return source_path


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
    '--verbose', dest='verbose', action='store_true', help='Print full stacktrace for all compiler errors.')
  parser.add_argument(
    '--Optimization', '-O', dest='opt', action='store', type=int, default=0, help='Optimize code.')
  parser.add_argument('--no-preamble', default=False, action='store_true', help='Do not add preamble to source code.')
  parser.add_argument('--debug', default=False, action='store_true', help='Add debug symbols.')
  parser.add_argument('--output', default=None, action='store', help='output file path')

  args = parser.parse_args()

  main_func_identifier = 'main'
  source_file_path: Path = Path(args.program)
  try:
    ast = make_translation_unit_ast(source_file_path)
    module_ir, symbol_table = ast.make_module_ir_and_symbol_table(
      module_name='default_module', emit_debug=args.debug, main_file_path=source_file_path, implicitly_exported_functions={main_func_identifier})
    if main_func_identifier not in symbol_table:
      raise CompilerError('Error: Entry point function %r not found' % main_func_identifier)
    main_func_symbol = symbol_table[main_func_identifier]
    if not isinstance(main_func_symbol, OverloadSet):
      raise CompilerError('Error: Entry point %r must be a function' % main_func_identifier)
    if len(main_func_symbol.signatures) != 1:
      raise CompilerError('Error: Must declare exactly one entry point function %r' % main_func_identifier)
  except CompilerError as ce:
    if args.verbose:
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

  object_file_name = _make_file_name(source_file_path, '.o', allow_exist=True)
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
    ir_file_name = _make_file_name(source_file_path, '.ll', allow_exist=True)
    with open(ir_file_name, 'w') as file:
      file.write(str(module_ir))
    return

  target = llvm.Target.from_default_triple()
  machine = target.create_target_machine(opt=args.opt)
  with open(object_file_name, 'wb') as file:
    file.write(machine.emit_object(module_ref))
  if args.emit_object:
    return

  if args.output is not None: exec_file_name = args.output
  else: exec_file_name = _make_file_name(source_file_path, '', allow_exist=True)

  import subprocess
  subprocess.run(
    ['gcc'] + (['-g'] if args.debug else [])
    + ['-o', exec_file_name, object_file_name, PREAMBLE_BINARIES_PATH + '_static.a']
    + ['-l%s' % lib_name for lib_name in args.compile_libs])


if __name__ == '__main__':
  main()
