from contextlib import contextmanager

import llvmlite.binding as llvm
from llvmlite.binding import ExecutionEngine


llvm.initialize()
llvm.initialize_native_target()
llvm.initialize_all_asmprinters()


std_func_identifiers = [
  ('print_char', 1), ('print_double', 1), ('allocate', 1), ('deallocate', 1), ('load', 1), ('store', 2)]
preamble = ''.join([
  'extern_func %s(%s);\n' % (identifier, ', '.join(['var%s' % num for num in range(num_args)]))
  for identifier, num_args in std_func_identifiers])


@contextmanager
def make_execution_engine():
  """
  Initialize just-in-time execution engine.

  :rtype: ExecutionEngine
  """
  target = llvm.Target.from_default_triple()
  target_machine = target.create_target_machine()
  backing_mod = llvm.parse_assembly('')
  engine = llvm.create_mcjit_compiler(backing_mod, target_machine)
  llvm.load_library_permanently('sleepy/std/build/libstd.so')
  yield engine
  del engine


def compile_ir(engine, llvm_ir):
  """
  :param ExecutionEngine engine:
  :param str|Any llvm_ir:
  """
  if not isinstance(llvm_ir, str):
    llvm_ir = llvm_ir.__repr__()
  mod = llvm.parse_assembly(llvm_ir)
  mod.verify()
  engine.add_module(mod)
  engine.finalize_object()
  engine.run_static_constructors()
