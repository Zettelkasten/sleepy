import os
from contextlib import contextmanager

import llvmlite.binding as llvm
from llvmlite.binding import ExecutionEngine


llvm.initialize()
llvm.initialize_native_target()
llvm.initialize_all_asmprinters()


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
  lib_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'std/build/libstd.so')
  llvm.load_library_permanently(lib_path)
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
