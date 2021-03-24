from typing import Optional

import llvmlite.binding as llvm
from llvmlite.binding import ExecutionEngine


default_execution_engine = None  # type: Optional[ExecutionEngine]


def get_execution_engine():
  """
  Initialize just-in-time execution engine.
  Will only be initialized once and then reused.

  :rtype: ExecutionEngine
  """
  global default_execution_engine
  if default_execution_engine is not None:
    return default_execution_engine
  llvm.initialize()
  llvm.initialize_native_target()
  llvm.initialize_all_asmprinters()

  target = llvm.Target.from_default_triple()
  target_machine = target.create_target_machine()
  backing_mod = llvm.parse_assembly('')
  default_execution_engine = llvm.create_mcjit_compiler(backing_mod, target_machine)
  return default_execution_engine


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
