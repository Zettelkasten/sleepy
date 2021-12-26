import os
from contextlib import contextmanager

import llvmlite.binding as llvm
from llvmlite.binding import ExecutionEngine

LIBRARY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'std/lib')
PREAMBLE_BINARIES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'std/build/libstd')

llvm.initialize()
llvm.initialize_native_target()
llvm.initialize_all_asmprinters()

pass_manager = llvm.ModulePassManager()
pass_manager.add_instruction_combining_pass()
pass_manager.add_gvn_pass()
pass_manager.add_cfg_simplification_pass()


@contextmanager
def make_execution_engine() -> ExecutionEngine:
  """
  Initialize just-in-time execution engine.
  """
  target = llvm.Target.from_default_triple()
  target_machine = target.create_target_machine()
  backing_mod = llvm.parse_assembly('')
  engine = llvm.create_mcjit_compiler(backing_mod, target_machine)
  llvm.load_library_permanently(PREAMBLE_BINARIES_PATH + '.so')
  yield engine
  del engine


def compile_ir(engine: ExecutionEngine, llvm_ir: str) -> llvm.ModuleRef:
  mod = llvm.parse_assembly(llvm_ir)
  mod.verify()
  engine.add_module(mod)
  engine.finalize_object()
  engine.run_static_constructors()
  return mod


def get_func_address(engine: ExecutionEngine, ir_func: llvm.ir.Function) -> int:
  return engine.get_function_address(ir_func.name)
