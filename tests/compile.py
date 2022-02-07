from typing import Callable

from llvmlite.binding import ExecutionEngine

from sleepy.syntactical_analysis.grammar import DummyPath
from sleepy.jit import compile_ir
from sleepy.ast_generation import make_translation_unit_ast_from_str


def compile_program(engine: ExecutionEngine,
                    program: str,
                    main_func_identifier: str = 'main',
                    add_preamble: bool = True,
                    emit_debug=True) -> Callable:
  file_path = DummyPath("test")
  ast = make_translation_unit_ast_from_str(file_path=file_path, program=program, add_preamble=add_preamble)
  module_ir, symbol_table, exported_funcs = ast.make_module_ir_and_symbol_table(
    module_name='test_parse_ast', emit_debug=emit_debug, implicitly_exported_functions={main_func_identifier})
  print('---- module intermediate repr:')
  print(module_ir)
  compile_ir(engine, module_ir)

  main_func = next((func for func in exported_funcs if func.indentifier == main_func_identifier), None)
  assert main_func is not None
  py_func = main_func.make_py_func(engine)

  assert callable(py_func)
  return py_func
