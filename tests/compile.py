from typing import Callable

from llvmlite.binding import ExecutionEngine

from sleepy.jit import compile_ir
from sleepy.parse import make_ast
from sleepy.symbols import FunctionSymbol


def compile_program(engine: ExecutionEngine,
                    program: str,
                    main_func_identifier: str = 'main',
                    add_preamble: bool = True) -> Callable:

  ast = make_ast(program, add_preamble=add_preamble)
  module_ir, symbol_table = ast.make_module_ir_and_symbol_table(
    module_name='test_parse_ast', emit_debug=False)
  print('---- module intermediate repr:')
  print(module_ir)
  optimized_module_ir = compile_ir(engine, module_ir)
  print('---- optimized module intermediate repr:')
  print(optimized_module_ir)
  assert main_func_identifier in symbol_table
  main_func_symbol = symbol_table[main_func_identifier]
  assert isinstance(main_func_symbol, FunctionSymbol)
  py_func = main_func_symbol.get_single_concrete_func().make_py_func(engine)
  assert callable(py_func)
  return py_func
