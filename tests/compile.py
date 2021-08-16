from sleepy.jit import compile_ir
from sleepy.symbols import FunctionSymbol
from tests.parse import parse_ast


def compile_program(engine, program, main_func_identifier='main', debug=True, optimize=True, add_preamble=True):
  """
  :param ExecutionEngine engine:
  :param str program:
  :param str main_func_identifier:
  :param bool debug:
  :param bool optimize:
  :param bool add_preamble:
  :rtype: Callable[[], float]
  """
  ast = parse_ast(program, add_preamble=add_preamble)
  module_ir, symbol_table = ast.make_module_ir_and_symbol_table(module_name='test_parse_ast', emit_debug=debug)
  print('---- module intermediate repr:')
  print(module_ir)
  optimized_module_ir = compile_ir(engine, module_ir, optimize=optimize)
  if optimize:
    print('---- optimized module intermediate repr:')
    print(optimized_module_ir)
  assert main_func_identifier in symbol_table
  main_func_symbol = symbol_table[main_func_identifier]
  assert isinstance(main_func_symbol, FunctionSymbol)
  py_func = main_func_symbol.get_single_concrete_func().make_py_func(engine)
  assert callable(py_func)
  return py_func