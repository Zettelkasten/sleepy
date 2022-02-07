import os
from pathlib import Path
from typing import Tuple, List, Optional

from sleepy.jit import make_execution_engine, compile_ir
from sleepy.ast_generation import make_translation_unit_ast
from sleepy.types import OverloadSet


def run_example(code_file_name: Optional[str] = None):
  if code_file_name is None:
    pass

  print('\nLoading example from %s.' % code_file_name)
  with make_execution_engine() as engine:
    ast = make_translation_unit_ast(file_path=Path(code_file_name))
    module_ir, symbol_table, exported_functions = ast.make_module_ir_and_symbol_table(
      module_name='test_parse_ast', emit_debug=False,
      implicitly_exported_functions={'main'})
    compile_ir(engine, module_ir)

    concrete_main_func = next((func for func in exported_functions if func.identifier == 'main'), None)
    assert concrete_main_func is not None, 'Need to declare a main function'

    py_func = concrete_main_func.make_py_func(engine)

    print('Now executing:')
    return_val = py_func()
    print('Returned value: %r of type %r' % (return_val, concrete_main_func.return_type))


def find_all_example_files(dir_name: str) -> Tuple[str, List[str]]:
  """
  :returns: code_file_root and a list of code_file_names
  """
  examples_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), dir_name)
  code_file_root, _, code_file_names = next(os.walk(examples_path))
  code_file_names = [code_file_name for code_file_name in code_file_names if code_file_name.endswith('.slp')]
  assert len(code_file_names) >= 1
  return code_file_root, code_file_names


def make_example_cases(dir_name: str):
  code_file_root, code_file_names = find_all_example_files(dir_name)
  for code_file_name in code_file_names:
    yield run_example, os.path.join(code_file_root, code_file_name)
