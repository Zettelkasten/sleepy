import os
from pathlib import Path
from typing import Tuple, List, Optional

from sleepy.jit import make_execution_engine, compile_ir
from sleepy.ast_generation import make_translation_unit_ast
from sleepy.symbols import OverloadSet


def run_example(code_file_name: Optional[str] = None):
  if code_file_name is None:
    pass

  print('\nLoading example from %s.' % code_file_name)
  with make_execution_engine() as engine:
    ast = make_translation_unit_ast(file_path=Path(code_file_name))
    module_ir, symbol_table = ast.make_module_ir_and_symbol_table(module_name='test_parse_ast', emit_debug=False,
                                                                  implicitly_exported_functions={'main'})
    compile_ir(engine, module_ir)
    assert 'main' in symbol_table, 'Need to declare a main function'
    main_func_symbol = symbol_table['main']
    assert isinstance(main_func_symbol, OverloadSet), 'main needs to be a function'
    assert len(main_func_symbol.signatures) == 1, 'need to declare exactly one main function'
    concrete_main_func = main_func_symbol.get_single_concrete_func()
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
