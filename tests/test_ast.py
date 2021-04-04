import _setup_test_env  # noqa
from better_exchook import better_exchook
import sys
import unittest

from llvmlite import ir
from nose.tools import assert_equal, assert_almost_equal
from ctypes import CFUNCTYPE, c_double

from sleepy.ast import TopLevelStatementAst, FunctionDeclarationAst, ReturnStatementAst, \
  OperatorValueAst, ConstantValueAst, VariableValueAst, SLEEPY_LEXER, SLEEPY_ATTR_GRAMMAR, SLEEPY_PARSER, \
  add_preamble_to_ast
from sleepy.jit import make_execution_engine, compile_ir
from sleepy.symbols import SLEEPY_DOUBLE


def _test_parse_ast(program):
  """
  :param str program:
  :rtype: TopLevelStatementAst
  """
  print('---- input program:')
  print(program)
  tokens, tokens_pos = SLEEPY_LEXER.tokenize(program)
  print('---- tokens:')
  print(tokens)
  analysis, eval = SLEEPY_PARSER.parse_syn_attr_analysis(SLEEPY_ATTR_GRAMMAR, program, tokens, tokens_pos)
  ast = eval['ast']
  print('---- right-most analysis:')
  print(analysis)
  print('---- abstract syntax tree (without preamble):')
  print(ast)
  assert isinstance(ast, TopLevelStatementAst)
  ast = add_preamble_to_ast(ast)
  assert isinstance(ast, TopLevelStatementAst)
  return ast


def test_ast_parser():
  program1 = 'hello_world(123);'
  _test_parse_ast(program1)
  program2 = """# This function will just return 4.
func do_stuff(val) {
  return 4;
}
do_stuff(7.5);
"""
  _test_parse_ast(program2)
  program3 = """
  # Compute 0 + 1 + ... + n
  func sum_all(n) {
    if n <= 0 { return 0; }
    else { return sum_all(n-1) + n; }
  }
  
  sum_all(12);
  """
  _test_parse_ast(program3)


def _get_py_func_from_ast(engine, ast):
  """
  :param ExecutionEngine engine:
  :param FunctionDeclarationAst ast:
  :rtype: Callable
  """
  assert isinstance(ast, FunctionDeclarationAst)
  module = ir.Module(name='_test_last_declared_func')
  symbol_table = {}
  ast.build_expr_ir(module=module, builder=None, symbol_table=symbol_table)
  assert ast.identifier in symbol_table
  compile_ir(engine, module)
  func_ptr = engine.get_function_address(ast.identifier)
  return CFUNCTYPE(*((c_double,) + (c_double,) * len(ast.arg_identifiers)))(func_ptr)


def test_FunctionDeclarationAst_build_expr_ir():
  with make_execution_engine() as engine:
    ast1 = FunctionDeclarationAst(
      identifier='foo', arg_identifiers=[], arg_type_identifiers=[], return_type_identifier='Double',
      stmt_list=[ReturnStatementAst([ConstantValueAst(42.0)])])
    func1 = _get_py_func_from_ast(engine, ast1)
    assert_equal(func1(), 42.0)
  with make_execution_engine() as engine:
    ast2 = FunctionDeclarationAst(
      identifier='foo', arg_identifiers=[], arg_type_identifiers=[], return_type_identifier='Double', stmt_list=[
        ReturnStatementAst([OperatorValueAst('+', ConstantValueAst(3.0), ConstantValueAst(5.0))])])
    func2 = _get_py_func_from_ast(engine, ast2)
    assert_equal(func2(), 8.0)
  with make_execution_engine() as engine:
    ast3 = FunctionDeclarationAst(
      identifier='sum', arg_identifiers=['a', 'b'], arg_type_identifiers=['Double', 'Double'],
      return_type_identifier='Double', stmt_list=[
        ReturnStatementAst([OperatorValueAst('+', VariableValueAst('a'), VariableValueAst('b'))])])
    func3 = _get_py_func_from_ast(engine, ast3)
    assert_equal(func3(7.0, 3.0), 10.0)


def _test_compile_program(engine, program, main_func_identifier='main', main_func_num_args=0):
  """
  :param ExecutionEngine engine:
  :param str program:
  :param str main_func_identifier:
  :rtype: Callable[[], float]
  """
  ast = _test_parse_ast(program)
  module_ir = ast.make_module_ir(module_name='test_parse_ast')
  print('---- module intermediate repr:')
  print(module_ir)
  compile_ir(engine, module_ir)
  main_func_ptr = engine.get_function_address(main_func_identifier)
  py_func = CFUNCTYPE(*((c_double,) + (c_double,) * main_func_num_args))(main_func_ptr)
  assert callable(py_func)
  return py_func


def test_simple_arithmetic():
  with make_execution_engine() as engine:
    program = """
    func main() -> Double {
      return 4.0 + 3.0;
    }
    """
    func = _test_compile_program(engine, program)
    assert_equal(func(), 4.0 + 3.0)
  with make_execution_engine() as engine:
    program = """
    func test() -> Double {
      return 2 * 4 - 3;
    }
    """
    func = _test_compile_program(engine, program, main_func_identifier='test')
    assert_equal(func(), 2.0 * 4.0 - 3.0)
  with make_execution_engine() as engine:
    program = """
    func sub(Double a, Double b) -> Double {
      return a - b;
    }
    """
    func = _test_compile_program(engine, program, main_func_identifier='sub', main_func_num_args=2)
    assert_equal(func(0.0, 1.0), 0.0 - 1.0)
    assert_equal(func(3.0, 5.0), 3.0 - 5.0)
    assert_equal(func(2.5, 2.5), 2.5 - 2.5)


def test_empty_func():
  with make_execution_engine() as engine:
    program = """
    func nothing() {
    }
    """
    nothing = _test_compile_program(engine, program, main_func_identifier='nothing')
    assert_equal(nothing(), 0.0)


def test_lerp():
  with make_execution_engine() as engine:
    program = """
    func lerp(Double x1, Double x2, Double time) -> Double {
      diff = x2 - x1;
      return x1 + diff * time;
    }
    """
    lerp = _test_compile_program(engine, program, main_func_identifier='lerp', main_func_num_args=3)
    assert_equal(lerp(0.0, 1.0, 0.3), 0.3)
    assert_equal(lerp(7.5, 3.2, 0.0), 7.5)
    assert_equal(lerp(7.5, 3.2, 1.0), 3.2)


def test_call_other_func():
  with make_execution_engine() as engine:
    program = """
    func square(Double x) -> Double {
      return x * x;
    }
    func dist_squared(Double x1, Double x2, Double y1, Double y2) -> Double {
      return square(x1 - y1) + square(x2 - y2);
    }
    """
    dist_squared = _test_compile_program(engine, program, main_func_identifier='dist_squared', main_func_num_args=4)
    assert_almost_equal(dist_squared(0.0, 0.0, 1.0, 0.0), 1.0)
    assert_almost_equal(dist_squared(3.0, 0.0, 0.0, 4.0), 25.0)
    assert_almost_equal(dist_squared(1.0, 2.0, 3.0, 4.0), (1.0 - 3.0)**2 + (2.0 - 4.0)**2)


@unittest.skip('global variables not yet implemented')
def test_global_var():
  with make_execution_engine() as engine:
    program = """
    PI = 3.1415;  # declare a global variable
    func cube(Double x) -> Double { return x * x * x; }
    func ball_volume(Double radius) -> Double {
      return 4/3 * PI * cube(radius);
    }
    """
    ball_volume = _test_compile_program(engine, program, main_func_identifier='ball_volume', main_func_num_args=1)
    for radius in [0.0, 2.0, 3.0, 124.343]:
      assert_almost_equal(ball_volume(radius), 4.0 / 3.0 * 3.1415 * radius ** 3.0)


def test_simple_mutable_assign():
  with make_execution_engine() as engine:
    program = """
    func main(Double x) {
      x = x + 1;
      x = x + 1;
      return x;
    }
    """
    main = _test_compile_program(engine, program, main_func_num_args=1)
    assert_equal(main(3), 3 + 2)


def test_nested_func_call():
  import numpy
  with make_execution_engine() as engine:
    program = """
    func ball_volume(Double radius) -> Double {
      func cube(Double x) -> Double { return x * x * x; }
      return 4/3 * 3.1415 * cube(radius);
    }
    # Compute relative volume difference of two balls.
    func main(Double radius1, Double radius2) -> Double {
      volume1 = ball_volume(radius1);
      volume2 = ball_volume(radius2);
      return volume1 / volume2;
    }
    """
    main = _test_compile_program(engine, program, main_func_num_args=2)
    for radius1 in [0.0, 2.0, 3.0, 124.343]:
      for radius2 in [0.0, 2.0, 3.0, 124.343]:
        volume1, volume2 = 4.0 / 3.0 * 3.1415 * radius1 ** 3.0, 4.0 / 3.0 * 3.1415 * radius2 ** 3.0
        numpy.testing.assert_almost_equal(main(radius1, radius2), numpy.divide(volume1, volume2))


def test_simple_if():
  with make_execution_engine() as engine:
    program = """
    func branch(Double cond, Double true_val, Double false_val) -> Double {
      if cond {
        return true_val;
      } else {
        return false_val;
      }
    }
    """
    branch = _test_compile_program(engine, program, main_func_identifier='branch', main_func_num_args=3)
    assert_equal(branch(0, 42, -13), -13)
    assert_equal(branch(1, 42, -13), 42)


def test_simple_if_max():
  with make_execution_engine() as engine:
    program = """
    func max(Double a, Double b) -> Double {
      if a < b {
        return b;
      } else {
        return a;
      }
    }
    """
    max_ = _test_compile_program(engine, program, main_func_identifier='max', main_func_num_args=2)
    assert_equal(max_(13, 18), 18)
    assert_equal(max_(-3, 4.23), 4.23)
    assert_equal(max_(0, 0), 0)
    assert_equal(max_(-4, -9), -4)


def test_simple_if_abs():
  with make_execution_engine() as engine:
    program = """ func abs(Double x) -> Double { if x < 0 { return -x; } else { return x; } } """
    abs_ = _test_compile_program(engine, program, main_func_identifier='abs', main_func_num_args=1)
    assert_equal(abs_(3.1415), 3.1415)
    assert_equal(abs_(0.0), 0.0)
    assert_equal(abs_(-5.1), 5.1)


def test_if_assign():
  with make_execution_engine() as engine:
    program = """
    func main(Int mode, Double x, Double y) -> Double {
      res = 0;
      if mode == 0 {  # addition
        res = x + y;
      } if mode == 1 {  # subtraction
        res = x - y;
      } if mode == 2 {  # distance squared
        a = x - y;
        res = a * a;
      }
      return res;
    }
    """
    ast = _test_parse_ast(program)
    assert isinstance(ast, TopLevelStatementAst)
    assert_equal(ast.get_declared_var_types(symbol_table={}), {})
    main_ast = ast.stmt_list[-1]
    assert isinstance(main_ast, FunctionDeclarationAst)
    assert_equal(main_ast.get_declared_var_types(symbol_table={}), {})
    assert_equal(main_ast.get_body_var_types(symbol_table={}, body_symbol_table={}), {
      'mode': SLEEPY_DOUBLE, 'x': SLEEPY_DOUBLE, 'y': SLEEPY_DOUBLE, 'res': SLEEPY_DOUBLE, 'a': SLEEPY_DOUBLE})
    main = _test_compile_program(engine, program, main_func_num_args=3)
    assert_equal(main(0, 4, 6), 10)
    assert_equal(main(1, 5, -3), 8)
    assert_equal(main(2, 0, 1), 1)


def test_simple_simple_recursion_factorial():
  import math
  with make_execution_engine() as engine:
    program = """
    func fac(Int x) -> Int {
      if x <= 1 {
        return x;
      } else {
        return x * fac(x-1);
      }
    }
    """
    fac = _test_compile_program(engine, program, main_func_identifier='fac', main_func_num_args=1)
    assert_equal(fac(3), 3 * 2 * 1)
    assert_equal(fac(9), math.prod(range(1, 9 + 1)))
    assert_equal(fac(0), 0)


def _reference_fibonacci(n):
  prev_fib, curr_fib = 0, 1
  for i in range(1, n):
    prev_fib, curr_fib = curr_fib, prev_fib + curr_fib
  return curr_fib


def test_simple_simple_recursion_fibonacci():
  with make_execution_engine() as engine:
    program = """
    func fibonacci(Int n) -> Int {
      # crashes if n <= 0 or if n is not integer :)
      if or(n == 1, n == 2) {
        return 1;
      } else {
        return fibonacci(n - 2) + fibonacci(n - 1);
      }
    }
    """
    fib = _test_compile_program(engine, program, main_func_identifier='fibonacci', main_func_num_args=1)
    for n in range(1, 15):
      assert_equal(fib(n), _reference_fibonacci(n))


def test_simple_simple_iterative_fibonacci():
  with make_execution_engine() as engine:
    program = """
    func fibonacci(Int n) -> Int {
      prev_fib = 0;
      current_fib = 1;
      i = 2;
      while i <= n {
        i = i + 1;
        next_fib = prev_fib + current_fib;
        prev_fib = current_fib;
        current_fib = next_fib;
      }
      return current_fib;
    }
    """
    fib = _test_compile_program(engine, program, main_func_identifier='fibonacci', main_func_num_args=1)
    for n in list(range(1, 15)) + [50]:
      assert_equal(fib(n), _reference_fibonacci(n))


def test_extern_func():
  import math
  with make_execution_engine() as engine:
    program = """
    extern_func cos(Double x) -> Double;
    func main(Double x) -> Double {
      return cos(x);
    }
    """

    cos_ = _test_compile_program(engine, program, main_func_num_args=1)
    for x in [0, 1, 2, 3, math.pi]:
      assert_almost_equal(cos_(x), math.cos(x))


def test_extern_func_simple_alloc():
  with make_execution_engine() as engine:
    program = """
    func main() -> Double {
      arr = allocate(3);
      store(arr, 42.0);
      res = load(arr);
      return res;
    }
    """

    main = _test_compile_program(engine, program)
    assert_equal(main(), 42)


def test_typed_assignments():
  with make_execution_engine() as engine:
    program = """
    func main() {
      Double x = 2.3;
      x = 5.0;
      y = x * 2;
    }
    """
    ast = _test_parse_ast(program)
    assert isinstance(ast, TopLevelStatementAst)
    main_ast = ast.stmt_list[-1]  # ignore preamble.
    assert isinstance(main_ast, FunctionDeclarationAst)
    assert len(main_ast.stmt_list) == 3
    assert_equal(
      main_ast.get_body_var_types(symbol_table={}, body_symbol_table={}), {'x': SLEEPY_DOUBLE, 'y': SLEEPY_DOUBLE})
    assert_equal(main_ast.make_arg_types(symbol_table={}), [])
    assert_equal(main_ast.get_declared_var_types(symbol_table={}), {})


def test_types_simple():
  with make_execution_engine() as engine:
    program = """
    func main() {
      Int my_int = 3;
      Double my_double = 4.0;
    }
    """
    main = _test_compile_program(engine, program)
    main()

if __name__ == "__main__":
  try:
    better_exchook.install()
    if len(sys.argv) <= 1:
      for k, v in sorted(globals().items()):
        if k.startswith("test_"):
          print("-" * 40)
          print("Executing: %s" % k)
          try:
            v()
          except unittest.SkipTest as exc:
            print("SkipTest:", exc)
          print("-" * 40)
      print("Finished all tests.")
    else:
      assert len(sys.argv) >= 2
      for arg in sys.argv[1:]:
        print("Executing: %s" % arg)
        if arg in globals():
          globals()[arg]()  # assume function and execute
        else:
          eval(arg)  # assume Python code and execute
  finally:
    pass
