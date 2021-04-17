from typing import Dict, List

import _setup_test_env  # noqa
import better_exchook
import sys
import unittest

from llvmlite import ir
from nose.tools import assert_equal, assert_almost_equal, assert_raises

from sleepy.ast import TopLevelStatementAst, FunctionDeclarationAst, ReturnStatementAst, \
  BinaryOperatorExpressionAst, ConstantExpressionAst, VariableExpressionAst, SLEEPY_LEXER, SLEEPY_ATTR_GRAMMAR, SLEEPY_PARSER, \
  add_preamble_to_ast
from sleepy.grammar import TreePosition, SemanticError
from sleepy.jit import make_execution_engine, compile_ir
from sleepy.symbols import SLEEPY_DOUBLE, FunctionSymbol, SymbolTable, make_initial_symbol_table, \
  build_initial_module_ir


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
func do_stuff(Double val) -> Int {
  return 4;
}
do_stuff(7.5);
"""
  _test_parse_ast(program2)
  program3 = """
  # Compute 0 + 1 + ... + n
  func sum_all(Int n) -> Int {
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
  symbol_table = make_initial_symbol_table()  # type: SymbolTable
  ast.build_symbol_table(symbol_table=symbol_table)
  build_initial_module_ir(module=module, symbol_table=symbol_table)
  ast.build_expr_ir(module=module, builder=None, symbol_table=symbol_table)
  assert ast.identifier in symbol_table
  func_symbol = symbol_table[ast.identifier]
  assert isinstance(func_symbol, FunctionSymbol)
  compile_ir(engine, module)
  return func_symbol.get_single_concrete_func().make_py_func(engine)


def test_FunctionDeclarationAst_build_expr_ir():
  with make_execution_engine() as engine:
    pos = TreePosition('', 0, 0)
    ast1 = FunctionDeclarationAst(
      pos, identifier='foo', arg_identifiers=[], arg_type_identifiers=[], arg_annotations=[],
      return_type_identifier='Double', return_annotation_list=[],
      stmt_list=[ReturnStatementAst(pos, [ConstantExpressionAst(pos, 42.0, SLEEPY_DOUBLE)])])
    func1 = _get_py_func_from_ast(engine, ast1)
    assert_equal(func1(), 42.0)
  with make_execution_engine() as engine:
    ast2 = FunctionDeclarationAst(
      pos, identifier='foo', arg_identifiers=[], arg_type_identifiers=[], arg_annotations=[],
      return_type_identifier='Double', return_annotation_list=[], stmt_list=[ReturnStatementAst(pos, [
        BinaryOperatorExpressionAst(pos, '+', ConstantExpressionAst(pos, 3.0, SLEEPY_DOUBLE), ConstantExpressionAst(pos, 5.0, SLEEPY_DOUBLE))])])
    func2 = _get_py_func_from_ast(engine, ast2)
    assert_equal(func2(), 8.0)
  with make_execution_engine() as engine:
    ast3 = FunctionDeclarationAst(
      pos, identifier='sum', arg_identifiers=['a', 'b'], arg_type_identifiers=['Double', 'Double'],
      arg_annotations=[[], []], return_type_identifier='Double', return_annotation_list=[], stmt_list=[
        ReturnStatementAst(pos, [BinaryOperatorExpressionAst(pos, '+', VariableExpressionAst(pos, 'a'), VariableExpressionAst(pos, 'b'))])])
    func3 = _get_py_func_from_ast(engine, ast3)
    assert_equal(func3(7.0, 3.0), 10.0)


def _test_compile_program(engine, program, main_func_identifier='main', optimize=True):
  """
  :param ExecutionEngine engine:
  :param str program:
  :param str main_func_identifier:
  :param bool optimize:
  :rtype: Callable[[], float]
  """
  ast = _test_parse_ast(program)
  module_ir, symbol_table = ast.make_module_ir_and_symbol_table(module_name='test_parse_ast')
  print('---- symbol table:')
  print(symbol_table)
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
    func test() -> Int {
      return 2 * 4 - 3;
    }
    """
    func = _test_compile_program(engine, program, main_func_identifier='test')
    assert_equal(func(), 2 * 4 - 3)
  with make_execution_engine() as engine:
    program = """
    func sub(Double a, Double b) -> Double {
      return a - b;
    }
    """
    func = _test_compile_program(engine, program, main_func_identifier='sub')
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
    assert_equal(nothing(), None)


def test_lerp():
  with make_execution_engine() as engine:
    program = """
    func lerp(Double x1, Double x2, Double time) -> Double {
      diff = x2 - x1;
      return x1 + diff * time;
    }
    """
    lerp = _test_compile_program(engine, program, main_func_identifier='lerp')
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
    dist_squared = _test_compile_program(engine, program, main_func_identifier='dist_squared')
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
    ball_volume = _test_compile_program(engine, program, main_func_identifier='ball_volume')
    for radius in [0.0, 2.0, 3.0, 124.343]:
      assert_almost_equal(ball_volume(radius), 4.0 / 3.0 * 3.1415 * radius ** 3.0)


def test_simple_mutable_assign():
  with make_execution_engine() as engine:
    program = """
    func main(Int x) -> Int {
      x = x + 1;
      x = x + 1;
      return x;
    }
    """
    main = _test_compile_program(engine, program)
    assert_equal(main(3), 3 + 2)


def test_nested_func_call():
  import numpy, warnings
  with make_execution_engine() as engine:
    program = """
    func ball_volume(Double radius) -> Double {
      func cube(Double x) -> Double { return x * x * x; }
      return 4.0/3.0 * 3.1415 * cube(radius);
    }
    # Compute relative volume difference of two balls.
    func main(Double radius1, Double radius2) -> Double {
      volume1 = ball_volume(radius1);
      volume2 = ball_volume(radius2);
      return volume1 / volume2;
    }
    """
    main = _test_compile_program(engine, program)
    with warnings.catch_warnings():
      warnings.filterwarnings('ignore')
      for radius1 in [0.0, 2.0, 3.0, 124.343]:
        for radius2 in [0.0, 2.0, 3.0, 124.343]:
          volume1, volume2 = 4.0 / 3.0 * 3.1415 * radius1 ** 3.0, 4.0 / 3.0 * 3.1415 * radius2 ** 3.0
          numpy.testing.assert_almost_equal(main(radius1, radius2), numpy.divide(volume1, volume2))


def test_simple_if():
  with make_execution_engine() as engine:
    program = """
    func branch(Bool cond, Double true_val, Double false_val) -> Double {
      if cond {
        return true_val;
      } else {
        return false_val;
      }
    }
    """
    branch = _test_compile_program(engine, program, main_func_identifier='branch')
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
    max_ = _test_compile_program(engine, program, main_func_identifier='max')
    assert_equal(max_(13, 18), 18)
    assert_equal(max_(-3, 4.23), 4.23)
    assert_equal(max_(0, 0), 0)
    assert_equal(max_(-4, -9), -4)


def test_simple_if_abs():
  with make_execution_engine() as engine:
    program = """ func abs(Double x) -> Double { if x < 0.0 { return -x; } else { return x; } } """
    abs_ = _test_compile_program(engine, program, main_func_identifier='abs')
    assert_equal(abs_(3.1415), 3.1415)
    assert_equal(abs_(0.0), 0.0)
    assert_equal(abs_(-5.1), 5.1)


def test_if_assign():
  with make_execution_engine() as engine:
    program = """
    func main(Int mode, Double x, Double y) -> Double {
      res = 0.0;
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
    main = _test_compile_program(engine, program)
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
    fac = _test_compile_program(engine, program, main_func_identifier='fac')
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
    fib = _test_compile_program(engine, program, main_func_identifier='fibonacci')
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
    fib = _test_compile_program(engine, program, main_func_identifier='fibonacci')
    for n in list(range(1, 15)) + [20]:
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

    cos_ = _test_compile_program(engine, program)
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


def test_wrong_return_type_should_be_void():
  with make_execution_engine() as engine:
    program = """
    func main() {
      return 6;
    }
    """
    with assert_raises(SemanticError):
      _test_compile_program(engine, program)


def test_wrong_return_type_should_not_be_void():
  with make_execution_engine() as engine:
    program = """
    func main() -> Double {
      return;
    }
    """
    with assert_raises(SemanticError):
      _test_compile_program(engine, program)


def test_wrong_return_type_not_matching():
  with make_execution_engine() as engine:
    program = """
    func main() -> Double {
      return True();
    }
    """
    with assert_raises(SemanticError):
      _test_compile_program(engine, program)


def test_redefine_variable_with_different_type():
  with make_execution_engine() as engine:
    program = """
    func main() {
      a = 3.0;
      a = True();  # should fail.
    }
    """
    with assert_raises(SemanticError):
      _test_compile_program(engine, program)


def test_define_variable_with_wrong_type():
  with make_execution_engine() as engine:
    program = """
    func main() {
      Bool a = 3.0;  # should fail.
    }
    """
    with assert_raises(SemanticError):
      _test_compile_program(engine, program)


def test_redefine_variable_with_function():
  with make_execution_engine() as engine:
    program = """
    func main() -> Int {
      val = 5;
      func val() -> Int { return 4; }
      return val();
    }
    """
    with assert_raises(SemanticError):
      _test_compile_program(engine, program)


def test_shadow_func_name_with_var():
  with make_execution_engine() as engine:
    program = """
    func main() -> Int {
      main = 4;  # should work!
      return main;
    }
    """
    main = _test_compile_program(engine, program)
    assert_equal(main(), 4)


def test_shadow_var_name_with_var_of_different_type():
  with make_execution_engine() as engine:
    program = """
    func main() -> Int {
      Int val = 5;
      func inner() -> Bool {
        Bool val = True();
        return val;
      }
      if inner() {
        return val;
      } else {
        return 3;
      }
    }
    """
    main = _test_compile_program(engine, program)
    assert_equal(main(), 5)


def test_struct_default_constructor():
  with make_execution_engine() as engine:
    program = """
    struct Vec2 {
      Int x = 0;
      Int y = 0;
    }
    func main() -> Vec2 {
      return Vec2(0, 0);
    }
    """
    main = _test_compile_program(engine, program)
    assert_equal(type(main()).__name__, 'Vec2_CType')


def test_struct_member_access():
  with make_execution_engine() as engine:
    program = """
    struct Vec3 { Double x = 1.0; Double y = 2.0; Double z = 3.0; }
    func main() -> Double {
      Vec3 my_vec = Vec3(1.0, 2.0, 3.0);
      middle = my_vec.y;
      return middle;
    }
    """
    main = _test_compile_program(engine, program)
    assert_equal(main(), 2.0)


def test_struct_with_struct_member():
  with make_execution_engine() as engine:
    program = """
    struct Vec2 {
      Double x = 0.0;
      Double y = 0.0;
    }
    struct Mat22 {
      Vec2 first = Vec2(0.0, 0.0);
      Vec2 second = Vec2(0.0, 0.0);
    }
    func mat_sum(Mat22 mat) -> Double {
      func vec_sum(Vec2 vec) -> Double {
        return vec.x + vec.y;
      }
      return vec_sum(mat.first) + vec_sum(mat.second);
    }
    func main() -> Double {
      Mat22 mat = Mat22(Vec2(1.0, 2.0), Vec2(3.0, 4.0));
      assert(mat.first.x == 1.0);
      return mat_sum(mat);
    }
    """
    main = _test_compile_program(engine, program)
    assert_equal(main(), 1.0 + 2.0 + 3.0 + 4.0)


def test_if_missing_return_branch():
  with make_execution_engine() as engine:
    program = """
    func foo(Int x) -> Int {
      if x == 0 { return 42; }
      if x == -1 { return -1; }
      # missing return here.
    }
    """
    with assert_raises(SemanticError):
      _test_compile_program(engine, program)


def test_pass_by_reference():
  with make_execution_engine() as engine:
    program = """
    @RefType
    struct Foo {
      Int val = 0;
    }
    func inc_val(@Mutable Foo of) {
      of.val = of.val + 1;
    }
    func main() -> Int {
      @Mutable my_foo = Foo(0);
      my_foo.val = 4;
      inc_val(my_foo);  # now my_foo.val should be 5.
      return my_foo.val;
    }
    """
    main = _test_compile_program(engine, program)
    assert_equal(main(), 5)


def test_pass_by_value():
  with make_execution_engine() as engine:
    program = """
    @RefType struct Foo { Int val = 0; }
    func inc_val(Foo of) {
      of.val = of.val + 1;  # cannot redefine a immutable parameter!
    }
    func main() { }
    """
    with assert_raises(SemanticError):
      _test_compile_program(engine, program)


def test_annotation_fail_contradiction():
  with make_execution_engine() as engine:
    program = """
    @ValType @RefType struct BadStruct { }
    func main() { }
    """
    with assert_raises(SemanticError):
      _test_compile_program(engine, program)


def test_annotation_fail_duplicated():
  with make_execution_engine() as engine:
    program = """
    @ValType @ValType struct BadStruct { }
    func main() { }
    """
    with assert_raises(SemanticError):
      _test_compile_program(engine, program)


def test_overload_func():
  with make_execution_engine() as engine:
    program = """
    func to_int(Bool b) -> Int {
      if b { return 1; }
      else { return 0; }
    }
    func to_int(Int i) -> Int {
      return i;
    }
    func main(Bool a, Int b) -> Int {
      return to_int(a) + to_int(b) * 2;
    }
    """
    main = _test_compile_program(engine, program)
    assert_equal(main(True, 2), 1 + 2 * 2)
    assert_equal(main(False, -5), 0 - 5 * 2)


def test_const_add_vs_assign_add():
  with make_execution_engine() as engine:
    program = """
    @RefType struct Vec2 { Double x = 0.0; Double y = 0.0; }
    func add(Vec2 a, Vec2 b) -> Vec2 {
      return Vec2(a.x + b.x, a.y + b.y);
    }
    func assign_add(@Mutable Vec2 a, Vec2 b) {
      a.x = a.x + b.x;
      a.y = a.y + b.y;
    }
    func main() -> Double {
      v1 = Vec2(5.0, 7.0);
      @Mutable v2 = Vec2(0.0, 3.0);
      v3 = add(v1, v2);  # should be (5.0, 10.0)
      assign_add(v2, v3);  # v2 = (5.0, 13.0);
      v4 = add(v1, v2);  # should be (10.0, 20.0)
      return v4.x + v4.y;
    }
    """
    main = _test_compile_program(engine, program)
    assert_equal(main(), 10.0 + 20.0)


def test_call_mutable_with_const_var():
  with make_execution_engine() as engine:
    program = """
    @RefType struct Counter { Int val = 0; }
    func inc(@Mutable Counter c) { c.val = c.val + 1; }
    func inc_wrapper(Counter c) { inc(c); }
    func main() {
      c = Counter(0);
      inc_wrapper(c);
    }
    """
    with assert_raises(SemanticError):
      _test_compile_program(engine, program)


def test_assign_mutable_to_const():
  with make_execution_engine() as engine:
    program = """
    @RefType struct Counter { Int val = 0; }
    func main() -> Int {
      @Const con = Counter(-42);
      @Mutable mut = Counter(123);
      con = mut;
      return con.val;
    }
    """
    main = _test_compile_program(engine, program)
    assert_equal(main(), 123)


def test_assign_const_to_mutable():
  with make_execution_engine() as engine:
    program = """
    @RefType struct Counter { Int val = 0; }
    func main() -> Int {
      @Const con = Counter(-42);
      @Mutable mut = Counter(123);
      mut = con;  # not allowed.
      return con.val;
    }
    """
    _test_compile_program(engine, program)


def test_counter_is_empty():
  with make_execution_engine() as engine:
    program = """
    @RefType struct Counter { Int val = 0; }
    func increase(@Mutable Counter c) {
      c.val = c.val + 1;
    }
    func is_empty(Counter c) -> Bool {
      return c.val == 0;
    }
    func increase_if_empty(@Mutable Counter c) {
      if is_empty(c) {
        increase(c);
      }
    }
    func main() -> Int {
      @Mutable c = Counter(0);
      increase(c);
      increase_if_empty(c);  # should not do anything
      c.val = c.val - 1;
      increase_if_empty(c);  # should increase
      increase(c);
      return c.val;
    }
    """
    main = _test_compile_program(engine, program)
    assert_equal(main(), 2)


def test_return_mutable_var_as_mutable():
  with make_execution_engine() as engine:
    program = """
    @RefType struct Container { Int value = 0; }
    func identity(@Mutable Container c) -> @Mutable Container {
      return c;
    }
    func main() { }
    """
    _test_compile_program(engine, program)  # just check that it compiles.


def test_return_const_var_as_mutable():
  with make_execution_engine() as engine:
    program = """
    @RefType struct Container { Int value = 0; }
    func identity(Container c) -> @Mutable Container {
      return c;  # shouldn't be allowed, cannot make something immutable suddenly mutable!
    }
    func main() { }
    """
    with assert_raises(SemanticError):
      _test_compile_program(engine, program)


def test_return_mutable_var_as_const():
  with make_execution_engine() as engine:
    program = """
    @RefType struct Container { Int value = 0; }
    func identity(@Mutable Container c) -> Container {
      return c;  # giving up that something can be edited is fine though.
    }
    func main() { }
    """
    _test_compile_program(engine, program)  # just check that compiles


def test_if_inside_while():
  with make_execution_engine() as engine:
    program = """
    func main() -> Int {
      x = 5;
      while False() {
        if True() { x = 7; } else { x = 5; }
      }
      return x;
    }
    """
    main = _test_compile_program(engine, program)
    assert_equal(main(), 5)


def test_if_inside_while2():
  with make_execution_engine() as engine:
    program = """
    func main(Int x) -> Int {
      while x <= 5 {
        if x <= 2 { x = x + 1; }
        else { x = x + 2; }
      }
      return x;
    }
    """
    main = _test_compile_program(engine, program)
    assert_equal(main(0), 7)


def test_wrong_assign_void():
  with make_execution_engine() as engine:
    program = """
    func nothing() { }
    func main() {
      x = nothing();  # cannot assign void to x.
    }
    """
    with assert_raises(SemanticError):
      _test_compile_program(engine, program)


def test_wrong_return_void():
  with make_execution_engine() as engine:
    program = """
    func nothing() { }
    func main() {
      return nothing();  # cannot return void.
    }
    """
    with assert_raises(SemanticError):
      _test_compile_program(engine, program)


def test_mutable_struct_member_const():
  with make_execution_engine() as engine:
    program = """
    @RefType struct Box { Int val = 42; }
    @RefType struct SuperBox { @Mutable Box b = Box(42); }
    func main() {
      @Mutable Box b = Box(42);
      SuperBox sb = SuperBox(Box(42));
      sb.b = b;  # should fail, sb is immutable.
    }
    """
    with assert_raises(SemanticError):
      _test_compile_program(engine, program)


def test_mutable_struct_member_mutable():
  with make_execution_engine() as engine:
    program = """
    @RefType struct Box { Int val = 42; }
    @RefType struct SuperBox { @Mutable Box b = Box(42); }
    func main() -> Int {
      @Mutable Box b = Box(42);
      @Mutable SuperBox sb = SuperBox(Box(42));
      sb.b = b;  # should work now.
      # can now even change b, and sb should be effected too:
      b.val = 123;
      return sb.b.val;
    }
    """
    main = _test_compile_program(engine, program)
    assert_equal(main(), 123)


def test_immutable_struct_member_assign_mutable_member():
  with make_execution_engine() as engine:
    program = """
    @RefType struct Box { Int val = 42; }
    @RefType struct SuperBox { Box b = Box(42); }
    func main() -> Int {
      @Mutable Box b = Box(42);
      b.val = 27;
      SuperBox sb = SuperBox(Box(42));
      sb.b = b;  # should fail as sb is immutable.
    }
    """
    with assert_raises(SemanticError):
      _test_compile_program(engine, program)


def test_func_operator():
  with make_execution_engine() as engine:
    program = """
    # use + as logical or, * as logical and.
    func +(Bool a, Bool b) -> Bool {
      return or(a, b);
    }
    func *(Bool a, Bool b) -> Bool {
      return and(a, b);
    }
    func main(Bool a, Bool b, Bool c) -> Bool {
      return a + b * c;
    }
    """
    main = _test_compile_program(engine, program)
    assert_equal(main(False, False, True), False)
    assert_equal(main(True, False, True), True)
    assert_equal(main(False, True, True), True)
    assert_equal(main(True, True, True), True)
    assert_equal(main(False, False, False), False)
    assert_equal(main(True, False, False), True)
    assert_equal(main(False, True, False), False)
    assert_equal(main(True, True, False), True)


def test_overload_func_twice():
  with make_execution_engine() as engine:
    program = """
    func +(Bool a, Bool b) -> Bool {
      return or(a, b);
    }
    func +(Bool left, Bool right) -> Bool { return True(); }  # not allowed!
    func main() { }
    """
    with assert_raises(SemanticError):
      _test_compile_program(engine, program)


def test_overload_with_different_structs():
  with make_execution_engine() as engine:
    program = """
    struct Vec2 { Double x = 0.0; Double y = 0.0; }
    struct Vec3 { Double x = 0.0; Double y = 0.0; Double z = 0.0; }
    func +(Vec2 left, Vec2 right) -> Vec2 {
      return Vec2(left.x + right.x, left.y + right.y);
    }
    func +(Vec3 left, Vec3 right) -> Vec3 {
      return Vec3(left.x + right.x, left.y + right.y, left.z + right.z);
    }
    func main() {
      res1 = Vec2(0.0, 0.0) + Vec2(1.0, 3.0);
      res2 = Vec3(0.0, 0.0, 0.4) + Vec3(1.0, 3.0, -3.4);
    }
    """
    _test_compile_program(engine, program)


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
