from typing import Dict, List

import _setup_test_env  # noqa
import better_exchook
import sys
import unittest

from llvmlite import ir
from nose.tools import assert_equal, assert_almost_equal, assert_raises

from sleepy.ast import TopLevelAst, FunctionDeclarationAst, SLEEPY_LEXER, SLEEPY_ATTR_GRAMMAR, \
  SLEEPY_PARSER, add_preamble_to_ast
from sleepy.grammar import SemanticError
from sleepy.jit import make_execution_engine, compile_ir
from sleepy.symbols import FunctionSymbol, SymbolTable


def _test_parse_ast(program, add_preamble=True):
  """
  :param str program:
  :param bool add_preamble:
  :rtype: TopLevelAst
  """
  print('---- input program:')
  print(program)
  tokens, tokens_pos = SLEEPY_LEXER.tokenize(program)
  analysis, eval = SLEEPY_PARSER.parse_syn_attr_analysis(SLEEPY_ATTR_GRAMMAR, program, tokens, tokens_pos)
  ast = eval['ast']
  assert isinstance(ast, TopLevelAst)
  if add_preamble:
    ast = add_preamble_to_ast(ast)
    assert isinstance(ast, TopLevelAst)
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


def _test_compile_program(engine, program, main_func_identifier='main', optimize=True, add_preamble=True):
  """
  :param ExecutionEngine engine:
  :param str program:
  :param str main_func_identifier:
  :param bool optimize:
  :param bool add_preamble:
  :rtype: Callable[[], float]
  """
  ast = _test_parse_ast(program, add_preamble=add_preamble)
  module_ir, symbol_table = ast.make_module_ir_and_symbol_table(module_name='test_parse_ast')
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
    lerp = _test_compile_program(engine, program, main_func_identifier='lerp', add_preamble=False)
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
    dist_squared = _test_compile_program(engine, program, main_func_identifier='dist_squared', add_preamble=False)
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
    ball_volume = _test_compile_program(engine, program, main_func_identifier='ball_volume', add_preamble=False)
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
    main = _test_compile_program(engine, program, add_preamble=False)
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
    main = _test_compile_program(engine, program, add_preamble=False)
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
    branch = _test_compile_program(engine, program, main_func_identifier='branch', add_preamble=False)
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
    max_ = _test_compile_program(engine, program, main_func_identifier='max', add_preamble=False)
    assert_equal(max_(13, 18), 18)
    assert_equal(max_(-3, 4.23), 4.23)
    assert_equal(max_(0, 0), 0)
    assert_equal(max_(-4, -9), -4)


def test_simple_if_abs():
  with make_execution_engine() as engine:
    program = """ func abs(Double x) -> Double { if x < 0.0 { return -x; } else { return x; } } """
    abs_ = _test_compile_program(engine, program, main_func_identifier='abs', add_preamble=False)
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
    main = _test_compile_program(engine, program, add_preamble=False)
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
    fac = _test_compile_program(engine, program, main_func_identifier='fac', add_preamble=False)
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
    fib = _test_compile_program(engine, program, main_func_identifier='fibonacci', add_preamble=True)
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
        i += 1;
        next_fib = prev_fib + current_fib;
        prev_fib = current_fib;
        current_fib = next_fib;
      }
      return current_fib;
    }
    """
    fib = _test_compile_program(engine, program, main_func_identifier='fibonacci', add_preamble=False)
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

    cos_ = _test_compile_program(engine, program, add_preamble=False)
    for x in [0, 1, 2, 3, math.pi]:
      assert_almost_equal(cos_(x), math.cos(x))


def test_extern_func_inside_inline_func():
  with make_execution_engine() as engine:
    program = """
    @Inline func foo_sqrt(Double x) -> Double {
      extern_func sqrt(Double x) -> Double;
      return sqrt(x);
    }
    func main(Double x) -> Double {
      return foo_sqrt(x);
    }
    """
    main = _test_compile_program(engine, program, add_preamble=False)
    from math import sqrt
    assert_almost_equal(main(12.0), sqrt(12.0))


def test_extern_func_simple_alloc():
  with make_execution_engine() as engine:
    program = """
    func main() -> Double {
      arr = allocate_double(3);
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


def test_operator_assign():
  with make_execution_engine() as engine:
    program = """
    func main(Int x) -> Int {
      x += 2;
      x *= -1;
      x += x;
      return x;  # 2 * -(x + 2)
    }
    """
    main = _test_compile_program(engine, program)
    assert_equal(main(5), 2 * -(5 + 2))


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


def test_mutable_val_type_local_var():
  with make_execution_engine() as engine:
    program = """
    struct Box { Int mem = 123; }
    func main() -> Int {
      @Mutable Box x = Box(42);
      x.mem = 17;
      return x.mem;
    }
    """
    main = _test_compile_program(engine, program, add_preamble=False)
    assert_equal(main(), 17)


def test_mutable_val_type_arg():
  with make_execution_engine() as engine:
    program = """
    struct Box { Int mem = 123; }
    func not_allowed(@Mutable Box b) { # <- cannot have mutable non-ref type as argument
    }
    func main() {
    }
    """
    with assert_raises(SemanticError):
      _test_compile_program(engine, program, add_preamble=False)


def test_mutable_val_type_return():
  with make_execution_engine() as engine:
    program = """
    struct Box { Int mem = 123; }
    func not_allowed() -> @Mutable Box { # <- cannot have mutable non-ref type as return type
      @Mutable Box cool = Box(123);
      return cool;
    }
    func main() {
    }
    """
    with assert_raises(SemanticError):
      _test_compile_program(engine, program, add_preamble=False)


def test_struct_free():
  with make_execution_engine() as engine:
    program = """
    @RefType struct Vec3 {
      Double x = 0.0;
      Double y = 0.0;
      Double z = 0.0;
    }
    func main(Double x, Double y, Double z) {
      left = 10000;
      while left > 0 {
        v = Vec3(x, y, z);
        free(v);
        left -= 1;
      }
    }
    """
    main = _test_compile_program(engine, program)
    main(0, 1, 2)


def test_struct_free_nested():
  with make_execution_engine() as engine:
    program = """
    @RefType struct Vec3 {
      Double x = 0.0;
      Double y = 0.0;
      Double z = 0.0;
    }
    @RefType struct Mat3x3 {
      @Mutable Vec3 x = Vec3(0.0, 0.0, 0.0);
      @Mutable Vec3 y = Vec3(0.0, 0.0, 0.0);
      @Mutable Vec3 z = Vec3(0.0, 0.0, 0.0);
    }
    func main() {
      @Mutable mat = Mat3x3(Vec3(0.0, 0.0, 0.0), Vec3(0.0, 0.0, 0.0), Vec3(0.0, 0.0, 0.0));
      mat.x.y = 42.0;
      free(mat);
    }
    """
    main = _test_compile_program(engine, program)
    main()


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
    func inc_wrapper(Counter c) { inc(c); }  # <-- cannot call a function taking sth mutable with const argument!
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
    struct Counter { Int val = 0; }
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
    struct Counter { Int val = 0; }
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
        if x <= 2 { x += 1; }
        else { x += 2; }
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
      free(b); free(sb);
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
      free(b); free(sb);
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


def test_index_operator():
  with make_execution_engine() as engine:
    program = """
    func get(DoublePtr ptr, Int pos) -> Double {
      return load(ptr + pos);
    }
    func set(DoublePtr ptr, Int pos, Double val) {
      store(ptr + pos, val);
    }
    func main(Double val) -> Double {
      ptr = allocate_double(8);
      ptr[0] = val;
      loaded = ptr[0];
      return loaded;
    }
    """
    main = _test_compile_program(engine, program)
    assert_equal(main(12.0), 12.0)


def test_func_inline():
  with make_execution_engine() as engine:
    program = """
    @Inline func important() -> Int {
      return 42;
    }
    func main() -> Int {
      Int what = important();
      return what + 5;
    }
    """
    main = _test_compile_program(engine, program, add_preamble=False)
    assert_equal(main(), 42 + 5)


def test_func_inline_with_branching():
  with make_execution_engine() as engine:
    program = """
    @Inline func ternary(Bool cond, Int if_true, Int if_false) -> Int {
      if cond {
        return if_true;
      } else {
        return if_false;
      }
    }
    # wrapper around ternary because we cannot inline main
    func main(Bool cond, Int if_true, Int if_false) -> Int {
      return ternary(cond, if_true, if_false);
    }
    """
    main = _test_compile_program(engine, program, add_preamble=False)
    assert_equal(main(True, 51, -43), 51)
    assert_equal(main(False, 51, -43), -43)


def test_func_inline_nested():
  with make_execution_engine() as engine:
    program = """
    @Inline func sum(Int a, Int b) -> Int { return a + b; }
    @Inline func ternary(Bool cond, Int if_true, Int if_false) -> Int {
      if cond { return if_true; } else { return if_false; }
    }
    func main(Bool cond, Int if_true, Int if_false) -> Int {
      return sum(ternary(cond, if_true, if_false), 42);
    }
    """
    main = _test_compile_program(engine, program)
    assert_equal(main(True, 51, -43), 51 + 42)
    assert_equal(main(False, 51, -43), -43 + 42)


def test_func_inline_sequence():
  with make_execution_engine() as engine:
    program = """
    @Inline func sum(Int a, Int b) -> Int { return a + b; }
    func main(Int a, Int b) -> Int {
      c = sum(a, b);
      d = sum(c, b);
      return d;
    }
    """
    main = _test_compile_program(engine, program, add_preamble=False)
    assert_equal(main(4, 6), 4 + 6 + 6)
    assert_equal(main(2, 4), 2 + 4 + 4)


def test_func_inline_void():
  with make_execution_engine() as engine:
    program = """
    @Inline func nothing() { }
    func main() {
      nothing();
      nothing();
      nothing();
    }
    """
    main = _test_compile_program(engine, program, add_preamble=False)
    main()


def test_func_inline_mutable_arg():
  with make_execution_engine() as engine:
    program = """
    @RefType struct Nothing {}
    @Inline func my_func(@Mutable Nothing wow) -> Int {
      return 32;
    }
    func main() -> Int { return my_func(Nothing()); }
    """
    main = _test_compile_program(engine, program)
    assert_equal(main(), 32)


def test_func_inline_own_symbol_table():
  with make_execution_engine() as engine:
    program = """
    # compute a + (a + 5)
    @Inline func foo(Int a) -> Int {
      Int b = a + 5;
      return a + b;
    }
    func main(Int x) -> Int {
      Int c = foo(x);
      Double b = 2.0;
      return c;
    }
    """
    main = _test_compile_program(engine, program)
    assert_equal(main(4), 4 + (4 + 5))


def test_func_inline_recursive():
  with make_execution_engine() as engine:
    program = """
    @Inline func foo(Int a) -> Int {
      return 2 * foo(a);
    }
    func main(Int x) {
      foo(x);
    }
    """
    with assert_raises(SemanticError):
      _test_compile_program(engine, program)


def test_func_inline_recursive_indirect():
  with make_execution_engine() as engine:
    program = """
    @Inline func foo(Int a) -> Int {
      @Inline func bar(Int b) -> Int {
        return foo(b) + 3;
      }
      return 2 * bar(a);
    }
    func main(Int x) {
      foo(x);
    }
    """
    with assert_raises(SemanticError):
      _test_compile_program(engine, program)


def test_if_scope():
  with make_execution_engine() as engine:
    program = """
    func main(Bool case) -> Bool {
      Bool result = False();
      if case {
        Bool bar = True();
        result = bar;
      } else {
        Double bar = 1.23;
        result = bar > 2.0;
      }
      return result;
    }
    """
    main = _test_compile_program(engine, program)
    assert_equal(main(True), True)
    assert_equal(main(False), False)


def test_if_scope_leak_var():
  with make_execution_engine() as engine:
    program = """
    func main(Bool case) -> Int {
      if case {
        Int fav_num = 123456;
      }
      return fav_num;  # should fail! if should not leak its local variables.
    }
    """
    with assert_raises(SemanticError):
      _test_compile_program(engine, program)


def test_scope_declare_variable_multiple_times():
  with make_execution_engine() as engine:
    program = """
    func main(Bool case) {
      if case {
        Int fav_num = 123456;
      } else {
        Double fav_num = 4.0;
      }
      Bool fav_num = True();
    }
    """
    main = _test_compile_program(engine, program)
    main(True)
    main(False)


def test_scope_capture_var():
  with make_execution_engine() as engine:
    program = """
    func main() -> Int {
      Int x = 100;
      func inner() {
          a = x;  # <- should raise error, variable captures not implemented yet
          print_line(a);
          x = 2;
          print_line(x);
      }
      inner();
      return x;
    }
    """
    with assert_raises(SemanticError):
      _test_compile_program(engine, program)


def test_union():
  with make_execution_engine() as engine:
    program = """
    struct MathError { }
    func safe_sqrt(Double x) -> Double|MathError {
      if x < 0.0 {
        return MathError();
      }
      extern_func sqrt(Double x) -> Double;
      return sqrt(x);
    }
    func main() {
      a = safe_sqrt(-123.0);
      b = safe_sqrt(1.0);
    }
    """
    main = _test_compile_program(engine, program)
    main()


def test_union_is_operator():
  with make_execution_engine() as engine:
    program = """
    struct Foo { Int lol = 1; }  # 1
    struct Bar { Int fav = 42; }  # 2
    func main_foo() -> Int {
      Bar|Foo value = Foo(0);
      if value is Foo { return 1; }
      if value is Bar { return 2; }
      return 0;
    }
    func main_bar() -> Int {
      Bar|Foo value = Bar(-123);
      if value is Foo { return 1; }
      if value is Bar { return 2; }
      return 0;
    }
    """
    main_foo = _test_compile_program(engine, program, main_func_identifier='main_foo')
    main_bar = _test_compile_program(engine, program, main_func_identifier='main_bar')
    assert_equal(main_foo(), 1)
    assert_equal(main_bar(), 2)


def test_union_is_operator_simple():
  with make_execution_engine() as engine:
    program = """
    func main() -> Bool {
      Int|Double test = 42;
      test = 1.234;
      return test is Double;  # should be True().
    }
    """
    main = _test_compile_program(engine, program, add_preamble=False, optimize=False)
    assert_equal(main(), True)


def test_union_is_operator_if_cond():
  with make_execution_engine() as engine:
    program = """
    func main() -> Bool {
      Int|Bool test = 42;
      if test is Bool {
        return True();
      } else {
        return False();
      }
    }
    """
    main = _test_compile_program(engine, program)
    assert_equal(main(), False)


def test_union_scope_assertions():
  with make_execution_engine() as engine:
    program = """
    func main(Double x, Double y) -> Double {
      struct MathError { }
      func safe_div(Double x, Double y) -> Double|MathError {
        eps = 0.0000001;
        if and(-eps <= y, y <= eps) {
          return MathError();
        }
        return x / y;
      }
      Double|MathError div = safe_div(x, y);
      if div is Double {
        return div;
      } else {
        return 0.0;
      }
    }
    """
    main = _test_compile_program(engine, program)
    assert_equal(main(5, 0), 0)
    assert_equal(main(5, 1), 5 / 1)
    assert_equal(main(-3, -6), -3 / -6)


def test_assign_to_union():
  with make_execution_engine() as engine:
    program = """
    func main() {
      Double|Int sth = 42;
    }
    """
    main = _test_compile_program(engine, program)
    main()


def test_assign_to_union2():
  with make_execution_engine() as engine:
    program = """
    struct MathError { }
    func safe_sqrt(Double x) -> Double|MathError {
      if x < 0.0 { return MathError(); }
      extern_func sqrt(Double x) -> Double;
      return sqrt(x);
    }
    func main() {
      a = safe_sqrt(-123.0);
      a = 5.0;
      Double|MathError a = MathError();  # can also explicitly specify type again
    }
    """
    main = _test_compile_program(engine, program)
    main()


def test_assign_union_to_single():
  with make_execution_engine() as engine:
    program = """
    func main() -> Int {
      Double|Int x = 3;  # at this point, the compiler asserts that x is an Int
      Int y = x;  # so this should work
      return y;
    }
    """
    main = _test_compile_program(engine, program, add_preamble=False)
    assert_equal(main(), 3)


def test_call_union_arg():
  with make_execution_engine() as engine:
    program = """
    func accepts_both(Int|Bool thing) -> Bool {
      if thing is Int { return thing >= 0; }
      if thing is Bool { return thing; }
      return False();  # never happens...
    }
    func main(Int a) -> Bool {
      Int|Bool thing = a;
      return accepts_both(thing);
    }
    """
    main = _test_compile_program(engine, program)
    assert_equal(main(4), True)
    assert_equal(main(-7), False)


def test_call_multiple_concrete_funcs_with_union_arg():
  with make_execution_engine() as engine:
    program = """
    func const() -> Bool|Int {
      # here declared separately such that the compiler does not know that it will actually always be an Int
      # in the future, we will probably add assertions so that the compiler does know that, but this will do for now.
      return 1 == 1;
    }
    func is_int(Int x) -> Bool { return True(); }
    func is_int(Bool x) -> Bool { return False(); }
    func main() -> Bool {
      Bool|Int alpha = const();
      return is_int(alpha);
    }
    """
    main = _test_compile_program(engine, program, optimize=False)
    assert_equal(main(), False)


def test_call_multiple_concrete_void_funcs_with_union_arg():
  with make_execution_engine() as engine:
    program = """
    func const() -> Bool|Int {
      # here declared separately such that the compiler does not know that it will actually always be an Int
      # in the future, we will probably add assertions so that the compiler does know that, but this will do for now.
      return 1 == 1;
    }
    func cool_func(Int x) { }
    func cool_func(Bool x) { }
    func main() -> Bool {
      Bool|Int alpha = const();
      cool_func(alpha);
      return True();
    }
    """
    main = _test_compile_program(engine, program)
    assert_equal(main(), True)


def test_union_folding():
  with make_execution_engine() as engine:
    program = """
    struct None { }
    func main() -> Double {
      None|Bool|Double a = 42.0;
      Double|(None|Bool) x = a;
      return x;
    }
    """
    main = _test_compile_program(engine, program)
    assert_equal(main(), 42.0)


def test_is_operator_incremental():
  with make_execution_engine() as engine:
    program = """
    struct None { }
    func make_val() -> None|Bool|Double { return 42.0; }
    func main() -> Bool {
      None|Bool|Double a = make_val();
      if a is None {
        if a is Bool {
          # this is dead code
          return False();
        }
      }
      return True();
    }
    """
    main = _test_compile_program(engine, program)
    assert_equal(main(), True)


def test_union_if_else_type_narrowing():
  with make_execution_engine() as engine:
    program = """
    func make_val() -> Int|Bool|Double { return 42.0; }
    func main() -> Double {
      val = make_val();
      if val is Int {
        val = val + 5;
        return 1.0;
      } else {
        # must be Bool or Double
        if val is Bool {
          val = not(val);
          return 2.0;
        } else {
          # must be Double
          extern_func sin(Double a) -> Double;
          val = sin(val);
          return val;
        }
      }
    }
    """
    main = _test_compile_program(engine, program)
    from math import sin
    assert_almost_equal(main(), sin(42.0))


def test_union_if_terminated_branch_type_narrowing():
  with make_execution_engine() as engine:
    program = """
    func make_val() -> Int|Double { return 42.0; }
    func main() -> Double {
      val = make_val();
      if val is Int {
        return 17.5;
      }
      # must be Double
      extern_func sin(Double a) -> Double;
      val = sin(val);
      return val;
    }
    """
    main = _test_compile_program(engine, program)
    from math import sin
    assert_almost_equal(main(), sin(42.0))


def test_while_cond_type_narrowing():
  with make_execution_engine() as engine:
    program = """
    func main(Int initial_value) -> Bool {
      Int|Bool value = initial_value;
      while value is Int {
        value = value - 1;
        if value < 0 { value = False(); }
        if value > 100 { value = True(); }
      }
      return value;
    }
    """
    main = _test_compile_program(engine, program)
    assert_equal(main(17), False)
    assert_equal(main(-2), False)
    assert_equal(main(117), True)


def test_assert_type_narrowing():
  with make_execution_engine() as engine:
    program = """
    func black_box() -> Int|Char {
      return 12;
    }
    func main() -> Int {
      Int|Char my_thing = black_box();
      assert(my_thing is Int);
      return my_thing;
    }
    """
    main = _test_compile_program(engine, program)
    assert_equal(main(), 12)


def test_unchecked_assert_type_narrowing():
  with make_execution_engine() as engine:
    program = """
    @RefType
    struct S {
      Double val = 0.0;
    }
    func cast_to_s(S|DoublePtr ptr) -> S {
      unchecked_assert(ptr is S);
      return ptr;
    }
    func main(Double val) -> Double {
      ptr = allocate_double(2);
      store(ptr, val);
      S s = cast_to_s(ptr);
      return s.val;
    }
    """
    main = _test_compile_program(engine, program)
    assert_equal(main(42.0), 42.0)
    assert_equal(main(0.123), 0.123)


def test_string_literal():
  with make_execution_engine() as engine:
    program = """
    func main(Int reps) -> Int {
      base = "test";
      out = "";
      while reps > 0 {
        out += base;
        reps -= 1;
      }
      return out.length;
    }
    """
    main = _test_compile_program(engine, program)
    assert_equal(main(10), 10 * 4)


def test_hex_int_literal():
  with make_execution_engine() as engine:
    program = """
    func main() -> Int {
      return 0x02A;
    }
    """
    main = _test_compile_program(engine, program)
    assert_equal(main(), 2 * 16 + 10)


def test_unreachable_code():
  with make_execution_engine() as engine:
    program = """
    func main() {
      return;
      i = 1;
    }
    """
    with assert_raises(SemanticError):
      _test_compile_program(engine, program)


def test_unreachable_code2():
  with make_execution_engine() as engine:
    program = """
    func main() {
      return;
      if 1 == 2 {
        x = 2;
      } else {
        return;
        x = 5;
      }
    }
    """
    with assert_raises(SemanticError):
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
