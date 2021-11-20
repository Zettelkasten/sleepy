import glob
import unittest

from nose.tools import assert_equal, assert_almost_equal, assert_raises

import _setup_test_env  # noqa
from sleepy.errors import SemanticError, ParseError
from sleepy.grammar import DummyPath
from sleepy.jit import make_execution_engine
from tests.compile import compile_program
from sleepy.parse import make_translation_unit_ast_from_str, make_file_ast_from_str


def test_ast_parser():
  dummy = DummyPath("dummy")
  program1 = 'hello_world(123)'
  make_translation_unit_ast_from_str(file_path=dummy, program=program1, add_preamble=False)
  program2 = """# This function will just return 4.
func do_stuff(val: Double) ->  Int  {
  return 4
}
do_stuff(7.5)
"""
  make_translation_unit_ast_from_str(file_path=dummy, program=program2, add_preamble=False)
  program3 = """
  # Compute 0 + 1 + ... + n
  func sum_all(n: Int) ->  Int  {
    if n <= 0 { return 0 }
    else { return sum_all(n-1) + n }
  }
  
  sum_all(12)
  """
  make_translation_unit_ast_from_str(file_path=dummy, program=program3, add_preamble=False)


def test_simple_arithmetic():
  with make_execution_engine() as engine:
    program = """
    func main() ->  Double  {
      return 4.0 + 3.0
    }
    """
    func = compile_program(engine, program, add_preamble=False)
    assert_equal(func(), 4.0 + 3.0)
  with make_execution_engine() as engine:
    program = """
    func test() ->  Int  {
      return 2 * 4 - 3
    }
    """
    func = compile_program(engine, program, main_func_identifier='test', add_preamble=False)
    assert_equal(func(), 2 * 4 - 3)
  with make_execution_engine() as engine:
    program = """
    func sub(a: Double, b: Double) ->  Double  {
      return a - b
    }
    """
    func = compile_program(engine, program, main_func_identifier='sub', add_preamble=False)
    assert_equal(func(0.0, 1.0), 0.0 - 1.0)
    assert_equal(func(3.0, 5.0), 3.0 - 5.0)
    assert_equal(func(2.5, 2.5), 2.5 - 2.5)
  with make_execution_engine() as engine:
    program = """
    func divide(a: Int, b: Int) ->  Int  {
      return a / b
    }
    """
    func = compile_program(engine, program, main_func_identifier='divide', add_preamble=False)
    assert_equal(func(1, 2), 0)
    assert_equal(func(14, 3), 4)
    assert_equal(func(25, 5), 5)
    assert_equal(func(-1, 2), 0)
    assert_equal(func(-14, 3), -4)
    assert_equal(func(-25, -5), 5)


def test_empty_func():
  with make_execution_engine() as engine:
    program = """
    func main()  {
    }
    """
    nothing = compile_program(engine, program, add_preamble=False)
    assert_equal(nothing(), None)


def test_empty_func_with_preamble():
  with make_execution_engine() as engine:
    program = """
    func main()  {
    }
    """
    nothing = compile_program(engine, program, add_preamble=True)
    assert_equal(nothing(), None)


def test_empty_func_with_arg():
  with make_execution_engine() as engine:
    program = """
    func main(x: Int)  {
    }
    """
    nothing = compile_program(engine, program, add_preamble=False)
    assert_equal(nothing(2), None)


def test_lerp():
  with make_execution_engine() as engine:
    program = """
    func lerp(x1: Double, x2: Double, time: Double) ->  Double  {
      diff = x2 - x1
      return x1 + diff * time
    }
    """
    lerp = compile_program(engine, program, main_func_identifier='lerp', add_preamble=False)
    assert_equal(lerp(0.0, 1.0, 0.3), 0.3)
    assert_equal(lerp(7.5, 3.2, 0.0), 7.5)
    assert_equal(lerp(7.5, 3.2, 1.0), 3.2)


def test_call_other_func():
  with make_execution_engine() as engine:
    program = """
    func square(x: Double) ->  Double  {
      return x * x
    }
    func dist_squared(x1: Double, x2: Double, y1: Double, y2: Double) ->  Double  {
      return square(x1 - y1) + square(x2 - y2)
    }
    """
    dist_squared = compile_program(engine, program, main_func_identifier='dist_squared', add_preamble=False)
    assert_almost_equal(dist_squared(0.0, 0.0, 1.0, 0.0), 1.0)
    assert_almost_equal(dist_squared(3.0, 0.0, 0.0, 4.0), 25.0)
    assert_almost_equal(dist_squared(1.0, 2.0, 3.0, 4.0), (1.0 - 3.0)**2 + (2.0 - 4.0)**2)


@unittest.skip('global variables not yet implemented')
def test_global_var():
  with make_execution_engine() as engine:
    program = """
    PI = 3.1415  # declare a global variable
    func cube(x: Double) ->  Double  { return x * x * x }
    func ball_volume(radius: Double) ->  Double  {
      return 4/3 * PI * cube(radius)
    }
    """
    ball_volume = compile_program(engine, program, main_func_identifier='ball_volume', add_preamble=False)
    for radius in [0.0, 2.0, 3.0, 124.343]:
      assert_almost_equal(ball_volume(radius), 4.0 / 3.0 * 3.1415 * radius ** 3.0)


def test_simple_mutable_assign():
  with make_execution_engine() as engine:
    program = """
    func main(x: Int) ->  Int  {
      x = x + 1
      x = x + 1
      return x
    }
    """
    main = compile_program(engine, program, add_preamble=False)
    assert_equal(main(3), 3 + 2)


def test_nested_func_call():
  import numpy
  import warnings
  with make_execution_engine() as engine:
    program = """
    func ball_volume(radius: Double) ->  Double  {
      func cube(x: Double) ->  Double  { return x * x * x }
      return 4.0/3.0 * 3.1415 * cube(radius)
    }
    # Compute relative volume difference of two balls.
    func main(radius1: Double, radius2: Double) ->  Double  {
      volume1 = ball_volume(radius1)
      volume2 = ball_volume(radius2)
      return volume1 / volume2
    }
    """
    main = compile_program(engine, program, add_preamble=False)
    with warnings.catch_warnings():
      warnings.filterwarnings('ignore')
      for radius1 in [0.0, 2.0, 3.0, 124.343]:
        for radius2 in [0.0, 2.0, 3.0, 124.343]:
          volume1, volume2 = 4.0 / 3.0 * 3.1415 * radius1 ** 3.0, 4.0 / 3.0 * 3.1415 * radius2 ** 3.0
          numpy.testing.assert_almost_equal(main(radius1, radius2), numpy.divide(volume1, volume2))


def test_simple_if():
  with make_execution_engine() as engine:
    program = """
    func branch(cond: Bool, true_val: Double, false_val: Double) ->  Double  {
      if cond {
        return true_val
      } else {
        return false_val
      }
    }
    """
    branch = compile_program(engine, program, main_func_identifier='branch', add_preamble=False)
    assert_equal(branch(0, 42, -13), -13)
    assert_equal(branch(1, 42, -13), 42)


def test_simple_if_max():
  with make_execution_engine() as engine:
    program = """
    func max(a: Double, b: Double) ->  Double  {
      if a < b {
        return b
      } else {
        return a
      }
    }
    """
    max_ = compile_program(engine, program, main_func_identifier='max', add_preamble=False)
    assert_equal(max_(13, 18), 18)
    assert_equal(max_(-3, 4.23), 4.23)
    assert_equal(max_(0, 0), 0)
    assert_equal(max_(-4, -9), -4)


def test_simple_if_abs():
  with make_execution_engine() as engine:
    program = """ func abs(x: Double) ->  Double  { if x < 0.0 { return -x } else { return x } } """
    abs_ = compile_program(engine, program, main_func_identifier='abs', add_preamble=False)
    assert_equal(abs_(3.1415), 3.1415)
    assert_equal(abs_(0.0), 0.0)
    assert_equal(abs_(-5.1), 5.1)


def test_if_assign():
  with make_execution_engine() as engine:
    program = """
    func main(mode: Int, x: Double, y: Double) ->  Double  {
      res = 0.0
      if mode == 0 {  # addition
        res = x + y
      } if mode == 1 {  # subtraction
        res = x - y
      } if mode == 2 {  # distance squared
        a = x - y
        res = a * a
      }
      return res
    }
    """
    main = compile_program(engine, program, add_preamble=False)
    assert_equal(main(0, 4, 6), 10)
    assert_equal(main(1, 5, -3), 8)
    assert_equal(main(2, 0, 1), 1)


def test_simple_simple_recursion_factorial():
  import math
  with make_execution_engine() as engine:
    program = """
    func fac(x: Int) ->  Int  {
      if x <= 1 {
        return x
      } else {
        return x * fac(x-1)
      }
    }
    """
    fac = compile_program(engine, program, main_func_identifier='fac', add_preamble=False)
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
    func fibonacci(n: Int) ->  Int  {
      # crashes if n <= 0 or if n is not integer :)
      if or(n == 1, n == 2) {
        return 1
      } else {
        return fibonacci(n - 2) + fibonacci(n - 1)
      }
    }
    """
    fib = compile_program(engine, program, main_func_identifier='fibonacci', add_preamble=True)
    for n in range(1, 15):
      assert_equal(fib(n), _reference_fibonacci(n))


def test_simple_simple_iterative_fibonacci():
  with make_execution_engine() as engine:
    program = """
    func fibonacci(n: Int) ->  Int  {
      prev_fib = 0
      current_fib = 1
      i = 2
      while i <= n {
        i += 1
        next_fib = prev_fib + current_fib
        prev_fib = current_fib
        current_fib = next_fib
      }
      return current_fib
    }
    """
    fib = compile_program(engine, program, main_func_identifier='fibonacci', add_preamble=False)
    for n in list(range(1, 15)) + [20]:
      assert_equal(fib(n), _reference_fibonacci(n))


def test_extern_func():
  import math
  with make_execution_engine() as engine:
    program = """
    extern_func cos(x: Double) ->  Double
    func main(x: Double) ->  Double  {
      return cos(x)
    }
    """

    cos_ = compile_program(engine, program, add_preamble=False)
    for x in [0, 1, 2, 3, math.pi]:
      assert_almost_equal(cos_(x), math.cos(x))


def test_extern_func_inside_inline_func():
  with make_execution_engine() as engine:
    program = """
    @Inline func foo_sqrt(x: Double) ->  Double  {
      extern_func sqrt(x: Double) ->  Double
      return sqrt(x)
    }
    func main(x: Double) ->  Double  {
      return foo_sqrt(x)
    }
    """
    main = compile_program(engine, program, add_preamble=False)
    from math import sqrt
    assert_almost_equal(main(12.0), sqrt(12.0))


def test_extern_func_simple_alloc():
  with make_execution_engine() as engine:
    program = """
    func main() ->  Double  {
      arr = allocate[Double](3)
      store(arr, 42.0)
      res = load(arr)
      return res
    }
    """

    main = compile_program(engine, program)
    assert_equal(main(), 42)


def test_types_simple():
  with make_execution_engine() as engine:
    program = """
    func main()  {
      my_int: Int = 3
      my_double: Double = 4.0
    }
    """
    main = compile_program(engine, program, add_preamble=False)
    main()


def test_wrong_return_type_should_be_void():
  with make_execution_engine() as engine:
    program = """
    func main()  {
      return 6
    }
    """
    with assert_raises(SemanticError):
      compile_program(engine, program, add_preamble=False)


def test_wrong_return_type_should_not_be_void():
  with make_execution_engine() as engine:
    program = """
    func main() ->  Double  {
      return
    }
    """
    with assert_raises(SemanticError):
      compile_program(engine, program, add_preamble=False)


def test_wrong_return_type_not_matching():
  with make_execution_engine() as engine:
    program = """
    func main() ->  Double  {
      return True()
    }
    """
    with assert_raises(SemanticError):
      compile_program(engine, program)


def test_redefine_variable_with_different_type():
  with make_execution_engine() as engine:
    program = """
    func main()  {
      a = 3.0
      a = True()  # should fail.
    }
    """
    with assert_raises(SemanticError):
      compile_program(engine, program)


def test_define_variable_with_wrong_type():
  with make_execution_engine() as engine:
    program = """
    func main()  {
      a: Bool = 3.0  # should fail.
    }
    """
    with assert_raises(SemanticError):
      compile_program(engine, program, add_preamble=False)


def test_redefine_variable_with_function():
  with make_execution_engine() as engine:
    program = """
    func main() ->  Int  {
      val = 5
      func val() ->  Int  { return 4 }
      return val()
    }
    """
    with assert_raises(SemanticError):
      compile_program(engine, program, add_preamble=False)


def test_shadow_func_name_with_var():
  with make_execution_engine() as engine:
    program = """
    func main() ->  Int  {
      main = 4  # should work!
      return main
    }
    """
    main = compile_program(engine, program, add_preamble=False)
    assert_equal(main(), 4)


def test_shadow_var_name_with_var_of_different_type():
  with make_execution_engine() as engine:
    program = """
    func main() ->  Int  {
      value: Int = 5
      func inner() ->  Bool  {
        value: Bool = True()
        return value
      }
      if inner() {
        return value
      } else {
        return 3
      }
    }
    """
    main = compile_program(engine, program)
    assert_equal(main(), 5)


def test_operator_assign():
  with make_execution_engine() as engine:
    program = """
    func main(x: Int) ->  Int  {
      x += 2
      x *= -1
      x += x
      return x  # 2 * -(x + 2)
    }
    """
    main = compile_program(engine, program, add_preamble=False)
    assert_equal(main(5), 2 * -(5 + 2))


def test_struct_default_constructor():
  with make_execution_engine() as engine:
    program = """
    struct Vec2 {
      x: Int = 0
      y: Int = 0
    }
    func main() ->  Vec2  {
      return Vec2(0, 0)
    }
    """
    main = compile_program(engine, program, add_preamble=False)
    res = main()
    assert_equal(res.x, 0)
    assert_equal(res.y, 0)


def test_struct_member_access():
  with make_execution_engine() as engine:
    program = """
    struct Vec3 { x: Double= 1.0; y: Double= 2.0; z: Double= 3.0; }
    func main() ->  Double  {
      my_vec: Vec3 = Vec3(1.0, 2.0, 3.0)
      middle = my_vec.y
      return middle
    }
    """
    main = compile_program(engine, program, add_preamble=False)
    assert_equal(main(), 2.0)


def test_struct_with_struct_member():
  with make_execution_engine() as engine:
    program = """
    struct Vec2 {
      x: Double = 0.0;
      y: Double = 0.0;
    }
    struct Mat22 {
      first: Vec2 = Vec2(0.0, 0.0);
      second: Vec2 = Vec2(0.0, 0.0);
    }
    func mat_sum(mat: Mat22) ->  Double  {
      func vec_sum(vec: Vec2) ->  Double  {
        return vec.x + vec.y
      }
      return vec_sum(mat.first) + vec_sum(mat.second)
    }
    func main() ->  Double  {
      mat: Mat22 = Mat22(Vec2(1.0, 2.0), Vec2(3.0, 4.0))
      assert(mat.first.x == 1.0)
      return mat_sum(mat)
    }
    """
    main = compile_program(engine, program)
    assert_equal(main(), 1.0 + 2.0 + 3.0 + 4.0)


def test_struct_member_not_assignable():
  with make_execution_engine() as engine:
    program = """
    struct Thing { val: Double; }
    func main() {
      Thing(2.0).val = 4.0  # should not work, struct is passed by value and thus not assignable
    }
    """
    with assert_raises(SemanticError):
      compile_program(engine, program, add_preamble=False)


def test_struct_member_load_assign():
  with make_execution_engine() as engine:
    program = """
    struct Thing { val: Double; }
    func main(value: Double) -> Double {  # should again return value
      ptr = allocate[Thing](1)
      store(ptr, Thing(0.0))
      load(ptr).val = value  # this should now change the memory where ptr points to.
      res = load(ptr).val
      deallocate(ptr)
      return res
    }
    """
    main = compile_program(engine, program, add_preamble=True)
    assert_equal(main(42.0), 42.0)


def test_struct_self_referencing_member():
  with make_execution_engine() as engine:
    program = """
    struct None { }
    struct LinkedList {
      value: Int;
      next: Ref[LinkedList]|None;
    }
    func main() {
      b = LinkedList(42, None());
      a = LinkedList(41, !b);
    }
    """
    main = compile_program(engine, program, add_preamble=False)
    main()


def test_if_missing_return_branch():
  with make_execution_engine() as engine:
    program = """
    func foo(x: Int) ->  Int  {
      if x == 0 { return 42 }
      if x == -1 { return -1 }
      # missing return here.
    }
    """
    with assert_raises(SemanticError):
      compile_program(engine, program)


def test_mutable_val_type_local_var():
  with make_execution_engine() as engine:
    program = """
    struct Box { mem: Int = 123; }
    func main() ->  Int  {
      x: Box = Box(42)
      x.mem = 17
      return x.mem
    }
    """
    main = compile_program(engine, program, add_preamble=False)
    assert_equal(main(), 17)


def test_unbind_operator():
  with make_execution_engine() as engine:
    program = """
    func main(x: Int) -> Int {
      y: Int = x
      !ref_y: Ref[Int] = !y
      return y
    }
    """
    main = compile_program(engine, program, add_preamble=False)
    assert_equal(main(6), 6)


def test_unbind_operator_func_call():
  with make_execution_engine() as engine:
    program = """
    func foo(x: Ref[Int]) -> Int {
      return x  # binds.
    }
    func main(x: Int) -> Int {
      y: Int = x
      return foo(!y)
    }
    """
    main = compile_program(engine, program, add_preamble=False)
    assert_equal(main(6), 6)


def test_unbind_operator_assign_variable():
  with make_execution_engine() as engine:
    program = """
    func main(x: Int) -> Int {
      !y: Ref[Int] = !x
      y = x + 1  # changes the memory y points to
      !y = !x  # changes the pointer itself (in this case a no-op)
      return y  # collapses automatically
    }
    """
    main = compile_program(engine, program, add_preamble=False)
    assert_equal(main(6), 6 + 1)


def test_unbind_operator_assign_member():
  with make_execution_engine() as engine:
    program = """
    struct Foo { amount: Int; }
    func main(x: Int) -> Int {
      my_foo = Foo(x)
      !ptr = !my_foo
      ptr.amount = ptr.amount + 1
      return my_foo.amount
    }
    """
    main = compile_program(engine, program, add_preamble=False)
    assert_equal(main(6), 6 + 1)


def test_unbind_operator_union_of_ref():
  with make_execution_engine() as engine:
    program = """
    func main(x: Int) -> Int {
      !x_ref: Ref[Int] = !x
      !any_ref: Ref[Int]|Ref[Double] = !x_ref
      any_collapsed: Int|Double = any_ref
      return any_collapsed
    }
    """
    main = compile_program(engine, program, add_preamble=False)
    assert_equal(main(6), 6)


def test_unbind_operator_union_of_ref_and_non_ref():
  with make_execution_engine() as engine:
    program = """
    func main(x: Int) -> Int {
      !foo: Ref[Int]|Double = !x
      return foo
    }
    """
    main = compile_program(engine, program, add_preamble=False)
    assert_equal(main(6), 6)


def test_unbind_operator_union_of_ref_and_non_ref_same():
  with make_execution_engine() as engine:
    program = """
    func main(x: Int) -> Int {
      !foo: Ref[Int]|Int = !x
      return foo
    }
    """
    main = compile_program(engine, program, add_preamble=False)
    assert_equal(main(6), 6)


def test_unbind_operator_union_of_ref_and_non_ref_assign():
  with make_execution_engine() as engine:
    program = """
    func main(x: Int) -> Int {
      foo: Ref[Int]|Double = 32.0  # some dummy
      !foo = !x
      return foo
    }
    """
    main = compile_program(engine, program, add_preamble=False)
    assert_equal(main(6), 6)


def test_swap_ref():
  with make_execution_engine() as engine:
    program = """
    func swap[T](a: Ref[T], b: Ref[T]) {
      tmp: T = a
      a = b
      b = tmp
    }
    func main(x: Int, y: Int) -> Int {
      swap(!x, !y)
      return x - y
    }
    """
    main = compile_program(engine, program, add_preamble=False)
    assert_equal(main(3, 4), 4 - 3)


def test_new_and_delete():
  with make_execution_engine() as engine:
    program = """
    func main(initial: Int) -> Int {
      !x: Ref[Int] = !new(initial)
      x += 1
      res: Int = x
      delete(!x)
      return res
    }
    """
    main = compile_program(engine, program, add_preamble=True)
    assert_equal(main(6), 6 + 1)


def test_struct_free():
  with make_execution_engine() as engine:
    program = """
    struct Vec3 {
      x: Double = 0.0
      y: Double = 0.0
      z: Double = 0.0
    }
    func main(x: Double, y: Double, z: Double)  {
      left = 10000
      while left > 0 {
        !v = !new(Vec3(x, y, z))
        delete(!v)
        left -= 1
      }
    }
    """
    main = compile_program(engine, program)
    main(0, 1, 2)


@unittest.skip('Destructors are not implemented currently')
def test_struct_free_nested():
  with make_execution_engine() as engine:
    program = """
    struct Vec3 {
      x: Double = 0.0
      y: Double = 0.0
      z: Double = 0.0
    }
    struct Mat3x3 {
      x: Vec3 = Vec3(0.0, 0.0, 0.0)
      y: Vec3 = Vec3(0.0, 0.0, 0.0)
      z: Vec3 = Vec3(0.0, 0.0, 0.0)
    }
    func main()  {
      mat = Mat3x3(Vec3(0.0, 0.0, 0.0), Vec3(0.0, 0.0, 0.0), Vec3(0.0, 0.0, 0.0))
      mat.x.y = 42.0
      free(mat)
    }
    """
    main = compile_program(engine, program)
    main()


def test_annotation_fail_duplicated():
  with make_execution_engine() as engine:
    program = """
    @Inline @Inline func weirdo() { }
    func main() { }
    """
    with assert_raises(SemanticError):
      compile_program(engine, program)


def test_overload_func():
  with make_execution_engine() as engine:
    program = """
    func to_int(b: Bool) ->  Int  {
      if b { return 1 }
      else { return 0 }
    }
    func to_int(i: Int) ->  Int  {
      return i
    }
    func main(a: Bool, b: Int) ->  Int  {
      return to_int(a) + to_int(b) * 2
    }
    """
    main = compile_program(engine, program)
    assert_equal(main(True, 2), 1 + 2 * 2)
    assert_equal(main(False, -5), 0 - 5 * 2)


def test_const_add_vs_assign_add():
  with make_execution_engine() as engine:
    program = """
    struct Vec2 { x: Double = 0.0; y: Double = 0.0; }
    func add(a: Vec2, b: Vec2) ->  Vec2  {
      return Vec2(a.x + b.x, a.y + b.y)
    }
    func assign_add(mutates a: Vec2, b: Vec2)  {
      a.x = a.x + b.x
      a.y = a.y + b.y
    }
    func main() ->  Double  {
      v1 = Vec2(5.0, 7.0)
      v2 = Vec2(0.0, 3.0)
      v3 = add(v1, v2)  # should be (5.0, 10.0)
      assign_add(v2, v3)  # v2 = (5.0, 13.0)
      v4 = add(v1, v2)  # should be (10.0, 20.0)
      return v4.x + v4.y
    }
    """
    main = compile_program(engine, program)
    assert_equal(main(), 10.0 + 20.0)


def test_counter_is_empty():
  with make_execution_engine() as engine:
    program = """
    struct Counter { value: Int = 0; }
    func increase(mutates c: Counter)  {
      c.value = c.value + 1
    }
    func is_empty(c: Counter) ->  Bool  {
      return c.value == 0
    }
    func increase_if_empty(mutates c: Counter)  {
      if is_empty(c) {
        increase(c)
      }
    }
    func main() ->  Int  {
      c = Counter(0)
      increase(c)
      increase_if_empty(c)  # should not do anything
      c.value = c.value - 1
      increase_if_empty(c)  # should increase
      increase(c)
      return c.value
    }
    """
    main = compile_program(engine, program)
    assert_equal(main(), 2)


def test_if_inside_while():
  with make_execution_engine() as engine:
    program = """
    func main() ->  Int  {
      x = 5
      while False() {
        if True() { x = 7 } else { x = 5 }
      }
      return x
    }
    """
    main = compile_program(engine, program)
    assert_equal(main(), 5)


def test_if_inside_while2():
  with make_execution_engine() as engine:
    program = """
    func main(x: Int) ->  Int  {
      while x <= 5 {
        if x <= 2 { x += 1 }
        else { x += 2 }
      }
      return x
    }
    """
    main = compile_program(engine, program)
    assert_equal(main(0), 7)


def test_wrong_assign_void():
  with make_execution_engine() as engine:
    program = """
    func nothing()  { }
    func main()  {
      x = nothing()  # cannot assign void to x.
    }
    """
    with assert_raises(SemanticError):
      compile_program(engine, program)


def test_wrong_return_void():
  with make_execution_engine() as engine:
    program = """
    func nothing()  { }
    func main()  {
      return nothing()  # cannot return void.
    }
    """
    with assert_raises(SemanticError):
      compile_program(engine, program)


def test_func_operator():
  with make_execution_engine() as engine:
    program = """
    # use + as logical or, * as logical and.
    func +(a: Bool, b: Bool) ->  Bool  {
      return or(a, b)
    }
    func *(a: Bool, b: Bool) ->  Bool  {
      return and(a, b)
    }
    func main(a: Bool, b: Bool, c: Bool) ->  Bool  {
      return a + b * c
    }
    """
    main = compile_program(engine, program)
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
    func +(a: Bool, b: Bool) ->  Bool  {
      return or(a, b)
    }
    func +(left: Bool, right: Bool) ->  Bool  { return True() }  # not allowed!
    func main()  { }
    """
    with assert_raises(SemanticError):
      compile_program(engine, program)


def test_overload_with_different_structs():
  with make_execution_engine() as engine:
    program = """
    struct Vec2 { x: Double = 0.0; y: Double = 0.0; }
    struct Vec3 { x: Double = 0.0; y: Double = 0.0; z: Double = 0.0; }
    func +(left: Vec2, right: Vec2) ->  Vec2  {
      return Vec2(left.x + right.x, left.y + right.y)
    }
    func +(left: Vec3, right: Vec3) ->  Vec3  {
      return Vec3(left.x + right.x, left.y + right.y, left.z + right.z)
    }
    func main()  {
      res1 = Vec2(0.0, 0.0) + Vec2(1.0, 3.0)
      res2 = Vec3(0.0, 0.0, 0.4) + Vec3(1.0, 3.0, -3.4)
    }
    """
    compile_program(engine, program)


def test_index_operator():
  with make_execution_engine() as engine:
    program = """
    func index(ptr: Ptr[Double], pos: Int) -> Ref[Double] {
      return !load(ptr + pos)
    }
    func main(val: Double) ->  Double  {
      ptr = allocate[Double](8)
      ptr[0] = val
      loaded = ptr[0]
      return loaded
    }
    """
    main = compile_program(engine, program)
    assert_equal(main(12.0), 12.0)


def test_index_operator_syntax():
  with make_execution_engine() as engine:
    program = """
    func (ptr: Ptr[Double])[pos: Int] -> Ref[Double] {
      return !load(ptr + pos)
    }
    func main(val: Double) ->  Double  {
      ptr = allocate[Double](8)
      ptr[0] = val
      loaded = ptr[0]
      return loaded
    }
    """
    main = compile_program(engine, program)
    assert_equal(main(12.0), 12.0)


def test_slice_syntax():
  with make_execution_engine() as engine:
    program = """
    import "slice.slp"
    struct Interval { max: Long; }
    func index(i: Interval, slice: Slice) -> Long {
      from = slice.from; to = slice.to
      if from is Unbounded { from = 0l }
      if to is Unbounded { to = i.max }
      return to - from + 1l
    }
    func main() -> Long {
      s = Interval(10l)
      a = s[5:6]  # == 2
      b = s[8:]  # = 3
      c = s[:3]  # = 4
      d = s[:]  # = 11
      return a + b + c + d
    }
    """
    main = compile_program(engine, program, add_preamble=True)
    assert_equal(main(), 2 + 3 + 4 + 11)


def test_func_inline():
  with make_execution_engine() as engine:
    program = """
    @Inline func important() ->  Int  {
      return 42
    }
    func main() ->  Int  {
      what: Int = important()
      return what + 5
    }
    """
    main = compile_program(engine, program, add_preamble=False)
    assert_equal(main(), 42 + 5)


def test_func_inline_with_branching():
  with make_execution_engine() as engine:
    program = """
    @Inline func ternary(cond: Bool, if_true: Int, if_false: Int) ->  Int  {
      if cond {
        return if_true
      } else {
        return if_false
      }
    }
    # wrapper around ternary because we cannot inline main
    func main(cond: Bool, if_true: Int, if_false: Int) ->  Int  {
      return ternary(cond, if_true, if_false)
    }
    """
    main = compile_program(engine, program, add_preamble=False)
    assert_equal(main(True, 51, -43), 51)
    assert_equal(main(False, 51, -43), -43)


def test_func_inline_nested():
  with make_execution_engine() as engine:
    program = """
    @Inline func sum(a: Int, b: Int) ->  Int  { return a + b }
    @Inline func ternary(cond: Bool, if_true: Int, if_false: Int) ->  Int  {
      if cond { return if_true } else { return if_false }
    }
    func main(cond: Bool, if_true: Int, if_false: Int) ->  Int  {
      return sum(ternary(cond, if_true, if_false), 42)
    }
    """
    main = compile_program(engine, program)
    assert_equal(main(True, 51, -43), 51 + 42)
    assert_equal(main(False, 51, -43), -43 + 42)


def test_func_inline_sequence():
  with make_execution_engine() as engine:
    program = """
    @Inline func sum(a: Int, b: Int) ->  Int  { return a + b }
    func main(a: Int, b: Int) ->  Int  {
      c = sum(a, b)
      d = sum(c, b)
      return d
    }
    """
    main = compile_program(engine, program, add_preamble=False)
    assert_equal(main(4, 6), 4 + 6 + 6)
    assert_equal(main(2, 4), 2 + 4 + 4)


def test_func_inline_void():
  with make_execution_engine() as engine:
    program = """
    @Inline func nothing()  { }
    func main()  {
      nothing()
      nothing()
      nothing()
    }
    """
    main = compile_program(engine, program, add_preamble=False)
    main()


def test_func_inline_own_symbol_table():
  with make_execution_engine() as engine:
    program = """
    # compute a + (a + 5)
    @Inline func foo(a: Int) ->  Int  {
      b: Int = a + 5
      return a + b
    }
    func main(x: Int) ->  Int  {
      c: Int = foo(x)
      b: Double = 2.0
      return c
    }
    """
    main = compile_program(engine, program)
    assert_equal(main(4), 4 + (4 + 5))


def test_func_inline_recursive():
  with make_execution_engine() as engine:
    program = """
    @Inline func foo(a: Int) ->  Int  {
      return 2 * foo(a)
    }
    func main(x: Int)  {
      foo(x)
    }
    """
    with assert_raises(SemanticError):
      compile_program(engine, program)


def test_func_inline_recursive_indirect():
  with make_execution_engine() as engine:
    program = """
    @Inline func foo(a: Int) ->  Int  {
      @Inline func bar(b: Int) ->  Int  {
        return foo(b) + 3
      }
      return 2 * bar(a)
    }
    func main(x: Int)  {
      foo(x)
    }
    """
    with assert_raises(SemanticError):
      compile_program(engine, program)


def test_if_scope():
  with make_execution_engine() as engine:
    program = """
    func main(case: Bool) ->  Bool  {
      result: Bool = False()
      if case {
        bar: Bool = True()
        result = bar
      } else {
        bar: Double = 1.23
        result = bar > 2.0
      }
      return result
    }
    """
    main = compile_program(engine, program)
    assert_equal(main(True), True)
    assert_equal(main(False), False)


def test_if_scope_leak_var():
  with make_execution_engine() as engine:
    program = """
    func main(case: Bool) ->  Int  {
      if case {
        fav_num: Int = 123456
      }
      return fav_num  # should fail! if should not leak its local variables.
    }
    """
    with assert_raises(SemanticError):
      compile_program(engine, program)


def test_scope_declare_variable_multiple_times():
  with make_execution_engine() as engine:
    program = """
    func main(case: Bool)  {
      if case {
        fav_num: Int = 123456
      } else {
        fav_num: Double = 4.0
      }
      fav_num: Bool = True()
    }
    """
    main = compile_program(engine, program)
    main(True)
    main(False)


def test_scope_capture_var():
  with make_execution_engine() as engine:
    program = """
    func main() ->  Int  {
      x: Int = 100
      func inner()  {
          a = x  # <- should raise error, variable captures not implemented yet
          print_line(a)
          x = 2
          print_line(x)
      }
      inner()
      return x
    }
    """
    with assert_raises(SemanticError):
      compile_program(engine, program)


def test_union():
  with make_execution_engine() as engine:
    program = """
    struct MathError { }
    func safe_sqrt(x: Double) ->  Double|MathError  {
      if x < 0.0 {
        return MathError()
      }
      extern_func sqrt(x: Double) ->  Double
      return sqrt(x)
    }
    func main()  {
      a = safe_sqrt(-123.0)
      b = safe_sqrt(1.0)
    }
    """
    main = compile_program(engine, program)
    main()


def test_union_is_operator():
  with make_execution_engine() as engine:
    program = """
    struct Foo { lol: Int = 1; }  # 1
    struct Bar { fav: Int = 42; }  # 2
    func main_foo() ->  Int  {
      value: Foo|Bar = Foo(0)
      if value is Foo { return 1 }
      if value is Bar { return 2 }
      return 0
    }
    func main_bar() ->  Int  {
      value: Bar|Foo  = Bar(-123)
      if value is Foo { return 1 }
      if value is Bar { return 2 }
      return 0
    }
    """
    main_foo = compile_program(engine, program, main_func_identifier='main_foo')
    main_bar = compile_program(engine, program, main_func_identifier='main_bar')
    assert_equal(main_foo(), 1)
    assert_equal(main_bar(), 2)


def test_union_is_operator_simple():
  with make_execution_engine() as engine:
    program = """
    func main() ->  Bool  {
      test: Int|Double = 42
      test = 1.234
      return test is Double  # should be True().
    }
    """
    main = compile_program(engine, program, add_preamble=False)
    assert_equal(main(), True)


def test_union_is_operator_if_cond():
  with make_execution_engine() as engine:
    program = """
    func main() ->  Bool  {
      test: Int|Bool = 42
      if test is Bool {
        return True()
      } else {
        return False()
      }
    }
    """
    main = compile_program(engine, program)
    assert_equal(main(), False)


def test_union_scope_assertions():
  with make_execution_engine() as engine:
    program = """
    func main(x: Double, y: Double) ->  Double  {
      struct MathError { }
      func safe_div(x: Double, y: Double) ->  Double|MathError  {
        eps = 0.0000001
        if and(-eps <= y, y <= eps) {
          return MathError()
        }
        return x / y
      }
      div: Double|MathError = safe_div(x, y)
      if div is Double {
        return div
      } else {
        return 0.0
      }
    }
    """
    main = compile_program(engine, program)
    assert_equal(main(5, 0), 0)
    assert_equal(main(5, 1), 5 / 1)
    assert_equal(main(-3, -6), -3 / -6)


def test_union_with_reference():
  with make_execution_engine() as engine:
    program = """
    func main(x: Int) -> Int {
      y: Ref[Bool]|Int = x
      return y
    }
    """
    main = compile_program(engine, program, add_preamble=False)
    assert_equal(main(6), 6)


def test_assign_to_union():
  with make_execution_engine() as engine:
    program = """
    func main()  {
      sth: Double|Int = 42
    }
    """
    main = compile_program(engine, program)
    main()


def test_assign_to_union2():
  with make_execution_engine() as engine:
    program = """
    struct MathError { }
    func safe_sqrt(x: Double) ->  Double|MathError  {
      if x < 0.0 { return MathError() }
      extern_func sqrt(x: Double) ->  Double
      return sqrt(x)
    }
    func main()  {
      a = safe_sqrt(-123.0)
      a = 5.0
      a: Double|MathError = MathError()  # can also explicitly specify type again
    }
    """
    main = compile_program(engine, program)
    main()


def test_assign_union_to_single():
  with make_execution_engine() as engine:
    program = """
    func main() -> Int {
      x: Double|Int = 3  # at this point, the compiler asserts that x is an Int
      y: Int = x  # so this should work
      return y
    }
    """
    main = compile_program(engine, program, add_preamble=False)
    assert_equal(main(), 3)


def test_call_union_arg():
  with make_execution_engine() as engine:
    program = """
    func accepts_both(thing: Int|Bool) ->  Bool  {
      if thing is Int { return thing >= 0 }
      if thing is Bool { return thing }
      return False()  # never happens...
    }
    func main(a: Int) ->  Bool  {
      thing: Int|Bool = a
      return accepts_both(thing)
    }
    """
    main = compile_program(engine, program)
    assert_equal(main(4), True)
    assert_equal(main(-7), False)


def test_call_multiple_concrete_funcs_with_union_arg():
  with make_execution_engine() as engine:
    program = """
    func const() ->  Bool|Int  {
      # here declared separately such that the compiler does not know that it will actually always be an Int
      # in the future, we will probably add assertions so that the compiler does know that, but this will do for now.
      return 1 == 1
    }
    func is_int(x: Int) ->  Bool  { return 0 == 0 }  # avoid needing preamble
    func is_int(x: Bool) ->  Bool  { return 1 == 0 }
    func main() ->  Bool  {
      alpha: Bool|Int = const()
      return is_int(alpha)
    }
    """
    main = compile_program(engine, program, add_preamble=False)
    assert_equal(main(), False)


def test_call_multiple_concrete_void_funcs_with_union_arg():
  with make_execution_engine() as engine:
    program = """
    func const() ->  Bool|Int  {
      # here declared separately such that the compiler does not know that it will actually always be an Int
      # in the future, we will probably add assertions so that the compiler does know that, but this will do for now.
      return 1 == 1
    }
    func cool_func(x: Int)  { }
    func cool_func(x: Bool)  { }
    func main() -> Int {
      alpha: Bool|Int = const()
      cool_func(alpha)
      return 1
    }
    """
    main = compile_program(engine, program, add_preamble=False)
    assert_equal(main(), 1)


def test_call_multiple_concrete_funcs_with_union_arg_different_return_type():
  with make_execution_engine() as engine:
    program = """
    func const() -> Double|Int {
      return 32.0
    }
    func neg(x: Int) -> Int { return -x }
    func neg(x: Double) -> Double { return -x }
    func main() {
      alpha: Double|Int = const()
      result: Double|Int = neg(alpha)
    }
    """
    main = compile_program(engine, program, add_preamble=False)
    main()


def test_union_folding():
  with make_execution_engine() as engine:
    program = """
    struct None { }
    func main() ->  Double  {
      a: None|Bool|Double = 42.0
      x: Double|(None|Bool) = a
      return x
    }
    """
    main = compile_program(engine, program)
    assert_equal(main(), 42.0)


def test_is_operator_incremental():
  with make_execution_engine() as engine:
    program = """
    struct None { }
    func make_val() ->  None|Bool|Double  { return 42.0 }
    func main() ->  Bool  {
      a: None|Bool|Double = make_val()
      if a is None {
        if a is Bool {
          # this is dead code
          return False()
        }
      }
      return True()
    }
    """
    main = compile_program(engine, program)
    assert_equal(main(), True)


def test_union_negative_else_clause():
  with make_execution_engine() as engine:
    program = """
    func is_int(b: Int) -> Int { return 0 }
    func is_bool(b: Bool) -> Int { return 1 }
    func make_val() -> Int|Bool { return 0 == 0 }
    func main() -> Int {
      val = make_val()
      if val is Int {
        return is_int(val)
      } else {
        # must be Bool
        return is_bool(val)
      }
    }
    """
    main = compile_program(engine, program, add_preamble=False)
    assert_equal(main(), 1)


def test_union_if_else_type_narrowing():
  with make_execution_engine() as engine:
    program = """
    func is_int(b: Int) -> Int { return 0 }
    func is_bool(b: Bool) -> Int { return 1 }
    func is_double(b: Double) -> Int { return 2 }
    func make_val() ->  Int|Bool|Double  { return 42.0 }
    func main() -> Int {
      val = make_val()
      if val is Int {
        return is_int(val)
      } else {
        # must be Bool or Double
        if val is Bool {
          return is_bool(val)
        } else {
          # must be Double
          return is_double(val)
        }
      }
    }
    """
    main = compile_program(engine, program, add_preamble=False)
    assert_equal(main(), 2)


def test_union_if_without_else_type_narrowing():
  with make_execution_engine() as engine:
    program = """
    struct Unbounded { }
    func normalized_from_index(index: Int|Unbounded, length: Int) ->  Int  {
      if index is Unbounded { index = 0 }
      if index < 0 { index += length }
      return index
    }
    func main(unbounded: Bool, index: Int, length: Int) ->  Int  {
      slice: Int|Unbounded = index
      if unbounded { slice = Unbounded() }
      return normalized_from_index(slice, length)
    }
    """
    main = compile_program(engine, program, add_preamble=False)
    assert_equal(main(False, 0, 10), 0)
    assert_equal(main(False, 4, 10), 4)
    assert_equal(main(False, -1, 10), 9)
    assert_equal(main(True, -1, 10), 0)


def test_union_if_terminated_branch_type_narrowing():
  with make_execution_engine() as engine:
    program = """
    func make_val() ->  Int|Double  { return 42.0 }
    func main() ->  Double  {
      val = make_val()
      if val is Int {
        return 17.5
      }
      # must be Double
      val = sin(val)
      return val
    }
    """
    main = compile_program(engine, program)
    from math import sin
    assert_almost_equal(main(), sin(42.0))


def test_union_store_into_smaller_value():
  with make_execution_engine() as engine:
    program = """
    struct None { }
    func main(int: Int) -> Int {
      x: Int|Double = int  # takes up 8 bytes
      y: Int|Bool = x      # only takes 4 bytes
      return y
    }
    """
    main = compile_program(engine, program, add_preamble=False)
    assert_equal(main(4), 4)


def test_union_member_access():
  with make_execution_engine() as engine:
    program = """
    struct Wrapper { val: Int; }
    func main(a: Int) -> Int {
      w: Int|Wrapper = Wrapper(a)
      return w.val
    }
    """
    main = compile_program(engine, program, add_preamble=False)
    assert_equal(main(5), 5)


def test_ref_of_union():
  with make_execution_engine() as engine:
    program = """
    struct Wrapper { val: Int; }
    func main(a: Int) -> Int {
      w: Int|Wrapper = a
      !ww: Ref[Int|Wrapper] = !w
      w2: Int = ww
      return w2
    }
    """
    main = compile_program(engine, program, add_preamble=False)
    assert_equal(main(5), 5)


def test_while_cond_type_narrowing():
  with make_execution_engine() as engine:
    program = """
    func main(initial_value: Int) ->  Bool  {
      value: Int|Bool = initial_value
      while value is Int {
        value -= 1
        if value < 0 { value = False() }
        else { if value > 100 { value = True() } }
      }
      return value
    }
    """
    main = compile_program(engine, program)
    assert_equal(main(17), False)
    assert_equal(main(-2), False)
    assert_equal(main(117), True)


def test_while_cond_type_narrowing_error():
  with make_execution_engine() as engine:
    program = """
    func main(initial_value: Int) ->  Bool  {
      value: Int|Bool = initial_value
      while value is Int {
        value -= 1
        if value < 0 { value = False() }
        if value > 100 { value = True() }  # <- this should not work, value might be an Int here
      }
      return value
    }
    """
    with assert_raises(SemanticError):
      compile_program(engine, program)


def test_assert_type_narrowing():
  with make_execution_engine() as engine:
    program = """
    func black_box() ->  Int|Char  {
      return 12
    }
    func main() ->  Int  {
      my_thing : Int|Char = black_box()
      assert(my_thing is Int)
      return my_thing
    }
    """
    main = compile_program(engine, program)
    assert_equal(main(), 12)


def test_assert_type_narrowing_not():
  with make_execution_engine() as engine:
    program = """
    func black_box() -> Int|Char {
      return 12
    }
    func main() ->  Int  {
      my_thing : Int|Char = black_box()
      assert(not(my_thing is Char))
      return my_thing
    }
    """
    main = compile_program(engine, program)
    assert_equal(main(), 12)


def test_unchecked_assert_type_narrowing():
  with make_execution_engine() as engine:
    program = """
    struct S {
      val: Ptr[Double] = 0.0
    }
    func cast_to_s(ptr: S|Ptr[Double]) ->  S  {
      unchecked_assert(ptr is S)
      return ptr
    }
    func main(val: Double) ->  Double  {
      ptr: Ptr[Double] = allocate[Double](2)
      store(ptr, val)
      s: S = cast_to_s(ptr)
      return load(s.val)
    }
    """
    main = compile_program(engine, program)
    assert_equal(main(42.0), 42.0)
    assert_equal(main(0.123), 0.123)


def test_string_literal():
  with make_execution_engine() as engine:
    program = """
    func main(reps: Int) ->  Int  {
      base = "test"
      out = ""
      while reps > 0 {
        out += base
        reps -= 1
      }
      return out.length
    }
    """
    main = compile_program(engine, program)
    assert_equal(main(10), 10 * 4)


def test_hex_int_literal():
  with make_execution_engine() as engine:
    program = """
    func main() ->  Int  {
      return 0x02A
    }
    """
    main = compile_program(engine, program)
    assert_equal(main(), 2 * 16 + 10)


def test_float_literal():
  with make_execution_engine() as engine:
    program = """
    func main() ->  Float  {
      return 0.5f
    }
    """
    main = compile_program(engine, program)
    assert_equal(main(), 0.5)


def test_unreachable_code():
  with make_execution_engine() as engine:
    program = """
    func main()  {
      return
      i = 1
    }
    """
    with assert_raises(SemanticError):
      compile_program(engine, program)


def test_unreachable_code2():
  with make_execution_engine() as engine:
    program = """
    func main()  {
      return
      if 1 == 2 {
        x = 2
      } else {
        return
        x = 5
      }
    }
    """
    with assert_raises(SemanticError):
      compile_program(engine, program)


def test_bitwise_or():
  with make_execution_engine() as engine:
    program = """
    func main(a: Int, b: Int) ->  Int  {
      return bitwise_or(a, b)
    }
    """
    main = compile_program(engine, program, add_preamble=False)
    assert_equal(main(15645, 4301), 15645 | 4301)


def test_mod():
  with make_execution_engine() as engine:
    program = """
    func mod_(a: Int, b: Int) ->  Int  {
      return mod(a, b)
    }
    """
    from math import fmod
    main = compile_program(engine, program, add_preamble=True, main_func_identifier='mod_')
    for a, b in [(4, 2), (-4, 2), (7, 3), (7, 1), (-5, 3), (-2, -2)]:
      assert_equal(main(a, b), fmod(a, b))

      
def test_char_not_comparable():
  for path in glob.glob('./examples_char_not_comparable/*'):
    with make_execution_engine() as engine, open(path) as file:
      program = file.read()
      with assert_raises(SemanticError):
        compile_program(engine, program, add_preamble=False)

        
def test_templ_ternary():
  with make_execution_engine() as engine:
    program = """
    func ternary[T](true: T, false: T, cond: Bool) -> T {
      if cond { return true } else { return false }
    }
    func main() -> Double {
      a = ternary(4, 6, False())
      b: Double = ternary(1.2, 5.6, True())
      return b
    }
    """
    main = compile_program(engine, program, add_preamble=True)
    assert_almost_equal(main(), 1.2)


def test_templ_max():
  with make_execution_engine() as engine:
    program = """
    func max_[T](a: T, b: T) -> T {
      if a < b {
        return b
      } else {
        return a
      }
    }
    func main(a: Int, b: Int) -> Int {
      return max_(a, b)
    }
    """
    main = compile_program(engine, program, add_preamble=False)
    assert_equal(main(2, 3), max(2, 3))
    assert_equal(main(5, 3), max(5, 3))


def test_templ_inner_func():
  with make_execution_engine() as engine:
    program = """
    func times_four[T](x: T) -> T {
      func sum(a: T, b: T) -> T {
        return a + b
      }
      double: T = sum(x, x)
      return sum(double, double)
    }
    func main(a: Int) -> Int {
      return times_four(a)
    }
    """
    main = compile_program(engine, program, add_preamble=False)
    assert_equal(main(3), 3 * 4)
    assert_equal(main(-2), -2 * 4)


def test_templ_inner_func2():
  with make_execution_engine() as engine:
    program = """
    func ToDouble(i: Int) -> Double {
      extern_func int_to_double(i: Int) -> Double
      return int_to_double(i)
    }
    func ToDouble(d: Double) -> Double { return d }
    func add_two_pi[T](x: T) -> Double {
      func double[U](y: U) -> Double {
        return ToDouble(y + y)
      }
      pi = 3.1415
      return double(pi) + ToDouble(x)
    }
    func main(a: Int) -> Double {
      return add_two_pi(a)
    }
    """
    main = compile_program(engine, program, add_preamble=False)
    assert_almost_equal(main(3), 2 * 3.1415 + 3)


def test_templ_local_var():
  with make_execution_engine() as engine:
    program = """
    func foo[T](local: T) -> T {
      x: T = local
      return x
    }
    func main(i: Int) -> Int {
      return foo(i)
    }
    """
    main = compile_program(engine, program, add_preamble=False)
    assert_equal(main(2), 2)
    assert_equal(main(5), 5)


def test_templ_ptr():
  with make_execution_engine() as engine:
    program = """
    func main(a: Int, b: Int) -> Int {  # returns a + b
      extern_func allocate(size: Int) -> Ptr[Int]
      ptr = allocate(2)
      store(ptr, a)
      store(ptr + 1, b)
      sum = load(ptr) + load(ptr + 1)
      extern_func deallocate(ptr: Ptr[Int])
      deallocate(ptr)
      return sum
    }
    """
    main = compile_program(engine, program, main_func_identifier='main', add_preamble=False)
    assert_equal(main(3, 5), 3 + 5)
    assert_equal(main(-34, 23), -34 + 23)


def test_templ_call_overloaded_union():
  with make_execution_engine() as engine:
    program = """
    func foo[T](a: Double, b: T) -> T {
      return b
    }
    func foo[T](a: Int, b: T) -> T {
      return b
    }
    func blackbox() -> Int|Double {
      return 2
    }
    func main(b: Bool) -> Bool {
      x: Int|Double = blackbox()
      y: Bool = foo(x, b)
      return y
    }
    """
    main = compile_program(engine, program, add_preamble=True)
    assert_equal(main(True), True)


def test_templ_struct():
  with make_execution_engine() as engine:
    program = """
    struct Wrapper[T] {
      value: T
    }
    func unwrap[T](wrapper: Wrapper[T]) -> T {
      return wrapper.value
    }
    func wrap[T](value: T) -> Wrapper[T] {
      return Wrapper(value)
    }
    func +[T](a: Wrapper[T], b: Wrapper[T]) -> Wrapper[T] {
      return wrap(unwrap(a) + unwrap(b))
    }
    func main(a: Int, b: Int) -> Int {
      a_ = wrap(a)
      b_ = wrap(b)
      c_ = a_ + b_
      return unwrap(c_)
    }
    """
    main = compile_program(engine, program, add_preamble=False)
    assert_equal(main(23, 43), 23 + 43)


def test_templ_explicit_types_redundant():
  with make_execution_engine() as engine:
    program = """
    func noop[T](x: T) -> T {
      return x
    }
    func main(a: Int) -> Int {
      b = noop[Int](a)
      return b
    }
    """
    main = compile_program(engine, program, add_preamble=False)
    assert_equal(main(3), 3)


def test_templ_explicit_types_multiple_times():
  with make_execution_engine() as engine:
    program = """
    func noop[T](x: T) -> T {
      return x
    }
    func main(a: Int) -> Int {
      b = noop[Int][Double](a)  # <- thats one too much!
      return b
    }
    """
    with assert_raises(SemanticError):
      compile_program(engine, program, add_preamble=False)


def test_templ_explicit_types_mismatch():
  with make_execution_engine() as engine:
    program = """
    func noop[T](x: T) -> T {
      return x
    }
    func main(a: Int) -> Int {
      b = noop[Double](a)  # <- can't call a Double version with an Int!
      return b
    }
    """
    with assert_raises(SemanticError):
      compile_program(engine, program, add_preamble=False)


def test_templ_explicit_types_needed():
  with make_execution_engine() as engine:
    program = """
    func noop[T]() {
      # does absolutely nothing.
    }
    func main() {
      noop[Int]()
    }
    """
    main = compile_program(engine, program, add_preamble=False)
    main()  # just check that it runs


def test_templ_explicit_types_struct_needed():
  with make_execution_engine() as engine:
    program = """
    struct Empty[T] { }
    func main() {
      x = Empty[Int]()
    }
    """
    main = compile_program(engine, program, add_preamble=False)
    main()  # just check that it runs


def test_templ_meta_programming():
  with make_execution_engine() as engine:
    program = """
    struct Nil {}
    struct Succ[T] {}
    
    func is_eq(x: Nil, y: Nil) -> Bool { return True() }
    func is_eq[A](x: Succ[A], y: Nil) -> Bool { return False() }
    func is_eq[A](x: Nil, y: Succ[A]) -> Bool { return False() }
    func is_eq[A, B](x: Succ[A], y: Succ[B]) -> Bool { return is_eq(A(), B()) }
    
    func three_eq_three() -> Bool {
      return is_eq(Succ[Succ[Succ[Nil]]](), Succ[Succ[Succ[Nil]]]())
    }
    func three_eq_two() -> Bool {
      return is_eq(Succ[Succ[Succ[Nil]]](), Succ[Succ[Nil]]())
    }
    func two_eq_three() -> Bool {
      return is_eq(Succ[Succ[Nil]](), Succ[Succ[Succ[Nil]]]())
    }
    """
    three_eq_three = compile_program(engine, program, main_func_identifier='three_eq_three')
    assert_equal(three_eq_three(), True)
    three_eq_two = compile_program(engine, program, main_func_identifier='three_eq_two')
    assert_equal(three_eq_two(), False)
    two_eq_three = compile_program(engine, program, main_func_identifier='two_eq_three')
    assert_equal(two_eq_three(), False)


def test_ptr_loop():
  with make_execution_engine() as engine:
    program = """
    func main(len: Int) -> Int {  # computes sum_i^{len-1} i
      arr: Ptr[Int] = allocate[Int](len)
      # store numbers 0 ... len-1
      pos = 0
      while pos < len {
        store(arr + pos, pos)
        pos += 1
      }
      
      # compute sum of all numbers
      sum = 0
      ptr = arr
      while ptr < arr + len {
        sum += load(ptr)
        ptr += 1
      }
      
      deallocate(arr)
      return sum
    }
    """
    main = compile_program(engine, program)
    assert_equal(main(5), sum(i for i in range(5)))
    assert_equal(main(32), sum(i for i in range(32)))


def test_ptr_arithmetic():
  with make_execution_engine() as engine:
    program = """
    func main(magic: Int) -> Int {
      orig = allocate[Int](20)
      store(orig, magic)
      ptr = orig
      ptr += 10  # = orig + 10
      assert(ptr > orig)
      assert(ptr >= orig)
      assert(not(ptr < orig))
      ptr += -3  # = orig + 7
      assert(ptr > orig)
      ptr += -7  # = orig
      assert(ptr == orig)
      assert(ptr <= orig)
      assert(not(ptr > orig))
      magic_ = load(ptr) 
      deallocate(orig)
      return magic_
    }
    """
    main = compile_program(engine, program)
    assert_equal(main(1234), 1234)


def test_ptr_casts():
  with make_execution_engine() as engine:
    program = """
    func main(val: Int) -> Int {
      ptr: Ptr[Int] = allocate[Int](3)
      store(ptr, val)
      raw_ptr = RawPtr(ptr)
      raw_ptr += 1
      raw_ptr -= 1l
      ptr1 = Ptr[Int](raw_ptr)
      if ptr != ptr1 { return -1 }
      val1 = load(ptr1)
      deallocate(ptr1)
      return val1
    }
    """
    main = compile_program(engine, program)
    assert_equal(main(1234), 1234)


def test_size_simple():
  with make_execution_engine() as engine:
    program = """
    func main() -> Int {
      if size(Bool) != 1l { return 0 }
      if size(Int) != 4l { return 0 }
      if size(Long) != 8l { return 0 }
      return 1
    }
    """
    main = compile_program(engine, program, add_preamble=False)
    assert_equal(main(), 1)


def test_size_templ():
  with make_execution_engine() as engine:
    program = """
    func templ_size[T]() -> Long {
      return size(T)
    }
    func main() -> Int {
      if templ_size[Bool]() != 1l { return 0 }
      if templ_size[Int]() != 4l { return 0 }
      if templ_size[Long]() != 8l { return 0 }
      if templ_size[Double]() != 8l { return 0 }
      struct Vec3[T] { x: T; y: T; z: T; }
      if templ_size[Vec3[Int]]() != 3l * templ_size[Int]() { return 0 }
      if templ_size[Vec3[Double]]() != 3l * templ_size[Double]() { return 0 }
      return 1
    }
    """
    main = compile_program(engine, program, add_preamble=False)
    assert_equal(main(), 1)


def test_scoped_function_redefinition():
  with make_execution_engine() as engine:
    program = """
    func main() -> Int {
      func a() -> Int {
          func inner() -> Int { return 1 }
          return inner() + 2
      }
  
      func b() -> Double {
          func inner() -> Double { return 5.5 }
          return inner() * 2.1
      }
  
      if a() == 3 {
          if b() >= 11.1 { return 1 }
      }
  
      return 0
    }
    """
    main = compile_program(engine, program, add_preamble=False)
    assert_equal(main(), 1)


def test_repeated_inline_calls():
  with make_execution_engine() as engine:
    program = """
    @Inline func f() {
      func g() {}
      g()
    }
    
    
    func main() {
      f()
      f()
      f()
    }
    """
    compile_program(engine, program, add_preamble=False)


def test_narrowing_with_shadowing():
  with make_execution_engine() as engine:
    program = """
    func main() {
      x: Int = 1
      func inner() {
        x: Double|Float = 2.0
        if x is Double { }
        else { assert(x is Double) }
      }
    }
    """
    compile_program(engine, program, add_preamble=True)


def test_arg_mutates():
  with make_execution_engine() as engine:
    program = """
    func increment(mutates x: Int) {
      x = x + 1
    }
    func main(a: Int) -> Int {
      increment(a)
      return a
    }
    """
    main = compile_program(engine, program, add_preamble=False)
    assert_equal(main(5), 6)
    assert_equal(main(10), 11)


def test_arg_mutates_inline():
  with make_execution_engine() as engine:
    program = """
    @Inline func increment(mutates x: Int) {
      x = x + 1
    }
    func main(a: Int) -> Int {
      increment(a)
      return a
    }
    """
    main = compile_program(engine, program, add_preamble=False)
    assert_equal(main(5), 6)
    assert_equal(main(10), 11)


def test_arg_mutates_called_with_non_referenable():
  with make_execution_engine() as engine:
    program = """
    func increment(mutates x: Int) { x += 1 }
    func main() {
      increment(5)  # doesn't work, 5 has no address.
    }
    """
    with assert_raises(SemanticError):
      compile_program(engine, program, add_preamble=False)


def test_arg_mutates_defined_with_mutates_and_non_mutates():
  with make_execution_engine() as engine:
    program = """
    func increment(x: Int) {
      # do nothing?
    }
    func increment(mutates x: Int) {
      x+= 1
    }
    """
    with assert_raises(SemanticError):
      compile_program(engine, program, add_preamble=False)


def test_arg_mutates_and_non_mutates():
  with make_execution_engine() as engine:
    program = """
    func foo(mutates x: Int, y: Int) {
      x += 1
      y -= 1
    }
    func main(a: Int) -> Int {
      foo(a, a)
      return a  # should be a + 1
    }
    """
    main = compile_program(engine, program, add_preamble=False)
    assert_equal(main(5), 5 + 1)


def test_mutates_struct_member():
  with make_execution_engine() as engine:
    program = """
    struct Foo {
      value: Int = 0
    }
    func inc_val(mutates of: Foo)  {
      of.value += 1
    }
    func main() ->  Int  {
      my_foo = Foo(0)
      my_foo.value = 4
      inc_val(my_foo)  # now my_foo.value should be 5.
      return my_foo.value
    }
    """
    main = compile_program(engine, program, add_preamble=False)
    assert_equal(main(), 5)


def test_mutates_called_with_ref():
  with make_execution_engine() as engine:
    program = """
    func inc_val(mutates i: Int)  { i += 1 }
    func main(my_int: Int) -> Int {
      !my_int_ref: Ref[Int] = !my_int
      inc_val(my_int_ref)
      return my_int_ref
    }
    """
    main = compile_program(engine, program, add_preamble=False)
    assert_equal(main(5), 5 + 1)


def test_mutates_called_with_ref_ref():
  with make_execution_engine() as engine:
    program = """
    func inc_val(mutates i: Int)  { i += 1 }
    func main(my_int: Int) -> Int {
      !my_int_ref: Ref[Int] = !my_int
      !!my_int_ref_ref: Ref[Ref[Int]] = !!my_int_ref
      inc_val(my_int_ref_ref)
      return my_int_ref  # could also use my_int or my_int_ref_ref
    }
    """
    main = compile_program(engine, program, add_preamble=False)
    assert_equal(main(5), 5 + 1)


def test_imports():
    program = """
      import "a", "b",
        "c", "d"
        , "e"
        ,
        "f"
    """
    ast = make_file_ast_from_str(file_path=DummyPath("test"), program=program)
    assert_equal(ast.imports_ast.imports, ["a", "b", "c", "d", "e", "f"])


def test_empty_imports_disallowed():
  program = """
    import 
  """
  with assert_raises(ParseError):
    make_translation_unit_ast_from_str(file_path=DummyPath("test"), program=program, add_preamble=False)


def test_no_import():
  program = """
  """
  ast = make_translation_unit_ast_from_str(file_path=DummyPath("test"), program=program, add_preamble=False)
  assert_equal(ast.file_asts[0].imports_ast.imports, [])


def test_operator_precedence():
  with make_execution_engine() as engine:
    program = """
    struct Foo { x: Ref[Int]; y: Int; }
    func index(x: Ref[Int]) -> Ref[Int] { return !x }
    func index(x: Int) -> Int { return 0; }
    func main() {
      x = 5
      foo = Foo(!x, x)
      a1 = -foo.y  # . binds stronger than -
      !a2 = !foo.x  # . binds stronger than !
      a3 = foo.y[]  # . binds stronger than []
      a4 = -foo.y[]  # . binds stronger than [] binds stronger than -
      a5 = (!index(!foo.x))[]  # here I have to use brackets
    }
    """
    compile_program(engine, program, add_preamble=False)  # just check that it compiles


def test_syntax_non_ascii_comment():
  with make_execution_engine() as engine:
    program = """
    func main() -> Int {
      # Hä, öhm, okay!
      return 1;
    }
    """
    main = compile_program(engine, program, add_preamble=False)
    assert_equal(main(), 1)
