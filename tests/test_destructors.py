#noqa
# noinspection PyUnresolvedReferences
import _setup_test_env

from sleepy.jit import make_execution_engine
from tests.compile import compile_program

from nose.tools import assert_equal


def test_str_destructor():
  with make_execution_engine() as engine:
    # language=Sleepy
    program = """
    func main() {
      str: Str = EmptyStr()
      str += "1234"
    }
    """
    main = compile_program(engine, program, main_func_identifier='main', add_preamble=True)
    main()


def test_destruct_if_clause():
  with make_execution_engine() as engine:
    # language=Sleepy
    program = """
    @destructible
    struct S { b: Ref[Int] }
    func destruct(self: S) { self.b += 1 }

    func main(should_construct: Bool) -> Int {
      num_destruct_calls = 0
      if should_construct {
        s = S(!num_destruct_calls)
      }
      return num_destruct_calls
    }
    """
    main = compile_program(engine, program, main_func_identifier='main', add_preamble=True)
    assert_equal(main(True), 1)
    assert_equal(main(False), 0)


def test_destruct_if_clause_with_template():
  with make_execution_engine() as engine:
    # language=Sleepy
    program = """
    @destructible
    struct S[T] { value: T; b: Ref[Int] }
    func destruct[T](self: S[T]) { self.b += 1 }

    func main(should_construct: Bool) -> Int {
      num_destruct_calls = 0
      if should_construct {
        s = S(12, !num_destruct_calls)
      }
      return num_destruct_calls
    }
    """
    main = compile_program(engine, program, main_func_identifier='main', add_preamble=True)
    assert_equal(main(True), 1)
    assert_equal(main(False), 0)


def test_destruct_function_body():
  with make_execution_engine() as engine:
    # language=Sleepy
    program = """
    @destructible
    struct S { num_destruct_calls: Ref[Int] }
    func destruct(self: S) { self.num_destruct_calls += 1 }

    func main() -> Int {
      func foo(mutates num_destruct_calls: Int) {
        s = S(!num_destruct_calls)
        # should directly be destructed again
      }
      num_destruct_calls = 0
      foo(num_destruct_calls)
      return num_destruct_calls
    }
    """
    main = compile_program(engine, program, add_preamble=True)
    assert_equal(main(), 1)
