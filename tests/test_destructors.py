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


def test_side_effect_in_destruct():
  with make_execution_engine() as engine:
    # language=Sleepy
    program = """
    @destructible
    struct S { b: Ref[Bool]}
    func destruct(self: S) { self.b = True() }

    func main() -> Bool {
      s_destructed = False()
      if True() {
        s = S(!s_destructed)
      }
      return s_destructed
    }
    """
    main = compile_program(engine, program, main_func_identifier='main', add_preamble=True)
    assert_equal(main(), True)

def test_destructible_templated_struct():
  with make_execution_engine() as engine:
    # language=Sleepy
    program = """
    @destructible
    struct S[T] { value: T; b: Ref[Bool] }
    func destruct[T](self: S[T]) { self.b = True() }

    func main() -> Bool {
      s_destructed = False()
      if True() {
        s = S(12, !s_destructed)
      }
      return s_destructed
    }
    """
    main = compile_program(engine, program, main_func_identifier='main', add_preamble=True)
    assert_equal(main(), True)