import _setup_test_env  # noqa
import better_exchook
import sys
import unittest

from nose.tools import assert_equal


def test_narrow_type():
  from sleepy.symbols import narrow_type, UnionType, SLEEPY_INT, SLEEPY_BOOL
  assert_equal(narrow_type(SLEEPY_INT, SLEEPY_INT), SLEEPY_INT)
  assert_equal(narrow_type(UnionType([SLEEPY_INT], [0], 4), SLEEPY_INT), UnionType([SLEEPY_INT], [0], 4))
  assert_equal(narrow_type(SLEEPY_INT, UnionType([SLEEPY_INT], [0], 4)), SLEEPY_INT)
  assert_equal(
    narrow_type(UnionType([SLEEPY_INT, SLEEPY_BOOL], [0, 1], 4), UnionType([SLEEPY_INT], [0], 4)),
    UnionType([SLEEPY_INT], [0], 4))
  assert_equal(
    narrow_type(UnionType([SLEEPY_INT, SLEEPY_BOOL], [0, 1], 4), SLEEPY_BOOL), UnionType([SLEEPY_BOOL], [1], 4))
  assert_equal(
    narrow_type(UnionType([SLEEPY_INT, SLEEPY_BOOL], [0, 1], 4), UnionType([SLEEPY_BOOL], [0], 1)),
    UnionType([SLEEPY_BOOL], [1], 4))


def test_get_common_type():
  from sleepy.symbols import get_common_type, UnionType, SLEEPY_INT, SLEEPY_BOOL, SLEEPY_DOUBLE
  assert_equal(get_common_type([SLEEPY_INT]), SLEEPY_INT)
  assert_equal(get_common_type([SLEEPY_INT, SLEEPY_INT]), SLEEPY_INT)
  int_bool_union = UnionType([SLEEPY_INT, SLEEPY_BOOL], [0, 1], 4)
  bool_int_union = UnionType([SLEEPY_INT, SLEEPY_BOOL], [1, 0], 4)
  assert_equal(get_common_type([SLEEPY_INT, SLEEPY_BOOL]), int_bool_union)
  assert_equal(get_common_type([SLEEPY_INT, SLEEPY_BOOL, int_bool_union]), int_bool_union)
  assert_equal(get_common_type([int_bool_union, bool_int_union]), int_bool_union)
  assert_equal(
    get_common_type([int_bool_union, SLEEPY_DOUBLE]), UnionType([SLEEPY_INT, SLEEPY_BOOL, SLEEPY_DOUBLE], [0, 1, 2], 8))


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
