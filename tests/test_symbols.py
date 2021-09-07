import _setup_test_env  # noqa
import better_exchook
import sys
import unittest

from nose.tools import assert_equal

from sleepy.symbols import UnionType


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


def test_exclude_type():
  from sleepy.symbols import exclude_type, UnionType, SLEEPY_INT, SLEEPY_BOOL, SLEEPY_DOUBLE, SLEEPY_NEVER
  assert_equal(exclude_type(SLEEPY_INT, SLEEPY_NEVER), SLEEPY_INT)
  assert_equal(exclude_type(SLEEPY_INT, SLEEPY_INT), SLEEPY_NEVER)
  assert_equal(
    exclude_type(UnionType([SLEEPY_INT, SLEEPY_DOUBLE], [0, 1], 8), SLEEPY_DOUBLE), UnionType([SLEEPY_INT], [0], 8))
  assert_equal(
    exclude_type(UnionType([SLEEPY_INT, SLEEPY_DOUBLE], [0, 1], 8), SLEEPY_INT), UnionType([SLEEPY_DOUBLE], [1], 8))
  assert_equal(
    exclude_type(UnionType([SLEEPY_INT, SLEEPY_DOUBLE], [0, 1], 8), SLEEPY_BOOL),
    UnionType([SLEEPY_INT, SLEEPY_DOUBLE], [0, 1], 8))


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


# noinspection PyPep8Naming
def test_try_infer_templ_types_simple():
  from sleepy.symbols import try_infer_templ_types, SLEEPY_INT, SLEEPY_DOUBLE, PlaceholderTemplateType
  T = PlaceholderTemplateType('T')
  U = PlaceholderTemplateType('U')
  assert_equal(try_infer_templ_types(calling_types=[], signature_types=[], placeholder_templ_types=[]), [])
  assert_equal(try_infer_templ_types(
    calling_types=[SLEEPY_INT, SLEEPY_DOUBLE], signature_types=[SLEEPY_INT, SLEEPY_DOUBLE], placeholder_templ_types=[]),
    [])
  assert_equal(try_infer_templ_types(
    calling_types=[SLEEPY_INT, SLEEPY_DOUBLE], signature_types=[SLEEPY_INT, SLEEPY_DOUBLE],
    placeholder_templ_types=[T]),
    None)
  assert_equal(try_infer_templ_types(
    calling_types=[SLEEPY_INT], signature_types=[T],
    placeholder_templ_types=[T]),
    [SLEEPY_INT])
  assert_equal(try_infer_templ_types(
    calling_types=[SLEEPY_INT, SLEEPY_INT], signature_types=[T, T],
    placeholder_templ_types=[T]),
    [SLEEPY_INT])
  assert_equal(try_infer_templ_types(
    calling_types=[SLEEPY_INT, SLEEPY_DOUBLE], signature_types=[T, SLEEPY_DOUBLE],
    placeholder_templ_types=[T]),
    [SLEEPY_INT])
  assert_equal(try_infer_templ_types(
    calling_types=[SLEEPY_INT, SLEEPY_DOUBLE], signature_types=[SLEEPY_INT, T],
    placeholder_templ_types=[T]),
    [SLEEPY_DOUBLE])
  assert_equal(try_infer_templ_types(
    calling_types=[SLEEPY_INT, SLEEPY_DOUBLE], signature_types=[T, U],
    placeholder_templ_types=[T, U]),
    [SLEEPY_INT, SLEEPY_DOUBLE])
  assert_equal(try_infer_templ_types(
    calling_types=[SLEEPY_INT, SLEEPY_DOUBLE], signature_types=[T, U],
    placeholder_templ_types=[U, T]),
    [SLEEPY_DOUBLE, SLEEPY_INT])


# noinspection PyPep8Naming
def test_try_infer_templ_types_ptr():
  from sleepy.symbols import try_infer_templ_types, PlaceholderTemplateType, PointerType, SLEEPY_INT, SLEEPY_CHAR
  T = PlaceholderTemplateType('T')
  assert_equal(
    try_infer_templ_types(
      calling_types=[PointerType(SLEEPY_INT)], signature_types=[PointerType(T)], placeholder_templ_types=[T]),
    [SLEEPY_INT])
  assert_equal(
    try_infer_templ_types(
      calling_types=[PointerType(SLEEPY_CHAR), SLEEPY_INT], signature_types=[PointerType(T), SLEEPY_INT],
      placeholder_templ_types=[T]),
    [SLEEPY_CHAR])


def test_try_infer_templ_types_union():
  from sleepy.symbols import try_infer_templ_types, SLEEPY_INT, SLEEPY_CHAR
  assert_equal(
    try_infer_templ_types(
      calling_types=[SLEEPY_INT],
      signature_types=[UnionType.from_types([SLEEPY_INT, SLEEPY_CHAR])], placeholder_templ_types=[]),
    [])
  assert_equal(
    try_infer_templ_types(
      calling_types=[SLEEPY_CHAR],
      signature_types=[UnionType.from_types([SLEEPY_INT, SLEEPY_CHAR])], placeholder_templ_types=[]),
    [])
  assert_equal(
    try_infer_templ_types(
      calling_types=[UnionType.from_types([SLEEPY_INT, SLEEPY_CHAR])],
      signature_types=[UnionType.from_types([SLEEPY_INT, SLEEPY_CHAR])], placeholder_templ_types=[]),
    [])
  assert_equal(
    try_infer_templ_types(
      calling_types=[UnionType.from_types([SLEEPY_INT, SLEEPY_CHAR])],
      signature_types=[UnionType.from_types([SLEEPY_CHAR, SLEEPY_INT])], placeholder_templ_types=[]),
    [])


# noinspection PyPep8Naming
def test_try_infer_templ_types_struct():
  from sleepy.symbols import try_infer_templ_types, PlaceholderTemplateType, StructType, SLEEPY_INT, SLEEPY_CHAR
  T = PlaceholderTemplateType('T')
  U = PlaceholderTemplateType('U')
  WrapperT = StructType(
    'Wrapper', templ_types=[T], member_identifiers=['value'], member_types=[T], member_mutables=[False],
    pass_by_ref=False)
  WrapperU = WrapperT.replace_types({T: U})
  WrapperInt = WrapperT.replace_types({T: SLEEPY_INT})
  WrapperChar = WrapperT.replace_types({T: SLEEPY_CHAR})
  assert_equal(
    try_infer_templ_types(
      calling_types=[WrapperInt], signature_types=[WrapperT], placeholder_templ_types=[T]),
    [SLEEPY_INT])
  assert_equal(
    try_infer_templ_types(
      calling_types=[WrapperInt], signature_types=[T], placeholder_templ_types=[T]),
    [WrapperInt])
  assert_equal(
    try_infer_templ_types(
      calling_types=[SLEEPY_INT], signature_types=[WrapperT], placeholder_templ_types=[T]),
    None)
  assert_equal(
    try_infer_templ_types(
      calling_types=[WrapperInt, WrapperInt], signature_types=[WrapperT, WrapperT], placeholder_templ_types=[T]),
    [SLEEPY_INT])
  assert_equal(
    try_infer_templ_types(
      calling_types=[WrapperInt, WrapperChar], signature_types=[WrapperT, WrapperU], placeholder_templ_types=[T, U]),
    [SLEEPY_INT, SLEEPY_CHAR])
  assert_equal(
    try_infer_templ_types(
      calling_types=[WrapperInt, SLEEPY_INT], signature_types=[WrapperT, T], placeholder_templ_types=[T]),
    [SLEEPY_INT])
  assert_equal(
    try_infer_templ_types(
      calling_types=[WrapperChar, SLEEPY_INT], signature_types=[WrapperT, T], placeholder_templ_types=[T]),
    None)
  assert_equal(
    try_infer_templ_types(
      calling_types=[SLEEPY_INT, SLEEPY_INT], signature_types=[WrapperT, T], placeholder_templ_types=[T]),
    None)

  # with templates in calling_types
  assert_equal(
    try_infer_templ_types(
      calling_types=[T, WrapperT], signature_types=[WrapperT, WrapperT], placeholder_templ_types=[T]),
    None)
  assert_equal(
    try_infer_templ_types(
      calling_types=[WrapperT, T], signature_types=[WrapperT, WrapperT], placeholder_templ_types=[T]),
    None)
  assert_equal(
    try_infer_templ_types(
      calling_types=[WrapperT, WrapperT], signature_types=[T, WrapperT], placeholder_templ_types=[T]),
    None)
  assert_equal(
    try_infer_templ_types(
      calling_types=[WrapperT, WrapperT], signature_types=[WrapperT, T], placeholder_templ_types=[T]),
    None)


def test_context_use_pos():
  from sleepy.symbols import CodegenContext, make_di_location
  from sleepy.grammar import TreePosition
  from llvmlite import ir
  module = ir.Module(name='module_name')
  io_func_type = ir.FunctionType(ir.VoidType(), ())
  ir_io_func = ir.Function(module, io_func_type, name='io')
  root_block = ir_io_func.append_basic_block(name='entry')
  context = CodegenContext(builder=ir.IRBuilder(root_block))
  context.current_di_scope = context.module.add_debug_info('DISubprogram', {'name': 'dummy'})
  program = '123456789'
  outer_pos = TreePosition(word=program, from_pos=0, to_pos=9)
  with context.use_pos(outer_pos):
    assert context.current_pos == outer_pos
    assert context.builder.debug_metadata == make_di_location(outer_pos, context=context)
    inner_pos = TreePosition(word=program, from_pos=2, to_pos=4)
    with context.use_pos(inner_pos):
      assert context.current_pos == inner_pos
      assert context.builder.debug_metadata == make_di_location(inner_pos, context=context)
    assert context.current_pos == outer_pos
    assert context.builder.debug_metadata == make_di_location(outer_pos, context=context)
  assert context.current_pos is None


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
