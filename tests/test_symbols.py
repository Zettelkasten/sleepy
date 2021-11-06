import _setup_test_env  # noqa

from nose.tools import assert_equal

from llvmlite import ir
from sleepy.grammar import DummyPath
from sleepy.symbols import UnionType, SLEEPY_NEVER, StructIdentity, CodegenContext


def make_test_context(emits_ir: bool = True) -> CodegenContext:
  module = ir.Module(name="test_module")
  builder = ir.IRBuilder() if emits_ir else None
  return CodegenContext(builder=builder, module=module, emits_debug=True, file_path=DummyPath("test"))


# noinspection PyPep8Naming
def test_can_implicit_cast_to():
  from sleepy.symbols import can_implicit_cast_to, ReferenceType, UnionType, StructType, PlaceholderTemplateType
  from sleepy.builtin_symbols import SLEEPY_INT, SLEEPY_DOUBLE
  context = make_test_context()
  assert_equal(can_implicit_cast_to(SLEEPY_INT, SLEEPY_DOUBLE), False)
  assert_equal(can_implicit_cast_to(SLEEPY_INT, SLEEPY_INT), True)
  assert_equal(can_implicit_cast_to(UnionType([SLEEPY_INT], [0], 8), SLEEPY_INT), True)
  assert_equal(can_implicit_cast_to(
    UnionType([SLEEPY_INT, SLEEPY_DOUBLE], [0, 1], 8), SLEEPY_DOUBLE), False)
  assert_equal(can_implicit_cast_to(
    SLEEPY_DOUBLE, UnionType([SLEEPY_INT, SLEEPY_DOUBLE], [0, 1], 8)), True)
  assert_equal(can_implicit_cast_to(ReferenceType(SLEEPY_INT), ReferenceType(SLEEPY_DOUBLE)), False)
  assert_equal(can_implicit_cast_to(ReferenceType(SLEEPY_INT), ReferenceType(SLEEPY_INT)), True)
  assert_equal(can_implicit_cast_to(
    ReferenceType(UnionType([SLEEPY_INT, SLEEPY_DOUBLE], [0, 1], 8)), ReferenceType(SLEEPY_INT)), False)
  assert_equal(can_implicit_cast_to(
    ReferenceType(SLEEPY_INT), ReferenceType(UnionType([SLEEPY_INT, SLEEPY_DOUBLE], [0, 1], 8))), True)
  T = PlaceholderTemplateType('T')
  List = StructType(
    identity=StructIdentity('List', context=context), templ_types=[T], member_identifiers=[], member_types=[])
  assert_equal(can_implicit_cast_to(
    ReferenceType(UnionType([SLEEPY_INT, SLEEPY_DOUBLE], [0, 1], 8)), ReferenceType(SLEEPY_INT)), False)
  assert_equal(can_implicit_cast_to(
    ReferenceType(SLEEPY_INT), ReferenceType(UnionType([SLEEPY_INT, SLEEPY_DOUBLE], [0, 1], 8))), True)
  assert_equal(
    can_implicit_cast_to(
      List.replace_types({T: UnionType([SLEEPY_INT, SLEEPY_DOUBLE], [0, 1], 8)}),
      List.replace_types({T: SLEEPY_INT})),
    False)
  assert_equal(
    can_implicit_cast_to(
      List.replace_types({T: SLEEPY_INT}),
      List.replace_types({T: UnionType([SLEEPY_INT, SLEEPY_DOUBLE], [0, 1], 8)})),
    False)


def test_narrow_type():
  from sleepy.symbols import narrow_type, UnionType
  from sleepy.builtin_symbols import SLEEPY_INT, SLEEPY_BOOL
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
  assert_equal(narrow_type(SLEEPY_INT, SLEEPY_BOOL), SLEEPY_NEVER)


# noinspection PyPep8Naming
def test_narrow_type_templates():
  from sleepy.symbols import narrow_type, UnionType, PlaceholderTemplateType, StructType
  from sleepy.builtin_symbols import SLEEPY_INT, SLEEPY_BOOL
  context = make_test_context()
  Int, Bool = SLEEPY_INT, SLEEPY_BOOL
  T = PlaceholderTemplateType('T')
  List = StructType(
    identity=StructIdentity('List', context=context), templ_types=[T], member_identifiers=[], member_types=[])
  # narrow(0:List[Int]|1:List[Bool], List[Int]) = 0:List[Int]
  assert_equal(
    narrow_type(
      UnionType.from_types([List.replace_types({T: Int}), List.replace_types({T: Bool})]),
      List.replace_types({T: Int})),
    UnionType.from_types([List.replace_types({T: Int})]))
  # narrow(List[Int|Bool], List[Int]) = List[Int|Bool]
  assert_equal(
    narrow_type(List.replace_types({T: UnionType.from_types([Int, Bool])}), List.replace_types({T: Int})),
    List.replace_types({T: UnionType.from_types([Int, Bool])}))
  # narrow(List[Int|Bool]|List[Int], List[Int]) = List[Int|Bool]|List[Int]
  assert_equal(
    narrow_type(
      UnionType.from_types([List.replace_types({T: UnionType.from_types([Int, Bool])}), List.replace_types({T: Int})]),
      List.replace_types({T: Int})),
    UnionType.from_types([List.replace_types({T: UnionType.from_types([Int, Bool])}), List.replace_types({T: Int})]))


# noinspection PyPep8Naming
def test_narrow_type_references():
  from sleepy.symbols import narrow_type, UnionType, ReferenceType
  from sleepy.builtin_symbols import SLEEPY_INT, SLEEPY_BOOL
  Int, Bool = SLEEPY_INT, SLEEPY_BOOL
  Ref = ReferenceType
  # narrow(Ref[A], Ref[A]) = Ref[A]
  assert_equal(narrow_type(Ref(Int), Ref(Int)), Ref(Int))
  # narrow(Ref[0:A|1:B], Ref[A]) = Ref[0:A]
  assert_equal(narrow_type(Ref(UnionType([Int, Bool], [0, 1], 4)), Ref(Int)), Ref(UnionType.from_types([Int])))
  # narrow(0:Ref[0:A|1:B]|1:Ref[A], Ref[B]) = 0:Ref[1:B]
  assert_equal(
    narrow_type(UnionType.from_types([Ref(UnionType([Int, Bool], [0, 1], 4)), Ref(Int)]), Ref(Bool)),
    UnionType([Ref(UnionType([Bool], [1], 4))], [0], 8))
  # narrow(Ref[0:A|1:B], Ref[A]|Ref[B]) = Ref[0:A|1:B]
  assert_equal(
    narrow_type(Ref(UnionType([Int, Bool], [0, 1], 8)), UnionType([Ref(Int), Ref(Bool)], [0, 1], 8)),
    Ref(UnionType([Int, Bool], [0, 1], 8)))
  # narrow(Ref[A]|Ref[B], Ref[A|B]) = never
  assert_equal(
    narrow_type(UnionType.from_types([Ref(Int), Ref(Bool)]), Ref(UnionType.from_types([Int, Bool]))),
    UnionType([], [], 8))
  # narrow(Ref[0:A|1:B]|Ref[A], Ref[A]) = Ref[0:A]|Ref[A]
  assert_equal(
    narrow_type(UnionType.from_types([Ref(UnionType.from_types([Int, Bool])), Ref(Int)]), Ref(Int)),
    UnionType.from_types([Ref(UnionType.from_types([Int])), Ref(Int)]))


def test_exclude_type():
  from sleepy.symbols import exclude_type, UnionType, ReferenceType
  from sleepy.builtin_symbols import SLEEPY_INT, SLEEPY_BOOL, SLEEPY_DOUBLE
  assert_equal(exclude_type(SLEEPY_INT, SLEEPY_NEVER), SLEEPY_INT)
  assert_equal(exclude_type(SLEEPY_INT, SLEEPY_INT), SLEEPY_NEVER)
  assert_equal(
    exclude_type(UnionType([SLEEPY_INT, SLEEPY_DOUBLE], [0, 1], 8), SLEEPY_DOUBLE), UnionType([SLEEPY_INT], [0], 8))
  assert_equal(
    exclude_type(UnionType([SLEEPY_INT, SLEEPY_DOUBLE], [0, 1], 8), SLEEPY_INT), UnionType([SLEEPY_DOUBLE], [1], 8))
  assert_equal(
    exclude_type(UnionType([SLEEPY_INT, SLEEPY_DOUBLE], [0, 1], 8), SLEEPY_BOOL),
    UnionType([SLEEPY_INT, SLEEPY_DOUBLE], [0, 1], 8))
  assert_equal(
    exclude_type(ReferenceType(SLEEPY_INT), ReferenceType(SLEEPY_INT)), SLEEPY_NEVER)
  assert_equal(
    exclude_type(ReferenceType(SLEEPY_INT), SLEEPY_NEVER), ReferenceType(SLEEPY_INT))
  assert_equal(
    exclude_type(ReferenceType(UnionType([SLEEPY_INT, SLEEPY_BOOL], [0, 1], 4)), ReferenceType(SLEEPY_INT)),
    ReferenceType(UnionType([SLEEPY_BOOL], [1], 4)))
  assert_equal(
    exclude_type(ReferenceType(UnionType([SLEEPY_INT, SLEEPY_BOOL], [0, 1], 4)), ReferenceType(SLEEPY_BOOL)),
    ReferenceType(UnionType([SLEEPY_INT], [0], 4)))


def test_get_common_type():
  from sleepy.symbols import get_common_type, UnionType
  from sleepy.builtin_symbols import SLEEPY_INT
  from sleepy.builtin_symbols import SLEEPY_BOOL
  from sleepy.builtin_symbols import SLEEPY_DOUBLE
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
  from sleepy.symbols import try_infer_templ_types, PlaceholderTemplateType
  from sleepy.builtin_symbols import SLEEPY_INT
  from sleepy.builtin_symbols import SLEEPY_DOUBLE
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
  from sleepy.symbols import try_infer_templ_types, PlaceholderTemplateType, PointerType
  from sleepy.builtin_symbols import SLEEPY_CHAR
  from sleepy.builtin_symbols import SLEEPY_INT
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
  from sleepy.symbols import try_infer_templ_types
  from sleepy.builtin_symbols import SLEEPY_CHAR
  from sleepy.builtin_symbols import SLEEPY_INT
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
  from sleepy.symbols import try_infer_templ_types, PlaceholderTemplateType, StructType
  from sleepy.builtin_symbols import SLEEPY_CHAR, SLEEPY_INT
  context = make_test_context()
  T = PlaceholderTemplateType('T')
  U = PlaceholderTemplateType('U')
  WrapperT = StructType(
    StructIdentity('Wrapper', context=context), templ_types=[T], member_identifiers=['value'], member_types=[T])
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
  file_path = DummyPath("test")
  context = CodegenContext(builder=ir.IRBuilder(), module=module, emits_debug=True, file_path=file_path)
  program = '123456789'
  outer_pos = TreePosition(word=program, from_pos=0, to_pos=9, file_path=file_path)
  with context.use_pos(outer_pos):
    assert context.current_pos == outer_pos
    assert context.builder.debug_metadata == make_di_location(outer_pos, context=context)
    inner_pos = TreePosition(word=program, from_pos=2, to_pos=4, file_path=file_path)
    with context.use_pos(inner_pos):
      assert context.current_pos == inner_pos
      assert context.builder.debug_metadata == make_di_location(inner_pos, context=context)
    assert context.current_pos == outer_pos
    assert context.builder.debug_metadata == make_di_location(outer_pos, context=context)


def test_bind_and_unbind():
  from sleepy.symbols import TypedValue, ReferenceType
  from sleepy.builtin_symbols import SLEEPY_INT
  context = make_test_context(emits_ir=False)

  ref_int = TypedValue(typ=ReferenceType(SLEEPY_INT), ir_val=None, num_unbindings=0)
  assert ref_int.num_possible_binds() == 1
  int = ref_int.copy_collapse(context=context, name='bind')
  assert int.type == SLEEPY_INT
  assert int.num_possible_binds() == 0
  assert int.num_unbindings == 0

  unbound_ref_int = ref_int.copy_unbind()
  assert unbound_ref_int.num_unbindings == 1
  assert unbound_ref_int.num_possible_binds() == 1

  ref_int_ = unbound_ref_int.copy_collapse(context=context, name='bind')
  assert ref_int_.type == ReferenceType(SLEEPY_INT)
  assert ref_int_.num_unbindings == 1
  assert ref_int_.num_possible_binds() == 1


# noinspection PyPep8Naming
def test_struct_self_referencing():
  from sleepy.symbols import StructType, PlaceholderTemplateType, PartialIdentifiedStructType, ReferenceType
  from sleepy.builtin_symbols import SLEEPY_INT
  context = make_test_context()

  struct_identity = StructIdentity("List", context=context)
  T = PlaceholderTemplateType(identifier="T")
  ListTPartial = PartialIdentifiedStructType(identity=struct_identity, templ_types=[T])
  ListT = StructType(
    identity=struct_identity, templ_types=[T], member_identifiers=["val", "next"],
    member_types=[T, ReferenceType(ListTPartial)], partial_struct_type=ListTPartial)
  assert_equal(ListT.member_types, [T, ReferenceType(ListT)])

  Int = SLEEPY_INT
  ListInt = ListT.replace_types({T: Int})
  assert isinstance(ListInt, StructType)
  assert_equal(ListInt.member_types, [Int, ReferenceType(ListInt)])
