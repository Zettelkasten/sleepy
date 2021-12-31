import _setup_test_env  # noqa

from nose.tools import assert_equal

from llvmlite import ir

from sleepy.builtin_symbols import SLEEPY_DOUBLE
from sleepy.syntactical_analysis.grammar import DummyPath
from sleepy.types import UnionType, SLEEPY_NEVER, StructIdentity, CodegenContext, ReferenceType, TypedValue, \
  narrow_with_collapsed_type, narrow_type


def make_test_context(emits_ir: bool = True) -> CodegenContext:
  module = ir.Module(name="test_module")
  builder = ir.IRBuilder() if emits_ir else None
  return CodegenContext(builder=builder, module=module, emits_debug=True, file_path=DummyPath("test"))


# noinspection PyPep8Naming
def test_can_implicit_cast_to():
  from sleepy.types import can_implicit_cast_to, ReferenceType, UnionType, StructType, PlaceholderTemplateType
  from sleepy.builtin_symbols import SLEEPY_INT, SLEEPY_DOUBLE
  context = make_test_context()
  assert_equal(can_implicit_cast_to(SLEEPY_INT, SLEEPY_DOUBLE), False)
  assert_equal(can_implicit_cast_to(SLEEPY_INT, SLEEPY_INT), True)
  assert_equal(can_implicit_cast_to(UnionType([SLEEPY_INT], [0], 8), SLEEPY_INT), True)
  assert_equal(
    can_implicit_cast_to(
      UnionType([SLEEPY_INT, SLEEPY_DOUBLE], [0, 1], 8), SLEEPY_DOUBLE), False)
  assert_equal(
    can_implicit_cast_to(
      SLEEPY_DOUBLE, UnionType([SLEEPY_INT, SLEEPY_DOUBLE], [0, 1], 8)), True)
  assert_equal(can_implicit_cast_to(ReferenceType(SLEEPY_INT), ReferenceType(SLEEPY_DOUBLE)), False)
  assert_equal(can_implicit_cast_to(ReferenceType(SLEEPY_INT), ReferenceType(SLEEPY_INT)), True)
  assert_equal(
    can_implicit_cast_to(
      ReferenceType(UnionType([SLEEPY_INT, SLEEPY_DOUBLE], [0, 1], 8)), ReferenceType(SLEEPY_INT)), False)
  assert_equal(
    can_implicit_cast_to(
      ReferenceType(SLEEPY_INT), ReferenceType(UnionType([SLEEPY_INT, SLEEPY_DOUBLE], [0, 1], 8))), False)
  T = PlaceholderTemplateType('T')
  List = StructType(
    identity=StructIdentity('List', context=context), template_param_or_arg=[T], member_identifiers=[], member_types=[])
  assert_equal(
    can_implicit_cast_to(
      ReferenceType(SLEEPY_INT), ReferenceType(UnionType([SLEEPY_INT, SLEEPY_DOUBLE], [0, 1], 8))), False)
  assert_equal(
    can_implicit_cast_to(
      ReferenceType(UnionType([SLEEPY_INT, SLEEPY_DOUBLE], [0, 1], 8)), ReferenceType(SLEEPY_INT)), False)
  assert_equal(
    can_implicit_cast_to(
      ReferenceType(UnionType([SLEEPY_INT, SLEEPY_DOUBLE], [0, 1], 8)),
      UnionType.from_types([ReferenceType(SLEEPY_INT), ReferenceType(SLEEPY_DOUBLE)])), False)
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
  from sleepy.types import narrow_type, UnionType
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
  from sleepy.types import narrow_type, UnionType, PlaceholderTemplateType, StructType
  from sleepy.builtin_symbols import SLEEPY_INT, SLEEPY_BOOL
  context = make_test_context()
  Int, Bool = SLEEPY_INT, SLEEPY_BOOL
  T = PlaceholderTemplateType('T')
  List = StructType(
    identity=StructIdentity('List', context=context), template_param_or_arg=[T], member_identifiers=[], member_types=[])
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
  from sleepy.types import narrow_type, UnionType, ReferenceType
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
  from sleepy.types import exclude_type, UnionType, ReferenceType
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
  from sleepy.types import get_common_type, UnionType
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
def test_narrow_with_collapsed_type():
  from sleepy.builtin_symbols import SLEEPY_INT, SLEEPY_DOUBLE
  Int = SLEEPY_INT
  RefInt = ReferenceType(Int)
  RefRefInt = ReferenceType(RefInt)
  Int_RefInt = UnionType.from_types([Int, RefInt])
  Double = SLEEPY_DOUBLE
  RefDouble = ReferenceType(Double)
  Int_Double = UnionType.from_types([Int, Double])
  RefDouble_RefInt = UnionType.from_types([RefDouble, RefInt])
  assert_equal(narrow_with_collapsed_type(Int, Int), Int)
  assert_equal(narrow_with_collapsed_type(RefInt, Int), RefInt)
  assert_equal(narrow_with_collapsed_type(RefRefInt, Int), RefRefInt)
  assert_equal(narrow_with_collapsed_type(Int_RefInt, Int), Int_RefInt)
  assert_equal(narrow_with_collapsed_type(RefInt, RefInt), RefInt)
  assert_equal(narrow_with_collapsed_type(Int, RefInt), SLEEPY_NEVER)
  assert_equal(narrow_with_collapsed_type(Int_RefInt, RefInt), narrow_type(Int_RefInt, RefInt))
  assert_equal(narrow_with_collapsed_type(Int_Double, Int), narrow_type(Int_Double, Int))
  assert_equal(narrow_with_collapsed_type(RefDouble_RefInt, Int), narrow_type(RefDouble_RefInt, RefInt))
  assert_equal(narrow_with_collapsed_type(ReferenceType(Int_Double), Int), ReferenceType(narrow_type(Int_Double, Int)))
  assert_equal(
    narrow_with_collapsed_type(ReferenceType(ReferenceType(Int_Double)), Int),
    ReferenceType(ReferenceType(narrow_type(Int_Double, Int))))
  assert_equal(
    narrow_with_collapsed_type(ReferenceType(ReferenceType(Int_Double)), ReferenceType(Int)),
    ReferenceType(ReferenceType(narrow_type(Int_Double, Int))))


# noinspection PyPep8Naming
def test_try_infer_templ_types_simple():
  from sleepy.types import try_infer_template_arguments, PlaceholderTemplateType
  from sleepy.builtin_symbols import SLEEPY_INT
  from sleepy.builtin_symbols import SLEEPY_DOUBLE
  T = PlaceholderTemplateType('T')
  U = PlaceholderTemplateType('U')
  assert_equal(try_infer_template_arguments(calling_types=[], signature_types=[], template_parameters=[]), [])
  assert_equal(
    try_infer_template_arguments(
      calling_types=[SLEEPY_INT, SLEEPY_DOUBLE], signature_types=[SLEEPY_INT, SLEEPY_DOUBLE], template_parameters=[]),
    [])
  assert_equal(
    try_infer_template_arguments(
      calling_types=[SLEEPY_INT, SLEEPY_DOUBLE], signature_types=[SLEEPY_INT, SLEEPY_DOUBLE],
      template_parameters=[T]),
    None)
  assert_equal(
    try_infer_template_arguments(
      calling_types=[SLEEPY_INT], signature_types=[T],
      template_parameters=[T]),
    [SLEEPY_INT])
  assert_equal(
    try_infer_template_arguments(
      calling_types=[SLEEPY_INT, SLEEPY_INT], signature_types=[T, T],
      template_parameters=[T]),
    [SLEEPY_INT])
  assert_equal(
    try_infer_template_arguments(
      calling_types=[SLEEPY_INT, SLEEPY_DOUBLE], signature_types=[T, SLEEPY_DOUBLE],
      template_parameters=[T]),
    [SLEEPY_INT])
  assert_equal(
    try_infer_template_arguments(
      calling_types=[SLEEPY_INT, SLEEPY_DOUBLE], signature_types=[SLEEPY_INT, T],
      template_parameters=[T]),
    [SLEEPY_DOUBLE])
  assert_equal(
    try_infer_template_arguments(
      calling_types=[SLEEPY_INT, SLEEPY_DOUBLE], signature_types=[T, U],
      template_parameters=[T, U]),
    [SLEEPY_INT, SLEEPY_DOUBLE])
  assert_equal(
    try_infer_template_arguments(
      calling_types=[SLEEPY_INT, SLEEPY_DOUBLE], signature_types=[T, U],
      template_parameters=[U, T]),
    [SLEEPY_DOUBLE, SLEEPY_INT])


# noinspection PyPep8Naming
def test_try_infer_template_arguments_ptr():
  from sleepy.types import try_infer_template_arguments, PlaceholderTemplateType, PointerType
  from sleepy.builtin_symbols import SLEEPY_CHAR
  from sleepy.builtin_symbols import SLEEPY_INT
  T = PlaceholderTemplateType('T')
  assert_equal(
    try_infer_template_arguments(
      calling_types=[PointerType(SLEEPY_INT)], signature_types=[PointerType(T)], template_parameters=[T]),
    [SLEEPY_INT])
  assert_equal(
    try_infer_template_arguments(
      calling_types=[PointerType(SLEEPY_CHAR), SLEEPY_INT], signature_types=[PointerType(T), SLEEPY_INT],
      template_parameters=[T]),
    [SLEEPY_CHAR])


def test_try_infer_templ_types_union():
  from sleepy.types import try_infer_template_arguments
  from sleepy.builtin_symbols import SLEEPY_CHAR
  from sleepy.builtin_symbols import SLEEPY_INT
  assert_equal(
    try_infer_template_arguments(
      calling_types=[SLEEPY_INT],
      signature_types=[UnionType.from_types([SLEEPY_INT, SLEEPY_CHAR])], template_parameters=[]),
    [])
  assert_equal(
    try_infer_template_arguments(
      calling_types=[SLEEPY_CHAR],
      signature_types=[UnionType.from_types([SLEEPY_INT, SLEEPY_CHAR])], template_parameters=[]),
    [])
  assert_equal(
    try_infer_template_arguments(
      calling_types=[UnionType.from_types([SLEEPY_INT, SLEEPY_CHAR])],
      signature_types=[UnionType.from_types([SLEEPY_INT, SLEEPY_CHAR])], template_parameters=[]),
    [])
  assert_equal(
    try_infer_template_arguments(
      calling_types=[UnionType.from_types([SLEEPY_INT, SLEEPY_CHAR])],
      signature_types=[UnionType.from_types([SLEEPY_CHAR, SLEEPY_INT])], template_parameters=[]),
    [])


# noinspection PyPep8Naming
def test_try_infer_templ_types_struct():
  from sleepy.types import try_infer_template_arguments, PlaceholderTemplateType, StructType
  from sleepy.builtin_symbols import SLEEPY_CHAR, SLEEPY_INT
  context = make_test_context()
  T = PlaceholderTemplateType('T')
  U = PlaceholderTemplateType('U')
  WrapperT = StructType(
    StructIdentity('Wrapper', context=context),
    template_param_or_arg=[T],
    member_identifiers=['value'],
    member_types=[T])
  WrapperU = WrapperT.replace_types({T: U})
  WrapperInt = WrapperT.replace_types({T: SLEEPY_INT})
  WrapperChar = WrapperT.replace_types({T: SLEEPY_CHAR})
  assert_equal(
    try_infer_template_arguments(
      calling_types=[WrapperInt], signature_types=[WrapperT], template_parameters=[T]),
    [SLEEPY_INT])
  assert_equal(
    try_infer_template_arguments(
      calling_types=[WrapperInt], signature_types=[T], template_parameters=[T]),
    [WrapperInt])
  assert_equal(
    try_infer_template_arguments(
      calling_types=[SLEEPY_INT], signature_types=[WrapperT], template_parameters=[T]),
    None)
  assert_equal(
    try_infer_template_arguments(
      calling_types=[WrapperInt, WrapperInt], signature_types=[WrapperT, WrapperT], template_parameters=[T]),
    [SLEEPY_INT])
  assert_equal(
    try_infer_template_arguments(
      calling_types=[WrapperInt, WrapperChar], signature_types=[WrapperT, WrapperU], template_parameters=[T, U]),
    [SLEEPY_INT, SLEEPY_CHAR])
  assert_equal(
    try_infer_template_arguments(
      calling_types=[WrapperInt, SLEEPY_INT], signature_types=[WrapperT, T], template_parameters=[T]),
    [SLEEPY_INT])
  assert_equal(
    try_infer_template_arguments(
      calling_types=[WrapperChar, SLEEPY_INT], signature_types=[WrapperT, T], template_parameters=[T]),
    None)
  assert_equal(
    try_infer_template_arguments(
      calling_types=[SLEEPY_INT, SLEEPY_INT], signature_types=[WrapperT, T], template_parameters=[T]),
    None)

  # with templates in calling_types
  assert_equal(
    try_infer_template_arguments(
      calling_types=[T, WrapperT], signature_types=[WrapperT, WrapperT], template_parameters=[T]),
    None)
  assert_equal(
    try_infer_template_arguments(
      calling_types=[WrapperT, T], signature_types=[WrapperT, WrapperT], template_parameters=[T]),
    None)
  assert_equal(
    try_infer_template_arguments(
      calling_types=[WrapperT, WrapperT], signature_types=[T, WrapperT], template_parameters=[T]),
    None)
  assert_equal(
    try_infer_template_arguments(
      calling_types=[WrapperT, WrapperT], signature_types=[WrapperT, T], template_parameters=[T]),
    None)


def test_context_use_pos():
  from sleepy.types import CodegenContext, make_di_location
  from sleepy.syntactical_analysis.grammar import TreePosition
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
  from sleepy.types import TypedValue, ReferenceType
  from sleepy.builtin_symbols import SLEEPY_INT
  context = make_test_context(emits_ir=False)

  ref_int = TypedValue(typ=ReferenceType(SLEEPY_INT), ir_val=None, num_unbindings=0)
  assert ref_int.num_possible_unbindings() == 1
  int = ref_int.copy_collapse(context=context, name='bind')
  assert int.type == SLEEPY_INT
  assert int.num_possible_unbindings() == 0
  assert int.num_unbindings == 0

  unbound_ref_int = ref_int.copy_unbind()
  assert unbound_ref_int.num_unbindings == 1
  assert unbound_ref_int.num_possible_unbindings() == 1

  ref_int_ = unbound_ref_int.copy_collapse(context=context, name='bind')
  assert ref_int_.type == ReferenceType(SLEEPY_INT)
  assert ref_int_.num_unbindings == 1
  assert ref_int_.num_possible_unbindings() == 1


# noinspection PyPep8Naming
def test_struct_self_referencing():
  from sleepy.types import StructType, PlaceholderTemplateType, PartialIdentifiedStructType, ReferenceType
  from sleepy.builtin_symbols import SLEEPY_INT
  context = make_test_context()

  struct_identity = StructIdentity("List", context=context)
  T = PlaceholderTemplateType(identifier="T")
  ListTPartial = PartialIdentifiedStructType(identity=struct_identity, template_param_or_arg=[T])
  ListT = StructType(
    identity=struct_identity, template_param_or_arg=[T], member_identifiers=["val", "next"],
    member_types=[T, ReferenceType(ListTPartial)], partial_struct_type=ListTPartial)
  assert_equal(ListT.member_types, [T, ReferenceType(ListT)])

  Int = SLEEPY_INT
  ListInt = ListT.replace_types({T: Int})
  assert isinstance(ListInt, StructType)
  assert_equal(ListInt.member_types, [Int, ReferenceType(ListInt)])


# noinspection PyPep8Naming
def test_copy_collapse():
  from sleepy.builtin_symbols import SLEEPY_INT
  Int = TypedValue(typ=SLEEPY_INT, num_unbindings=0, ir_val=None)
  RefInt = TypedValue(typ=ReferenceType(SLEEPY_INT), num_unbindings=0, ir_val=None)
  Int_RefInt = TypedValue(typ=UnionType.from_types([Int.type, RefInt.type]), num_unbindings=0, ir_val=None)
  RefInt_Int = TypedValue(typ=UnionType.from_types([RefInt.type, Int.type]), num_unbindings=0, ir_val=None)
  RefInt_Double = TypedValue(typ=UnionType.from_types([RefInt.type, SLEEPY_DOUBLE]), num_unbindings=0, ir_val=None)

  assert_equal(RefInt.collapsed_type(), Int.type)
  assert_equal(Int.collapsed_type(), Int.type)
  assert_equal(RefInt_Int.collapsed_type(), Int.type)
  assert_equal(Int_RefInt.collapsed_type(), Int.type)
  assert_equal(
    RefInt_Double.collapsed_type(),
    UnionType.from_types([SLEEPY_INT, SLEEPY_DOUBLE]))


def test_union_replace_types():
  from sleepy.builtin_symbols import SLEEPY_INT, SLEEPY_CHAR
  union = UnionType.from_types(possible_types=[SLEEPY_DOUBLE])
  replaced_union = union.replace_types({SLEEPY_DOUBLE: UnionType.from_types([SLEEPY_INT, SLEEPY_CHAR])})
  assert_equal(replaced_union, UnionType.from_types([SLEEPY_INT, SLEEPY_CHAR], val_size=union.val_size))
