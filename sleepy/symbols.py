"""
Implements a symbol table.
"""
from __future__ import annotations
import ctypes
from typing import Dict, Optional, List, Tuple, Set, Union, Callable

from llvmlite import ir
from llvmlite.binding import ExecutionEngine

LLVM_SIZE_TYPE = ir.types.IntType(64)
LLVM_VOID_POINTER_TYPE = ir.PointerType(ir.types.IntType(8))


class Symbol:
  """
  A declared symbol, with an identifier.
  """
  def __init__(self):
    """
    Initialize the symbol.
    """
    self.base = self  # type: Symbol


class Type:
  """
  A type of a declared variable.
  """
  def __init__(self, ir_type, pass_by_ref, c_type):
    """
    :param ir.types.Type|None ir_type: may be None if this is non-realizable (e.g. template types / void)
    :param bool pass_by_ref:
    :param ctypes._CData|None c_type: may be None if this is non-realizable (e.g. template types / void)
    """
    assert not pass_by_ref or isinstance(ir_type, ir.types.PointerType)
    self.ir_type = ir_type
    self.pass_by_ref = pass_by_ref
    self.c_type = c_type

  def __repr__(self):
    return self.__class__.__name__

  @property
  def size(self):
    """
    :rtype int:
    """
    return ctypes.sizeof(self.c_type)

  def is_pass_by_ref(self):
    """
    :rtype: bool
    """
    return self.pass_by_ref

  def make_ir_size(self):
    """
    :rtype: ir.values.Value
    """
    return ir.Constant(LLVM_SIZE_TYPE, self.size)

  def make_ir_alloca(self, context):
    """
    :param CodegenContext context:
    :rtype: ir.instructions.Instruction
    """
    assert context.emits_ir
    if self.is_pass_by_ref():  # use malloc
      assert context.ir_func_malloc is not None
      self_ir_alloca_raw = context.builder.call(
        context.ir_func_malloc, [self.make_ir_size()], name='self_raw_ptr')
      return context.builder.bitcast(self_ir_alloca_raw, self.ir_type, name='self')
    else:  # pass by value, use alloca
      return context.alloca_at_entry(self.ir_type, name='self')

  def is_realizable(self) -> bool:
    return True

  def replace_types(self, replacements: Dict[Type, Type]):
    if self in replacements:
      return replacements[self]
    return self

  def has_templ_placeholder(self) -> bool:
    return False


class VoidType(Type):
  """
  Type returned when nothing is returned.
  """
  def __init__(self):
    super().__init__(ir.VoidType(), pass_by_ref=False, c_type=None)


class DoubleType(Type):
  """
  A double.
  """
  def __init__(self):
    super().__init__(ir.DoubleType(), pass_by_ref=False, c_type=ctypes.c_double)


class FloatType(Type):
  """
  A double.
  """
  def __init__(self):
    super().__init__(ir.FloatType(), pass_by_ref=False, c_type=ctypes.c_float)


class BoolType(Type):
  """
  A 1-bit integer.
  """
  def __init__(self):
    super().__init__(ir.IntType(bits=1), pass_by_ref=False, c_type=ctypes.c_bool)


class IntType(Type):
  """
  An 32-bit integer.
  """
  def __init__(self):
    super().__init__(ir.IntType(bits=32), pass_by_ref=False, c_type=ctypes.c_int32)


class LongType(Type):
  """
  A 64-bit integer.
  """
  def __init__(self):
    super().__init__(ir.IntType(bits=64), pass_by_ref=False, c_type=ctypes.c_int64)


class CharType(Type):
  """
  An 8-bit character.
  """
  def __init__(self):
    super().__init__(ir.IntType(bits=8), pass_by_ref=False, c_type=ctypes.c_char)


class DoublePtrType(Type):
  """
  A double pointer.
  """
  def __init__(self):
    super().__init__(ir.PointerType(ir.DoubleType()), pass_by_ref=False, c_type=ctypes.POINTER(ctypes.c_double))


class FloatPtrType(Type):
  """
  A double pointer.
  """
  def __init__(self):
    super().__init__(ir.PointerType(ir.FloatType()), pass_by_ref=False, c_type=ctypes.POINTER(ctypes.c_float))


class CharPtrType(Type):
  """
  A char pointer.
  """
  def __init__(self):
    super().__init__(ir.PointerType(ir.IntType(bits=8)), pass_by_ref=False, c_type=ctypes.POINTER(ctypes.c_char))


class IntPtrType(Type):
  """
  An int pointer.
  """
  def __init__(self):
    super().__init__(ir.PointerType(ir.IntType(bits=32)), pass_by_ref=False, c_type=ctypes.POINTER(ctypes.c_int32))


class LongPtrType(Type):
  """
  A long pointer.
  """
  def __init__(self):
    super().__init__(ir.PointerType(ir.IntType(bits=64)), pass_by_ref=False, c_type=ctypes.POINTER(ctypes.c_int64))


class UnionType(Type):
  """
  A tagged union, i.e. a type that can be one of a set of different types.
  """
  def __init__(self, possible_types, possible_type_nums, val_size):
    """
    :param list[Type] possible_types:
    :param list[int] possible_type_nums:
    :param int val_size: size of untagged union in bytes
    """
    assert len(possible_types) == len(possible_type_nums)
    assert SLEEPY_VOID not in possible_types
    assert len(set(possible_types)) == len(possible_types)
    assert all(val_size >= possible_type.size for possible_type in possible_types)
    self.possible_types = possible_types
    self.possible_type_nums = possible_type_nums
    self.identifier = 'Union(%s)' % '_'.join(str(possible_type) for possible_type in possible_types)
    self.val_size = val_size

    self.tag_c_type = ctypes.c_uint8
    self.tag_ir_type = ir.types.IntType(8)
    # TODO: The C Type should match the IR type: Also make this an byte array.
    self.untagged_union_c_type = type(
      '%s_UntaggedCType' % self.identifier, (ctypes.Union,),
      {'_fields_': [('variant%s' % num, possible_type.c_type) for num, possible_type in enumerate(possible_types)]})
    self.untagged_union_ir_type = ir.types.ArrayType(ir.types.IntType(8), val_size)
    c_type = type(
      '%s_CType' % self.identifier, (ctypes.Structure,),
      {'_fields_': [('tag', self.tag_c_type), ('untagged_union', self.untagged_union_c_type)]})
    ir_type = ir.types.LiteralStructType([self.tag_ir_type, self.untagged_union_ir_type])
    super().__init__(ir_type, pass_by_ref=False, c_type=c_type)

  def __repr__(self):
    if len(self.possible_types) == 0:
      return 'NeverType'
    return 'UnionType(%s)' % ', '.join(
      '%s:%s' % (possible_type_num, possible_type)
      for possible_type_num, possible_type in zip(self.possible_type_nums, self.possible_types))

  def __eq__(self, other):
    """
    :param Any other:
    """
    if not isinstance(other, UnionType):
      return False
    self_types_dict = dict(zip(self.possible_types, self.possible_type_nums))
    other_types_dict = dict(zip(other.possible_types, other.possible_type_nums))
    return self_types_dict == other_types_dict and self.val_size == other.val_size

  def is_realizable(self):
    """
    :rtype: bool
    """
    return len(self.possible_types) > 0

  def replace_types(self, replacements: Dict[Type, Type]) -> UnionType:
    if len(self.possible_types) == 0:
      return self
    new_possible_types = [
      replacements.get(possible_type, possible_type) for possible_type in self.possible_types]
    new_possible_type_nums = self.possible_type_nums.copy()
    duplicate_indices = [
      index for index, possible_type in enumerate(new_possible_types)
      if possible_type in new_possible_types[:index]]
    for duplicate_index in reversed(duplicate_indices):
      del new_possible_types[duplicate_index]
      del new_possible_type_nums[duplicate_index]
    val_size = max(ctypes.sizeof(possible_type.c_type) for possible_type in new_possible_types)
    return UnionType(possible_types=new_possible_types, possible_type_nums=new_possible_type_nums, val_size=val_size)

  def has_templ_placeholder(self) -> bool:
    return any(possible_type.has_templ_placeholder() for possible_type in self.possible_types)

  def contains(self, contained_type):
    """
    :param Type contained_type:
    :rtype: bool
    """
    if isinstance(contained_type, UnionType):
      contained_possible_types = set(contained_type.possible_types)
    else:
      contained_possible_types = {contained_type}
    return contained_possible_types.issubset(self.possible_types)

  def get_variant_num(self, variant_type):
    """
    :param Type variant_type:
    :rtype: int
    """
    assert variant_type in self.possible_types
    return self.possible_type_nums[self.possible_types.index(variant_type)]

  def make_tag_ptr(self, union_ir_alloca, context, name):
    """
    :param ir.instructions.AllocaInstr union_ir_alloca:
    :param CodegenContext context:
    :rtype: ir.instructions.Instruction
    """
    assert context.emits_ir
    tag_gep_indices = (ir.Constant(ir.IntType(32), 0), ir.Constant(ir.IntType(32), 0))
    return context.builder.gep(union_ir_alloca, tag_gep_indices, name=name)

  def make_untagged_union_void_ptr(self, union_ir_alloca, context, name):
    """
    :param ir.instructions.AllocaInstr union_ir_alloca:
    :param CodegenContext context:
    :param str name:
    :rtype: ir.instructions.Instruction
    """
    assert context.emits_ir
    untagged_union_gep_indices = (ir.Constant(ir.IntType(32), 0), ir.Constant(ir.IntType(32), 1))
    return context.builder.gep(union_ir_alloca, untagged_union_gep_indices, name=name)

  def make_untagged_union_ptr(self, union_ir_alloca, variant_type, context, name):
    """
    :param ir.instructions.AllocaInstr union_ir_alloca:
    :param Type variant_type:
    :param CodegenContext context:
    :param str name:
    :rtype: ir.instructions.Instruction
    """
    assert variant_type in self.possible_types
    untagged_union_ptr = self.make_untagged_union_void_ptr(
      union_ir_alloca=union_ir_alloca, context=context, name='%s_raw' % name)
    return context.builder.bitcast(
      untagged_union_ptr, ir.types.PointerType(variant_type.ir_type),
      name=name)

  def make_extract_tag(self, union_ir_val, context, name):
    """
    :param ir.values.Value union_ir_val:
    :param CodegenContext context:
    :param str name:
    :rtype: ir.values.Value
    """
    return context.builder.extract_value(union_ir_val, 0, name=name)

  def make_extract_void_val(self, union_ir_val, context, name):
    """
    :param ir.values.Value union_ir_val:
    :param CodegenContext context:
    :param str name:
    :rtype: ir.values.Value
    """
    assert context.emits_ir
    return context.builder.extract_value(union_ir_val, idx=1, name=name)

  def make_extract_val(self, union_ir_val, variant_type, context, name):
    """
    :param ir.values.Value union_ir_val:
    :param Type variant_type:
    :param CodegenContext context:
    :param str name:
    :rtype: ir.values.Value
    """
    assert context.emits_ir
    assert variant_type in self.possible_types
    union_ir_alloca = context.alloca_at_entry(self.ir_type, name='%s_ptr' % name)
    context.builder.store(union_ir_val, union_ir_alloca)
    untagged_union_ptr = self.make_untagged_union_ptr(
      union_ir_alloca, variant_type=variant_type, context=context, name='%s_val_ptr' % name)
    return context.builder.load(untagged_union_ptr, name=name)

  def copy_with_narrowed_types(self, narrow_to_types):
    """
    :param list[Type] narrow_to_types:
    :rtype: UnionType
    """
    possible_types = [
      possible_type for possible_type in self.possible_types if possible_type in narrow_to_types]
    possible_type_nums = [
      possible_type_num
      for possible_type, possible_type_num in zip(self.possible_types, self.possible_type_nums)
      if possible_type in narrow_to_types]
    return UnionType(possible_types=possible_types, possible_type_nums=possible_type_nums, val_size=self.val_size)

  def copy_with_extended_types(self, extended_types, extended_type_nums=None):
    """
    :param list[Type] extended_types:
    :param None|list[int|None] extended_type_nums: suggestions for new extended type nums
    :rtype: UnionType
    """
    assert all(not isinstance(extended_type, UnionType) for extended_type in extended_types)
    if extended_type_nums is None:
      extended_type_nums = [None] * len(extended_types)
    assert len(extended_types) == len(extended_type_nums)
    new_possible_types = self.possible_types.copy()
    new_possible_type_nums = self.possible_type_nums.copy()
    for extended_type, extended_type_num in zip(extended_types, extended_type_nums):
      if extended_type in new_possible_types:
        continue
      new_possible_types.append(extended_type)
      if extended_type_num is not None and extended_type_num not in new_possible_type_nums:
        new_possible_type_nums.append(extended_type_num)
      else:
        next_type_num = max(new_possible_type_nums) + 1 if len(new_possible_type_nums) > 0 else 0
        new_possible_type_nums.append(next_type_num)
    new_val_size = max([self.val_size] + [extended_type.size for extended_type in extended_types])
    return UnionType(
      possible_types=new_possible_types, possible_type_nums=new_possible_type_nums, val_size=new_val_size)

  @classmethod
  def from_types(cls, possible_types):
    """
    :param list[Type] possible_types:
    :rtype: UnionType
    """
    possible_types = list(dict.fromkeys(possible_types))  # possibly remove duplicates
    possible_type_nums = list(range(len(possible_types)))
    val_size = max(ctypes.sizeof(possible_type.c_type) for possible_type in possible_types)
    return UnionType(possible_types=possible_types, possible_type_nums=possible_type_nums, val_size=val_size)


class StructType(Type):
  """
  A struct.
  """
  def __init__(self, struct_identifier: str, templ_types: List[Type], member_identifiers: List[str],
               member_types: List[Type], member_mutables: List[bool], pass_by_ref: bool):
    assert len(member_identifiers) == len(member_types) == len(member_mutables)
    member_ir_types = [member_type.ir_type for member_type in member_types]
    ir_val_type = ir.types.LiteralStructType(member_ir_types)
    member_c_types = [
      (member_identifier, member_type.c_type)
      for member_identifier, member_type in zip(member_identifiers, member_types)]
    c_type = type('%s_CType' % struct_identifier, (ctypes.Structure,), {'_fields_': member_c_types})
    if pass_by_ref:
      super().__init__(ir.types.PointerType(ir_val_type), pass_by_ref=True, c_type=ctypes.POINTER(c_type))
    else:
      super().__init__(ir_val_type, pass_by_ref=False, c_type=c_type)
    self.struct_identifier = struct_identifier
    self.templ_types = templ_types
    self.member_identifiers = member_identifiers
    self.member_types = member_types
    self.member_mutables = member_mutables

  def get_member_num(self, member_identifier):
    """
    :param str member_identifier:
    :rtype: int
    """
    assert member_identifier in self.member_identifiers
    return self.member_identifiers.index(member_identifier)

  def replace_types(self, replacements: Dict[Type, Type]) -> StructType:
    new_templ_types = [replacements.get(templ_type, templ_type) for templ_type in self.templ_types]
    new_member_types = [replacements.get(member_type, member_type) for member_type in self.member_types]
    return StructType(
      struct_identifier=self.struct_identifier, templ_types=new_templ_types, member_identifiers=self.member_identifiers,
      member_types=new_member_types, member_mutables=self.member_mutables, pass_by_ref=self.pass_by_ref)

  def has_templ_placeholder(self) -> bool:
    return any(templ_type.has_templ_placeholder() for templ_type in self.templ_types)

  def __eq__(self, other) -> bool:
    if not isinstance(other, StructType):
      return False
    return (
      self.struct_identifier == other.struct_identifier and self.templ_types == other.templ_types and
      self.member_identifiers == other.member_identifiers and self.member_types == other.member_types and
      self.member_mutables == other.member_mutables and self.pass_by_ref == other.pass_by_ref)

  def __hash__(self) -> int:
    return hash((self.__class__, self.struct_identifier, tuple(self.templ_types), tuple(self.member_identifiers),
    tuple(self.member_types), tuple(self.member_mutables), self.pass_by_ref))

  def __repr__(self):
    """
    :rtype: str
    """
    return '%sType' % self.struct_identifier

  def make_extract_member_val_ir(self, member_identifier, struct_ir_val, context):
    """
    :param str member_identifier:
    :param ir.values.Value struct_ir_val:
    :param CodegenContext context:
    """
    if self.is_pass_by_ref():
      member_num = self.get_member_num(member_identifier)
      gep_indices = (ir.Constant(ir.IntType(32), 0), ir.Constant(ir.IntType(32), member_num))
      member_ptr = context.builder.gep(struct_ir_val, gep_indices, name='member_ptr_%s' % member_identifier)
      return context.builder.load(member_ptr, name='member_%s' % member_identifier)
    else:  # pass by value
      return context.builder.extract_value(
        struct_ir_val, self.get_member_num(member_identifier), name='member_%s' % member_identifier)

  def make_store_members_ir(self, member_ir_vals, struct_ir_alloca, context):
    """
    :param list[ir.values.Value] member_ir_vals:
    :param ir.instructions.Instruction struct_ir_alloca:
    :param CodegenContext context:
    """
    assert len(member_ir_vals) == len(self.member_identifiers)
    assert context.emits_ir
    for member_num, (member_identifier, ir_func_arg) in enumerate(zip(self.member_identifiers, member_ir_vals)):
      gep_indices = (ir.Constant(ir.IntType(32), 0), ir.Constant(ir.IntType(32), member_num))
      member_ptr = context.builder.gep(struct_ir_alloca, gep_indices, name='%s_ptr' % member_identifier)
      context.builder.store(ir_func_arg, member_ptr)

  def build_constructor(self, parent_symbol_table: SymbolTable, parent_context: CodegenContext) -> FunctionSymbol:
    class ConstructorConcreteFuncFactor(ConcreteFunctionFactory):
      def build_concrete_func_ir(self_, concrete_func: ConcreteFunction):
        concrete_struct_type = concrete_func.return_type
        assert isinstance(concrete_struct_type, StructType)

        if parent_context.emits_ir:
          ir_func_name = parent_symbol_table.make_ir_func_name(
            self.struct_identifier, extern=False, concrete_func=concrete_func)
          concrete_func.ir_func = ir.Function(
            parent_context.module, concrete_func.make_ir_function_type(), name=ir_func_name)
          constructor_block = concrete_func.ir_func.append_basic_block(name='entry')
          context = parent_context.copy_with_builder(ir.IRBuilder(constructor_block))
          self_ir_alloca = concrete_struct_type.make_ir_alloca(context=context)

          for member_identifier, ir_func_arg in zip(self.member_identifiers, concrete_func.ir_func.args):
            ir_func_arg.identifier = member_identifier
          concrete_struct_type.make_store_members_ir(
            member_ir_vals=concrete_func.ir_func.args, struct_ir_alloca=self_ir_alloca, context=context)

          if concrete_struct_type.is_pass_by_ref():
            context.builder.ret(self_ir_alloca)
          else:  # pass by value
            context.builder.ret(context.builder.load(self_ir_alloca, name='self'))

        return concrete_func

    constructor_symbol = FunctionSymbol(returns_void=False)
    placeholder_templ_types = [templ_type for templ_type in self.templ_types if isinstance(templ_type, TemplateType)]
    signature_ = FunctionSignature(
      concrete_func_factory=ConstructorConcreteFuncFactor(),
      placeholder_templ_types=placeholder_templ_types, return_type=self, return_mutable=True,
      arg_identifiers=self.member_identifiers, arg_types=self.member_types, arg_mutables=self.member_mutables,
      arg_type_narrowings=self.member_types, is_inline=False)
    constructor_symbol.add_signature(signature_)
    return constructor_symbol

  def build_destructor(self, parent_symbol_table: SymbolTable, parent_context: CodegenContext) -> FunctionSymbol:
    destructor_symbol = parent_symbol_table['free']
    assert isinstance(destructor_symbol, FunctionSymbol)

    class DestructorConcreteFuncFactory(ConcreteFunctionFactory):
      def build_concrete_func_ir(self_, concrete_func: ConcreteFunction):
        assert len(concrete_func.arg_types) == 1
        concrete_struct_type = concrete_func.arg_types[0]
        assert isinstance(concrete_struct_type, StructType)
        if parent_context.emits_ir:
          ir_func_name = parent_symbol_table.make_ir_func_name('free', extern=False, concrete_func=concrete_func)
          concrete_func.ir_func = ir.Function(
            parent_context.module, concrete_func.make_ir_function_type(), name=ir_func_name)
          destructor_block = concrete_func.ir_func.append_basic_block(name='entry')
          context = parent_context.copy_with_builder(ir.IRBuilder(destructor_block))

          assert len(concrete_func.ir_func.args) == 1
          self_ir_alloca = concrete_func.ir_func.args[0]
          self_ir_alloca.identifier = 'self_ptr'
          # Free members in reversed order
          for member_num in reversed(range(len(self.member_identifiers))):
            member_identifier = self.member_identifiers[member_num]
            signature_member_type = self.member_types[member_num]
            concrete_member_type = concrete_struct_type.member_types[member_num]
            member_ir_val = self.make_extract_member_val_ir(
              member_identifier, struct_ir_val=self_ir_alloca, context=context)
            assert not (isinstance(signature_member_type, StructType) and len(signature_member_type.templ_types) > 0), (
              'not implemented yet')
            make_func_call_ir(
              func_identifier='free', func_symbol=destructor_symbol, templ_types=[],
              calling_arg_types=[signature_member_type],
              calling_ir_args=[member_ir_val], context=context)
          if self.is_pass_by_ref():
            assert context.ir_func_free is not None
            self_ir_alloca_raw = context.builder.bitcast(self_ir_alloca, LLVM_VOID_POINTER_TYPE, name='self_raw_ptr')
            context.builder.call(context.ir_func_free, args=[self_ir_alloca_raw], name='free_self')
          context.builder.ret_void()

    placeholder_templ_types = [templ_type for templ_type in self.templ_types if isinstance(templ_type, TemplateType)]
    # TODO: Narrow type to something more meaningful then SLEEPY_NEVER
    # E.g. make a copy of the never union type and give that a name ("Freed" or sth)
    signature_ = FunctionSignature(
      concrete_func_factory=DestructorConcreteFuncFactory(),
      placeholder_templ_types=placeholder_templ_types, return_type=SLEEPY_VOID, return_mutable=False,
      arg_types=[self], arg_identifiers=['var'], arg_type_narrowings=[SLEEPY_NEVER],
      arg_mutables=[False])
    destructor_symbol.add_signature(signature_)
    return destructor_symbol


class TemplateType(Type):
  """
  A template parameter.
  """
  def __init__(self, identifier: str):
    super().__init__(ir_type=None, pass_by_ref=False, c_type=None)
    self.identifier = identifier

  def has_templ_placeholder(self) -> bool:
    return True


SLEEPY_VOID = VoidType()
SLEEPY_NEVER = UnionType(possible_types=[], possible_type_nums=[], val_size=0)
SLEEPY_DOUBLE = DoubleType()
SLEEPY_FLOAT = FloatType()
SLEEPY_BOOL = BoolType()
SLEEPY_INT = IntType()
SLEEPY_LONG = LongType()
SLEEPY_CHAR = CharType()
SLEEPY_DOUBLE_PTR = DoublePtrType()
SLEEPY_FLOAT_PTR = FloatPtrType()
SLEEPY_CHAR_PTR = CharPtrType()
SLEEPY_INT_PTR = IntPtrType()
SLEEPY_LONG_PTR = LongPtrType()

SLEEPY_TYPES = {
  'Void': SLEEPY_VOID, 'Double': SLEEPY_DOUBLE, 'Float': SLEEPY_FLOAT, 'Bool': SLEEPY_BOOL, 'Int': SLEEPY_INT,
  'Long': SLEEPY_LONG, 'Char': SLEEPY_CHAR,
  'DoublePtr': SLEEPY_DOUBLE_PTR, 'FloatPtr': SLEEPY_FLOAT_PTR, 'CharPtr': SLEEPY_CHAR_PTR,
  'IntPtr': SLEEPY_INT_PTR, 'LongPtr': SLEEPY_LONG_PTR}  # type: Dict[str, Type]
SLEEPY_NUMERICAL_TYPES = {SLEEPY_DOUBLE, SLEEPY_FLOAT, SLEEPY_INT, SLEEPY_LONG}
SLEEPY_POINTER_TYPES = {SLEEPY_DOUBLE_PTR, SLEEPY_FLOAT_PTR, SLEEPY_CHAR_PTR, SLEEPY_INT_PTR, SLEEPY_LONG_PTR}


def can_implicit_cast_to(from_type, to_type):
  """
  :param Type from_type:
  :param Type to_type:
  :rtype bool:
  """
  if from_type == to_type:
    return True
  if from_type == SLEEPY_VOID or to_type == SLEEPY_VOID:
    return False
  if isinstance(from_type, UnionType):
    possible_from_types = from_type.possible_types
  else:
    possible_from_types = [from_type]
  if isinstance(to_type, UnionType):
    possible_to_types = to_type.possible_types
  else:
    possible_to_types = [to_type]
  return all(possible_from_type in possible_to_types for possible_from_type in possible_from_types)


def make_implicit_cast_to_ir_val(from_type, to_type, from_ir_val, context, name):
  """
  :param Type from_type:
  :param Type to_type:
  :param ir.values.Value from_ir_val:
  :param CodegenContext context:
  :param str name:
  :rtype: ir.values.Value
  """
  assert context.emits_ir
  assert can_implicit_cast_to(from_type, to_type)
  if from_type == to_type:
    return from_ir_val
  if isinstance(to_type, UnionType):
    to_ir_alloca = context.alloca_at_entry(to_type.ir_type, name='%s_ptr' % name)
    if isinstance(from_type, UnionType):
      tag_mapping_ir_type = ir.types.VectorType(to_type.tag_ir_type, max(from_type.possible_type_nums) + 1)
      tag_mapping = [-1] * (max(from_type.possible_type_nums) + 1)
      for from_variant_num, from_variant_type in zip(from_type.possible_type_nums, from_type.possible_types):
        assert to_type.contains(from_variant_type)
        tag_mapping[from_variant_num] = to_type.get_variant_num(from_variant_type)
      ir_tag_mapping = ir.values.Constant(tag_mapping_ir_type, tag_mapping_ir_type.wrap_constant_value(tag_mapping))
      ir_from_tag = from_type.make_extract_tag(from_ir_val, context=context, name='%s_from_tag' % name)
      ir_to_tag = context.builder.extract_element(ir_tag_mapping, ir_from_tag, name='%s_to_tag' % name)
      context.builder.store(
        ir_to_tag, to_type.make_tag_ptr(to_ir_alloca, context=context, name='%s_tag_ptr' % name))
      assert to_type.val_size >= from_type.val_size
      ir_from_untagged_union = from_type.make_extract_void_val(from_ir_val, context=context, name='%s_from_val' % name)
      ir_to_untagged_union_ptr = to_type.make_untagged_union_void_ptr(
        to_ir_alloca, context=context, name='%s_to_val_raw' % name)
      ir_to_untagged_union_ptr_casted = context.builder.bitcast(
        ir_to_untagged_union_ptr, ir.PointerType(to_type.untagged_union_ir_type), name='%s_to_val' % name)
      context.builder.store(ir_from_untagged_union, ir_to_untagged_union_ptr_casted)
    else:
      assert not isinstance(from_type, UnionType)
      ir_to_tag = ir.Constant(to_type.tag_ir_type, to_type.get_variant_num(from_type))
      context.builder.store(
        ir_to_tag, to_type.make_tag_ptr(to_ir_alloca, context=context, name='%s_tag_ptr' % name))
      context.builder.store(
        from_ir_val,
        to_type.make_untagged_union_ptr(to_ir_alloca, from_type, context=context, name='%s_val_ptr' % name))
    return context.builder.load(to_ir_alloca, name=name)
  else:
    assert not isinstance(to_type, UnionType)
    # this is only possible when from_type is a single-type union
    assert isinstance(from_type, UnionType)
    assert all(possible_from_type == to_type for possible_from_type in from_type.possible_types)
    return from_type.make_extract_val(from_ir_val, to_type, context=context, name=name)


def narrow_type(from_type, narrow_to):
  """
  :param Type from_type:
  :param Type narrow_to:
  :rtype: Type
  """
  if from_type == narrow_to:
    return from_type
  if isinstance(narrow_to, UnionType):
    narrow_to_types = narrow_to.possible_types
  else:
    narrow_to_types = [narrow_to]
  if not isinstance(from_type, UnionType):
    if from_type in narrow_to_types:
      return from_type
    else:
      return SLEEPY_NEVER
  return from_type.copy_with_narrowed_types(narrow_to_types=narrow_to_types)


def exclude_type(from_type, excluded_type):
  """
  :param Type from_type:
  :param Type excluded_type:
  :rtype: Type
  """
  if from_type == excluded_type:
    return SLEEPY_NEVER
  if excluded_type == SLEEPY_NEVER:
    return from_type
  if isinstance(excluded_type, UnionType):
    excluded_types = excluded_type.possible_types
  else:
    excluded_types = [excluded_type]
  if not isinstance(from_type, UnionType):
    if from_type in excluded_types:
      return SLEEPY_NEVER
    else:
      return from_type
  narrow_to_types = [possible_type for possible_type in from_type.possible_types if possible_type not in excluded_types]
  return from_type.copy_with_narrowed_types(narrow_to_types=narrow_to_types)


def get_common_type(possible_types):
  """
  :param list[Type] possible_types:
  :rtype: Type
  """
  assert len(possible_types) >= 1
  common_type = possible_types[0]
  for i, other_type in enumerate(possible_types):
    if i == 0 or other_type in possible_types[:i]:
      continue
    if isinstance(common_type, UnionType):
      if isinstance(other_type, UnionType):
        common_type = common_type.copy_with_extended_types(
          extended_types=other_type.possible_types, extended_type_nums=other_type.possible_type_nums)
      else:
        common_type = common_type.copy_with_extended_types(extended_types=[other_type])
    else:
      if isinstance(other_type, UnionType):
        common_type = other_type.copy_with_extended_types(extended_types=[common_type])
      else:
        common_type = UnionType.from_types(possible_types=[common_type, other_type])
  return common_type


def make_ir_val_is_type(ir_val, known_type, check_type, context):
  """
  :param ir.values.Value ir_val:
  :param Type known_type:
  :param Type check_type:
  :param CodegenContext context:
  :rtype: ir.values.Value
  """
  assert context.emits_ir
  if known_type == check_type:
    return ir.Constant(ir.IntType(1), True)
  if not isinstance(known_type, UnionType):
    return ir.Constant(ir.IntType(1), False)
  if not known_type.contains(check_type):
    return ir.Constant(ir.IntType(1), False)
  assert not isinstance(check_type, UnionType), 'not implemented yet'
  union_tag = context.builder.extract_value(ir_val, 0, name='tmp_is_val')
  cmp_val = context.builder.icmp_signed(
    '==', union_tag, ir.Constant(known_type.tag_ir_type, known_type.get_variant_num(check_type)), name='tmp_is_check')
  return cmp_val


class VariableSymbol(Symbol):
  """
  A declared variable.
  """
  def __init__(self, ir_alloca, var_type, mutable):
    """
    :param ir.instructions.AllocaInstr|None ir_alloca:
    :param Type var_type:
    :param bool mutable:
    """
    super().__init__()
    assert ir_alloca is None or isinstance(ir_alloca, ir.instructions.AllocaInstr)
    assert var_type != SLEEPY_VOID
    self.ir_alloca = ir_alloca
    self.declared_var_type = var_type
    self.narrowed_var_type = var_type
    self.mutable = mutable

  def copy_with_narrowed_type(self, narrow_to):
    """
    :param Type narrow_to:
    :rtype: VariableSymbol
    """
    new_var_symbol = VariableSymbol(self.ir_alloca, self.declared_var_type, self.mutable)
    new_var_symbol.base = self if self.base is None else self.base
    new_var_symbol.narrowed_var_type = narrow_type(self.narrowed_var_type, narrow_to)
    return new_var_symbol

  def copy_reset_narrowed_type(self):
    """
    :rtype: VariableSymbol
    """
    new_var_symbol = VariableSymbol(self.ir_alloca, self.declared_var_type, self.mutable)
    new_var_symbol.base = self if self.base is None else self.base
    new_var_symbol.narrowed_var_type = self.declared_var_type
    return new_var_symbol

  def copy_with_excluded_type(self, excluded):
    """
    :param Type excluded:
    :rtype: VariableSymbol
    """
    new_var_symbol = VariableSymbol(self.ir_alloca, self.declared_var_type, self.mutable)
    new_var_symbol.base = self if self.base is None else self.base
    new_var_symbol.narrowed_var_type = exclude_type(self.narrowed_var_type, excluded)
    return new_var_symbol

  def build_ir_alloca(self, context, identifier):
    """
    :param CodegenContext context:
    :param str identifier:
    """
    assert self.ir_alloca is None
    if not context.emits_ir:
      return
    self.ir_alloca = context.alloca_at_entry(self.declared_var_type.ir_type, name='%s_ptr' % identifier)

  def __repr__(self):
    """
    :rtype: str
    """
    return 'VariableSymbol(ir_alloca=%r, declared_var_type=%r, narrowed_var_type=%r, mutable=%r)' % (
      self.ir_alloca, self.declared_var_type, self.narrowed_var_type, self.mutable)


class ConcreteFunction:
  """
  An actual function implementation.
  """
  def __init__(self,
               signature: FunctionSignature,
               ir_func: Optional[ir.Function],
               make_inline_func_call_ir_callback: Optional[Callable[[List[ir.values.Value], CodegenContext], Optional[ir.values.Value]]],  # noqa
               concrete_templ_types: List[Type], return_type: Type,
               arg_types: List[Type], arg_type_narrowings: List[Type]):
    """
    :param signature:
    :param ir_func:
    :param make_inline_func_call_ir_callback: ir_func_args + caller_context -> return value
    :param concrete_templ_types:
    :param return_type:
    :param arg_types:
    :param arg_type_narrowings:
    """
    assert ir_func is None or isinstance(ir_func, ir.Function)
    assert make_inline_func_call_ir_callback is None or callable(make_inline_func_call_ir_callback)
    assert len(signature.arg_identifiers) == len(arg_types) == len(arg_type_narrowings)
    assert all(not templ_type.has_templ_placeholder() for templ_type in concrete_templ_types)
    assert not return_type.has_templ_placeholder()
    assert all(not arg_type.has_templ_placeholder() for arg_type in arg_types)
    assert all(not arg_type.has_templ_placeholder() for arg_type in arg_type_narrowings)
    self.signature = signature
    self.make_inline_func_call_ir = make_inline_func_call_ir_callback
    self.ir_func = ir_func
    self.concrete_templ_types = concrete_templ_types
    self.return_type = return_type
    self.arg_types = arg_types
    self.arg_type_narrowings = arg_type_narrowings

  def get_c_arg_types(self) -> Tuple[Callable]:
    return (self.return_type.c_type,) + tuple(arg_type.c_type for arg_type in self.arg_types)

  def make_ir_function_type(self) -> ir.FunctionType:
    return ir.FunctionType(self.return_type.ir_type, [arg_type.ir_type for arg_type in self.arg_types])

  def make_py_func(self, engine: ExecutionEngine) -> Callable:
    assert self.ir_func is not None
    from sleepy.jit import get_func_address
    main_func_ptr = get_func_address(engine, self.ir_func)
    py_func = ctypes.CFUNCTYPE(
      self.return_type.c_type, *[arg_type.c_type for arg_type in self.arg_types])(main_func_ptr)
    assert callable(py_func)
    return py_func

  @property
  def arg_identifiers(self) -> List[str]:
    return self.signature.arg_identifiers

  @property
  def arg_mutables(self) -> List[bool]:
    return self.signature.arg_mutables

  @property
  def return_mutable(self) -> bool:
    return self.signature.return_mutable

  @property
  def is_inline(self) -> bool:
    return self.signature.is_inline

  def __repr__(self) -> str:
    return (
      'ConcreteFunction(signature=%r, concrete_templ_types=%r, return_type=%r, arg_types=%r, '
      'arg_type_narrowings=%r)' % (
        self.signature, self.concrete_templ_types, self.return_type, self.arg_types, self.arg_type_narrowings))

  def has_same_signature_as(self, other: ConcreteFunction) -> bool:
    return (
      self.return_type == other.return_type and self.signature.return_mutable == other.signature.return_mutable and
      self.arg_types == other.arg_types and self.signature.arg_mutables == other.signature.arg_mutables and
      self.arg_type_narrowings == other.arg_type_narrowings)


class ConcreteFunctionFactory:
  """
  Generates concrete function implementations.
  """
  def build_concrete_func_ir(self, concrete_func: ConcreteFunction):
    raise NotImplementedError()


class FunctionSignature:
  """
  Given template arguments, this builds a concrete function implementation on demand.
  """
  def __init__(self,
               concrete_func_factory: ConcreteFunctionFactory,
               placeholder_templ_types: List[TemplateType],
               return_type: Type, return_mutable: bool,
               arg_identifiers: List[str], arg_types: List[Type], arg_mutables: List[bool],
               arg_type_narrowings: List[Type],
               is_inline: bool = False):
    assert isinstance(return_type, Type)
    assert len(arg_identifiers) == len(arg_types) == len(arg_mutables) == len(arg_type_narrowings)
    self.concrete_func_factory = concrete_func_factory
    self.placeholder_templ_types = placeholder_templ_types
    self.return_type = return_type
    self.return_mutable = return_mutable
    self.arg_identifiers = arg_identifiers
    self.arg_types = arg_types
    self.arg_mutables = arg_mutables
    self.arg_type_narrowings = arg_type_narrowings
    self.is_inline = is_inline
    self.initialized_templ_funcs: Dict[Tuple[Type], ConcreteFunction] = {}

  def to_signature_str(self) -> str:
    templ_args = '' if len(self.placeholder_templ_types) == 0 else '[%s]' % (
      ', '.join([templ_type.identifier for templ_type in self.placeholder_templ_types]))
    args = ', '.join(['%s %s' % arg_tuple for arg_tuple in zip(self.arg_types, self.arg_identifiers)])
    return '%s(%s) -> %s' % (templ_args, args, self.return_type)

  def get_concrete_func(self, concrete_templ_types: List[Type]) -> ConcreteFunction:
    assert len(concrete_templ_types) == len(self.placeholder_templ_types)
    concrete_templ_types = tuple(concrete_templ_types)
    if concrete_templ_types in self.initialized_templ_funcs:
      return self.initialized_templ_funcs[concrete_templ_types]
    concrete_type_replacements = dict(zip(self.placeholder_templ_types, concrete_templ_types))
    concrete_return_type = self.return_type.replace_types(replacements=concrete_type_replacements)
    concrete_arg_types = [
      arg_type.replace_types(replacements=concrete_type_replacements) for arg_type in self.arg_types]
    concrete_arg_types_narrowings = [
      arg_type.replace_types(replacements=concrete_type_replacements) for arg_type in self.arg_type_narrowings]
    concrete_func = ConcreteFunction(
      signature=self, ir_func=None, make_inline_func_call_ir_callback=None,
      concrete_templ_types=list(concrete_templ_types), return_type=concrete_return_type, arg_types=concrete_arg_types,
      arg_type_narrowings=concrete_arg_types_narrowings)
    self.initialized_templ_funcs[concrete_templ_types] = concrete_func
    self.concrete_func_factory.build_concrete_func_ir(concrete_func=concrete_func)
    return concrete_func

  @property
  def returns_void(self):
    """
    :rtype: bool
    """
    return self.return_type == SLEEPY_VOID

  def can_call_with_expanded_arg_types(self, concrete_templ_types: List[Type], expanded_arg_types: List[Type]):
    assert all(not isinstance(arg_type, UnionType) for arg_type in expanded_arg_types)
    assert all(not templ_type.has_templ_placeholder() for templ_type in concrete_templ_types)
    assert all(not arg_type.has_templ_placeholder() for arg_type in expanded_arg_types)
    if len(concrete_templ_types) != len(self.placeholder_templ_types):
      return False
    if len(expanded_arg_types) != len(self.arg_types):
      return False
    replacements = dict(zip(self.placeholder_templ_types, concrete_templ_types))
    concrete_arg_types = [arg_type.replace_types(replacements=replacements) for arg_type in self.arg_types]
    assert all(not arg_type.has_templ_placeholder() for arg_type in concrete_arg_types)
    return all(
      can_implicit_cast_to(call_type, arg_type) for call_type, arg_type in zip(expanded_arg_types, concrete_arg_types))

  def __repr__(self) -> str:
    return 'FunctionSignature(placeholder_templ_types=%r, return_type=%r, arg_identifiers=%r, arg_types=%r)' % (
      self.placeholder_templ_types, self.return_type, self.arg_identifiers, self.arg_types)


class FunctionSymbol(Symbol):
  """
  A set of declared overloaded function signatures with the same name.
  Can have one or multiple overloaded signatures accepting different parameter types (FunctionSignature).
  Each of these signatures itself can have a set of concrete implementations,
  where template types have been replaced with concrete types.
  """
  def __init__(self, returns_void: bool):
    super().__init__()
    self.signatures_by_number_of_templ_args: Dict[int, List[FunctionSignature]] = {}
    self.returns_void = returns_void
    
  @property
  def signatures(self) -> List[FunctionSignature]:
    return [signature for signatures in self.signatures_by_number_of_templ_args.values() for signature in signatures]

  def can_call_with_expanded_arg_types(self, concrete_templ_types: List[Type], expanded_arg_types: List[Type]):
    assert all(not isinstance(arg_type, UnionType) for arg_type in expanded_arg_types)
    assert all(not templ_type.has_templ_placeholder() for templ_type in concrete_templ_types)
    assert all(not arg_type.has_templ_placeholder() for arg_type in expanded_arg_types)
    signatures = self.signatures_by_number_of_templ_args.get(len(concrete_templ_types), [])
    return any(
      signature.can_call_with_expanded_arg_types(
        concrete_templ_types=concrete_templ_types, expanded_arg_types=expanded_arg_types)
      for signature in signatures)

  def can_call_with_arg_types(self, concrete_templ_types: List[Type],
                              arg_types: Union[List[Type], Tuple[Type]]) -> bool:
    assert all(not templ_type.has_templ_placeholder() for templ_type in concrete_templ_types)
    assert all(not arg_type.has_templ_placeholder() for arg_type in arg_types)
    all_expanded_arg_types = self._iter_expanded_possible_arg_types(arg_types)
    return all(
      self.can_call_with_expanded_arg_types(concrete_templ_types=concrete_templ_types, expanded_arg_types=arg_types)
      for arg_types in all_expanded_arg_types)

  def is_undefined_for_types(self, placeholder_templ_types: List[TemplateType], arg_types: List[Type]):
    signatures = self.signatures_by_number_of_templ_args.get(len(placeholder_templ_types), [])
    if len(signatures) == 0:
      return True
    if len(placeholder_templ_types) > 0:
      return False
    return not any(
      signature.can_call_with_expanded_arg_types(concrete_templ_types=[], expanded_arg_types=expanded_arg_types)
      for signature in signatures for expanded_arg_types in self._iter_expanded_possible_arg_types(arg_types))

  def get_concrete_funcs(self, templ_types: List[Type], arg_types: List[Type]) -> List[ConcreteFunction]:
    signatures = self.signatures_by_number_of_templ_args.get(len(templ_types), [])
    possible_concrete_funcs = []
    for expanded_arg_types in self._iter_expanded_possible_arg_types(arg_types):
      for signature in signatures:
        if signature.can_call_with_expanded_arg_types(
            concrete_templ_types=templ_types, expanded_arg_types=expanded_arg_types):
          concrete_func = signature.get_concrete_func(concrete_templ_types=templ_types)
          if concrete_func not in possible_concrete_funcs:
            possible_concrete_funcs.append(concrete_func)
    return possible_concrete_funcs

  def add_signature(self, signature: FunctionSignature):
    assert self.is_undefined_for_types(
      placeholder_templ_types=signature.placeholder_templ_types, arg_types=signature.arg_types)
    assert signature.returns_void == self.returns_void
    if len(signature.placeholder_templ_types) not in self.signatures_by_number_of_templ_args:
      self.signatures_by_number_of_templ_args[len(signature.placeholder_templ_types)] = []
    self.signatures_by_number_of_templ_args[len(signature.placeholder_templ_types)].append(signature)

  def get_single_concrete_func(self) -> ConcreteFunction:
    assert len(self.signatures) == 1
    signature = self.signatures[0]
    assert len(signature.placeholder_templ_types) == 0
    return signature.get_concrete_func(concrete_templ_types=[])

  @classmethod
  def _iter_expanded_possible_arg_types(cls, arg_types):
    """
    :param list[Type]|tuple[Type] arg_types:
    :rtype: Iterator[Tuple[Type]]
    """
    import itertools
    return itertools.product(*[
      arg_type.possible_types if isinstance(arg_type, UnionType) else [arg_type] for arg_type in arg_types])

  def __repr__(self) -> str:
    return 'FunctionSymbol(signatures=%r)' % self.signatures


class TypeFactory:
  """
  Lazily generates concrete types with initialized template arguments.
  """
  def __init__(self, placeholder_templ_types: List[TemplateType], signature_type: Type):
    self.placeholder_templ_types = placeholder_templ_types
    self.signature_type = signature_type

  def make_concrete_type(self, concrete_templ_types: List[Type]) -> Type:
    replacements = dict(zip(self.placeholder_templ_types, concrete_templ_types))
    concrete_type = self.signature_type.replace_types(replacements)
    assert not concrete_type.has_templ_placeholder()
    return concrete_type


class TypeSymbol(Symbol):
  """
  A (statically) declared (possibly) template type.
  Can have one or multiple template initializations that yield different concrete types.
  These are initialized lazily.
  """
  def __init__(self, type_factory: TypeFactory, constructor_symbol: Optional[FunctionSymbol]):
    super().__init__()
    assert isinstance(type_factory, TypeFactory)
    self.type_factory = type_factory
    self.constructor_symbol = constructor_symbol
    self.initialized_templ_types: Dict[Tuple[Type], Type] = {}

  def get_type(self, concrete_templ_types: List[Type]) -> Type:
    concrete_templ_types = tuple(concrete_templ_types)
    if concrete_templ_types in self.initialized_templ_types:
      return self.initialized_templ_types[concrete_templ_types]
    templ_type = self.type_factory.make_concrete_type(concrete_templ_types=concrete_templ_types)
    self.initialized_templ_types[concrete_templ_types] = templ_type
    return templ_type


class SymbolTable:
  """
  Basically a dict mapping identifier names to symbols.
  Also contains information about the current scope.
  """
  def __init__(self, copy_from=None, copy_new_current_func=None):
    """
    :param SymbolTable|None copy_from:
    :param ConcreteFunction|None copy_new_current_func:
    """
    if copy_from is None:
      assert copy_new_current_func is None
      self.symbols = {}  # type: Dict[str, Symbol]
      self.current_func = None  # type: Optional[ConcreteFunction]
      self.current_scope_identifiers = []  # type: List[str]
      self.used_ir_func_names = set()  # type: Set[str]
      self.known_extern_funcs = {}  # type: Dict[str, ConcreteFunction]
      self.inbuilt_symbols = {}  # type: Dict[str, Symbol]
    else:
      self.symbols = copy_from.symbols.copy()  # type: Dict[str, Symbol]
      if copy_new_current_func is None:
        self.current_func = copy_from.current_func  # type: Optional[ConcreteFunction]
        self.current_scope_identifiers = copy_from.current_scope_identifiers.copy()  # type: List[str]
      else:
        self.current_func = copy_new_current_func  # type: Optional[ConcreteFunction]
        self.current_scope_identifiers = []  # type: List[str]
      self.used_ir_func_names = copy_from.used_ir_func_names  # type: Set[str]
      # do not copy known_extern_funcs, but reference back as we want those to be shared globally
      self.known_extern_funcs = copy_from.known_extern_funcs  # type: Dict[str, ConcreteFunction]
      self.inbuilt_symbols = copy_from.inbuilt_symbols  # type: Dict[str, Symbol]

  def __setitem__(self, identifier, symbol):
    """
    :param str identifier:
    :param Symbol symbol:
    """
    self.symbols[identifier] = symbol

  def __getitem__(self, identifier):
    """
    :param str identifier:
    :rtype: Symbol
    """
    return self.symbols[identifier]

  def __contains__(self, identifier):
    """
    :param str identifier:
    :rtype: bool
    """
    return identifier in self.symbols

  def copy(self):
    """
    :rtype: SymbolTable
    """
    return SymbolTable(self)

  def copy_with_new_current_func(self, new_current_func):
    """
    :param ConcreteFunction new_current_func:
    :rtype: SymbolTable
    """
    assert new_current_func is not None
    return SymbolTable(self, copy_new_current_func=new_current_func)

  def __repr__(self):
    """
    :rtype: str
    """
    return 'SymbolTable%r' % self.__dict__

  def make_ir_func_name(self, func_identifier, extern, concrete_func):
    """
    :param str func_identifier:
    :param bool extern:
    :param ConcreteFunction concrete_func:
    :rtype: str
    """
    if extern:
      ir_func_name = func_identifier
      if ir_func_name in self.known_extern_funcs:
        assert self.known_extern_funcs[ir_func_name].has_same_signature_as(concrete_func)
    else:
      ir_func_name = '_'.join([func_identifier] + [str(arg_type) for arg_type in concrete_func.arg_types])
      assert ir_func_name not in self.used_ir_func_names
    self.used_ir_func_names.add(ir_func_name)
    return ir_func_name

  def apply_type_narrowings_from(self, *other_symbol_tables):
    """
    For all variable symbols, copy common type of all other_symbol_tables.

    :param SymbolTable other_symbol_tables: containing all variables of self
    """
    for symbol_identifier, self_symbol in self.symbols.items():
      if not isinstance(self_symbol, VariableSymbol):
        continue
      assert all(symbol_identifier in symbol_table for symbol_table in other_symbol_tables)
      other_symbols = [
        symbol_table[symbol_identifier] for symbol_table in other_symbol_tables]
      other_symbols = [other_symbol for other_symbol in other_symbols if other_symbol.base == self_symbol.base]
      assert all(isinstance(other_symbol, VariableSymbol) for other_symbol in other_symbols)
      if len(other_symbols) == 0:
        continue
      common_type = get_common_type([other_symbol.narrowed_var_type for other_symbol in other_symbols])
      self[symbol_identifier] = self_symbol.copy_with_narrowed_type(common_type)

  def reset_narrowed_types(self):
    """
    Applies symbol.copy_reset_narrowed_type() for all variable symbols.
    """
    for symbol_identifier, symbol in self.symbols.items():
      if isinstance(symbol, VariableSymbol):
        if symbol.declared_var_type != symbol.narrowed_var_type:
          self[symbol_identifier] = symbol.copy_reset_narrowed_type()

  def has_extern_func(self, func_identifier):
    """
    :param str func_identifier:
    :rtype: bool
    """
    return func_identifier in self.known_extern_funcs

  def get_extern_func(self, func_identifier):
    """
    :param str func_identifier:
    :rtype: ConcreteFunction
    """
    assert self.has_extern_func(func_identifier)
    return self.known_extern_funcs[func_identifier]

  def add_extern_func(self, func_identifier, concrete_func):
    """
    :param str func_identifier:
    :param ConcreteFunction concrete_func:
    """
    assert not self.has_extern_func(func_identifier)
    self.known_extern_funcs[func_identifier] = concrete_func

  @property
  def free_symbol(self) -> FunctionSymbol:
    assert 'free' in self
    free_symbol = self['free']
    assert isinstance(free_symbol, FunctionSymbol)
    return free_symbol


class CodegenContext:
  """
  Used to keep track where code is currently generated.
  This is essentially a pointer to an ir.IRBuilder.
  """
  def __init__(self, builder, copy_from=None):
    """
    :param ir.IRBuilder|None builder:
    :param CodegenContext copy_from:
    """
    self.builder = builder

    if copy_from is None:
      self.emits_ir = builder is not None  # type: bool
      self.is_terminated = False

      self.current_func_inline_return_collect_block = None  # type: Optional[ir.Block]
      self.current_func_inline_return_ir_alloca = None  # type: Optional[ir.instructions.AllocaInstr]
      self.inline_func_call_stack = []  # type: List[ConcreteFunction]
      self.ir_func_malloc = None  # type: Optional[ir.Function]
      self.ir_func_free = None  # type: Optional[ir.Function]
    else:
      self.emits_ir = copy_from.emits_ir
      self.is_terminated = copy_from.is_terminated

      self.current_func_inline_return_collect_block = copy_from.current_func_inline_return_collect_block  # type: Optional[ir.Block]  # noqa
      self.current_func_inline_return_ir_alloca = copy_from.current_func_inline_return_ir_alloca  # type: Optional[ir.instructions.AllocaInstr]  # noqa
      self.inline_func_call_stack = copy_from.inline_func_call_stack.copy()  # type: List[ConcreteFunction]
      self.ir_func_malloc = copy_from.ir_func_malloc  # type: Optional[ir.Function]
      self.ir_func_free = copy_from.ir_func_free  # type: Optional[ir.Function]

    assert all(inline_func.is_inline for inline_func in self.inline_func_call_stack)

  @property
  def module(self):
    """
    :rtype: ir.Module
    """
    assert self.emits_ir
    assert self.builder is not None
    return self.builder.module

  @property
  def block(self):
    """
    :rtype: ir.Block
    """
    assert self.emits_ir
    assert self.builder is not None
    return self.builder.block

  def __repr__(self):
    """
    :rtype: str
    """
    return 'CodegenContext(builder=%r, emits_ir=%r, is_terminated=%r)' % (
      self.builder, self.emits_ir, self.is_terminated)

  def copy_with_builder(self, new_builder):
    """
    :param ir.IRBuilder|None new_builder:
    :rtype: CodegenContext
    """
    return CodegenContext(builder=new_builder, copy_from=self)

  def copy(self):
    """
    :rtype: CodegenContext
    """
    return self.copy_with_builder(self.builder)

  def copy_without_builder(self):
    """
    :rtype: CodegenContext
    """
    new_context = self.copy_with_builder(None)
    new_context.emits_ir = False
    return new_context

  def copy_with_inline_func(self, concrete_func, return_ir_alloca, return_collect_block):
    """
    :param ConcreteFunction concrete_func:
    :param ir.instructions.AllocaInstr return_ir_alloca:
    :param ir.Block return_collect_block:
    """
    assert concrete_func.is_inline
    assert concrete_func not in self.inline_func_call_stack
    new_context = self.copy()
    new_context.current_func_inline_return_ir_alloca = return_ir_alloca
    new_context.current_func_inline_return_collect_block = return_collect_block
    new_context.inline_func_call_stack.append(concrete_func)
    return new_context

  def alloca_at_entry(self, ir_type, name):
    """
    Add alloca instruction at entry block of the current function.

    :param ir.types.Type ir_type:
    :param str name:
    :rtype: ir.instructions.AllocaInstr
    """
    assert self.emits_ir
    entry_block = self.block.function.entry_basic_block  # type: ir.Block
    entry_builder = ir.IRBuilder(entry_block)
    if len(entry_block.instructions) == 0 or not isinstance(entry_block.instructions[0], ir.instructions.AllocaInstr):
      entry_builder.position_at_start(entry_block)
    else:
      last_alloca_num = 0
      while (last_alloca_num < len(entry_block.instructions) and
             isinstance(entry_block.instructions[last_alloca_num], ir.instructions.AllocaInstr)):
        last_alloca_num += 1
      assert isinstance(entry_block.instructions[last_alloca_num - 1], ir.instructions.AllocaInstr)
      entry_builder.position_after(entry_block.instructions[last_alloca_num - 1])
    ir_alloca = entry_builder.alloca(typ=ir_type, name=name)
    if self.builder.block == entry_builder.block:
      # assume that builders are always at the last instruction of a block
      self.builder.position_at_end(self.builder.block)
    return ir_alloca


def make_func_call_ir(func_identifier: str, func_symbol: FunctionSymbol, templ_types: List[Type],
                      calling_arg_types: List[Type], calling_ir_args: List[ir.values.Value],
                      context: CodegenContext) -> Optional[ir.values.Value]:
  assert context.emits_ir
  assert len(calling_arg_types) == len(calling_ir_args)

  def make_call_func(concrete_func, concrete_calling_arg_types_, caller_context):
    """
    :param ConcreteFunction concrete_func:
    :param list[Type] concrete_calling_arg_types_:
    :param CodegenContext caller_context:
    :rtype: ir.values.Value
    """
    assert len(concrete_func.arg_types) == len(calling_arg_types)
    casted_ir_args = [make_implicit_cast_to_ir_val(
      from_type=called_arg_type, to_type=declared_arg_type, from_ir_val=ir_func_arg, context=caller_context,
      name='call_arg_%s_cast' % arg_identifier)
      for arg_identifier, ir_func_arg, called_arg_type, declared_arg_type in zip(
        concrete_func.arg_identifiers, calling_ir_args, concrete_calling_arg_types_, concrete_func.arg_types)]
    assert len(calling_ir_args) == len(casted_ir_args)
    if concrete_func.is_inline:
      assert concrete_func not in caller_context.inline_func_call_stack
      assert callable(concrete_func.make_inline_func_call_ir)
      return concrete_func.make_inline_func_call_ir(ir_func_args=casted_ir_args, caller_context=caller_context)
    else:
      ir_func = concrete_func.ir_func
      assert ir_func is not None and len(ir_func.args) == len(calling_arg_types)
      return caller_context.builder.call(ir_func, casted_ir_args, name='call_%s' % func_identifier)

  assert func_symbol.can_call_with_arg_types(concrete_templ_types=templ_types, arg_types=calling_arg_types)
  possible_concrete_funcs = func_symbol.get_concrete_funcs(templ_types=templ_types, arg_types=calling_arg_types)
  if len(possible_concrete_funcs) == 1:
    return make_call_func(
      concrete_func=possible_concrete_funcs[0], concrete_calling_arg_types_=calling_arg_types, caller_context=context)
  else:
    import numpy as np, itertools
    # The arguments which we need to look at to determine which concrete function to call
    # TODO: This could use a better heuristic, only because something is a union type does not mean that it
    # distinguishes different concrete funcs.
    distinguishing_arg_nums = [
      arg_num for arg_num, calling_arg_type in enumerate(calling_arg_types)
      if isinstance(calling_arg_type, UnionType)]
    assert len(distinguishing_arg_nums) >= 1
    # noinspection PyTypeChecker
    distinguishing_calling_arg_types = [
      calling_arg_types[arg_num] for arg_num in distinguishing_arg_nums]  # type: List[UnionType]
    assert all(isinstance(calling_arg, UnionType) for calling_arg in distinguishing_calling_arg_types)
    # To distinguish which concrete func to call, use this table
    block_addresses_distinguished_mapping = np.ndarray(
      shape=tuple(max(calling_arg.possible_type_nums) + 1 for calling_arg in distinguishing_calling_arg_types),
      dtype=ir.values.BlockAddress)

    # Go through all concrete functions, and add one block for each
    concrete_func_caller_contexts = [
      context.copy_with_builder(ir.IRBuilder(context.builder.append_basic_block("call_%s_%s" % (
        func_identifier, '_'.join(str(arg_type) for arg_type in concrete_func.arg_types)))))
      for concrete_func in possible_concrete_funcs]
    for concrete_func, concrete_caller_context in zip(possible_concrete_funcs, concrete_func_caller_contexts):
      concrete_func_block = concrete_caller_context.block
      concrete_func_block_address = ir.values.BlockAddress(context.builder.function, concrete_func_block)
      concrete_func_distinguishing_args = [concrete_func.arg_types[arg_num] for arg_num in distinguishing_arg_nums]
      concrete_func_possible_types_per_arg = [
        [possible_type
          for possible_type in (arg_type.possible_types if isinstance(arg_type, UnionType) else [arg_type])
          if possible_type in calling_arg_type.possible_types]
        for arg_type, calling_arg_type in zip(concrete_func_distinguishing_args, distinguishing_calling_arg_types)]
      assert len(distinguishing_arg_nums) == len(concrete_func_possible_types_per_arg)

      # Register the concrete function in the table
      for expanded_func_types in itertools.product(*concrete_func_possible_types_per_arg):
        assert len(expanded_func_types) == len(distinguishing_calling_arg_types)
        distinguishing_variant_nums = tuple(
          calling_arg_type.get_variant_num(concrete_arg_type)
            for calling_arg_type, concrete_arg_type in zip(distinguishing_calling_arg_types, expanded_func_types))
        assert block_addresses_distinguished_mapping[distinguishing_variant_nums] is None
        block_addresses_distinguished_mapping[distinguishing_variant_nums] = concrete_func_block_address

    # Compute the index we have to look at in the table
    tag_ir_type = ir.types.IntType(8)
    call_block_index_ir = ir.Constant(tag_ir_type, 0)
    for arg_num, calling_arg_type in zip(distinguishing_arg_nums, distinguishing_calling_arg_types):
      ir_func_arg = calling_ir_args[arg_num]
      base = np.prod(block_addresses_distinguished_mapping.shape[arg_num + 1:], dtype='int32')
      base_ir = ir.Constant(tag_ir_type, base)
      tag_ir = calling_arg_type.make_extract_tag(
        ir_func_arg, context=context, name='call_%s_arg%s_tag_ptr' % (func_identifier, arg_num))
      call_block_index_ir = context.builder.add(call_block_index_ir, context.builder.mul(base_ir, tag_ir))
    call_block_index_ir = context.builder.zext(
      call_block_index_ir, LLVM_SIZE_TYPE, name='call_%s_block_index' % func_identifier)

    # Look it up in the table and call the function
    ir_block_addresses_type = ir.types.VectorType(
      LLVM_VOID_POINTER_TYPE, np.prod(block_addresses_distinguished_mapping.shape))
    ir_block_addresses = ir.values.Constant(ir_block_addresses_type, ir_block_addresses_type.wrap_constant_value(
      list(block_addresses_distinguished_mapping.flatten())))
    ir_call_block_target = context.builder.extract_element(ir_block_addresses, call_block_index_ir)
    indirect_branch = context.builder.branch_indirect(ir_call_block_target)
    for concrete_caller_context in concrete_func_caller_contexts:
      indirect_branch.add_destination(concrete_caller_context.block)

    # Execute the concrete functions and collect their return value
    collect_block = context.builder.append_basic_block("collect_%s_overload" % func_identifier)
    context.builder = ir.IRBuilder(collect_block)
    concrete_func_return_ir_vals = []  # type: List[ir.values.Value]
    for concrete_func, concrete_caller_context in zip(possible_concrete_funcs, concrete_func_caller_contexts):
      concrete_calling_arg_types = [
        narrow_type(calling_arg_type, concrete_arg_type)
        for calling_arg_type, concrete_arg_type in zip(calling_arg_types, concrete_func.arg_types)]
      concrete_return_ir_val = make_call_func(
        concrete_func, concrete_calling_arg_types_=concrete_calling_arg_types, caller_context=concrete_caller_context)
      concrete_func_return_ir_vals.append(concrete_return_ir_val)
      assert not concrete_caller_context.is_terminated
      concrete_caller_context.builder.branch(collect_block)
      assert concrete_func.signature.returns_void == func_symbol.returns_void
    assert len(possible_concrete_funcs) == len(concrete_func_return_ir_vals)

    if func_symbol.returns_void:
      return None
    else:
      common_return_type = get_common_type([concrete_func.return_type for concrete_func in possible_concrete_funcs])
      collect_return_ir_phi = context.builder.phi(
        common_return_type.ir_type, name="collect_%s_overload_return" % func_identifier)
      for concrete_return_ir_val, concrete_caller_context in zip(
          concrete_func_return_ir_vals, concrete_func_caller_contexts):
        collect_return_ir_phi.add_incoming(concrete_return_ir_val, concrete_caller_context.block)
      return collect_return_ir_phi


def _make_builtin_op_arg_names(op, op_arg_types):
  """
  :param str op:
  :param list[Type] op_arg_types:
  :rtype: list[str]
  """
  if len(op_arg_types) == 1:
    return ['arg']
  if len(op_arg_types) == 2:
    return ['left', 'right']
  assert False, 'not implemented'


SLEEPY_INBUILT_BINARY_OPS = ['+', '-', '*', '/', '==', '!=', '<', '>', '<=', '>=', 'bitwise_or']  # type: List[str]
SLEEPY_INBUILT_BINARY_OPS_ARG_TYPES = ([
  [
    [SLEEPY_DOUBLE, SLEEPY_DOUBLE], [SLEEPY_FLOAT, SLEEPY_FLOAT], [SLEEPY_INT, SLEEPY_INT], [SLEEPY_LONG, SLEEPY_LONG],
    [SLEEPY_DOUBLE_PTR, SLEEPY_INT], [SLEEPY_FLOAT_PTR, SLEEPY_INT], [SLEEPY_CHAR_PTR, SLEEPY_INT],
    [SLEEPY_INT_PTR, SLEEPY_INT], [SLEEPY_DOUBLE], [SLEEPY_FLOAT], [SLEEPY_INT], [SLEEPY_LONG]
  ], [
    [SLEEPY_DOUBLE, SLEEPY_DOUBLE], [SLEEPY_FLOAT, SLEEPY_FLOAT], [SLEEPY_INT, SLEEPY_INT], [SLEEPY_LONG, SLEEPY_LONG],
    [SLEEPY_DOUBLE], [SLEEPY_FLOAT], [SLEEPY_INT], [SLEEPY_LONG]
  ],
  [[SLEEPY_DOUBLE, SLEEPY_DOUBLE], [SLEEPY_FLOAT, SLEEPY_FLOAT], [SLEEPY_INT, SLEEPY_INT], [SLEEPY_LONG, SLEEPY_LONG]],
  [[SLEEPY_DOUBLE, SLEEPY_DOUBLE], [SLEEPY_FLOAT, SLEEPY_FLOAT]]] + [
    [[SLEEPY_DOUBLE, SLEEPY_DOUBLE], [SLEEPY_FLOAT, SLEEPY_FLOAT], [SLEEPY_INT, SLEEPY_INT], [SLEEPY_LONG, SLEEPY_LONG],
    [SLEEPY_DOUBLE_PTR, SLEEPY_DOUBLE_PTR], [SLEEPY_FLOAT_PTR, SLEEPY_FLOAT_PTR], [SLEEPY_CHAR_PTR, SLEEPY_CHAR_PTR],
    [SLEEPY_INT_PTR, SLEEPY_INT_PTR]]] * 6 +
  [[[SLEEPY_INT, SLEEPY_INT], [SLEEPY_LONG, SLEEPY_LONG]]])  # type: List[List[List[Type]]]
SLEEPY_INBUILT_BINARY_OPS_RETURN_TYPES = ([
  [
    SLEEPY_DOUBLE, SLEEPY_FLOAT, SLEEPY_INT, SLEEPY_LONG, SLEEPY_DOUBLE_PTR, SLEEPY_FLOAT_PTR, SLEEPY_CHAR_PTR,
    SLEEPY_INT_PTR, SLEEPY_DOUBLE, SLEEPY_FLOAT, SLEEPY_INT, SLEEPY_LONG],
  [SLEEPY_DOUBLE, SLEEPY_FLOAT, SLEEPY_INT, SLEEPY_LONG, SLEEPY_DOUBLE, SLEEPY_FLOAT, SLEEPY_INT, SLEEPY_LONG],
  [SLEEPY_DOUBLE, SLEEPY_FLOAT, SLEEPY_INT, SLEEPY_LONG],
  [SLEEPY_DOUBLE, SLEEPY_FLOAT]] + [[SLEEPY_BOOL] * 8] * 6 +
  [[SLEEPY_INT, SLEEPY_LONG]])  # type: List[List[Type]]
assert (
  len(SLEEPY_INBUILT_BINARY_OPS) == len(SLEEPY_INBUILT_BINARY_OPS_ARG_TYPES) ==
  len(SLEEPY_INBUILT_BINARY_OPS_RETURN_TYPES))
assert all(len(arg_types) == len(return_types) for arg_types, return_types in (
  zip(SLEEPY_INBUILT_BINARY_OPS_ARG_TYPES, SLEEPY_INBUILT_BINARY_OPS_ARG_TYPES)))


def _make_builtin_op_ir_val(op, op_arg_types, ir_func_args, caller_context):
  """
  :param str op:
  :param list[Type] op_arg_types:
  :param list[ir.values.Value] ir_func_args:
  :param CodegenContext caller_context:
  :rtype: ir.values.Value
  """
  assert caller_context.emits_ir
  assert not caller_context.is_terminated
  op_arg_types = tuple(op_arg_types)
  assert len(op_arg_types) == len(ir_func_args)

  def make_binary_op(single_type_instr, instr_name):
    """
    :param Dict[Type,Callable] single_type_instr:
    :param str instr_name:
    :rtype: ir.values.Value
    """
    assert len(op_arg_types) == len(ir_func_args) == 2
    type_instr = {(arg_type, arg_type): instr for arg_type, instr in single_type_instr.items()}
    return make_op(type_instr=type_instr, instr_name=instr_name)

  def make_op(type_instr, instr_name):
    """
    :param Dict[Tuple[Type],Callable] type_instr:
    :param str instr_name:
    :rtype: ir.values.Value
    """
    assert op_arg_types in type_instr
    return type_instr[op_arg_types](*ir_func_args, name=instr_name)

  body_builder = caller_context.builder
  if op == '*':
    return make_binary_op({
      SLEEPY_DOUBLE: body_builder.fmul, SLEEPY_FLOAT: body_builder.fmul, SLEEPY_INT: body_builder.mul,
      SLEEPY_LONG: body_builder.mul}, instr_name='mul')
  if op == '/':
    return make_binary_op({SLEEPY_DOUBLE: body_builder.fdiv, SLEEPY_FLOAT: body_builder.fdiv}, instr_name='div')
  if op == '+':
    if len(op_arg_types) == 1:
      assert len(ir_func_args) == 1
      return ir_func_args[0]
    assert len(op_arg_types) == len(ir_func_args) == 2
    left_type, right_type = op_arg_types
    left_val, right_val = ir_func_args
    if left_type in SLEEPY_POINTER_TYPES and right_type == SLEEPY_INT:
      return body_builder.gep(left_val, (right_val,), name='add')
    return make_binary_op({
      SLEEPY_DOUBLE: body_builder.fadd, SLEEPY_FLOAT: body_builder.fadd, SLEEPY_INT: body_builder.add,
      SLEEPY_LONG: body_builder.add}, instr_name='add')
  if op == '-':
    if len(op_arg_types) == 1:
      assert len(ir_func_args) == 1
      val_type, ir_val = op_arg_types[0], ir_func_args[0]
      constant_minus_one = ir.Constant(val_type.ir_type, -1)
      if val_type in {SLEEPY_DOUBLE, SLEEPY_FLOAT}:
        return body_builder.fmul(constant_minus_one, ir_val, name='neg')
      if val_type in {SLEEPY_INT, SLEEPY_LONG}:
        return body_builder.mul(constant_minus_one, ir_val, name='neg')
    return make_binary_op({
      SLEEPY_DOUBLE: body_builder.fsub, SLEEPY_FLOAT: body_builder.fsub, SLEEPY_INT: body_builder.sub,
      SLEEPY_LONG: body_builder.sub}, instr_name='sub')
  if op in {'==', '!=', '<', '>', '<=', '>='}:
    from functools import partial
    return make_binary_op({
      SLEEPY_DOUBLE: partial(body_builder.fcmp_ordered, op), SLEEPY_FLOAT: partial(body_builder.fcmp_ordered, op),
      SLEEPY_INT: partial(body_builder.icmp_signed, op), SLEEPY_LONG: partial(body_builder.icmp_signed, op),
      SLEEPY_DOUBLE_PTR: partial(body_builder.icmp_unsigned, op),
      SLEEPY_FLOAT_PTR: partial(body_builder.icmp_unsigned, op),
      SLEEPY_CHAR_PTR: partial(body_builder.icmp_unsigned, op),
      SLEEPY_INT_PTR: partial(body_builder.icmp_unsigned, op)},
      instr_name='cmp')
  if op == 'bitwise_or':
    return make_binary_op({SLEEPY_INT: body_builder.or_, SLEEPY_LONG: body_builder.or_}, instr_name='bitwise_or')
  assert False, 'Operator %s not handled!' % op


def _make_str_symbol(symbol_table, context):
  """
  :param SymbolTable symbol_table:
  :param CodegenContext context:
  :rtype: TypeSymbol
  """
  str_type = StructType(
    struct_identifier='Str', member_identifiers=['start', 'length', 'alloc_length'], templ_types=[],
    member_types=[SLEEPY_CHAR_PTR, SLEEPY_INT, SLEEPY_INT], member_mutables=[False, False, False],
    pass_by_ref=True)
  constructor_symbol = str_type.build_constructor(parent_symbol_table=symbol_table, parent_context=context)
  type_factory = TypeFactory(placeholder_templ_types=[], signature_type=str_type)
  struct_symbol = TypeSymbol(type_factory=type_factory, constructor_symbol=constructor_symbol)
  str_type.build_destructor(parent_symbol_table=symbol_table, parent_context=context)
  return struct_symbol


class InbuiltOpConcreteFuncFactory(ConcreteFunctionFactory):
  def __init__(self, op: str, op_arg_types: List[Type], op_return_type: Type, context: CodegenContext):
    self.op = op
    self.op_arg_types = op_arg_types
    self.op_return_type = op_return_type
    self.context = context

  def build_concrete_func_ir(self, concrete_func: ConcreteFunction):
    if self.context.emits_ir:
      from functools import partial
      concrete_func.make_inline_func_call_ir = partial(_make_builtin_op_ir_val, self.op, self.op_arg_types)
    return concrete_func


class InbuiltOpDestructorFuncFactory(ConcreteFunctionFactory):
  def __init__(self, symbol_table: SymbolTable, context: CodegenContext):
    self.symbol_table = symbol_table
    self.context = context

  def build_concrete_func_ir(self, concrete_func: ConcreteFunction):
    if self.context.emits_ir:
      ir_func_name = self.symbol_table.make_ir_func_name('free', extern=False, concrete_func=concrete_func)
      concrete_func.ir_func = ir.Function(self.context.module, concrete_func.make_ir_function_type(), name=ir_func_name)
      destructor_block = concrete_func.ir_func.append_basic_block(name='entry')
      destructor_context = self.context.copy_with_builder(ir.IRBuilder(destructor_block))
      destructor_context.builder.ret_void()  # destructor does not do anything for value types


def build_initial_ir(symbol_table: SymbolTable, context: CodegenContext):
  assert 'free' not in symbol_table
  free_symbol = FunctionSymbol(returns_void=True)
  symbol_table['free'] = free_symbol
  for type_identifier, inbuilt_type in SLEEPY_TYPES.items():
    type_generator = TypeFactory(placeholder_templ_types=[], signature_type=inbuilt_type)
    assert type_identifier not in symbol_table
    symbol_table[type_identifier] = TypeSymbol(type_generator, constructor_symbol=None)
    if inbuilt_type == SLEEPY_VOID:
      continue

    # add destructor
    destructor_factory = InbuiltOpDestructorFuncFactory(symbol_table=symbol_table, context=context)
    signature_ = FunctionSignature(
      concrete_func_factory=destructor_factory, placeholder_templ_types=[], return_type=SLEEPY_VOID,
      return_mutable=False, arg_types=[inbuilt_type], arg_identifiers=['var'], arg_type_narrowings=[SLEEPY_NEVER],
      arg_mutables=[False])
    symbol_table.free_symbol.add_signature(signature_)

  for assert_identifier in ['assert', 'unchecked_assert']:
    assert assert_identifier not in symbol_table
    assert_symbol = FunctionSymbol(returns_void=True)
    symbol_table[assert_identifier] = assert_symbol
    symbol_table.inbuilt_symbols[assert_identifier] = assert_symbol

  if context.emits_ir:
    module = context.builder.module
    context.ir_func_malloc = ir.Function(
      module, ir.FunctionType(LLVM_VOID_POINTER_TYPE, [LLVM_SIZE_TYPE]), name='malloc')
    context.ir_func_free = ir.Function(module, ir.FunctionType(ir.VoidType(), [LLVM_VOID_POINTER_TYPE]), name='free')

  assert 'Str' not in symbol_table
  str_symbol = _make_str_symbol(symbol_table=symbol_table, context=context)
  symbol_table['Str'] = str_symbol
  symbol_table.inbuilt_symbols['Str'] = str_symbol

  for op, op_arg_type_list, op_return_type_list in zip(
    SLEEPY_INBUILT_BINARY_OPS, SLEEPY_INBUILT_BINARY_OPS_ARG_TYPES, SLEEPY_INBUILT_BINARY_OPS_RETURN_TYPES):
    assert op not in symbol_table
    func_symbol = FunctionSymbol(returns_void=False)
    symbol_table[op] = func_symbol
    assert len(op_arg_type_list) == len(op_return_type_list)
    for op_arg_types, op_return_type in zip(op_arg_type_list, op_return_type_list):
      assert func_symbol.is_undefined_for_types(placeholder_templ_types=[], arg_types=op_arg_types)
      op_arg_identifiers = _make_builtin_op_arg_names(op, op_arg_types)
      assert len(op_arg_types) == len(op_arg_identifiers)

      concrete_func_factory = InbuiltOpConcreteFuncFactory(
        op=op, op_arg_types=op_arg_types, op_return_type=op_return_type, context=context)
      signature_ = FunctionSignature(
        concrete_func_factory=concrete_func_factory, placeholder_templ_types=[], return_type=op_return_type,
        return_mutable=False, arg_identifiers=op_arg_identifiers, arg_types=op_arg_types,
        arg_mutables=[False] * len(op_arg_types), arg_type_narrowings=op_arg_types, is_inline=True)
      func_symbol.add_signature(signature_)
