"""
Implements a symbol table.
"""
from __future__ import annotations

import ctypes
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Optional, List, Tuple, Set, Union, Callable

from llvmlite import ir
from llvmlite.binding import ExecutionEngine
from llvmlite.ir import IRBuilder, Value

from sleepy.grammar import TreePosition
from sleepy.util import concat_dicts

LLVM_SIZE_TYPE = ir.types.IntType(64)
LLVM_VOID_POINTER_TYPE = ir.PointerType(ir.types.IntType(8))


class Symbol(ABC):
  """
  A declared symbol, with an identifier.
  """
  kind = None

  def __init__(self):
    """
    Initialize the symbol.
    """
    self.base = self  # type: Symbol
    assert self.kind is not None

  @abstractmethod
  def copy_replace_unbound_templ_types(self, templ_type_replacements: Dict[TemplateType, Type]) -> Symbol:
    raise NotImplementedError()

  class Kind(Enum):
    """
    Possible symbol types.
    """
    VARIABLE = 'var'
    TYPE = 'type'
    FUNCTION = 'func'


class Type(ABC):
  """
  A type of a declared variable.
  """
  def __init__(self, ir_type: Optional[ir.types.Type], pass_by_ref: bool, c_type, constructor: Optional[FunctionSymbol]):
    """
    :param ctypes._CData|None c_type: may be None if this is non-realizable (e.g. template types / void)
    """
    assert not pass_by_ref or isinstance(ir_type, ir.types.PointerType)
    self.ir_type = ir_type
    self.pass_by_ref = pass_by_ref
    self.c_type = c_type
    self._constructor: Optional[FunctionSymbol] = constructor

  def __repr__(self):
    return self.__class__.__name__

  @property
  def size(self) -> int:
    return ctypes.sizeof(self.c_type)

  def is_pass_by_ref(self):
    """
    :rtype: bool
    """
    return self.pass_by_ref

  def make_ir_size(self, size: int) -> ir.values.Value:
    return ir.Constant(LLVM_SIZE_TYPE, size)

  def is_realizable(self) -> bool:
    return True

  def replace_types(self, replacements: Dict[Type, Type]):
    if self in replacements:
      return replacements[self]
    return self

  def has_templ_placeholder(self) -> bool:
    return False

  @abstractmethod
  def children(self) -> List[Type]:
    raise NotImplementedError()

  @property
  @abstractmethod
  def templ_types(self) -> List[Type]:
    raise NotImplementedError()

  @abstractmethod
  def has_same_symbol_as(self, other: Type) -> bool:
    raise NotImplementedError()

  @abstractmethod
  def make_di_type(self, context: CodegenContext) -> ir.DIValue:
    raise NotImplementedError()

  @property
  def constructor(self) -> Optional[FunctionSymbol]:
    return self._constructor

  @property
  def constructor_caller(self) -> FunctionSymbolCaller:
    assert self.constructor is not None
    # TODO: We do not currently keep track if the user specifies template types anywhere.
    # this means saying MyType[][][] is okay currently.
    templ_types = self.templ_types if len(self.templ_types) > 0 else None
    return FunctionSymbolCaller(self.constructor, templ_types=templ_types)

  @constructor.setter
  def constructor(self, new_constructor: FunctionSymbol):
    assert new_constructor is None or not new_constructor.returns_void
    self._constructor = new_constructor


class VoidType(Type):
  """
  Type returned when nothing is returned.
  """
  def __init__(self):
    super().__init__(ir.VoidType(), pass_by_ref=False, c_type=None, constructor=None)

  def __repr__(self) -> str:
    return 'Void'

  def children(self) -> List[Type]:
    return []

  @property
  def templ_types(self) -> List[Type]:
    return []

  def __eq__(self, other) -> bool:
    return isinstance(other, VoidType)

  def __hash__(self) -> int:
    return id(type(self))

  def has_same_symbol_as(self, other: Type) -> bool:
    return self == other

  def make_di_type(self, context: CodegenContext):
    assert False, self


class DoubleType(Type):
  """
  A double.
  """
  def __init__(self):
    super().__init__(ir.DoubleType(), pass_by_ref=False, c_type=ctypes.c_double, constructor=None)

  def make_di_type(self, context: CodegenContext) -> ir.DIValue:
    assert context.emits_debug
    return context.module.add_debug_info(
      'DIBasicType',
      {'name': repr(self), 'size': self.size, 'align': self.size, 'encoding': ir.DIToken('DW_ATE_float')})

  def __repr__(self) -> str:
    return 'Double'

  def children(self) -> List[Type]:
    return []

  @property
  def templ_types(self) -> List[Type]:
    return []

  def __eq__(self, other) -> bool:
    return isinstance(other, DoubleType)

  def __hash__(self) -> int:
    return id(type(self))

  def has_same_symbol_as(self, other: Type) -> bool:
    return self == other


class FloatType(Type):
  """
  A double.
  """
  def __init__(self):
    super().__init__(ir.FloatType(), pass_by_ref=False, c_type=ctypes.c_float, constructor=None)

  def make_di_type(self, context: CodegenContext) -> ir.DIValue:
    assert context.emits_debug
    return context.module.add_debug_info(
      'DIBasicType',
      {'name': repr(self), 'size': self.size, 'align': self.size, 'encoding': ir.DIToken('DW_ATE_float')})

  def __repr__(self) -> str:
    return 'Float'

  def children(self) -> List[Type]:
    return []

  @property
  def templ_types(self) -> List[Type]:
    return []

  def __eq__(self, other) -> bool:
    return isinstance(other, FloatType)

  def __hash__(self) -> int:
    return id(type(self))

  def has_same_symbol_as(self, other: Type) -> bool:
    return self == other


class BoolType(Type):
  """
  A 1-bit integer.
  """
  def __init__(self):
    super().__init__(ir.IntType(bits=1), pass_by_ref=False, c_type=ctypes.c_bool, constructor=None)

  def make_di_type(self, context: CodegenContext) -> ir.DIValue:
    assert context.emits_debug
    return context.module.add_debug_info(
      'DIBasicType',
      {'name': repr(self), 'size': self.size, 'align': self.size, 'encoding': ir.DIToken('DW_ATE_boolean')})

  def __repr__(self) -> str:
    return 'Bool'

  def children(self) -> List[Type]:
    return []

  @property
  def templ_types(self) -> List[Type]:
    return []

  def __eq__(self, other) -> bool:
    return isinstance(other, BoolType)

  def __hash__(self) -> int:
    return id(type(self))

  def has_same_symbol_as(self, other: Type) -> bool:
    return isinstance(other, BoolType)


class IntType(Type):
  """
  An 32-bit integer.
  """
  def __init__(self):
    super().__init__(ir.IntType(bits=32), pass_by_ref=False, c_type=ctypes.c_int32, constructor=None)

  def make_di_type(self, context: CodegenContext) -> ir.DIValue:
    assert context.emits_debug
    return context.module.add_debug_info(
      'DIBasicType',
      {'name': repr(self), 'size': self.size, 'align': self.size, 'encoding': ir.DIToken('DW_ATE_signed')})

  def __repr__(self) -> str:
    return 'Int'

  def children(self) -> List[Type]:
    return []

  @property
  def templ_types(self) -> List[Type]:
    return []

  def __eq__(self, other) -> bool:
    return isinstance(other, IntType)

  def __hash__(self) -> int:
    return id(type(self))

  def has_same_symbol_as(self, other: Type) -> bool:
    return self == other


class LongType(Type):
  """
  A 64-bit integer.
  """
  def __init__(self):
    super().__init__(ir.IntType(bits=64), pass_by_ref=False, c_type=ctypes.c_int64, constructor=None)

  def make_di_type(self, context: CodegenContext) -> ir.DIValue:
    assert context.emits_debug
    return context.module.add_debug_info(
      'DIBasicType',
      {'name': repr(self), 'size': self.size, 'align': self.size, 'encoding': ir.DIToken('DW_ATE_signed')})

  def __repr__(self) -> str:
    return 'Long'

  def children(self) -> List[Type]:
    return []

  @property
  def templ_types(self) -> List[Type]:
    return []

  def __eq__(self, other) -> bool:
    return isinstance(other, LongType)

  def __hash__(self) -> int:
    return id(type(self))

  def has_same_symbol_as(self, other: Type) -> bool:
    return self == other


class CharType(Type):
  """
  An 8-bit character.
  """
  def __init__(self):
    super().__init__(ir.IntType(bits=8), pass_by_ref=False, c_type=ctypes.c_char, constructor=None)

  def make_di_type(self, context: CodegenContext) -> ir.DIValue:
    assert context.emits_debug
    return context.module.add_debug_info(
      'DIBasicType',
      {'name': repr(self), 'size': self.size, 'align': self.size, 'encoding': ir.DIToken('DW_ATE_unsigned_char')})

  def __repr__(self) -> str:
    return 'Char'

  def children(self) -> List[Type]:
    return []

  @property
  def templ_types(self) -> List[Type]:
    return []

  def __eq__(self, other) -> bool:
    return isinstance(other, CharType)

  def __hash__(self) -> int:
    return id(type(self))

  def has_same_symbol_as(self, other: Type) -> bool:
    return self == other


class PointerType(Type):
  """
  A (not-raw) pointer with a known underlying type.
  """
  def __init__(self, pointee_type: Type, constructor: Optional[FunctionSymbol] = None):
    super().__init__(
      ir.PointerType(pointee_type.ir_type), pass_by_ref=False, c_type=ctypes.POINTER(pointee_type.c_type),
      constructor=constructor)
    self.pointee_type = pointee_type

  def make_di_type(self, context: CodegenContext) -> ir.DIValue:
    assert context.emits_debug
    return context.module.add_debug_info(
      'DIBasicType',
      {'name': repr(self), 'size': self.size, 'align': self.size, 'encoding': ir.DIToken('DW_ATE_address')})

  def __repr__(self) -> str:
    return 'Ptr[%r]' % self.pointee_type

  def __eq__(self, other) -> bool:
    if not isinstance(other, PointerType):
      return False
    return self.pointee_type == other.pointee_type

  def __hash__(self) -> int:
    return hash((self.__class__, self.pointee_type))

  def replace_types(self, replacements: Dict[Type, Type]) -> PointerType:
    return PointerType(pointee_type=self.pointee_type.replace_types(replacements), constructor=self.constructor)

  def children(self) -> List[Type]:
    return []

  @property
  def templ_types(self) -> List[Type]:
    return [self.pointee_type]

  def has_same_symbol_as(self, other: Type) -> bool:
    return isinstance(other, PointerType)


class RawPointerType(Type):
  """
  A raw (void) pointer from which we do not know the underlying type.
  """
  def __init__(self):
    super().__init__(LLVM_VOID_POINTER_TYPE, pass_by_ref=False, c_type=ctypes.c_void_p, constructor=None)

  def make_di_type(self, context: CodegenContext) -> ir.DIValue:
    assert context.emits_debug
    return context.module.add_debug_info(
      'DIBasicType',
      {'name': repr(self), 'size': self.size, 'align': self.size, 'encoding': ir.DIToken('DW_ATE_address')})

  def __repr__(self) -> str:
    return 'RawPtr'

  def replace_types(self, replacements: Dict[Type, Type]) -> RawPointerType:
    return self

  def children(self) -> List[Type]:
    return []

  @property
  def templ_types(self) -> List[Type]:
    return []

  def __eq__(self, other) -> bool:
    return isinstance(other, RawPointerType)

  def __hash__(self) -> int:
    return id(type(self))

  def has_same_symbol_as(self, other: Type) -> bool:
    return self == other


class UnionType(Type):
  """
  A tagged union, i.e. a type that can be one of a set of different types.
  """
  def __init__(self, possible_types: List[Type], possible_type_nums: List[int], val_size: int,
               constructor: Optional[FunctionSymbol] = None):
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
    self.tag_size = 8
    # TODO: The C Type should match the IR type: Also make this an byte array.
    self.untagged_union_c_type = type(
      '%s_UntaggedCType' % self.identifier, (ctypes.Union,),
      {'_fields_': [('variant%s' % num, possible_type.c_type) for num, possible_type in enumerate(possible_types)]})
    self.untagged_union_ir_type = ir.types.ArrayType(ir.types.IntType(8), val_size)
    c_type = type(
      '%s_CType' % self.identifier, (ctypes.Structure,),
      {'_fields_': [('tag', self.tag_c_type), ('untagged_union', self.untagged_union_c_type)]})
    ir_type = ir.types.LiteralStructType([self.tag_ir_type, self.untagged_union_ir_type])
    super().__init__(ir_type, pass_by_ref=False, c_type=c_type, constructor=constructor)

  def __repr__(self):
    if len(self.possible_types) == 0:
      return 'NeverType'
    return '|'.join(
      '%s:%s' % (possible_type_num, possible_type)
      for possible_type_num, possible_type in zip(self.possible_type_nums, self.possible_types))

  def __eq__(self, other) -> bool:
    if not isinstance(other, UnionType):
      return False
    self_types_dict = dict(zip(self.possible_types, self.possible_type_nums))
    other_types_dict = dict(zip(other.possible_types, other.possible_type_nums))
    return self_types_dict == other_types_dict and self.val_size == other.val_size

  def __hash__(self) -> int:
    return hash(tuple(sorted(zip(self.possible_type_nums, self.possible_types))) + (self.val_size,))

  def make_di_type(self, context: CodegenContext) -> ir.DIValue:
    assert context.emits_debug
    di_derived_types = [
      context.module.add_debug_info(
        'DIDerivedType', {
          'tag': ir.DIToken('DW_TAG_inheritance'), 'baseType': member_type.make_di_type(context=context),
          'size': member_type.size, 'align': member_type.size})
      for member_type in self.possible_types]
    di_tag_type = context.module.add_debug_info(
      'DIBasicType',
      {'name': 'tag', 'size': self.tag_size, 'align': self.tag_size, 'encoding': ir.DIToken('DW_ATE_unsigned')})
    di_untagged_union_type = context.module.add_debug_info(
      'DICompositeType', {
        'name': repr(self), 'size': self.val_size, 'align': self.val_size, 'tag': ir.DIToken('DW_TAG_union_type'),
        'file': context.current_di_file, 'elements': [di_derived_types]})
    return context.module.add_debug_info(
      'DICompositeType', {
        'name': repr(self), 'size': self.size, 'align': self.size, 'tag': ir.DIToken('DW_TAG_structure_type'),
        'elements': [di_tag_type, di_untagged_union_type]})

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
    return UnionType(
      possible_types=new_possible_types, possible_type_nums=new_possible_type_nums, val_size=val_size,
      constructor=self.constructor)

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

  def copy_with_extended_types(self, extended_types: List[Type], extended_type_nums: Union[List[Union[int, None]], None]=None) -> UnionType:
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
        # noinspection PyTypeChecker
        new_possible_type_nums.append(extended_type_num) # inspection is wrong here
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

  def children(self) -> List[Type]:
    return self.possible_types

  @property
  def templ_types(self) -> List[Type]:
    return []

  def has_same_symbol_as(self, other: Type) -> bool:
    return isinstance(other, UnionType)


class StructType(Type):
  """
  A struct.
  """
  def __init__(self, struct_identifier: str, templ_types: List[Type], member_identifiers: List[str],
               member_types: List[Type], member_mutables: List[bool], pass_by_ref: bool,
               constructor: Optional[FunctionSymbol] = None):
    self.struct_identifier = struct_identifier
    self._templ_types = templ_types
    self.member_identifiers = member_identifiers
    self.member_types = member_types
    self.member_mutables = member_mutables
    self._constructor = None  # type: Optional[FunctionSymbol]
    if self.has_templ_placeholder():
      self.ir_val_type = None
      self.c_val_type = None
      super().__init__(None, pass_by_ref=pass_by_ref, c_type=None, constructor=constructor)
    else:  # default case
      assert not self.has_templ_placeholder()
      member_ir_types = [member_type.ir_type for member_type in member_types]
      self.ir_val_type = ir.types.LiteralStructType(member_ir_types)
      member_c_types = [
        (member_identifier, member_type.c_type)
        for member_identifier, member_type in zip(member_identifiers, member_types)]
      self.c_val_type = type('%s_CType' % struct_identifier, (ctypes.Structure,), {'_fields_': member_c_types})
      if pass_by_ref:
        super().__init__(
          ir.types.PointerType(self.ir_val_type), pass_by_ref=True, c_type=ctypes.POINTER(self.c_val_type),
          constructor=constructor)
      else:
        super().__init__(self.ir_val_type, pass_by_ref=False, c_type=self.c_val_type, constructor=constructor)

  @property
  def templ_types(self) -> List[Type]:
    return self._templ_types

  def get_member_num(self, member_identifier):
    """
    :param str member_identifier:
    :rtype: int
    """
    assert member_identifier in self.member_identifiers
    return self.member_identifiers.index(member_identifier)

  def replace_types(self, replacements: Dict[Type, Type]) -> StructType:
    if len(replacements) == 0:
      return self
    new_templ_types = [replacements.get(templ_type, templ_type) for templ_type in self.templ_types]
    new_member_types = [replacements.get(member_type, member_type) for member_type in self.member_types]
    new_struct = StructType(
      struct_identifier=self.struct_identifier, templ_types=new_templ_types, member_identifiers=self.member_identifiers,
      member_types=new_member_types, member_mutables=self.member_mutables, pass_by_ref=self.pass_by_ref,
      constructor=self.constructor)
    return new_struct

  def has_templ_placeholder(self) -> bool:
    return any(templ_type.has_templ_placeholder() for templ_type in self.templ_types)

  def make_ir_alloca(self, context: CodegenContext) -> ir.instructions.Instruction:
    assert context.emits_ir
    if self.is_pass_by_ref():  # use malloc
      assert context.ir_func_malloc is not None
      self_ir_alloca_raw = context.builder.call(
        context.ir_func_malloc, [self.make_ir_size(ctypes.sizeof(self.c_val_type))], name='self_raw_ptr')
      return context.builder.bitcast(self_ir_alloca_raw, self.ir_type, name='self')
    else:  # pass by value, use alloca
      return context.alloca_at_entry(self.ir_type, name='self')

  def make_di_type(self, context: CodegenContext) -> ir.DIValue:
    assert context.emits_debug
    di_derived_types = [
      context.module.add_debug_info(
        'DIDerivedType', {
          'tag': ir.DIToken('DW_TAG_inheritance'), 'baseType': member_type.make_di_type(context=context),
          'size': member_type.size, 'align': member_type.size})
      for member_type in self.member_types]
    return context.module.add_debug_info(
      'DICompositeType', {
        'name': repr(self), 'size': self.size, 'align': self.size, 'tag': ir.DIToken('DW_TAG_structure_type'),
        'file': context.current_di_file, 'elements': di_derived_types})

  def __eq__(self, other) -> bool:
    if not isinstance(other, StructType):
      return False
    # TODO: we do not want duck-typing here: keep track of which TypeSymbol this actually belongs to and compare those
    # instead of comparing the identifiers.
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
    return self.struct_identifier + ('' if len(self.templ_types) == 0 else str(self.templ_types))

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
    assert not parent_context.emits_debug or parent_context.builder.debug_metadata is not None
    pos = parent_context.current_pos

    class ConstructorConcreteFuncFactor(ConcreteFunctionFactory):
      def build_concrete_func_ir(self_, concrete_func: ConcreteFunction):
        with parent_context.use_pos(pos):
          concrete_struct_type = concrete_func.return_type
          assert isinstance(concrete_struct_type, StructType)

          if parent_context.emits_ir:
            concrete_func.make_ir_func(
              identifier=self.struct_identifier, extern=False, symbol_table=parent_symbol_table, context=parent_context)
            constructor_block = concrete_func.ir_func.append_basic_block(name='entry')
            context = parent_context.copy_with_func(concrete_func, builder=ir.IRBuilder(constructor_block))
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

    constructor_symbol = FunctionSymbol(identifier=self.struct_identifier, returns_void=False)
    placeholder_templ_types = [templ_type for templ_type in self.templ_types if isinstance(templ_type, TemplateType)]
    signature_ = FunctionTemplate(
      concrete_func_factory=ConstructorConcreteFuncFactor(),
      placeholder_templ_types=placeholder_templ_types, return_type=self, return_mutable=True,
      arg_identifiers=self.member_identifiers, arg_types=self.member_types, arg_mutables=self.member_mutables,
      arg_type_narrowings=self.member_types, is_inline=False)
    constructor_symbol.add_signature(signature_)
    return constructor_symbol

  def build_destructor(self, parent_symbol_table: SymbolTable, parent_context: CodegenContext) -> FunctionSymbol:
    destructor_symbol = parent_symbol_table['free']
    assert isinstance(destructor_symbol, FunctionSymbol)
    assert not parent_context.emits_debug or parent_context.builder.debug_metadata is not None
    pos = parent_context.current_pos

    class DestructorConcreteFuncFactory(ConcreteFunctionFactory):
      def build_concrete_func_ir(self_, concrete_func: ConcreteFunction):
        with parent_context.use_pos(pos):
          assert len(concrete_func.arg_types) == 1
          concrete_struct_type = concrete_func.arg_types[0]
          assert isinstance(concrete_struct_type, StructType)
          if parent_context.emits_ir:
            concrete_func.make_ir_func(
              identifier='free', extern=False, symbol_table=parent_symbol_table, context=parent_context)
            destructor_block = concrete_func.ir_func.append_basic_block(name='entry')
            context = parent_context.copy_with_func(concrete_func, builder=ir.IRBuilder(destructor_block))

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
              # TODO: properly infer templ types, also for struct members
              assert not (isinstance(signature_member_type, StructType) and len(signature_member_type.templ_types) > 0), (  # noqa
                'not implemented yet')
              templ_types: List[Type] = []
              if isinstance(concrete_member_type, PointerType):
                templ_types = [concrete_member_type.pointee_type]
              make_func_call_ir(
                func=destructor_symbol, templ_types=templ_types,
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
    signature_ = FunctionTemplate(
      concrete_func_factory=DestructorConcreteFuncFactory(),
      placeholder_templ_types=placeholder_templ_types, return_type=SLEEPY_VOID, return_mutable=False,
      arg_types=[self], arg_identifiers=['var'], arg_type_narrowings=[SLEEPY_NEVER],
      arg_mutables=[False])
    destructor_symbol.add_signature(signature_)
    return destructor_symbol

  def children(self) -> List[Type]:
    return self.member_types

  def has_same_symbol_as(self, other: Type) -> bool:
    # TODO: either store the actual symbol, or keep track of a type base (= the signature type)
    # note: cannot compare types, those might change if they are templates.
    return (
      isinstance(other, StructType) and other.struct_identifier == self.struct_identifier
      and self.member_identifiers == other.member_identifiers)


class TemplateType(Type):
  """
  A template parameter.
  """

  def __init__(self, identifier: str):
    super().__init__(ir_type=None, pass_by_ref=False, c_type=None, constructor=None)
    self.identifier = identifier

  def make_di_type(self, context: CodegenContext):
    assert False, self

  def __repr__(self):
    return self.identifier

  def has_templ_placeholder(self) -> bool:
    return True

  def children(self) -> List[Type]:
    return []

  @property
  def templ_types(self) -> List[Type]:
    return []

  def has_same_symbol_as(self, other: Type) -> bool:
    return self == other

  @property
  def constructor(self) -> Optional[FunctionSymbol]:
    return None


SLEEPY_VOID = VoidType()
SLEEPY_NEVER = UnionType(possible_types=[], possible_type_nums=[], val_size=0)
SLEEPY_DOUBLE = DoubleType()
SLEEPY_FLOAT = FloatType()
SLEEPY_BOOL = BoolType()
SLEEPY_INT = IntType()
SLEEPY_LONG = LongType()
SLEEPY_CHAR = CharType()
SLEEPY_RAW_PTR = RawPointerType()
SLEEPY_CHAR_PTR = PointerType(SLEEPY_CHAR)

SLEEPY_TYPES: Dict[str, Type] = {
  'Void': SLEEPY_VOID, 'Double': SLEEPY_DOUBLE, 'Float': SLEEPY_FLOAT, 'Bool': SLEEPY_BOOL, 'Int': SLEEPY_INT,
  'Long': SLEEPY_LONG, 'Char': SLEEPY_CHAR}

INT_TYPES: Set[Type] = {SLEEPY_INT, SLEEPY_LONG}
FLOAT_TYPES: Set[Type] = {SLEEPY_FLOAT, SLEEPY_DOUBLE}

SLEEPY_NUMERICAL_TYPES: Set[Type] = INT_TYPES | FLOAT_TYPES

COMPARABLE_TYPES = SLEEPY_NUMERICAL_TYPES | {SLEEPY_BOOL}


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
  kind = Symbol.Kind.VARIABLE

  def __init__(self, ir_alloca: Optional[ir.instructions.AllocaInstr], var_type: Type, mutable: bool):
    super().__init__()
    assert ir_alloca is None or isinstance(ir_alloca, ir.instructions.AllocaInstr)
    assert var_type != SLEEPY_VOID
    self.ir_alloca = ir_alloca
    self.declared_var_type = var_type
    self.narrowed_var_type = var_type
    self.mutable = mutable

  def copy_replace_unbound_templ_types(self, templ_type_replacements: Dict[TemplateType, Type]) -> VariableSymbol:
    assert not self.declared_var_type.has_templ_placeholder()
    assert not self.narrowed_var_type.has_templ_placeholder()
    return self

  def copy_with_narrowed_type(self, new_narrow_type: Type) -> VariableSymbol:
    new_var_symbol = VariableSymbol(self.ir_alloca, self.declared_var_type, self.mutable)
    new_var_symbol.base = self if self.base is None else self.base
    # explicitly apply narrowing from declared type here: always stay compatible to the base type
    new_var_symbol.narrowed_var_type = narrow_type(from_type=self.declared_var_type, narrow_to=new_narrow_type)
    return new_var_symbol

  def copy_narrow_type(self, narrow_to: Type) -> VariableSymbol:
    return self.copy_with_narrowed_type(new_narrow_type=narrow_type(self.narrowed_var_type, narrow_to))

  def copy_reset_narrowed_type(self) -> VariableSymbol:
    return self.copy_with_narrowed_type(new_narrow_type=self.declared_var_type)

  def copy_exclude_type(self, excluded: Type) -> VariableSymbol:
    return self.copy_with_narrowed_type(new_narrow_type=exclude_type(self.narrowed_var_type, excluded))

  def build_ir_alloca(self, context: CodegenContext, identifier: str):
    assert self.ir_alloca is None
    if not context.emits_ir:
      return
    self.ir_alloca = context.alloca_at_entry(self.declared_var_type.ir_type, name='%s_ptr' % identifier)
    if context.emits_debug:
      assert context.di_declare_func is not None
      di_local_var = context.module.add_debug_info(
        'DILocalVariable', {
          'name': identifier, 'scope': context.current_di_scope, 'file': context.current_di_file,
          'line': context.current_pos.get_from_line(), 'type': self.declared_var_type.make_di_type(context=context)})
      di_expression = context.module.add_debug_info('DIExpression', {})
      assert context.builder.debug_metadata is not None
      context.builder.call(context.di_declare_func, args=[self.ir_alloca, di_local_var, di_expression])

  def __repr__(self) -> str:
    return 'VariableSymbol(ir_alloca=%r, declared_var_type=%r, narrowed_var_type=%r, mutable=%r)' % (
      self.ir_alloca, self.declared_var_type, self.narrowed_var_type, self.mutable)


class ConcreteFunction:
  """
  An actual function implementation.
  """
  def __init__(self,
               signature: FunctionTemplate,
               ir_func: Optional[ir.Function],
               make_inline_func_call_ir_callback: Optional[Callable[[List[ir.values.Value], CodegenContext, TreePosition], Optional[ir.values.Value]]],  # noqa
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
    self.di_subprogram: Optional[ir.DIValue] = None

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

  def make_ir_func(self, identifier: str, extern: bool, symbol_table: SymbolTable, context: CodegenContext):
    assert context.emits_ir
    assert not self.is_inline
    assert self.ir_func is None
    ir_func_name = symbol_table.make_ir_func_name(identifier, extern=extern, concrete_func=self)
    ir_func_type = self.make_ir_function_type()
    self.ir_func = ir.Function(context.module, ir_func_type, name=ir_func_name)
    if context.emits_debug and not extern:
      assert self.di_subprogram is None
      di_func_type = context.module.add_debug_info(
        'DISubroutineType', {'types': context.module.add_metadata([None])})
      self.di_subprogram = context.module.add_debug_info(
        'DISubprogram', {
          'name': ir_func_name, 'file': context.current_di_file, 'scope': context.current_di_scope,
          'line': context.current_pos.get_from_line(),
          'type': di_func_type,
          'isLocal': False, 'isDefinition': True,
          'unit': context.current_di_compile_unit
        }, is_distinct=True)
      self.ir_func.set_metadata('dbg', self.di_subprogram)


class ConcreteFunctionFactory:
  """
  Generates concrete function implementations.
  """
  def build_concrete_func_ir(self, concrete_func: ConcreteFunction):
    raise NotImplementedError()


class FunctionTemplate:
  """
  Given template arguments, this builds a concrete function implementation on demand.
  """
  def __init__(self,
               concrete_func_factory: ConcreteFunctionFactory,
               placeholder_templ_types: List[TemplateType],
               return_type: Type, return_mutable: bool,
               arg_identifiers: List[str], arg_types: List[Type], arg_mutables: List[bool],
               arg_type_narrowings: List[Type],
               is_inline: bool = False,
               base: FunctionTemplate = None):
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
    if base is not None:
      assert base.concrete_func_factory == self.concrete_func_factory
      assert base.arg_identifiers == self.arg_identifiers
      assert base.arg_mutables == self.arg_mutables
      assert base.is_inline == self.is_inline
      # we share the initialized_templ_funcs here, so that we do not create a ConcreteFunction multiple times
      self.initialized_templ_funcs: Dict[Tuple[Type], ConcreteFunction] = base.initialized_templ_funcs
    else:
      self.initialized_templ_funcs: Dict[Tuple[Type], ConcreteFunction] = {}

  def copy_replace_unbound_templ_types(self, templ_type_replacements: Dict[TemplateType, Type]) -> FunctionTemplate:
    assert all(isinstance(key, TemplateType) for key in templ_type_replacements)
    unbound_templ_type_replacements = {
      templ_type: replacement for templ_type, replacement in templ_type_replacements.items()
      if templ_type not in self.placeholder_templ_types}
    return self.copy_replace_types(unbound_templ_type_replacements)

  def copy_replace_types(self, type_replacements: Dict[Type, Type]) -> FunctionTemplate:
    new_placeholder_templ_types = [
      type_replacements.get(templ_type, templ_type) for templ_type in self.placeholder_templ_types]
    new_placeholder_templ_types = [
      templ_type for templ_type in new_placeholder_templ_types if isinstance(templ_type, TemplateType)]
    return FunctionTemplate(
      concrete_func_factory=self.concrete_func_factory,
      placeholder_templ_types=new_placeholder_templ_types,
      return_type=self.return_type.replace_types(type_replacements), return_mutable=self.return_mutable,
      arg_identifiers=self.arg_identifiers.copy(),
      arg_types=[arg_type.replace_types(type_replacements) for arg_type in self.arg_types.copy()],
      arg_mutables=self.arg_mutables.copy(), arg_type_narrowings=self.arg_type_narrowings.copy(),
      is_inline=self.is_inline,
      base=self)

  def to_signature_str(self) -> str:
    templ_args = '' if len(self.placeholder_templ_types) == 0 else '[%s]' % (
      ', '.join([templ_type.identifier for templ_type in self.placeholder_templ_types]))
    args = ', '.join(['%s: %s' % arg_tuple for arg_tuple in zip(self.arg_identifiers, self.arg_types)])
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
  def returns_void(self) -> bool:
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

  def is_undefined_for_expanded_arg_types(self, placeholder_templ_types: List[TemplateType],
                                          expanded_arg_types: List[Type]) -> bool:
    assert all(not isinstance(arg_type, UnionType) for arg_type in expanded_arg_types)
    if len(placeholder_templ_types) != len(self.placeholder_templ_types):
      return True
    if len(expanded_arg_types) != len(self.arg_types):
      return True
    # convert own template variables to calling template variables s.t. we can try to unify them
    templ_to_templ_replacements = dict(zip(self.placeholder_templ_types, placeholder_templ_types))
    own_arg_types = [arg_type.replace_types(templ_to_templ_replacements) for arg_type in self.arg_types]
    inferred_templ_types = try_infer_templ_types(
      calling_types=expanded_arg_types, signature_types=own_arg_types, placeholder_templ_types=placeholder_templ_types)
    return inferred_templ_types is None


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
  kind = Symbol.Kind.FUNCTION

  def __init__(self, identifier: str, returns_void: bool, base: Optional[FunctionSymbol] = None):
    super().__init__()
    self.identifier = identifier
    self.signatures_by_number_of_templ_args: Dict[int, List[FunctionTemplate]] = {}
    self.returns_void = returns_void
    if base is None:
      base = self
    elif base.base is not None:
      base = base.base
    self.base = base

  def copy_replace_types(self, type_replacements: Dict[Type, Type]) -> FunctionSymbol:
    new_func_symbol = FunctionSymbol(identifier=self.identifier, returns_void=self.returns_void, base=self)
    new_func_symbol.signatures_by_number_of_templ_args = {
      num_templ_args: [signature.copy_replace_types(type_replacements) for signature in signatures]
      for num_templ_args, signatures in self.signatures_by_number_of_templ_args.items()}
    return new_func_symbol

  def copy_replace_unbound_templ_types(self, templ_type_replacements: Dict[TemplateType, Type]) -> FunctionSymbol:
    """
    e.g. if replacements is T => Int:
     - from foo(T val) -> T make foo(Int val)
     - do not change bound variables: foo[T](T val) -> T stays the same
    """
    new_func_symbol = FunctionSymbol(identifier=self.identifier, returns_void=self.returns_void, base=self)
    new_func_symbol.signatures_by_number_of_templ_args = {
      num_templ_args: [signature.copy_replace_unbound_templ_types(templ_type_replacements) for signature in signatures]
      for num_templ_args, signatures in self.signatures_by_number_of_templ_args.items()}
    return new_func_symbol

  @property
  def signatures(self) -> List[FunctionTemplate]:
    return [signature for signatures in self.signatures_by_number_of_templ_args.values() for signature in signatures]

  def can_call_with_expanded_arg_types(self, concrete_templ_types: List[Type], expanded_arg_types: List[Type]) -> bool:
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
    all_expanded_arg_types = self.iter_expanded_possible_arg_types(arg_types)
    return all(
      self.can_call_with_expanded_arg_types(concrete_templ_types=concrete_templ_types, expanded_arg_types=arg_types)
      for arg_types in all_expanded_arg_types)

  def is_undefined_for_arg_types(self, placeholder_templ_types: List[TemplateType], arg_types: List[Type]):
    signatures = self.signatures_by_number_of_templ_args.get(len(placeholder_templ_types), [])
    return all(
      signature.is_undefined_for_expanded_arg_types(
        placeholder_templ_types=placeholder_templ_types, expanded_arg_types=expanded_arg_types)
      for signature in signatures for expanded_arg_types in self.iter_expanded_possible_arg_types(arg_types))

  def get_concrete_funcs(self, templ_types: List[Type], arg_types: List[Type]) -> List[ConcreteFunction]:
    signatures = self.signatures_by_number_of_templ_args.get(len(templ_types), [])
    possible_concrete_funcs = []
    for expanded_arg_types in self.iter_expanded_possible_arg_types(arg_types):
      for signature in signatures:
        if signature.can_call_with_expanded_arg_types(
            concrete_templ_types=templ_types, expanded_arg_types=expanded_arg_types):
          concrete_func = signature.get_concrete_func(concrete_templ_types=templ_types)
          if concrete_func not in possible_concrete_funcs:
            possible_concrete_funcs.append(concrete_func)
    return possible_concrete_funcs

  def add_signature(self, signature: FunctionTemplate):
    assert self.is_undefined_for_arg_types(
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
  def iter_expanded_possible_arg_types(cls, arg_types):
    """
    :param list[Type]|tuple[Type] arg_types:
    :rtype: Iterator[Tuple[Type]]
    """
    import itertools
    return itertools.product(*[
      arg_type.possible_types if isinstance(arg_type, UnionType) else [arg_type] for arg_type in arg_types])

  def __repr__(self) -> str:
    return 'FunctionSymbol(identifier=%r, signatures=%r)' % (self.identifier, self.signatures)

  def make_signature_list_str(self) -> str:
    return '\n'.join([' - ' + signature.to_signature_str() for signature in self.signatures])


class FunctionSymbolCaller:
  """
  A FunctionSymbol plus template types it is called with.
  """
  def __init__(self, func: FunctionSymbol, templ_types: Optional[List[Type]] = None):
    self.func = func
    self.templ_types = templ_types

  def copy_with_templ_types(self, templ_types: List[Type]) -> FunctionSymbolCaller:
    assert self.templ_types is None
    return FunctionSymbolCaller(func=self.func, templ_types=templ_types)


class TypeFactory:
  """
  Lazily generates concrete types with initialized template arguments.
  """
  def __init__(self, placeholder_templ_types: List[TemplateType], signature_type: Type):
    self.placeholder_templ_types = placeholder_templ_types
    self.signature_type = signature_type

  def make_concrete_type(self, concrete_templ_types: List[Type]) -> Type:
    replacements = dict(zip(self.placeholder_templ_types, concrete_templ_types))
    return self.signature_type.replace_types(replacements)


class TypeSymbol(Symbol):
  """
  A (statically) declared (possibly) template type.
  Can have one or multiple template initializations that yield different concrete types.
  These are initialized lazily.
  """
  kind = Symbol.Kind.TYPE

  def __init__(self, type_factory: TypeFactory):
    super().__init__()
    assert isinstance(type_factory, TypeFactory)
    self.type_factory = type_factory
    self.initialized_templ_types: Dict[Tuple[Type], Type] = {}

  def copy_replace_unbound_templ_types(self, templ_type_replacements: Dict[TemplateType, Type]) -> TypeSymbol:
    """
    e.g. if replacements is T -> Int:
     - from T make Int
     - from List[T] make List[Int]
     - from U make U
    """
    new_signature_type = self.type_factory.signature_type.replace_types(replacements=templ_type_replacements)
    new_type_factory = TypeFactory(
      placeholder_templ_types=[
        templ for templ in self.type_factory.placeholder_templ_types if templ not in templ_type_replacements],
      signature_type=new_signature_type)
    return TypeSymbol(type_factory=new_type_factory)

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
      if copy_new_current_func is None:
        self.symbols = copy_from.symbols.copy()  # type: Dict[str, Symbol]
        self.current_func = copy_from.current_func  # type: Optional[ConcreteFunction]
        self.current_scope_identifiers = copy_from.current_scope_identifiers.copy()  # type: List[str]
      else:
        templ_type_replacements = dict(zip(
          copy_new_current_func.signature.placeholder_templ_types, copy_new_current_func.concrete_templ_types))
        self.symbols = {
          identifier: symbol.copy_replace_unbound_templ_types(templ_type_replacements)
          for identifier, symbol in copy_from.symbols.items()}
        self.current_func = copy_new_current_func  # type: Optional[ConcreteFunction]
        self.current_scope_identifiers = []  # type: List[str]
      self.used_ir_func_names = copy_from.used_ir_func_names  # type: Set[str]
      # do not copy known_extern_funcs, but reference back as we want those to be shared globally
      self.known_extern_funcs = copy_from.known_extern_funcs  # type: Dict[str, ConcreteFunction]
      self.inbuilt_symbols = copy_from.inbuilt_symbols  # type: Dict[str, Symbol]

  def __setitem__(self, identifier: str, symbol: Symbol):
    self.symbols[identifier] = symbol

  def __getitem__(self, identifier: str) -> Symbol:
    return self.symbols[identifier]

  def __contains__(self, identifier: str) -> bool:
    return identifier in self.symbols

  def copy(self) -> SymbolTable:
    return SymbolTable(self)

  def copy_with_new_current_func(self, new_current_func: ConcreteFunction) -> SymbolTable:
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
      ir_func_name = '_'.join(
        [func_identifier]
        + [str(arg_type) for arg_type in concrete_func.concrete_templ_types + concrete_func.arg_types])
      assert ir_func_name not in self.used_ir_func_names
    self.used_ir_func_names.add(ir_func_name)
    return ir_func_name

  def apply_type_narrowings_from(self, *other_symbol_tables: SymbolTable):
    """
    For all variable symbols, copy common type of all other_symbol_tables.
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

  def has_extern_func(self, func_identifier: str) -> bool:
    return func_identifier in self.known_extern_funcs

  def get_extern_func(self, func_identifier: str) -> ConcreteFunction:
    assert self.has_extern_func(func_identifier)
    return self.known_extern_funcs[func_identifier]

  def add_extern_func(self, func_identifier: str, concrete_func: ConcreteFunction):
    assert not self.has_extern_func(func_identifier)
    self.known_extern_funcs[func_identifier] = concrete_func

  @property
  def free_symbol(self) -> FunctionSymbol:
    assert 'free' in self
    free_symbol = self['free']
    assert isinstance(free_symbol, FunctionSymbol)
    return free_symbol

  def add_to_current_scope(self, identifier: str, symbol: Symbol):
    self.current_scope_identifiers.append(identifier)
    self[identifier] = symbol


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
      self.emits_debug = self.emits_ir  # type: bool
      self.is_terminated = False

      self.current_pos: Optional[TreePosition] = None
      self.current_func: Optional[ConcreteFunction] = None
      self.current_func_inline_return_collect_block = None  # type: Optional[ir.Block]
      self.current_func_inline_return_ir_alloca = None  # type: Optional[ir.instructions.AllocaInstr]
      self.inline_func_call_stack = []  # type: List[ConcreteFunction]
      self.ir_func_malloc = None  # type: Optional[ir.Function]
      self.ir_func_free = None  # type: Optional[ir.Function]

      self.current_di_file: Optional[ir.DIValue] = None
      self.current_di_compile_unit: Optional[ir.DIValue] = None
      self.current_di_scope: Optional[ir.DIValue] = None
      self.di_declare_func: Optional[ir.Function] = None
    else:
      self.emits_ir = copy_from.emits_ir
      self.emits_debug = copy_from.emits_debug  # type: bool
      self.is_terminated = copy_from.is_terminated

      self.current_pos: Optional[TreePosition] = copy_from.current_pos
      self.current_func: Optional[ConcreteFunction] = copy_from.current_func
      self.current_func_inline_return_collect_block = copy_from.current_func_inline_return_collect_block  # type: Optional[ir.Block]  # noqa
      self.current_func_inline_return_ir_alloca = copy_from.current_func_inline_return_ir_alloca  # type: Optional[ir.instructions.AllocaInstr]  # noqa
      self.inline_func_call_stack = copy_from.inline_func_call_stack.copy()  # type: List[ConcreteFunction]
      self.ir_func_malloc = copy_from.ir_func_malloc  # type: Optional[ir.Function]
      self.ir_func_free = copy_from.ir_func_free  # type: Optional[ir.Function]

      self.current_di_file: Optional[ir.DIValue] = copy_from.current_di_file
      self.current_di_compile_unit: Optional[ir.DIValue] = copy_from.current_di_compile_unit
      self.current_di_scope: Optional[ir.DIValue] = copy_from.current_di_scope
      self.di_declare_func: Optional[ir.Function] = copy_from.di_declare_func

      if self.builder is not None:
        self.builder.debug_metadata = copy_from.builder.debug_metadata

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
    new_context.emits_debug = False
    return new_context

  def copy_with_func(self, concrete_func: ConcreteFunction, builder: Optional[ir.IRBuilder]):
    assert not concrete_func.is_inline
    new_context = self.copy_with_builder(builder)
    new_context.current_func = concrete_func
    new_context.current_func_inline_return_ir_alloca = None
    new_context.current_func_inline_return_collect_block = None
    if builder is None:
      new_context.emits_ir = False
      new_context.emits_debug = False
    if new_context.emits_debug:
      assert concrete_func.di_subprogram is not None
      new_context.current_di_scope = concrete_func.di_subprogram
      # as the current scope changed, reset the debug_metadata
      new_context.builder.debug_metadata = make_di_location(pos=new_context.current_pos, context=new_context)
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
    new_context.current_func = concrete_func
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

  def use_pos(self, pos: TreePosition) -> UsePosRuntimeContext:
    return UsePosRuntimeContext(pos, context=self)


class UsePosRuntimeContext:
  def __init__(self, pos: TreePosition, context: CodegenContext):
    self.pos = pos
    self.context = context

  def __enter__(self):
    if self.context.is_terminated:
      return
    self.prev_pos = self.context.current_pos
    self.context.current_pos = self.pos
    if self.context.emits_debug:
      self.prev_debug_metadata = self.context.builder.debug_metadata
      self.context.builder.debug_metadata = make_di_location(self.pos, context=self.context)

  def __exit__(self, exc_type, exc_val, exc_tb):
    if self.context.is_terminated:
      return
    self.context.current_pos = self.prev_pos
    if self.context.emits_debug:
      self.context.builder.debug_metadata = self.prev_debug_metadata


def make_func_call_ir(func: FunctionSymbol, templ_types: List[Type],
                      calling_arg_types: List[Type], calling_ir_args: List[ir.values.Value],
                      context: CodegenContext) -> Optional[ir.values.Value]:
  assert context.emits_ir
  assert not context.emits_debug or context.builder.debug_metadata is not None
  assert len(calling_arg_types) == len(calling_ir_args)

  def make_call_func(concrete_func, concrete_calling_arg_types_, caller_context):
    """
    :param ConcreteFunction concrete_func:
    :param list[Type] concrete_calling_arg_types_:
    :param CodegenContext caller_context:
    :rtype: ir.values.Value
    """
    assert caller_context.emits_ir
    assert not caller_context.emits_debug or caller_context.builder.debug_metadata is not None
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
      return caller_context.builder.call(ir_func, casted_ir_args, name='call_%s' % func.identifier)

  assert func.can_call_with_arg_types(concrete_templ_types=templ_types, arg_types=calling_arg_types)
  possible_concrete_funcs = func.get_concrete_funcs(templ_types=templ_types, arg_types=calling_arg_types)
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
        func.identifier, '_'.join(str(arg_type) for arg_type in concrete_func.arg_types)))))
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
        ir_func_arg, context=context, name='call_%s_arg%s_tag_ptr' % (func.identifier, arg_num))
      call_block_index_ir = context.builder.add(call_block_index_ir, context.builder.mul(base_ir, tag_ir))
    call_block_index_ir = context.builder.zext(
      call_block_index_ir, LLVM_SIZE_TYPE, name='call_%s_block_index' % func.identifier)

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
    collect_block = context.builder.append_basic_block("collect_%s_overload" % func.identifier)
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
      assert concrete_func.signature.returns_void == func.returns_void
    assert len(possible_concrete_funcs) == len(concrete_func_return_ir_vals)

    if func.returns_void:
      return None
    else:
      common_return_type = get_common_type([concrete_func.return_type for concrete_func in possible_concrete_funcs])
      collect_return_ir_phi = context.builder.phi(
        common_return_type.ir_type, name="collect_%s_overload_return" % func.identifier)
      for concrete_return_ir_val, concrete_caller_context in zip(
          concrete_func_return_ir_vals, concrete_func_caller_contexts):
        collect_return_ir_phi.add_incoming(concrete_return_ir_val, concrete_caller_context.block)
      return collect_return_ir_phi


class BuiltinBinaryOps(Enum):
  Addition = '+'
  Subtraction = '-'
  Multiplication = '*'
  Division = '/'
  Equality = '=='
  Inequality = '!='
  Less = '<'
  Greater = '>'
  LessOrEqual = '<='
  GreaterOrEqual = '>='
  BitwiseOr = 'bitwise_or'
  Mod = 'mod'


Simple_Arithmetic_Ops: List[BuiltinBinaryOps] = \
  [BuiltinBinaryOps.Addition, BuiltinBinaryOps.Subtraction, BuiltinBinaryOps.Multiplication, BuiltinBinaryOps.Division]


Simple_Comparison_Ops: List[BuiltinBinaryOps] = \
 [BuiltinBinaryOps.Equality, BuiltinBinaryOps.Inequality, BuiltinBinaryOps.Less, BuiltinBinaryOps.Greater,
  BuiltinBinaryOps.GreaterOrEqual, BuiltinBinaryOps.LessOrEqual]


class BitcastFunctionFactory(ConcreteFunctionFactory):
  def __init__(self, emits_ir: bool):
    self.emits_ir = emits_ir

  def build_concrete_func_ir(self, concrete_func: ConcreteFunction):
    assert concrete_func.is_inline
    if self.emits_ir:
      to_type = concrete_func.return_type
      concrete_func.make_inline_func_call_ir = lambda caller_context, ir_func_args: (
        caller_context.builder.bitcast(ir_func_args[0], typ=to_type.ir_type, name='bitcast'))


def _make_str_symbol(symbol_table: SymbolTable, context: CodegenContext) -> TypeSymbol:
  str_type = StructType(
    struct_identifier='Str', member_identifiers=['start', 'length', 'alloc_length'], templ_types=[],
    member_types=[SLEEPY_CHAR_PTR, SLEEPY_INT, SLEEPY_INT], member_mutables=[False, False, False],
    pass_by_ref=True)
  constructor_symbol = str_type.build_constructor(parent_symbol_table=symbol_table, parent_context=context)
  str_type.constructor = constructor_symbol
  type_factory = TypeFactory(placeholder_templ_types=[], signature_type=str_type)
  struct_symbol = TypeSymbol(type_factory=type_factory)
  str_type.build_destructor(parent_symbol_table=symbol_table, parent_context=context)
  return struct_symbol


def _make_ptr_symbol(symbol_table: SymbolTable, context: CodegenContext) -> TypeSymbol:
  pointee_type = TemplateType(identifier='T')
  ptr_type = PointerType(pointee_type=pointee_type)

  assert 'load' not in symbol_table
  load_symbol = FunctionSymbol(identifier='load', returns_void=False)
  load_factory = InbuiltOpConcreteFuncFactory(
    instruction=lambda builder, ptr: builder.load(ptr, name='load'), emits_ir=context.emits_ir)
  load_signature = FunctionTemplate(
    concrete_func_factory=load_factory, placeholder_templ_types=[pointee_type], return_type=pointee_type,
    return_mutable=False, arg_identifiers=['ptr'], arg_types=[ptr_type], arg_mutables=[False],
    arg_type_narrowings=[ptr_type], is_inline=True)
  load_symbol.add_signature(signature=load_signature)
  symbol_table['load'] = load_symbol

  assert 'store' not in symbol_table
  store_symbol = FunctionSymbol(identifier='store', returns_void=True)
  store_factory = InbuiltOpConcreteFuncFactory(
    instruction=lambda builder, ptr, value: builder.store(value=value, ptr=ptr),
    emits_ir=context.emits_ir)
  store_signature = FunctionTemplate(
    concrete_func_factory=store_factory, placeholder_templ_types=[pointee_type], return_type=SLEEPY_VOID,
    return_mutable=True, arg_identifiers=['ptr', 'value'], arg_types=[ptr_type, pointee_type],
    arg_mutables=[False, False], arg_type_narrowings=[ptr_type, pointee_type], is_inline=True)
  store_symbol.add_signature(store_signature)
  symbol_table['store'] = store_symbol
  type_factory = TypeFactory(placeholder_templ_types=[pointee_type], signature_type=ptr_type)

  # cast from RawPtr -> Ptr[T]
  constructor_symbol = FunctionSymbol(identifier='Ptr', returns_void=False)
  constructor_factory = BitcastFunctionFactory(emits_ir=context.emits_ir)
  constructor_signature = FunctionTemplate(
    concrete_func_factory=constructor_factory, placeholder_templ_types=[pointee_type], return_type=ptr_type,
    return_mutable=False, arg_identifiers=['raw_ptr'], arg_types=[SLEEPY_RAW_PTR], arg_mutables=[False],
    arg_type_narrowings=[ptr_type], is_inline=True)
  constructor_symbol.add_signature(signature=constructor_signature)
  ptr_type.constructor = constructor_symbol

  PTR_OP_DECL = [(
    BuiltinBinaryOps.Addition,
    [(lambda builder, lhs, rhs: builder.gep(lhs, (rhs,)), [ptr_type, i], ptr_type) for i in INT_TYPES] +
    [(lambda builder, lhs, rhs: builder.gep(rhs, (lhs,)), [i, ptr_type], ptr_type) for i in INT_TYPES]),
    (BuiltinBinaryOps.Subtraction,
    [(lambda builder, lhs, rhs: builder.gep(lhs, (builder.mul(ir.Constant(i.ir_type, -1), rhs),)), [ptr_type, i], ptr_type) for i in INT_TYPES])]  # noqa
  PTR_OP_DECL += [
    (
      op,
      [(lambda builder, lhs, rhs, op=op: builder.icmp_unsigned(op.value, lhs, rhs), [ptr_type, ptr_type], SLEEPY_BOOL)])
    for op in Simple_Comparison_Ops]
  for operator, overloads in PTR_OP_DECL:
    if operator.value not in symbol_table:
      symbol_table[operator.value] = FunctionSymbol(identifier=operator.value, returns_void=False)
    function_symbol = symbol_table[operator.value]
    assert isinstance(function_symbol, FunctionSymbol)

    for instruction, arg_types, return_type in overloads:
      signature = make_function_signature(
        instruction, op_placeholder_templ_types=[pointee_type], op_arg_types=arg_types, op_return_type=return_type,
        emits_ir=context.emits_ir)
      function_symbol.add_signature(signature)

  destructor_factory = get_dummy_destructor_function_factory(emits_ir=context.emits_ir)
  free_signature = FunctionTemplate(
    concrete_func_factory=destructor_factory, placeholder_templ_types=[pointee_type], return_type=SLEEPY_VOID,
    return_mutable=False, arg_types=[ptr_type], arg_identifiers=['ptr'], arg_type_narrowings=[SLEEPY_NEVER],
    arg_mutables=[False], is_inline=True)
  symbol_table.free_symbol.add_signature(free_signature)

  return TypeSymbol(type_factory=type_factory)


def _make_raw_ptr_symbol(symbol_table: SymbolTable, context: CodegenContext) -> TypeSymbol:
  # add destructor
  destructor_factory = get_dummy_destructor_function_factory(emits_ir=context.emits_ir)
  destructor_signature = FunctionTemplate(
    concrete_func_factory=destructor_factory, placeholder_templ_types=[], return_type=SLEEPY_VOID,
    return_mutable=False, arg_types=[SLEEPY_RAW_PTR], arg_identifiers=['raw_ptr'], arg_type_narrowings=[SLEEPY_NEVER],
    arg_mutables=[False], is_inline=True)
  symbol_table.free_symbol.add_signature(destructor_signature)

  pointee_type = TemplateType(identifier='T')
  ptr_type = PointerType(pointee_type=pointee_type)
  constructor_symbol = FunctionSymbol(identifier='RawPtr', returns_void=False)
  # RawPtr[T](Ptr[T]) -> RawPtr
  from_specific_factory = InbuiltOpConcreteFuncFactory(
    instruction=lambda builder, typed_ptr: builder.bitcast(typed_ptr, typ=SLEEPY_RAW_PTR.ir_type, name='ptr_cast'),
    emits_ir=context.emits_ir)
  from_specific_signature = FunctionTemplate(
    concrete_func_factory=from_specific_factory, placeholder_templ_types=[pointee_type], return_type=SLEEPY_RAW_PTR,
    return_mutable=False, arg_identifiers=['ptr'], arg_types=[ptr_type], arg_mutables=[False],
    arg_type_narrowings=[ptr_type], is_inline=True)
  constructor_symbol.add_signature(signature=from_specific_signature)
  # RawPtr(Int) -> RawPtr
  for int_type in INT_TYPES:
    from_int_factory = InbuiltOpConcreteFuncFactory(
      instruction=lambda builder, int: builder.inttoptr(int, typ=SLEEPY_RAW_PTR.ir_type, name='int_to_ptr'),
      emits_ir=context.emits_ir)
    from_int_signature = FunctionTemplate(
      concrete_func_factory=from_int_factory, placeholder_templ_types=[], return_type=SLEEPY_RAW_PTR,
      return_mutable=False, arg_identifiers=['int'], arg_types=[int_type], arg_mutables=[False],
      arg_type_narrowings=[int_type], is_inline=True)
    constructor_symbol.add_signature(from_int_signature)
  SLEEPY_RAW_PTR.constructor = constructor_symbol

  type_generator = TypeFactory(placeholder_templ_types=[], signature_type=SLEEPY_RAW_PTR)
  raw_ptr_symbol = TypeSymbol(type_generator)
  return raw_ptr_symbol


def _make_bitcast_symbol(symbol_table: SymbolTable, context: CodegenContext) -> FunctionSymbol:
  bitcast_func = FunctionSymbol(identifier='bitcast', returns_void=False)
  func_factory = BitcastFunctionFactory(emits_ir=context.emits_ir)
  to_type, from_type = TemplateType('T'), TemplateType('U')
  bitcast_signature = FunctionTemplate(
    concrete_func_factory=func_factory, placeholder_templ_types=[to_type, from_type], return_type=to_type,
    return_mutable=False, arg_identifiers=['from'], arg_types=[from_type], arg_mutables=[False],
    arg_type_narrowings=[to_type], is_inline=True)
  bitcast_func.add_signature(bitcast_signature)
  return bitcast_func


def make_di_location(pos: TreePosition, context: CodegenContext):
  assert context.emits_debug
  assert context.current_di_scope is not None
  line, col = pos.get_from_line_col()
  return context.module.add_debug_info(
    'DILocation', {'line': line, 'column': col, 'scope': context.current_di_scope})


class InbuiltOpConcreteFuncFactory(ConcreteFunctionFactory):
  def __init__(self, instruction: Callable[..., Optional[ir.Instruction]], emits_ir: bool):
    self.instruction = instruction
    self.emits_ir = emits_ir

  def _make_inline_func_call_ir(self, ir_func_args: List[ir.Instruction],
                                caller_context: CodegenContext) -> ir.instructions.Instruction:
    return self.instruction(caller_context.builder, *ir_func_args)

  def build_concrete_func_ir(self, concrete_func: ConcreteFunction):
    assert concrete_func.is_inline
    if self.emits_ir:
      concrete_func.make_inline_func_call_ir = self._make_inline_func_call_ir
    return concrete_func


def get_dummy_destructor_function_factory(emits_ir: bool) -> ConcreteFunctionFactory:
  return InbuiltOpConcreteFuncFactory(instruction=lambda builder, value: None, emits_ir=emits_ir)


SLEEPY_SIZE_FUNC = FunctionSymbol(identifier='size', returns_void=False)


def build_initial_ir(symbol_table: SymbolTable, context: CodegenContext):
  if context.emits_debug:
    assert context.di_declare_func is None
    di_declare_func_type = ir.FunctionType(ir.VoidType(), [ir.MetaDataType()] * 3)
    context.di_declare_func = ir.Function(context.builder.module, di_declare_func_type, 'llvm.dbg.declare')

    # TODO: Proper filename / directory
    # TODO: Add compiler version
    producer = 'sleepy compiler'
    assert context.current_di_file is None and context.current_di_compile_unit is None
    context.current_di_file = context.module.add_debug_info(
      'DIFile', {'filename': 'tmp.slp', 'directory': '.'})
    context.current_di_compile_unit = context.module.add_debug_info(
      'DICompileUnit', {
        'language': ir.DIToken('DW_LANG_C'), 'file': context.current_di_file, 'producer': producer,
        'isOptimized': False, 'runtimeVersion': 1},
      is_distinct=True)
    context.current_di_scope = context.current_di_file

    context.module.add_named_metadata('llvm.dbg.cu', context.current_di_compile_unit)
    di_dwarf_version = [ir.Constant(ir.IntType(32), 2), 'Dwarf Version', ir.Constant(ir.IntType(32), 2)]
    di_debug_info_version = [ir.Constant(ir.IntType(32), 2), 'Debug Info Version', ir.Constant(ir.IntType(32), 3)]
    context.module.add_named_metadata('llvm.module.flags', di_dwarf_version)
    context.module.add_named_metadata('llvm.module.flags', di_debug_info_version)
    context.module.add_named_metadata('llvm.ident', [producer])

  assert 'free' not in symbol_table
  free_symbol = FunctionSymbol(identifier='free', returns_void=True)
  symbol_table['free'] = free_symbol

  assert 'get' not in symbol_table
  symbol_table['get'] = FunctionSymbol(identifier='get', returns_void=False)

  assert 'size' not in symbol_table
  symbol_table['size'] = SLEEPY_SIZE_FUNC

  for type_identifier, inbuilt_type in SLEEPY_TYPES.items():
    type_generator = TypeFactory(placeholder_templ_types=[], signature_type=inbuilt_type)
    assert type_identifier not in symbol_table
    symbol_table[type_identifier] = TypeSymbol(type_generator)
    if inbuilt_type == SLEEPY_VOID:
      continue

    # add destructor
    destructor_factory = get_dummy_destructor_function_factory(emits_ir=context.emits_ir)
    destructor_signature = FunctionTemplate(
      concrete_func_factory=destructor_factory, placeholder_templ_types=[], return_type=SLEEPY_VOID,
      return_mutable=False, arg_types=[inbuilt_type], arg_identifiers=['var'], arg_type_narrowings=[SLEEPY_NEVER],
      arg_mutables=[False], is_inline=True)
    symbol_table.free_symbol.add_signature(destructor_signature)

  for assert_identifier in ['assert', 'unchecked_assert']:
    assert assert_identifier not in symbol_table
    assert_symbol = FunctionSymbol(identifier=assert_identifier, returns_void=True)
    symbol_table[assert_identifier] = assert_symbol
    symbol_table.inbuilt_symbols[assert_identifier] = assert_symbol

  if context.emits_ir:
    context.ir_func_malloc = ir.Function(
      context.module, ir.FunctionType(LLVM_VOID_POINTER_TYPE, [LLVM_SIZE_TYPE]), name='malloc')
    context.ir_func_free = ir.Function(
      context.module, ir.FunctionType(ir.VoidType(), [LLVM_VOID_POINTER_TYPE]), name='free')

  # TODO: currently, some inbuilt free() functions are not inlined.
  # This means that we need to add debug information to these functions, but they do not have any line numbers.
  # We use this dummy here.
  inbuilt_pos = TreePosition('', 0, 0)

  inbuilt_symbols = {
    'Str': _make_str_symbol, 'Ptr':_make_ptr_symbol, 'RawPtr': _make_raw_ptr_symbol, 'bitcast': _make_bitcast_symbol}
  with context.use_pos(inbuilt_pos):
    for symbol_identifier, setup_func in inbuilt_symbols.items():
      assert symbol_identifier not in symbol_table
      symbol = setup_func(symbol_table=symbol_table, context=context)
      symbol_table[symbol_identifier] = symbol
      symbol_table.inbuilt_symbols[symbol_identifier] = symbol

    make_builtin_operator_functions(symbol_table, context.emits_ir)


Instructions: Dict[Tuple[BuiltinBinaryOps, Type], Callable[[CodegenContext, Value, Value], Value]] = concat_dicts([
  {(BuiltinBinaryOps.Addition, Int): IRBuilder.add for Int in INT_TYPES},
  {(BuiltinBinaryOps.Subtraction, Int): IRBuilder.sub for Int in INT_TYPES},
  {(BuiltinBinaryOps.Multiplication, Int): IRBuilder.mul for Int in INT_TYPES},
  {(BuiltinBinaryOps.Division, Int): IRBuilder.sdiv for Int in INT_TYPES},

  {(BuiltinBinaryOps.Addition, T): IRBuilder.fadd for T in FLOAT_TYPES},
  {(BuiltinBinaryOps.Subtraction, T): IRBuilder.fsub for T in FLOAT_TYPES},
  {(BuiltinBinaryOps.Multiplication, T): IRBuilder.fmul for T in FLOAT_TYPES},
  {(BuiltinBinaryOps.Division, T): IRBuilder.fdiv for T in FLOAT_TYPES},

  {(op, T): lambda builder, lhs, rhs, op=op: builder.icmp_signed(op.value, lhs, rhs) for op in Simple_Comparison_Ops for T in INT_TYPES},
  {(op, T): lambda builder, lhs, rhs, op=op: builder.icmp_unsigned(op.value, lhs, rhs) for op in Simple_Comparison_Ops for T in {SLEEPY_CHAR, SLEEPY_BOOL}},
  {(op, T):
     lambda builder, lhs, rhs, op=op: builder.fcmp_ordered(op.value, lhs, rhs)
   for op in Simple_Comparison_Ops for T in FLOAT_TYPES}
  ])


BINARY_OP_DECL = (
        [(operator, [(Instructions[(operator, arith_t)], [arith_t, arith_t], arith_t) for arith_t in SLEEPY_NUMERICAL_TYPES])
   for operator in Simple_Arithmetic_Ops] +  # simple arithmetic on all arithmetic types
        [
    (BuiltinBinaryOps.Addition,
      [(lambda builder, arg: arg, [arith_t], arith_t) for arith_t in SLEEPY_NUMERICAL_TYPES] +
      [(lambda builder, lhs, rhs: builder.gep(lhs, (rhs,)), [SLEEPY_RAW_PTR, i], SLEEPY_RAW_PTR) for i in INT_TYPES] +
      [(lambda builder, lhs, rhs: builder.gep(rhs, (lhs,)), [i, SLEEPY_RAW_PTR], SLEEPY_RAW_PTR) for i in INT_TYPES]
    ),
    (BuiltinBinaryOps.Subtraction, [(lambda builder, arg, arith_t=arith_t: builder.mul(ir.Constant(arith_t.ir_type, -1), arg), [arith_t], arith_t)
                                    for arith_t in INT_TYPES]),
    (BuiltinBinaryOps.Subtraction, [(lambda builder, arg, arith_t=arith_t: builder.fmul(ir.Constant(arith_t.ir_type, -1), arg), [arith_t], arith_t)
                                    for arith_t in FLOAT_TYPES]),
    (BuiltinBinaryOps.Subtraction, [(lambda builder, lhs, rhs: builder.gep(lhs, (builder.mul(ir.Constant(arith_t.ir_type, -1), rhs),)), [SLEEPY_RAW_PTR, arith_t], SLEEPY_RAW_PTR)  # noqa
      for arith_t in INT_TYPES])]  # noqa
     +  # unary plus and minus on all arithmetic types
        [(operator, [(Instructions[(operator, comp_t)], [comp_t, comp_t], SLEEPY_BOOL) for comp_t in COMPARABLE_TYPES])
   for operator in Simple_Comparison_Ops] +  # comparisons on all types except void and char
        [(operator, [(lambda builder, lhs, rhs, op=operator: builder.icmp_unsigned(op.value, lhs, rhs), [SLEEPY_CHAR, SLEEPY_CHAR], SLEEPY_BOOL)])
   for operator in [BuiltinBinaryOps.Equality, BuiltinBinaryOps.Inequality]] +
        [(BuiltinBinaryOps.BitwiseOr, [(IRBuilder.or_, [int_t, int_t], int_t) for int_t in INT_TYPES])] +  # bitwise_or on integer types
        [(BuiltinBinaryOps.Mod, [(IRBuilder.srem, [int_t, int_t], int_t) for int_t in INT_TYPES])]  # modulo on integer types
  )


def make_builtin_operator_functions(symbol_table: SymbolTable, emits_ir: bool):
  for operator, overloads in BINARY_OP_DECL:
    if operator.value not in symbol_table:
      symbol_table[operator.value] = FunctionSymbol(identifier=operator.value, returns_void=False)
    function_symbol = symbol_table[operator.value]
    assert isinstance(function_symbol, FunctionSymbol)

    for instruction, arg_types, return_type in overloads:
      signature = make_function_signature(
        instruction, op_placeholder_templ_types=[], op_arg_types=arg_types, op_return_type=return_type,
        emits_ir=emits_ir)
      function_symbol.add_signature(signature)


def make_function_signature(instruction: Callable[..., Value],
                            op_placeholder_templ_types: Union[Tuple[TemplateType], List[TemplateType]],
                            op_arg_types: List[Type], op_return_type: Type, emits_ir: bool) -> FunctionTemplate:
  assert len(op_arg_types) in {1, 2}
  unary: bool = len(op_arg_types) == 1
  op_arg_identifiers = ['arg'] if unary else ['lhs', 'rhs']
  assert len(op_arg_types) == len(op_arg_identifiers)
  # ir_func will be set in build_initial_module_ir
  factory = InbuiltOpConcreteFuncFactory(instruction, emits_ir=emits_ir)

  signature = FunctionTemplate(
    factory, placeholder_templ_types=list(op_placeholder_templ_types), return_type=op_return_type, return_mutable=False,
    arg_identifiers=op_arg_identifiers, arg_types=op_arg_types, arg_mutables=[False] * len(op_arg_types),
    arg_type_narrowings=op_arg_types, is_inline=True)

  return signature


def try_infer_templ_types(calling_types: List[Type], signature_types: List[Type],
                          placeholder_templ_types: List[TemplateType]) -> Optional[List[Type]]:
  """
  Implements unification.
  Mostly symmetric, however uses can_implicit_cast_to(calling_type, signature_type) only in this direction.
  """
  if len(calling_types) != len(signature_types):
    return None
  templ_type_replacements: Dict[TemplateType, Type] = {}

  def check_deep_type_contains(type: Type, contains: Type) -> bool:
    return any(
      can_implicit_cast_to(child, contains) or check_deep_type_contains(child, contains=contains)
      for child in type.templ_types)

  def infer_type(calling_type: Type, signature_type: Type) -> bool:
    calling_type = calling_type.replace_types(templ_type_replacements)
    signature_type = signature_type.replace_types(templ_type_replacements)
    if isinstance(calling_type, UnionType) and len(calling_type.possible_types) == 1:
      calling_type = calling_type.possible_types[0]
    if isinstance(signature_type, UnionType) and len(signature_type.possible_types) == 1:
      signature_type = signature_type.possible_types[0]

    if can_implicit_cast_to(calling_type, signature_type):
      return True
    if calling_type.has_same_symbol_as(signature_type):
      assert len(calling_type.templ_types) == len(signature_type.templ_types)
      return all(
        infer_type(calling_type=call_type, signature_type=sig_type)
        for call_type, sig_type in zip(calling_type.templ_types, signature_type.templ_types))
    if signature_type in placeholder_templ_types or calling_type in placeholder_templ_types:  # template variable.
      if signature_type in placeholder_templ_types:
        template_type, other_type = signature_type, calling_type
      else:
        assert calling_type in placeholder_templ_types
        template_type, other_type = calling_type, signature_type
      assert isinstance(template_type, TemplateType)
      if check_deep_type_contains(other_type, contains=template_type):  # recursively contains itself
        return False
      if template_type not in templ_type_replacements:
        templ_type_replacements[template_type] = other_type
      assert templ_type_replacements[template_type] == other_type
      return True
    return False

  for calling_type_, signature_type_ in zip(calling_types, signature_types):
    possible = infer_type(calling_type=calling_type_, signature_type=signature_type_)
    if not possible:
      return None
  if any(templ_type not in templ_type_replacements for templ_type in placeholder_templ_types):
    return None
  return [templ_type_replacements[templ_type] for templ_type in placeholder_templ_types]
