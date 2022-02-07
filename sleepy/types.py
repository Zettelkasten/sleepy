from __future__ import annotations

import copy
import ctypes
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, Optional, List, Tuple, Set, Union, Callable, Iterable, cast, Any, Iterator, MutableSet

import llvmlite
from llvmlite import ir
from llvmlite.binding import ExecutionEngine

from sleepy.syntactical_analysis.grammar import TreePosition, DummyPath

LLVM_POINTER_SIZE = 8
LLVM_SIZE_TYPE = ir.IntType(LLVM_POINTER_SIZE * 8)
LLVM_VOID_POINTER_TYPE = ir.PointerType(ir.types.IntType(8))


class Type(ABC):
  """
  The type of a declared variable.
  """

  def __init__(self, *,
               template_param_or_arg: List[Type],
               ir_type: Optional[ir.types.Type],
               c_type: Optional[Any],
               constructor: Optional[OverloadSet]):
    """
    :param c_type: may be None if this is non-realizable (e.g. template types / void)
    """
    self.template_param_or_arg = template_param_or_arg
    self.ir_type = ir_type
    self.c_type = c_type
    self._constructor: Optional[OverloadSet] = constructor

  def __repr__(self) -> str:
    return self.__class__.__name__

  @property
  def size(self) -> int:
    return ctypes.sizeof(self.c_type)

  def is_realizable(self) -> bool:
    return True

  def replace_types(self, replacements: Dict[Type, Type]) -> Type:
    if self in replacements:
      return replacements[self]
    return self

  def has_unfilled_template_parameters(self) -> bool:
    return False

  @abstractmethod
  def children(self) -> List[Type]:
    raise NotImplementedError()

  @abstractmethod
  def has_same_symbol_as(self, other: Type) -> bool:
    raise NotImplementedError()

  @abstractmethod
  def _make_di_type(self, context: CodegenContext) -> ir.DIValue:
    raise NotImplementedError()

  def make_di_type(self, context: CodegenContext) -> ir.DIValue:
    if self not in context.known_di_types:
      context.known_di_types[self] = self._make_di_type(context=context)
    return context.known_di_types[self]

  @property
  def constructor(self) -> Optional[OverloadSet]:
    return self._constructor

  @property
  def constructor_caller(self) -> FunctionSymbolCaller:
    assert self.constructor is not None
    # TODO: We do not currently keep track if the user specifies template types anywhere.
    # this means saying MyType[][][] is okay currently.
    template_parameters = (
      self.template_param_or_arg
      if len(self.template_param_or_arg) > 0 and all(
        not t_param.has_unfilled_template_parameters() for t_param in self.template_param_or_arg)
      else None)
    return FunctionSymbolCaller(self.constructor, template_parameters=template_parameters)

  @constructor.setter
  def constructor(self, new_constructor: OverloadSet):
    assert new_constructor is None or all(not s.return_type is SLEEPY_UNIT for s in new_constructor.signatures)
    self._constructor = new_constructor

  def copy(self) -> Type:
    return copy.copy(self)

  def is_referenceable(self) -> bool:
    if isinstance(self, UnionType):
      return all(possible_type.is_referenceable() for possible_type in self.possible_types)
    return isinstance(self, ReferenceType)

  def num_possible_unbindings(self):
    if isinstance(self, UnionType):
      return min((possible_type.num_possible_unbindings() for possible_type in self.possible_types), default=0)
    if isinstance(self, ReferenceType):
      return 1 + self.pointee_type.num_possible_unbindings()
    return 0


class UnitType(Type):
  """
  Type returned when nothing is returned.
  """

  def __init__(self):
    super().__init__(
      template_param_or_arg=[], ir_type=ir.types.LiteralStructType(()), c_type=ctypes.c_char * 0,
      constructor=None)

  def __repr__(self) -> str:
    return 'Unit'

  def children(self) -> List[Type]:
    return []

  def __eq__(self, other) -> bool:
    return isinstance(other, UnitType)

  def __hash__(self) -> int:
    return id(type(self))

  def has_same_symbol_as(self, other: Type) -> bool:
    return self == other

  def _make_di_type(self, context: CodegenContext) -> ir.DIValue:
    return context.module.add_debug_info(
      'DICompositeType', {
        'name': repr(self), 'size': 0, 'tag': ir.DIToken('DW_TAG_structure_type'),
        'file': context.current_di_file, 'elements': []
      }
    )

  def unit_constant(self) -> ir.Constant:
    return ir.Constant(self.ir_type, ())


class DoubleType(Type):
  """
  A double.
  """

  def __init__(self):
    super().__init__(template_param_or_arg=[], ir_type=ir.DoubleType(), c_type=ctypes.c_double, constructor=None)

  def _make_di_type(self, context: CodegenContext) -> ir.DIValue:
    assert context.emits_debug
    return context.module.add_debug_info(
      'DIBasicType',
      {'name': repr(self), 'size': self.size * 8, 'encoding': ir.DIToken('DW_ATE_float')})

  def __repr__(self) -> str:
    return 'Double'

  def children(self) -> List[Type]:
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
    super().__init__(template_param_or_arg=[], ir_type=ir.FloatType(), c_type=ctypes.c_float, constructor=None)

  def _make_di_type(self, context: CodegenContext) -> ir.DIValue:
    assert context.emits_debug
    return context.module.add_debug_info(
      'DIBasicType',
      {'name': repr(self), 'size': self.size * 8, 'encoding': ir.DIToken('DW_ATE_float')})

  def __repr__(self) -> str:
    return 'Float'

  def children(self) -> List[Type]:
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
    super().__init__(template_param_or_arg=[], ir_type=ir.IntType(bits=1), c_type=ctypes.c_bool, constructor=None)

  def _make_di_type(self, context: CodegenContext) -> ir.DIValue:
    assert context.emits_debug
    return context.module.add_debug_info(
      'DIBasicType',
      {'name': repr(self), 'size': self.size * 8, 'encoding': ir.DIToken('DW_ATE_boolean')})

  def __repr__(self) -> str:
    return 'Bool'

  def children(self) -> List[Type]:
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
    super().__init__(template_param_or_arg=[], ir_type=ir.IntType(bits=32), c_type=ctypes.c_int32, constructor=None)

  def _make_di_type(self, context: CodegenContext) -> ir.DIValue:
    assert context.emits_debug
    return context.module.add_debug_info(
      'DIBasicType',
      {'name': repr(self), 'size': self.size * 8, 'encoding': ir.DIToken('DW_ATE_signed')})

  def __repr__(self) -> str:
    return 'Int'

  def children(self) -> List[Type]:
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
    super().__init__(template_param_or_arg=[], ir_type=ir.IntType(bits=64), c_type=ctypes.c_int64, constructor=None)

  def _make_di_type(self, context: CodegenContext) -> ir.DIValue:
    assert context.emits_debug
    return context.module.add_debug_info(
      'DIBasicType',
      {'name': repr(self), 'size': self.size * 8, 'encoding': ir.DIToken('DW_ATE_signed')})

  def __repr__(self) -> str:
    return 'Long'

  def children(self) -> List[Type]:
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
    super().__init__(template_param_or_arg=[], ir_type=ir.IntType(bits=8), c_type=ctypes.c_uint8, constructor=None)

  def _make_di_type(self, context: CodegenContext) -> ir.DIValue:
    assert context.emits_debug
    return context.module.add_debug_info(
      'DIBasicType',
      {'name': repr(self), 'size': self.size * 8, 'encoding': ir.DIToken('DW_ATE_unsigned_char')})

  def __repr__(self) -> str:
    return 'Char'

  def children(self) -> List[Type]:
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

  def __init__(self, pointee_type: Type, constructor: Optional[OverloadSet] = None):
    super().__init__(
      template_param_or_arg=[pointee_type], ir_type=ir.PointerType(pointee_type.ir_type),
      c_type=ctypes.POINTER(pointee_type.c_type), constructor=constructor)
    self.pointee_type = pointee_type

  def _make_di_type(self, context: CodegenContext) -> ir.DIValue:
    assert context.emits_debug
    return context.module.add_debug_info(
      'DIDerivedType', {
        'tag': ir.DIToken('DW_TAG_pointer_type'), 'baseType': self.pointee_type.make_di_type(context=context),
        'size': LLVM_POINTER_SIZE
      })

  def __repr__(self) -> str:
    return 'Ptr[%r]' % self.pointee_type

  def __eq__(self, other) -> bool:
    if not isinstance(other, PointerType):
      return False
    return self.pointee_type == other.pointee_type

  def __hash__(self) -> int:
    return hash((self.__class__, self.pointee_type))

  def replace_types(self, replacements: Dict[Type, Type]) -> Type:
    if self in replacements:
      return replacements[self]
    return PointerType(pointee_type=self.pointee_type.replace_types(replacements), constructor=self.constructor)

  def children(self) -> List[Type]:
    return []

  def has_same_symbol_as(self, other: Type) -> bool:
    return isinstance(other, PointerType)


class RawPointerType(Type):
  """
  A raw (void) pointer from which we do not know the underlying type.
  """

  def __init__(self):
    super().__init__(template_param_or_arg=[], ir_type=LLVM_VOID_POINTER_TYPE, c_type=ctypes.c_void_p, constructor=None)

  def _make_di_type(self, context: CodegenContext) -> ir.DIValue:
    assert context.emits_debug
    return context.module.add_debug_info(
      'DIBasicType',
      {'name': repr(self), 'size': self.size * 8, 'encoding': ir.DIToken('DW_ATE_address')})

  def __repr__(self) -> str:
    return 'RawPtr'

  def children(self) -> List[Type]:
    return []

  def __eq__(self, other) -> bool:
    return isinstance(other, RawPointerType)

  def __hash__(self) -> int:
    return id(type(self))

  def has_same_symbol_as(self, other: Type) -> bool:
    return self == other


class ReferenceType(PointerType):
  """
  A reference to some other type.
  """

  def __init__(self, pointee_type: Type, constructor: Optional[OverloadSet] = None):
    super().__init__(pointee_type=pointee_type, constructor=constructor)

  def __repr__(self) -> str:
    return 'Ref[%r]' % self.pointee_type

  def __eq__(self, other) -> bool:
    if not isinstance(other, ReferenceType):
      return False
    return self.pointee_type == other.pointee_type

  def replace_types(self, replacements: Dict[Type, Type]) -> Type:
    if self in replacements:
      return replacements[self]
    return ReferenceType(pointee_type=self.pointee_type.replace_types(replacements), constructor=self.constructor)

  def __hash__(self) -> int:
    return hash((self.__class__, self.pointee_type))

  def has_same_symbol_as(self, other: Type) -> bool:
    return isinstance(other, ReferenceType)

  @classmethod
  def wrap(cls, typ: Type, times: int) -> Type:
    assert times >= 0
    if times == 0:
      return typ
    return ReferenceType(pointee_type=ReferenceType.wrap(typ, times=times - 1))


class UnionType(Type):
  """
  A tagged union, i.e. a type that can be one of a set of different types.
  """

  def __init__(self, possible_types: List[Type], possible_type_nums: List[int], val_size: Optional[int],
               constructor: Optional[OverloadSet] = None):
    assert len(possible_types) == len(possible_type_nums)
    assert len(set(possible_types)) == len(possible_types)
    self.possible_types = possible_types
    self.possible_type_nums = possible_type_nums
    self.identifier = 'Union(%s)' % '_'.join(str(possible_type) for possible_type in possible_types)
    self.val_size = val_size

    self.tag_c_type = ctypes.c_uint8
    self.tag_ir_type = ir.types.IntType(8)
    self.tag_size = 8
    if any(possible_type.has_unfilled_template_parameters() for possible_type in possible_types):
      assert val_size is None
      self.tag_c_type = None
      self.tag_ir_type = None
      self.untagged_union_ir_type: Optional[ir.Type] = None
      self.untagged_union_c_type: Optional[type] = None
      ir_type, c_type = None, None
    else:  # default case, non-template
      assert val_size is not None
      assert all(val_size >= possible_type.size for possible_type in possible_types)
      self.untagged_union_ir_type = ir.types.ArrayType(ir.types.IntType(8), val_size)
      self.untagged_union_c_type = ctypes.c_ubyte * max(
        (ctypes.sizeof(possible_type.c_type) for possible_type in possible_types), default=0)
      c_type = type(
        '%s_CType' % self.identifier, (ctypes.Structure,),
        {'_fields_': [('tag', self.tag_c_type), ('untagged_union', self.untagged_union_c_type)]})
      ir_type = ir.types.LiteralStructType([self.tag_ir_type, self.untagged_union_ir_type])
    super().__init__(template_param_or_arg=[], ir_type=ir_type, c_type=c_type, constructor=constructor)

  def __repr__(self) -> str:
    if len(self.possible_types) == 0:
      return 'NeverType'
    return '|'.join(
      '%s:%s' % (possible_type_num, possible_type)
      for possible_type_num, possible_type in zip(self.possible_type_nums, self.possible_types))

  def __eq__(self, other) -> bool:
    if not isinstance(other, UnionType):
      return False
    if len(self.possible_types) == len(other.possible_types) == 0:  # self = SLEEPY_NEVER
      return True
    self_types_dict = dict(zip(self.possible_types, self.possible_type_nums))
    other_types_dict = dict(zip(other.possible_types, other.possible_type_nums))
    return self_types_dict == other_types_dict and self.val_size == other.val_size

  def __hash__(self) -> int:
    if len(self.possible_types) == 0:  # self = SLEEPY_NEVER
      return 0
    return hash(tuple(sorted(zip(self.possible_type_nums, self.possible_types))) + (self.val_size,))

  def _make_di_type(self, context: CodegenContext) -> ir.DIValue:
    assert context.emits_debug
    di_derived_types = [
      context.module.add_debug_info(
        'DIDerivedType', {
          'tag': ir.DIToken('DW_TAG_inheritance'), 'baseType': member_type.make_di_type(context=context),
          'size': member_type.size * 8
        })
      for member_type in self.possible_types]
    di_tag_type = context.module.add_debug_info(
      'DIBasicType',
      {'name': 'tag', 'size': self.tag_size * 8, 'encoding': ir.DIToken('DW_ATE_unsigned')})
    di_untagged_union_type = context.module.add_debug_info(
      'DICompositeType', {
        'name': repr(self), 'size': self.val_size * 8, 'tag': ir.DIToken('DW_TAG_union_type'),
        'file': context.current_di_file, 'elements': [di_derived_types]
      })
    return context.module.add_debug_info(
      'DICompositeType', {
        'name': repr(self), 'size': self.size * 8, 'tag': ir.DIToken('DW_TAG_structure_type'),
        'elements': [di_tag_type, di_untagged_union_type]
      })

  def is_realizable(self) -> bool:
    return len(self.possible_types) > 0

  def replace_types(self, replacements: Dict[Type, Type]) -> Type:
    if self in replacements:
      return replacements[self]
    if len(self.possible_types) == 0:
      return self
    new_possible_types = [possible_type.replace_types(replacements) for possible_type in self.possible_types]
    new_possible_type_nums = self.possible_type_nums.copy()
    duplicate_indices = [
      index for index, possible_type in enumerate(new_possible_types)
      if possible_type in new_possible_types[:index]]
    for duplicate_index in reversed(duplicate_indices):
      del new_possible_types[duplicate_index]
      del new_possible_type_nums[duplicate_index]
    if any(possible_type.has_unfilled_template_parameters() for possible_type in new_possible_types):
      val_size = None
    else:  # default case
      val_size = max([ctypes.sizeof(possible_type.c_type) for possible_type in new_possible_types])
      # Note: We don't decrease the size of the union value to stay compatible to before.
      if self.val_size is not None:
        val_size = max(val_size, self.val_size)

    return UnionType(
      possible_types=new_possible_types, possible_type_nums=new_possible_type_nums, val_size=val_size,
      constructor=self.constructor)

  def has_unfilled_template_parameters(self) -> bool:
    return any(possible_type.has_unfilled_template_parameters() for possible_type in self.possible_types)

  def contains(self, contained_type: Type) -> bool:
    if isinstance(contained_type, UnionType):
      contained_possible_types = set(contained_type.possible_types)
    else:
      contained_possible_types = {contained_type}
    return contained_possible_types.issubset(self.possible_types)

  def get_variant_num(self, variant_type: Type) -> int:
    assert variant_type in self.possible_types
    return self.possible_type_nums[self.possible_types.index(variant_type)]

  @staticmethod
  def make_tag_ptr(union_ir_alloca: ir.AllocaInstr,
                   context: CodegenContext,
                   name: str) -> ir.Instruction:
    assert context.emits_ir
    tag_gep_indices = (ir.Constant(ir.IntType(32), 0), ir.Constant(ir.IntType(32), 0))
    return context.builder.gep(union_ir_alloca, tag_gep_indices, name=name)

  @staticmethod
  def make_untagged_union_void_ptr(union_ir_alloca: ir.Value,
                                   context: CodegenContext,
                                   name: str) -> ir.Instruction:
    assert context.emits_ir
    untagged_union_gep_indices = (ir.Constant(ir.IntType(32), 0), ir.Constant(ir.IntType(32), 1))
    return context.builder.gep(union_ir_alloca, untagged_union_gep_indices, name=name)

  def make_untagged_union_ptr(self,
                              union_ir_alloca: ir.AllocaInstr,
                              variant_type: Type,
                              context: CodegenContext,
                              name: str) -> ir.Instruction:
    assert variant_type in self.possible_types
    untagged_union_ptr = self.make_untagged_union_void_ptr(
      union_ir_alloca=union_ir_alloca, context=context, name='%s_raw' % name)
    return context.builder.bitcast(
      untagged_union_ptr, ir.types.PointerType(variant_type.ir_type),
      name=name)

  @staticmethod
  def make_extract_tag(union_ir_val: ir.Value, context: CodegenContext, name: str) -> ir.Value:
    return context.builder.extract_value(union_ir_val, 0, name=name)

  @staticmethod
  def make_extract_void_val(union_ir_val: ir.Value, context: CodegenContext, name: str) -> ir.Value:
    assert context.emits_ir
    return context.builder.extract_value(union_ir_val, idx=1, name=name)

  def make_extract_val(self, union_ir_val: ir.Value,
                       variant_type: Type,
                       context: CodegenContext,
                       name: str) -> ir.Value:
    assert context.emits_ir
    assert variant_type in self.possible_types
    union_ir_alloca = context.alloca_at_entry(self.ir_type, name='%s_ptr' % name)
    context.builder.store(union_ir_val, union_ir_alloca)
    untagged_union_ptr = self.make_untagged_union_ptr(
      union_ir_alloca, variant_type=variant_type, context=context, name='%s_val_ptr' % name)
    return context.builder.load(untagged_union_ptr, name=name)

  def copy_with_narrowed_types(self, narrow_to_types: List[Type] | Set[Type]) -> UnionType:
    possible_types = [possible_type for possible_type in self.possible_types if possible_type in narrow_to_types]
    possible_type_nums = [
      possible_type_num
      for possible_type, possible_type_num in zip(self.possible_types, self.possible_type_nums)
      if possible_type in narrow_to_types]
    return UnionType(possible_types=possible_types, possible_type_nums=possible_type_nums, val_size=self.val_size)

  def copy_with_extended_types(self, extended_types: List[Type],
                               extended_type_nums: Union[List[Union[int, None]], None] = None) -> UnionType:
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
        new_possible_type_nums.append(extended_type_num)
      else:
        next_type_num = max(new_possible_type_nums) + 1 if len(new_possible_type_nums) > 0 else 0
        new_possible_type_nums.append(next_type_num)
    if self.val_size is None or any(
      extended_type.has_unfilled_template_parameters() for extended_type in extended_types):
      new_val_size = None
    else:
      new_val_size = max([self.val_size] + [extended_type.size for extended_type in extended_types])
    return UnionType(
      possible_types=new_possible_types, possible_type_nums=new_possible_type_nums, val_size=new_val_size)

  @classmethod
  def from_types(cls, possible_types: List[Type]) -> UnionType:
    possible_types = list(dict.fromkeys(possible_types))  # possibly remove duplicates
    possible_type_nums = list(range(len(possible_types)))
    if any(possible_type.has_unfilled_template_parameters() for possible_type in possible_types):
      val_size = None
    else:  # default case, no template
      val_size = max((ctypes.sizeof(possible_type.c_type) for possible_type in possible_types), default=0)
    return UnionType(possible_types=possible_types, possible_type_nums=possible_type_nums, val_size=val_size)

  def children(self) -> List[Type]:
    return self.possible_types

  def has_same_symbol_as(self, other: Type) -> bool:
    return isinstance(other, UnionType)


class PartialIdentifiedStructType(Type):
  """
  A struct while it is being declared. Needed for self-referencing struct members.
  """

  def __init__(self, identity: StructIdentity, template_param_or_arg: List[Type]):
    self.identity = identity
    self.templ_types = template_param_or_arg
    if identity.context is not None and not any(
      templ_type.has_unfilled_template_parameters() for templ_type in template_param_or_arg):
      ir_type = identity.context.make_struct_ir_type(identity=self.identity, templ_types=template_param_or_arg)
      c_type = identity.context.make_struct_c_type(identity=self.identity, templ_types=template_param_or_arg)
    else:
      ir_type, c_type = None, None
    super().__init__(template_param_or_arg=[], ir_type=ir_type, c_type=c_type, constructor=None)

  @property
  def struct_identifier(self) -> str:
    return self.identity.struct_identifier

  def _make_di_type(self, context: CodegenContext):
    assert False, self

  def __repr__(self) -> str:
    return self.struct_identifier + 'Partial'

  def children(self) -> List[Type]:
    return []

  def has_same_symbol_as(self, other: Type) -> bool:
    return self == other


class StructIdentity:
  """
  To distinguish different structs. Similar to TypeSymbol.
  """

  def __init__(self, struct_identifier: str, context: CodegenContext):
    self.struct_identifier = struct_identifier
    assert context is not None
    self.context = context

  def __repr__(self):
    return '%s@%s' % (self.struct_identifier, hex(id(self)))


class StructType(Type):
  """
  A struct.
  """

  def __init__(self,
               identity: StructIdentity,
               template_param_or_arg: List[Type],
               member_identifiers: List[str],
               member_types: List[Type],
               partial_struct_type: Optional[PartialIdentifiedStructType] = None,
               constructor: Optional[OverloadSet] = None):
    assert len(member_identifiers) == len(member_types)
    self.identity = identity
    self.member_identifiers = member_identifiers
    self.member_types = member_types
    self._constructor: Optional[OverloadSet] = None
    assert partial_struct_type is None or partial_struct_type.identity == self.identity

    if any(t.has_unfilled_template_parameters() for t in template_param_or_arg):
      super().__init__(template_param_or_arg=template_param_or_arg, ir_type=None, c_type=None, constructor=constructor)
    else:
      member_ir_types = [member_type.ir_type for member_type in member_types]
      member_c_types = [
        (member_identifier, member_type.c_type)
        for member_identifier, member_type in zip(member_identifiers, member_types)]
      assert None not in member_ir_types
      assert None not in member_c_types

      if partial_struct_type is None:
        partial_struct_type = PartialIdentifiedStructType(
          identity=self.identity,
          template_param_or_arg=template_param_or_arg)
      ir_type, c_type = partial_struct_type.ir_type, partial_struct_type.c_type
      assert ir_type is not None
      assert c_type is not None
      assert isinstance(ir_type, ir.types.IdentifiedStructType)
      if ir_type.is_opaque:
        ir_type.set_body(*member_ir_types)
        c_type._fields_ = member_c_types
      else:  # already defined before
        assert ir_type.elements == tuple(member_ir_types)
        # Note that we do not check c_type._fields_ here because ctypes do not properly compare for equality always
      super().__init__(
        template_param_or_arg=template_param_or_arg, ir_type=ir_type, c_type=c_type,
        constructor=constructor)

    if partial_struct_type is not None:
      self.member_types = [member_type.replace_types({partial_struct_type: self}) for member_type in self.member_types]

  @property
  def struct_identifier(self) -> str:
    return self.identity.struct_identifier

  def get_member_num(self, member_identifier: str) -> int:
    assert member_identifier in self.member_identifiers
    return self.member_identifiers.index(member_identifier)

  def replace_types(self, replacements: Dict[Type, Type]) -> Type:
    if self in replacements: return replacements[self]
    if len(replacements) == 0: return self

    new_templ_types = [templ_type.replace_types(replacements) for templ_type in self.template_param_or_arg]
    # in case we have a self-referencing member, first replace all occurrences of ourself with something temporary
    # to avoid infinite recursion.
    partial_self_replaced = PartialIdentifiedStructType(identity=self.identity, template_param_or_arg=new_templ_types)
    replacements = {**replacements, self: partial_self_replaced}
    new_member_types = [member_type.replace_types(replacements) for member_type in self.member_types]
    new_struct = StructType(
      identity=self.identity, template_param_or_arg=new_templ_types, member_identifiers=self.member_identifiers,
      member_types=new_member_types, partial_struct_type=partial_self_replaced, constructor=self.constructor)
    return new_struct

  def has_unfilled_template_parameters(self) -> bool:
    return any(param_or_arg.has_unfilled_template_parameters() for param_or_arg in self.template_param_or_arg)

  def _make_di_type(self, context: CodegenContext) -> ir.DIValue:
    assert context.emits_debug

    # register placeholder for self so that members can reference this struct
    placeholder = DebugValueIrPatcher.make_placeholder(
      repr(self.identity) + '_'.join(map(str, self.template_param_or_arg)))
    context.known_di_types[self] = placeholder

    di_derived_types = []
    for member_identifier, member_type in zip(self.member_identifiers, self.member_types):
      di_derived_types.append(
        context.module.add_debug_info(
          'DIDerivedType', {
            'tag': ir.DIToken('DW_TAG_member'), 'baseType': member_type.make_di_type(context=context),
            'name': member_identifier, 'size': member_type.size * 8,
            'offset': getattr(self.c_type, member_identifier).offset * 8
          }))
    debug_value = context.module.add_debug_info(
      'DICompositeType', {
        'name': repr(self), 'size': self.size * 8, 'tag': ir.DIToken('DW_TAG_structure_type'),
        'file': context.current_di_file, 'elements': di_derived_types
      })

    # set replacement for placeholder for this type
    context.debug_ir_patcher.add_replacement(placeholder, debug_value.get_reference())
    return debug_value

  def __eq__(self, other) -> bool:
    if not isinstance(other, StructType):
      return False
    return self.identity == other.identity and self.template_param_or_arg == other.template_param_or_arg

  def __hash__(self) -> int:
    return hash((self.__class__, self.identity) + tuple(self.template_param_or_arg))

  def __repr__(self) -> str:
    return self.struct_identifier + ('' if len(self.template_param_or_arg) == 0 else str(self.template_param_or_arg))

  def make_extract_member_val_ir(self, member_identifier: str,
                                 struct_ir_val: ir.Value,
                                 context: CodegenContext) -> ir.Instruction:
    return context.builder.extract_value(
      struct_ir_val, self.get_member_num(member_identifier), name='member_%s' % member_identifier)

  def make_store_members_ir(self, member_ir_vals: List[ir.Value],
                            struct_ir_alloca: ir.Instruction,
                            context: CodegenContext):
    assert len(member_ir_vals) == len(self.member_identifiers)
    assert context.emits_ir
    for member_num, (member_identifier, ir_func_arg) in enumerate(zip(self.member_identifiers, member_ir_vals)):
      gep_indices = (ir.Constant(ir.IntType(32), 0), ir.Constant(ir.IntType(32), member_num))
      member_ptr = context.builder.gep(struct_ir_alloca, gep_indices, name='%s_ptr' % member_identifier)
      context.builder.store(ir_func_arg, member_ptr)

  def children(self) -> List[Type]:
    return self.member_types

  def has_same_symbol_as(self, other: Type) -> bool:
    if not isinstance(other, StructType):
      return False
    return self.identity == other.identity


class TemplateParameter:
  def __init__(self, identifier: str):
    self.identifier = identifier

  def __repr__(self) -> str:
    return self.identifier


class PlaceholderTemplateType(Type):
  """
  A template parameter.
  """

  def __init__(self, identifier: str):
    super().__init__(template_param_or_arg=[], ir_type=None, c_type=None, constructor=None)
    self.identifier = identifier

  def _make_di_type(self, context: CodegenContext):
    assert False, self

  def __repr__(self) -> str:
    return self.identifier

  def has_unfilled_template_parameters(self) -> bool:
    return True

  def children(self) -> List[Type]:
    return []

  def has_same_symbol_as(self, other: Type) -> bool:
    return self == other

  @property
  def constructor(self) -> Optional[OverloadSet]:
    return None


def can_implicit_cast_ref_to(from_pointee_type: Type, to_pointee_type: Type) -> bool:
  """
  Whether `Ref[from_pointee_type]` can easily be treated as `Ref[to_pointee_type]`.
  """
  if not is_subtype(from_pointee_type, to_pointee_type):
    return False
  if from_pointee_type == to_pointee_type:
    return True

  if not isinstance(to_pointee_type, UnionType):
    return True
  if isinstance(from_pointee_type, UnionType) and isinstance(to_pointee_type, UnionType):
    simple_subset = all(
      to_pointee_type.contains(a_type) and to_pointee_type.get_variant_num(a_type) == a_num
      for a_num, a_type in zip(from_pointee_type.possible_type_nums, from_pointee_type.possible_types))
    if simple_subset:
      return True
  return False


def can_implicit_cast_to(from_type: Type, to_type: Type) -> bool:
  """
  Whether you can "easily" treat memory of `from_type` as `to_type`.
  "Easily" means via `TypedValue.copy_with_implicit_cast`, without calling any function / allocating new memory / etc.
  E.g., if `from_type` is a union, you can treat it as each of each possible member types
  by simply dropping the union tag.
  But e.g. List[Int] cannot easily be converted to List[Int|Bool],
  because that would need to allocate an entirely new list (Int has a different value size than just Int|Bool).

  References are a special case here: If A is a subtype of B, we allow Ref[A] to be implicitly casted to Ref[B].
  In general, for other template types, this is not true.
  """
  if from_type == to_type:
    return True

  possible_from_types = set(from_type.possible_types) if isinstance(from_type, UnionType) else {from_type}
  possible_to_types = set(to_type.possible_types) if isinstance(to_type, UnionType) else {to_type}
  # handle references specially
  for a in list(possible_from_types):
    if not isinstance(a, ReferenceType):
      continue
    if not any(
      isinstance(b, ReferenceType) and can_implicit_cast_ref_to(a.pointee_type, b.pointee_type) for b in
      possible_to_types):  # noqa
      return False
    possible_from_types.remove(a)
  # Note: These elements really need to match exactly,
  # it is not enough for their template types to be implicitly convertible alone.
  # E.g. even if A|B can be cast to B implicitly, List[A|B] cannot be to List[B].
  # That would require creating an entire new list.
  return possible_from_types.issubset(possible_to_types)


def is_subtype(a: Type, b: Type) -> bool:
  """
  Whether all possible values of type `a` are also logically of type `b`.
  """
  if a == b:  # quick path
    return True
  if isinstance(a, UnionType):
    return all(is_subtype(possible_type, b) for possible_type in a.possible_types)
  possible_b_types = set(b.possible_types) if isinstance(b, UnionType) else {b}
  if a in possible_b_types:  # quick path
    return True

  for possible_b in possible_b_types:
    assert not isinstance(possible_b, UnionType)
    if not a.has_same_symbol_as(possible_b):
      continue
    assert len(a.template_param_or_arg) == len(possible_b.template_param_or_arg)
    if all(
      is_subtype(a_templ, b_templ) for a_templ, b_templ in
      zip(a.template_param_or_arg, possible_b.template_param_or_arg)):
      return True
  return False


def narrow_type(from_type: Type, narrow_to: Type) -> Type:
  """
  Computes `intersection(from_type, narrow_to)`, but stays compatible to `from_type`,
  meaning that memory in the form of `from_type` can directly be interpreted as `narrow_type`.

  In case there is no representation of `intersection(from_type, narrow_to)` that is compatible to `from_type`,
  we use some superset of the intersection which remains a subset of `from_type`.
  E.g. `narrow_type(List[Int|Bool], List[Int]|List[Bool])` will just return `List[Int|Bool]`,
  because you cannot implicitly convert `List[Int|Bool]` to `List[Int]|List[Bool]`
  (you would need to construct an entirely new list).
  In this sense we over-approximate the actual type.
  """
  return _narrow_type(from_type=from_type, narrow_to=narrow_to, narrow_to_complement=False)


def exclude_type(from_type: Type, excluded_type: Type) -> Type:
  return _narrow_type(from_type=from_type, narrow_to=excluded_type, narrow_to_complement=True)


def _narrow_type(from_type: Type, narrow_to: Type, narrow_to_complement: bool = False) -> Type:
  if from_type == narrow_to:
    return SLEEPY_NEVER if narrow_to_complement else from_type
  all_possible_from_types = set(from_type.possible_types) if isinstance(from_type, UnionType) else {from_type}
  possible_to_types = set(narrow_to.possible_types) if isinstance(narrow_to, UnionType) else {narrow_to}

  new_possible_from_types: Set[Type] = set()
  new_possible_ref_pointee_types: Dict[ReferenceType, List[Type]] = {
    ref: [] for ref in all_possible_from_types if isinstance(ref, ReferenceType)}
  for possible_to_type in possible_to_types:
    assert not isinstance(possible_to_type, UnionType)
    for possible_from_type in all_possible_from_types:
      if not is_subtype(possible_to_type, possible_from_type):
        continue
      if isinstance(possible_to_type, ReferenceType):
        # references are special: we can always implicitly cast between them, so just be as concrete as possible.
        assert isinstance(possible_from_type, ReferenceType)
        new_possible_ref_pointee_types[possible_from_type].append(possible_to_type.pointee_type)
      else:
        new_possible_from_types.add(possible_from_type)

  if narrow_to_complement:
    new_possible_from_types = all_possible_from_types - new_possible_from_types

  # always keep all Ref elements for now (no matter if narrow_to_complement is set)
  new_possible_from_types.update(
    ref for ref, possible_types in new_possible_ref_pointee_types.items() if len(possible_types) > 0)

  assert len(new_possible_from_types) <= len(all_possible_from_types)
  if isinstance(from_type, UnionType):
    narrowed_type = from_type.copy_with_narrowed_types(narrow_to_types=new_possible_from_types)
  elif len(new_possible_from_types) == 0:
    # Note: if a variable has type "never", the memory can have every form. So we can safely narrow to this.
    return SLEEPY_NEVER
  else:
    assert len(new_possible_from_types) == 1
    narrowed_type = list(new_possible_from_types)[0]

  # Apply Ref[T] specializations
  if len(new_possible_ref_pointee_types) > 0:
    ref_replacements = {
      ref: ReferenceType(_narrow_type(ref.pointee_type, get_common_type(possible_types), narrow_to_complement))
      for ref, possible_types in new_possible_ref_pointee_types.items()
      if len(possible_types) > 0}
    narrowed_type = narrowed_type.replace_types(ref_replacements)
  return narrowed_type


def get_common_type(possible_types: List[Type]) -> Type:
  if len(possible_types) == 0: return SLEEPY_NEVER

  common_type = possible_types[0]
  common_type = common_type if isinstance(common_type, UnionType) \
    else UnionType.from_types(possible_types=[possible_types[0]])

  for other_type in possible_types[1:]:
    if isinstance(other_type, UnionType):
      common_type = common_type.copy_with_extended_types(
        extended_types=other_type.possible_types, extended_type_nums=other_type.possible_type_nums)
    else:
      common_type = common_type.copy_with_extended_types(extended_types=[other_type])

  if isinstance(common_type, UnionType) and len(common_type.possible_types) == 1:
    return common_type.possible_types[0]  # stay as simple as possible
  return common_type


def _uncollapse_type(from_type: Type, collapsed_type: Type) -> Type:
  min_ref_depth, max_ref_depth = min_max_ref_depth(from_type)
  to_min_ref_depth, to_max_ref_depth = min_max_ref_depth(collapsed_type)
  check_from = max(min_ref_depth - to_max_ref_depth, 0)
  check_to = max(max_ref_depth - to_min_ref_depth + 1, 0)
  uncollapsed_type = get_common_type(
    [ReferenceType.wrap(collapsed_type, depth) for depth in range(check_from, check_to)])
  return uncollapsed_type


def narrow_with_collapsed_type(from_type: Type, collapsed_type: Type):
  """
  Narrows all types away which are not `collapsed_type`, `Ref[collapsed_type]`, etc.
  E.g. if self is Ref[A]|Ref[B], and collapsed_type=A, this returns Ref[A].
  However, if self is Ref[A]|A, and collapsed_type=A, then self will remain unchanged.
  """
  return narrow_type(
    from_type=from_type, narrow_to=_uncollapse_type(from_type=from_type, collapsed_type=collapsed_type))


def exclude_with_collapsed_type(from_type: Type, collapsed_type: Type):
  return exclude_type(
    from_type=from_type, excluded_type=_uncollapse_type(from_type=from_type, collapsed_type=collapsed_type))


def make_ir_val_is_type(ir_val: ir.values.Value,
                        known_type: Type,
                        check_type: Type,
                        context: CodegenContext) -> ir.values.Value:
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


class ConcreteFunction:
  """
  An actual function implementation.
  """

  def __init__(self,
               signature: FunctionTemplate,
               ir_func: Optional[ir.Function],
               template_arguments: List[Type],
               return_type: Type,
               parameter_types: List[Type],
               narrowed_parameter_types: List[Type],
               parameter_mutates: List[bool]):
    assert ir_func is None or isinstance(ir_func, ir.Function)
    assert (len(signature.arg_identifiers) == len(parameter_types)
            == len(narrowed_parameter_types) == len(parameter_mutates))
    assert all(not templ_type.has_unfilled_template_parameters() for templ_type in template_arguments)
    assert not return_type.has_unfilled_template_parameters()
    assert all(not arg_type.has_unfilled_template_parameters() for arg_type in parameter_types)
    assert all(not arg_type.has_unfilled_template_parameters() for arg_type in narrowed_parameter_types)
    self.signature = signature
    self.ir_func = ir_func
    self.template_arguments = template_arguments
    self.return_type = return_type
    self.arg_types = parameter_types
    self.arg_type_narrowings = narrowed_parameter_types
    self.arg_mutates = parameter_mutates
    self.di_subprogram: Optional[ir.DIValue] = None

  def get_c_arg_types(self) -> Tuple[Any]:
    return (cast(Any, self.return_type.c_type),) + tuple(cast(Any, arg_type.c_type) for arg_type in self.arg_types)

  def make_ir_function_type(self) -> ir.FunctionType:
    arg_types = [arg_type.ir_type for arg_type in self.arg_types]
    arg_types = [
      ir.PointerType(arg_type) if arg_mutates else arg_type
      for arg_type, arg_mutates in zip(arg_types, self.arg_mutates)]
    return ir.FunctionType(self.return_type.ir_type, arg_types)

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
  def is_inline(self) -> bool:
    return False

  def __repr__(self) -> str:
    return (
      'ConcreteFunction(signature=%r, concrete_templ_types=%r, return_type=%r, arg_types=%r, '
      'arg_type_narrowings=%r, arg_mutates=%r)' % (
        self.signature, self.template_arguments, self.return_type, self.arg_types, self.arg_type_narrowings,
        self.arg_mutates))

  @property
  def uncollapsed_arg_types(self) -> List[Type]:
    return [
      ReferenceType(param_type) if mutates else param_type
      for param_type, mutates in zip(self.arg_types, self.arg_mutates)]

  def has_same_signature_as(self, other: ConcreteFunction) -> bool:
    return (
      self.return_type == other.return_type and self.arg_types == other.arg_types and
      self.arg_type_narrowings == other.arg_type_narrowings)

  def make_ir_func(self, identifier: str, extern: bool, context: CodegenContext):
    assert context.emits_ir
    assert not self.is_inline
    assert self.ir_func is None
    ir_func_name = context.make_ir_function_name(identifier, self, extern)
    ir_func_type = self.make_ir_function_type()
    self.ir_func = ir.Function(context.module, ir_func_type, name=ir_func_name)
    if context.emits_debug and not extern:
      assert self.di_subprogram is None
      di_return_type = self.return_type.make_di_type(context=context)
      di_arg_types = [arg_type.make_di_type(context=context) for arg_type in self.arg_types]
      di_arg_types = [
        context.module.add_debug_info(
          'DIDerivedType',
          {'tag': ir.DIToken('DW_TAG_reference_type'), 'baseType': di_type, 'size': LLVM_POINTER_SIZE * 8})
        if mutates else di_type
        for di_type, mutates in zip(di_arg_types, self.arg_mutates)]
      di_func_type = context.module.add_debug_info(
        'DISubroutineType', {'types': context.module.add_metadata([di_return_type] + di_arg_types)})
      self.di_subprogram = context.module.add_debug_info(
        'DISubprogram', {
          'name': ir_func_name, 'file': context.current_di_file, 'scope': context.current_di_scope,
          'line': context.current_pos.get_from_line(), 'scopeLine': context.current_pos.get_from_line(),
          'type': di_func_type,
          'isLocal': False, 'isDefinition': True,
          'unit': context.current_di_compile_unit
        }, is_distinct=True)
      self.ir_func.set_metadata('dbg', self.di_subprogram)

  # noinspection PyTypeChecker
  def make_inline_func_call_ir(self,
                               func_args: List[TypedValue],
                               caller_context: CodegenContext) -> ir.Instruction:
    assert self.is_inline
    assert False, 'not implemented!'


class ConcreteBuiltinOperationFunction(ConcreteFunction):
  def __init__(self,
               signature: FunctionTemplate,
               ir_func: Optional[ir.Function],
               template_arguments: List[Type],
               return_type: Type,
               parameter_types: List[Type],
               narrowed_parameter_types: List[Type],
               parameter_mutates: List[bool],
               instruction: Callable[..., Optional[ir.Instruction]],
               emits_ir: bool):
    super().__init__(
      signature=signature, ir_func=ir_func, template_arguments=template_arguments, return_type=return_type,
      parameter_types=parameter_types, narrowed_parameter_types=narrowed_parameter_types,
      parameter_mutates=parameter_mutates)
    assert callable(instruction)
    self.instruction = instruction
    self.emits_ir = emits_ir

  def make_inline_func_call_ir(self, func_args: List[TypedValue],
                               caller_context: CodegenContext) -> ir.Instruction:
    ir_func_args = [arg.ir_val for arg in func_args]
    assert None not in ir_func_args
    return self.instruction(caller_context.builder, *ir_func_args)

  @property
  def is_inline(self) -> bool:
    return True


class ConcreteBitcastFunction(ConcreteFunction):
  def __init__(self, signature: FunctionTemplate, ir_func: Optional[ir.Function], template_arguments: List[Type],
               return_type: Type, parameter_types: List[Type], narrowed_parameter_types: List[Type]):
    super().__init__(
      signature=signature, ir_func=ir_func, template_arguments=template_arguments, return_type=return_type,
      parameter_types=parameter_types, narrowed_parameter_types=narrowed_parameter_types,
      parameter_mutates=[False] * len(parameter_types))

  def make_inline_func_call_ir(self, func_args: List[TypedValue],
                               caller_context: CodegenContext) -> ir.Instruction:
    assert len(func_args) == 1
    assert func_args[0].ir_val is not None
    return caller_context.builder.bitcast(val=func_args[0].ir_val, typ=self.return_type.ir_type, name="bitcast")

  @property
  def is_inline(self) -> bool:
    return True


class FunctionTemplate:
  """
  Given template arguments, this builds a concrete function implementation on demand.
  """

  def __init__(self, placeholder_template_types: List[PlaceholderTemplateType],
               return_type: Type,
               arg_identifiers: List[str],
               arg_types: List[Type],
               arg_type_narrowings: List[Type],
               arg_mutates: List[bool]):
    assert isinstance(return_type, Type)
    assert len(arg_identifiers) == len(arg_types) == len(arg_type_narrowings) == len(arg_mutates)
    assert all(isinstance(templ_type, PlaceholderTemplateType) for templ_type in placeholder_template_types)
    self.placeholder_templ_types = placeholder_template_types
    self.return_type = return_type
    self.arg_identifiers = arg_identifiers
    self.arg_types = arg_types
    self.arg_type_narrowings = arg_type_narrowings
    self.arg_mutates = arg_mutates
    self.initialized_templ_funcs: Dict[Tuple[Type], ConcreteFunction] = {}

  def to_signature_str(self) -> str:
    templ_args = '' if len(self.placeholder_templ_types) == 0 else '[%s]' % (
      ', '.join([templ_type.identifier for templ_type in self.placeholder_templ_types]))
    args = ', '.join(
      [
        '%s%s: %s' % ('mutates ' if mutates else '', identifier, typ)
        for mutates, identifier, typ in zip(self.arg_mutates, self.arg_identifiers, self.arg_types)])
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

    return self._get_concrete_function(
      concrete_template_arguments=list(concrete_templ_types), concrete_parameter_types=concrete_arg_types,
      concrete_narrowed_parameter_types=concrete_arg_types_narrowings, concrete_return_type=concrete_return_type)

  @abstractmethod
  def _get_concrete_function(self, concrete_template_arguments: List[Type],
                             concrete_parameter_types: List[Type],
                             concrete_narrowed_parameter_types: List[Type],
                             concrete_return_type: Type) -> ConcreteFunction:
    raise NotImplementedError()

  def can_call_with_expanded_arg_types(self, concrete_templ_types: List[Type], expanded_arg_types: List[Type]):
    assert all(not isinstance(arg_type, UnionType) for arg_type in expanded_arg_types)
    assert all(not templ_type.has_unfilled_template_parameters() for templ_type in concrete_templ_types)
    assert all(not arg_type.has_unfilled_template_parameters() for arg_type in expanded_arg_types)
    if len(concrete_templ_types) != len(self.placeholder_templ_types):
      return False
    if len(expanded_arg_types) != len(self.arg_types):
      return False
    replacements = dict(zip(self.placeholder_templ_types, concrete_templ_types))
    concrete_arg_types = [arg_type.replace_types(replacements=replacements) for arg_type in self.arg_types]
    assert all(not arg_type.has_unfilled_template_parameters() for arg_type in concrete_arg_types)
    return all(
      can_implicit_cast_to(call_type, arg_type) for call_type, arg_type in zip(expanded_arg_types, concrete_arg_types))

  def is_undefined_for_expanded_arg_types(self, placeholder_templ_types: List[PlaceholderTemplateType],
                                          expanded_arg_types: List[Type]) -> bool:
    assert all(not isinstance(arg_type, UnionType) for arg_type in expanded_arg_types)
    if len(placeholder_templ_types) != len(self.placeholder_templ_types):
      return True
    if len(expanded_arg_types) != len(self.arg_types):
      return True
    # convert own template variables to calling template variables s.t. we can try to unify them
    templ_to_templ_replacements = dict(zip(self.placeholder_templ_types, placeholder_templ_types))
    own_arg_types = [arg_type.replace_types(templ_to_templ_replacements) for arg_type in self.arg_types]
    inferred_templ_types = try_infer_template_arguments(
      calling_types=expanded_arg_types, signature_types=own_arg_types, template_parameters=placeholder_templ_types)
    return inferred_templ_types is None

  def __repr__(self) -> str:
    return 'FunctionSignature(placeholder_templ_types=%r, return_type=%r, arg_identifiers=%r, arg_types=%r)' % (
      self.placeholder_templ_types, self.return_type, self.arg_identifiers, self.arg_types)


class DummyFunctionTemplate(FunctionTemplate):

  def __init__(self):
    super().__init__([], SLEEPY_NEVER, [], [], [], [])

  def _get_concrete_function(self, concrete_template_arguments: List[Type], concrete_parameter_types: List[Type],
                             concrete_narrowed_parameter_types: List[Type],
                             concrete_return_type: Type) -> ConcreteFunction:
    raise NotImplementedError()


class OverloadSet(MutableSet[FunctionTemplate]):
  """
  A set of declared overloaded function signatures with the same name.
  Can have one or multiple overloaded signatures accepting different parameter types (FunctionSignature).
  Each of these signatures itself can have a set of concrete implementations,
  where template types have been replaced with concrete types.
  """

  def __init__(self, identifier: str, signatures: Iterable[FunctionTemplate]):
    self.identifier = identifier
    self.signatures_by_number_of_templ_args: Dict[int, List[FunctionTemplate]] = {}
    for s in signatures: self.add(s)

  @property
  def signatures(self) -> List[FunctionTemplate]:
    return [signature for signatures in self.signatures_by_number_of_templ_args.values() for signature in signatures]

  def can_call_with_expanded_arg_types(self, concrete_templ_types: List[Type], expanded_arg_types: List[Type]) -> bool:
    assert all(not isinstance(arg_type, UnionType) for arg_type in expanded_arg_types)
    assert all(not templ_type.has_unfilled_template_parameters() for templ_type in concrete_templ_types)
    assert all(not arg_type.has_unfilled_template_parameters() for arg_type in expanded_arg_types)
    signatures = self.signatures_by_number_of_templ_args.get(len(concrete_templ_types), [])
    return any(
      signature.can_call_with_expanded_arg_types(
        concrete_templ_types=concrete_templ_types, expanded_arg_types=expanded_arg_types)
      for signature in signatures)

  def can_call_with_arg_types(self, template_arguments: List[Type],
                              arg_types: Union[List[Type], Tuple[Type]]) -> bool:
    assert all(not templ_type.has_unfilled_template_parameters() for templ_type in template_arguments)
    assert all(not arg_type.has_unfilled_template_parameters() for arg_type in arg_types)
    all_expanded_arg_types = self.iter_expanded_possible_arg_types(arg_types)
    return all(
      self.can_call_with_expanded_arg_types(concrete_templ_types=template_arguments, expanded_arg_types=list(arg_types))
      for arg_types in all_expanded_arg_types)

  def is_undefined_for_arg_types(self, placeholder_templ_types: List[PlaceholderTemplateType], arg_types: List[Type]):
    signatures = self.signatures_by_number_of_templ_args.get(len(placeholder_templ_types), [])
    return all(
      signature.is_undefined_for_expanded_arg_types(
        placeholder_templ_types=placeholder_templ_types, expanded_arg_types=list(expanded_arg_types))
      for signature in signatures for expanded_arg_types in self.iter_expanded_possible_arg_types(arg_types))

  def get_concrete_funcs(self, template_arguments: List[Type], arg_types: List[Type]) -> List[ConcreteFunction]:
    signatures = self.signatures_by_number_of_templ_args.get(len(template_arguments), [])
    possible_concrete_funcs = []
    for expanded_arg_types in self.iter_expanded_possible_arg_types(arg_types):
      for signature in signatures:
        if signature.can_call_with_expanded_arg_types(
          concrete_templ_types=template_arguments,
          expanded_arg_types=expanded_arg_types):  # noqa
          concrete_func = signature.get_concrete_func(concrete_templ_types=template_arguments)
          if concrete_func not in possible_concrete_funcs:
            possible_concrete_funcs.append(concrete_func)
    return possible_concrete_funcs

  def has_single_concrete_func(self) -> bool:
    return len(self.signatures) == 1 and len(self.signatures[0].placeholder_templ_types) == 0

  def get_single_concrete_func(self) -> ConcreteFunction:
    assert self.has_single_concrete_func()
    return self.signatures[0].get_concrete_func(concrete_templ_types=[])

  @staticmethod
  def iter_expanded_possible_arg_types(arg_types: Iterable[Type]) -> Iterable[Iterable[Type]]:
    import itertools
    return itertools.product(
      *[arg_type.possible_types if isinstance(arg_type, UnionType) else [arg_type] for arg_type in arg_types])

  def __repr__(self) -> str:
    return 'OverloadSet(identifier=%r, signatures=%r)' % (self.identifier, self.signatures)

  def make_signature_list_str(self) -> str:
    return '\n'.join([' - ' + signature.to_signature_str() for signature in self.signatures])

  ##
  ## MutableSet implementation
  ##

  def add(self, signature: FunctionTemplate):
    assert self.is_undefined_for_arg_types(
      placeholder_templ_types=signature.placeholder_templ_types,
      arg_types=signature.arg_types)
    self.signatures_by_number_of_templ_args.setdefault(len(signature.placeholder_templ_types), []).append(signature)

  def discard(self, value: FunctionTemplate):
    lst = self.signatures_by_number_of_templ_args.get(len(value.placeholder_templ_types))
    if lst is not None: lst.remove(value)

  def __contains__(self, x: object) -> bool:
    return isinstance(x, FunctionTemplate) and any(x in fs for fs in self.signatures_by_number_of_templ_args.values())

  def __len__(self) -> int:
    return sum(len(fs) for fs in self.signatures_by_number_of_templ_args.values())

  def __iter__(self) -> Iterator[FunctionTemplate]:
    return self.signatures.__iter__()

  # is used by the MutableSet mixin
  def _from_iterable(self, iterable: Iterable[FunctionTemplate]) -> OverloadSet:
    return OverloadSet(identifier=self.identifier, signatures=iterable)


class FunctionSymbolCaller:
  """
  A FunctionSymbol with the template arguments it is called with.
  """

  def __init__(self, func: OverloadSet, template_parameters: Optional[List[Type]] = None):
    if template_parameters is not None:
      assert all(not t_arg.has_unfilled_template_parameters() for t_arg in template_parameters)
    self.func = func
    self.template_parameters = template_parameters

  def copy_with_template_arguments(self, template_arguments: List[Type]) -> FunctionSymbolCaller:
    assert self.template_parameters is None
    return FunctionSymbolCaller(func=self.func, template_parameters=template_arguments)


class DebugValueIrPatcher:
  def __init__(self):
    self._patches: Dict[str, Optional[str]] = {}

  @staticmethod
  def make_placeholder(unique_name: str) -> str:
    return f"dbg_val_placeholder_{unique_name}"

  def add_replacement(self, placeholder: str, replacement: str):
    assert placeholder not in self._patches
    self._patches[placeholder] = replacement

  def patch_ir(self, ir_str: str) -> str:
    for placeholder, replacement in self._patches.items():
      ir_str = ir_str.replace(f"\"{placeholder}\"", str(replacement))
    return ir_str


class CodegenContext:
  """
  Used to keep track where code is currently generated.
  This is essentially a pointer to an ir.IRBuilder.
  """

  def __init__(self, builder: Optional[ir.IRBuilder],
               module: ir.Module,
               emits_debug: bool,
               file_path: Path | DummyPath):
    self._builder = builder
    self._module = module

    self._emits_debug: bool = emits_debug
    self.is_terminated: bool = False

    # use dummy just to set file_path
    self.current_pos = TreePosition(word="", from_pos=0, to_pos=0, file_path=file_path)
    self.current_func_inline_return_collect_block: Optional[ir.Block] = None
    self.current_func_inline_return_ir_alloca: Optional[ir.AllocaInstr] = None
    self.inline_func_call_stack: List[ConcreteFunction] = []
    self.ir_func_malloc: Optional[ir.Function] = None

    self.debug_ir_patcher = DebugValueIrPatcher()

    if not self.emits_debug:
      self.current_di_compile_unit: Optional[ir.DIValue] = None
      self.current_di_scope: Optional[ir.DIValue] = None
      self.di_declare_func: Optional[ir.Function] = None
      self.known_di_types: Optional[Dict[Type, ir.DIValue]] = None
    else:
      self._file_di_value_cache: Dict[Path | DummyPath, ir.DIValue] = dict()
      self.current_di_scope = self.current_di_file

      # TODO: Add compiler version
      producer = 'sleepy compiler'

      self.current_di_compile_unit: Optional[ir.DIValue] = module.add_debug_info(
        'DICompileUnit', {
          'language': ir.DIToken('DW_LANG_C'), 'file': self.current_di_file, 'producer': producer,
          'isOptimized': False, 'runtimeVersion': 1, 'emissionKind': ir.DIToken('FullDebug')
        },
        is_distinct=True)

      self.module.add_named_metadata('llvm.dbg.cu', self.current_di_compile_unit)
      di_dwarf_version = [ir.Constant(ir.IntType(32), 2), 'Dwarf Version', ir.Constant(ir.IntType(32), 4)]
      di_debug_info_version = [ir.Constant(ir.IntType(32), 2), 'Debug Info Version', ir.Constant(ir.IntType(32), 3)]
      self.module.add_named_metadata('llvm.module.flags', di_dwarf_version)
      self.module.add_named_metadata('llvm.module.flags', di_debug_info_version)
      self.module.add_named_metadata('llvm.ident', [producer])

      di_declare_func_type = ir.FunctionType(ir.VoidType(), [ir.MetaDataType()] * 3)
      self.di_declare_func = ir.Function(self.module, di_declare_func_type, 'llvm.dbg.declare')
      self.known_di_types: Dict[Type, Union[ir.DIValue, str]] = {}

    self._known_struct_c_types: Dict[(StructIdentity, Tuple[Type]), type] = {}

  @property
  def emits_debug(self) -> bool:
    return self._emits_debug and self.emits_ir

  @property
  def module(self) -> ir.Module:
    assert self.emits_ir
    assert self.builder is not None
    if self.block is not None:
      assert self.block.module == self._module
    return self._module

  @property
  def block(self) -> ir.Block:
    assert self.emits_ir
    assert self.builder is not None
    return self.builder.block

  @property
  def builder(self) -> ir.IRBuilder:
    assert self.emits_ir
    return self._builder

  @builder.setter
  def builder(self, builder):
    if self.emits_ir and builder is not None:
      builder.debug_metadata = self.builder.debug_metadata
    self._builder = builder

  @property
  def emits_ir(self) -> bool:
    return self._builder is not None

  @property
  def current_di_file(self) -> ir.DIValue:
    assert self.emits_debug
    path = self.current_pos.file_path
    if path not in self._file_di_value_cache:
      di_file = self._make_current_di_file(path=path)
      self._file_di_value_cache[path] = di_file
    else:
      di_file = self._file_di_value_cache[path]
    return di_file

  def _make_current_di_file(self, path: Path | DummyPath):
    assert self.emits_debug
    if isinstance(path, Path):
      file_name, file_dir = path.name, str(path.parent)
    else:
      file_name, file_dir = path.dummy_name, ""
    return self.module.add_debug_info('DIFile', {'filename': file_name, 'directory': file_dir})

  def __repr__(self) -> str:
    emits_ir = '' if not self.emits_ir else (', emits debug' if self.emits_debug else ', emits ir')
    return 'CodegenContext(builder=%r%s, is_terminated=%r)' % (
      self.builder, emits_ir, self.is_terminated)

  def copy_with_new_callstack_frame(self) -> CodegenContext:
    new_context = copy.copy(self)
    new_context.inline_func_call_stack = self.inline_func_call_stack.copy()
    assert all(inline_func.is_inline for inline_func in self.inline_func_call_stack)
    return new_context

  def copy_with_builder(self, new_builder: Optional[ir.IRBuilder]) -> CodegenContext:
    new_context = self.copy_with_new_callstack_frame()
    new_context.builder = new_builder
    return new_context

  def copy_without_builder(self) -> CodegenContext:
    return self.copy_with_builder(None)

  def copy_with_func(self, concrete_func: ConcreteFunction, builder: Optional[ir.IRBuilder]):
    assert not concrete_func.is_inline
    new_context = self.copy_with_builder(builder)
    new_context.current_func_inline_return_ir_alloca = None
    new_context.current_func_inline_return_collect_block = None
    if new_context.emits_debug:
      assert concrete_func.di_subprogram is not None
      new_context.current_di_scope = concrete_func.di_subprogram
      # as the current scope changed, reset the debug_metadata
      new_context.builder.debug_metadata = make_di_location(pos=new_context.current_pos, context=new_context)
    return new_context

  def copy_with_inline_func(self, concrete_func: ConcreteFunction,
                            return_ir_alloca: ir.AllocaInstr,
                            return_collect_block: ir.Block) -> CodegenContext:
    assert concrete_func.is_inline
    assert concrete_func not in self.inline_func_call_stack
    new_context = self.copy_with_new_callstack_frame()
    new_context.current_func_inline_return_ir_alloca = return_ir_alloca
    new_context.current_func_inline_return_collect_block = return_collect_block
    new_context.inline_func_call_stack.append(concrete_func)
    return new_context

  def alloca_at_entry(self, ir_type: ir.types.Type, name: str) -> ir.AllocaInstr:
    """
    Add alloca instruction at entry block of the current function.
    """
    assert self.emits_ir
    entry_block: ir.Block = self.block.function.entry_basic_block
    entry_builder = ir.IRBuilder(entry_block)
    if len(entry_block.instructions) == 0 or not isinstance(entry_block.instructions[0], ir.AllocaInstr):
      entry_builder.position_at_start(entry_block)
    else:
      last_alloca_num = 0
      while (last_alloca_num < len(entry_block.instructions) and
             isinstance(entry_block.instructions[last_alloca_num], ir.AllocaInstr)):
        last_alloca_num += 1
      assert isinstance(entry_block.instructions[last_alloca_num - 1], ir.AllocaInstr)
      entry_builder.position_after(entry_block.instructions[last_alloca_num - 1])
    ir_alloca = entry_builder.alloca(typ=ir_type, name=name)
    if self.builder.block == entry_builder.block:
      # assume that builders are always at the last instruction of a block
      self.builder.position_at_end(self.builder.block)
    return ir_alloca

  def use_pos(self, pos: TreePosition) -> UsePosRuntimeContext:
    return UsePosRuntimeContext(pos, context=self)

  def make_ir_function_name(self, identifier: str, concrete_function: ConcreteFunction, extern: bool):
    if extern:
      if any(f.name == identifier for f in self.module.functions):
        return None
      return identifier
    else:
      ir_func_name = '_'.join(
        [identifier]
        + [str(arg_type) for arg_type in concrete_function.template_arguments + concrete_function.arg_types])
      return self.module.get_unique_name(ir_func_name)

  @staticmethod
  def _make_struct_type_name(identity: StructIdentity, templ_types: List[Type] | Tuple[Type]) -> str:
    return '_'.join([str(identity)] + [str(templ_type) for templ_type in templ_types])

  def make_struct_ir_type(self, identity: StructIdentity, templ_types: List[Type]) -> llvmlite.ir.IdentifiedStructType:
    return self.module.context.get_identified_type(name=self._make_struct_type_name(identity, templ_types))

  def make_struct_c_type(self, identity: StructIdentity, templ_types: List[Type]) -> type:
    templ_types = tuple(templ_types)
    if (identity, templ_types) not in self._known_struct_c_types:
      c_type = type(self._make_struct_type_name(identity, templ_types), (ctypes.Structure,), {})
      self._known_struct_c_types[identity, templ_types] = c_type
    return self._known_struct_c_types[identity, templ_types]

  def get_patched_ir(self) -> str:
    return self.debug_ir_patcher.patch_ir(repr(self.module))


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
    if self.context.is_terminated: return
    self.context.current_pos = self.prev_pos
    if self.context.emits_debug:
      self.context.builder.debug_metadata = self.prev_debug_metadata


def make_ir_size(size: int) -> ir.values.Value:
  return ir.Constant(LLVM_SIZE_TYPE, size)


def make_func_call_ir(func: OverloadSet,
                      template_arguments: List[Type],
                      func_args: List[TypedValue],
                      context: CodegenContext) -> TypedValue:
  assert not context.emits_debug or context.builder.debug_metadata is not None

  def do_call(concrete_func: ConcreteFunction,
              caller_context: CodegenContext) -> TypedValue:
    assert caller_context.emits_ir
    assert not caller_context.emits_debug or caller_context.builder.debug_metadata is not None
    assert len(concrete_func.arg_types) == len(func_args)
    assert all(
      not arg_mutates or arg.is_referenceable() for arg, arg_mutates in zip(func_args, concrete_func.arg_mutates))
    # mutates arguments are handled here as Ref[T], all others normally as T.
    collapsed_func_args = [
      (arg_val.copy_unbind() if mutates else arg_val).copy_collapse(
        context=caller_context, name='call_arg_%s_collapse' % identifier)
      for arg_val, identifier, mutates in zip(func_args, concrete_func.arg_identifiers, concrete_func.arg_mutates)]
    # Note: We first copy with the param_type as narrowed type, because at this point we already checked that
    # the param actually has the correct param_type.
    # Then, copy_with_implicit_cast only needs to handle the cases which can actually occur.
    casted_calling_args = [
      arg_val.copy_set_narrowed_type(param_type).copy_with_implicit_cast(
        to_type=param_type, context=caller_context, name='call_arg_%s_cast' % arg_identifier)
      for arg_identifier, arg_val, param_type in zip(
        concrete_func.arg_identifiers, collapsed_func_args, concrete_func.uncollapsed_arg_types)]
    if concrete_func.is_inline:
      assert concrete_func not in caller_context.inline_func_call_stack
      return_ir = concrete_func.make_inline_func_call_ir(func_args=casted_calling_args, caller_context=caller_context)
    else:
      ir_func = concrete_func.ir_func
      assert ir_func is not None and len(ir_func.args) == len(casted_calling_args)
      casted_ir_args = [arg.ir_val for arg in casted_calling_args]
      assert None not in casted_ir_args
      return_ir = caller_context.builder.call(ir_func, casted_ir_args, name='call_%s' % func.identifier)

    assert isinstance(return_ir, ir.values.Value)
    return TypedValue(typ=concrete_func.return_type, ir_val=return_ir)

  calling_collapsed_args = [arg.copy_collapse(context=context) for arg in func_args]
  calling_arg_types = [arg.narrowed_type for arg in calling_collapsed_args]
  assert func.can_call_with_arg_types(template_arguments=template_arguments, arg_types=calling_arg_types)
  possible_concrete_funcs = func.get_concrete_funcs(template_arguments=template_arguments, arg_types=calling_arg_types)
  from functools import partial
  return make_union_switch_ir(
    case_funcs={
      tuple(concrete_func.arg_types): partial(do_call, concrete_func) for concrete_func in possible_concrete_funcs},
    calling_args=calling_collapsed_args, name=func.identifier, context=context)


def make_union_switch_ir(case_funcs: Dict[Tuple[Type], Callable[[CodegenContext], TypedValue]],
                         calling_args: List[TypedValue],
                         name: str,
                         context: CodegenContext) -> TypedValue:
  """
  Given different functions to execute for different argument types,
  this will generate IR to call the correct function for given `calling_args`.
  This is especially useful if the argument type contains unions,
  as then it is only clear at run time which function to actually call.
  """
  assert not context.emits_debug or context.builder.debug_metadata is not None
  if len(case_funcs) == 1:
    single_case = next(iter(case_funcs.values()))
    single_value = single_case(caller_context=context)  # noqa
    return single_value

  import numpy as np
  import itertools
  calling_arg_types = [arg.narrowed_type for arg in calling_args]
  # The arguments which we need to look at to determine which concrete function to call
  # TODO: This could use a better heuristic, only because something is a union type does not mean that it
  # distinguishes different concrete funcs.
  distinguishing_arg_nums = [
    arg_num for arg_num, calling_arg_type in enumerate(calling_arg_types)
    if isinstance(calling_arg_type, UnionType)]
  assert len(distinguishing_arg_nums) >= 1
  # noinspection PyTypeChecker
  distinguishing_calling_arg_types: List[UnionType] = [
    calling_arg_types[arg_num] for arg_num in distinguishing_arg_nums]
  assert all(isinstance(calling_arg, UnionType) for calling_arg in distinguishing_calling_arg_types)
  # To distinguish which concrete func to call, use this table
  block_addresses_distinguished_mapping = np.ndarray(
    shape=tuple(max(calling_arg.possible_type_nums) + 1 for calling_arg in distinguishing_calling_arg_types),
    dtype=ir.values.BlockAddress)

  # Go through all concrete functions, and add one block for each
  case_contexts: Dict[Tuple[Type], CodegenContext] = {
    case_arg_types: context.copy_with_builder(
      ir.IRBuilder(
        context.builder.append_basic_block(
          "call_%s_%s" % (name, '_'.join(str(arg_type) for arg_type in case_arg_types)))))
    for case_arg_types in case_funcs.keys()}
  for case_arg_types, case_context in case_contexts.items():
    assert not case_context.emits_debug or case_context.builder.debug_metadata is not None
    case_block = case_context.block
    case_block_address = ir.values.BlockAddress(context.builder.function, case_block)
    case_distinguishing_args = [case_arg_types[arg_num] for arg_num in distinguishing_arg_nums]
    case_possible_types_per_arg = [
      [
        possible_type
        for possible_type in (arg_type.possible_types if isinstance(arg_type, UnionType) else [arg_type])
        if possible_type in calling_arg_type.possible_types]
      for arg_type, calling_arg_type in zip(case_distinguishing_args, distinguishing_calling_arg_types)]
    assert len(distinguishing_arg_nums) == len(case_possible_types_per_arg)

    # Register the concrete function in the table
    for expanded_case_types in itertools.product(*case_possible_types_per_arg):
      assert len(expanded_case_types) == len(distinguishing_calling_arg_types)
      distinguishing_variant_nums = tuple(
        calling_arg_type.get_variant_num(concrete_arg_type)
        for calling_arg_type, concrete_arg_type in zip(distinguishing_calling_arg_types, expanded_case_types))
      assert block_addresses_distinguished_mapping[distinguishing_variant_nums] is None
      block_addresses_distinguished_mapping[distinguishing_variant_nums] = case_block_address

  # Compute the index we have to look at in the table
  tag_ir_type = ir.types.IntType(8)
  call_block_index_ir = ir.Constant(tag_ir_type, 0)
  for arg_num, calling_arg_type in zip(distinguishing_arg_nums, distinguishing_calling_arg_types):
    ir_func_arg = calling_args[arg_num].ir_val
    assert ir_func_arg is not None
    base = np.prod(block_addresses_distinguished_mapping.shape[arg_num + 1:], dtype='int32')
    base_ir = ir.Constant(tag_ir_type, base)
    tag_ir = calling_arg_type.make_extract_tag(
      ir_func_arg, context=context, name='call_%s_arg%s_tag_ptr' % (name, arg_num))
    call_block_index_ir = context.builder.add(call_block_index_ir, context.builder.mul(base_ir, tag_ir))
  call_block_index_ir = context.builder.zext(call_block_index_ir, LLVM_SIZE_TYPE, name='call_%s_block_index' % name)

  # Look it up in the table and call the function
  ir_block_addresses_type = ir.types.VectorType(
    LLVM_VOID_POINTER_TYPE, np.prod(block_addresses_distinguished_mapping.shape))
  ir_block_addresses = ir.values.Constant(
    ir_block_addresses_type, ir_block_addresses_type.wrap_constant_value(
      list(block_addresses_distinguished_mapping.flatten())))
  ir_call_block_target = context.builder.extract_element(ir_block_addresses, call_block_index_ir)
  indirect_branch = context.builder.branch_indirect(ir_call_block_target)
  for case_context in case_contexts.values():
    indirect_branch.add_destination(case_context.block)

  # Execute the concrete functions
  collect_block = context.builder.append_basic_block('collect_%s_overload' % name)
  context.builder = ir.IRBuilder(collect_block)
  return_vals: List[TypedValue] = []
  for case_func, case_context in zip(case_funcs.values(), case_contexts.values()):
    case_return_val = case_func(caller_context=case_context)  # noqa
    assert not case_context.is_terminated
    return_vals.append(case_return_val)
  assert len(case_funcs) == len(return_vals)

  common_return_type = get_common_type([return_val.narrowed_type for return_val in return_vals])
  collect_return_ir_phi = context.builder.phi(
    common_return_type.ir_type, name="collect_%s_overload_return" % name)
  for case_return_val, case_context in zip(return_vals, case_contexts.values()):
    case_return_val_compatible = case_return_val.copy_with_implicit_cast(
      to_type=common_return_type, context=case_context, name="collect_cast_%s" % name)
    collect_return_ir_phi.add_incoming(case_return_val_compatible.ir_val, case_context.block)
    case_context.builder.branch(collect_block)
  return TypedValue(typ=common_return_type, ir_val=collect_return_ir_phi)


def make_di_location(pos: TreePosition, context: CodegenContext):
  assert context.emits_debug
  assert context.current_di_scope is not None
  line, col = pos.get_from_line_col()
  return context.module.add_debug_info(
    'DILocation', {'line': line, 'column': col, 'scope': context.current_di_scope})


def try_infer_template_arguments(calling_types: List[Type], signature_types: List[Type],
                                 template_parameters: List[PlaceholderTemplateType]) -> Optional[List[Type]]:
  """
  Implements unification.
  Mostly symmetric, however uses can_implicit_cast_to(calling_type, signature_type) only in this direction.
  """
  if len(calling_types) != len(signature_types):
    return None
  template_parameter_replacements: Dict[PlaceholderTemplateType, Type] = {}

  def check_deep_type_contains(in_type: Type, contains: Type) -> bool:
    return any(
      can_implicit_cast_to(child, contains) or check_deep_type_contains(child, contains=contains)
      for child in in_type.template_param_or_arg)

  def infer_type(calling_type: Type, signature_type: Type) -> bool:
    calling_type = calling_type.replace_types(template_parameter_replacements)
    signature_type = signature_type.replace_types(template_parameter_replacements)
    if isinstance(calling_type, UnionType) and len(calling_type.possible_types) == 1:
      calling_type = calling_type.possible_types[0]
    if isinstance(signature_type, UnionType) and len(signature_type.possible_types) == 1:
      signature_type = signature_type.possible_types[0]

    if can_implicit_cast_to(calling_type, signature_type):
      return True
    if calling_type.has_same_symbol_as(signature_type):
      assert len(calling_type.template_param_or_arg) == len(signature_type.template_param_or_arg)
      return all(
        infer_type(calling_type=call_type, signature_type=sig_type)
        for call_type, sig_type in zip(calling_type.template_param_or_arg, signature_type.template_param_or_arg))
    if signature_type in template_parameters or calling_type in template_parameters:  # template variable.
      if signature_type in template_parameters:
        template_type, other_type = signature_type, calling_type
      else:
        assert calling_type in template_parameters
        template_type, other_type = calling_type, signature_type
      assert isinstance(template_type, PlaceholderTemplateType)
      if check_deep_type_contains(other_type, contains=template_type):  # recursively contains itself
        return False
      if template_type not in template_parameter_replacements:
        template_parameter_replacements[template_type] = other_type
      assert template_parameter_replacements[template_type] == other_type
      return True
    return False

  for calling_type_, signature_type_ in zip(calling_types, signature_types):
    possible = infer_type(calling_type=calling_type_, signature_type=signature_type_)
    if not possible:
      return None
  if any(template_parameter not in template_parameter_replacements for template_parameter in template_parameters):
    return None
  return [template_parameter_replacements[template_parameter] for template_parameter in template_parameters]


def min_max_ref_depth(typ: Type) -> (int, int):
  if typ == SLEEPY_NEVER:
    return 0, 0
  if isinstance(typ, UnionType):
    assert len(typ.possible_types) > 0  # handled above
    possible_types_min_max = [min_max_ref_depth(possible_type) for possible_type in typ.possible_types]
    possible_types_min, possible_types_max = zip(*possible_types_min_max)
    return min(possible_types_min), max(possible_types_max)
  if isinstance(typ, ReferenceType):
    pointee_min, pointee_max = min_max_ref_depth(typ.pointee_type)
    return pointee_min + 1, pointee_max + 1
  return 0, 0


SLEEPY_UNIT = UnitType()
SLEEPY_NEVER = UnionType(possible_types=[], possible_type_nums=[], val_size=0)


class TypedValue:
  """
  A value an expression returns.
  Has a type, and, if it emits_ir, also an IR value.

  Reference-like types bind themselves to the value they represent.
  E.g. Ref[T] behaves like it is a T.
  This is also recursive, e.g. Ref[Ref[T]] also behaves like T.
  If num_unbindings > 0, this prevents this behavior and gives access to the reference-like type itself.
  """

  def __init__(self, *,
               typ: Type,
               narrowed_type: Type = None,
               num_unbindings: int = 0,
               ir_val: Optional[ir.values.Value]):
    if narrowed_type is None:
      narrowed_type = typ
    assert narrowed_type == SLEEPY_NEVER or (isinstance(typ, ReferenceType) == isinstance(narrowed_type, ReferenceType))
    self.type = typ
    self.narrowed_type = narrowed_type
    assert num_unbindings <= self.num_possible_unbindings()
    self.num_unbindings = num_unbindings
    self.ir_val = ir_val

  def is_referenceable(self) -> bool:
    return self.narrowed_type.is_referenceable()

  def copy(self) -> TypedValue:
    return copy.copy(self)

  def copy_set_narrowed_type(self, narrow_to_type: Type) -> TypedValue:
    new = self.copy()
    new.narrowed_type = narrow_type(from_type=self.type, narrow_to=narrow_to_type)
    return new

  def copy_set_narrowed_collapsed_type(self, collapsed_type: Type) -> TypedValue:
    new = self.copy()
    new.narrowed_type = narrow_with_collapsed_type(from_type=self.type, collapsed_type=collapsed_type)
    return new

  def copy_with_implicit_cast(self, to_type: Type, context: CodegenContext, name: str) -> TypedValue:
    """
    Returns copy of self so that self.type really is to_type.
    This changes the memory layout, it will not be compatible to before.
    """
    from_type = self.narrowed_type
    assert can_implicit_cast_to(from_type, to_type)
    new = self.copy()
    if from_type == to_type:
      return new
    new.type, new.narrowed_type = to_type, to_type
    new.ir_val = None
    if self.ir_val is None or not context.emits_ir:
      return new

    def do_simple_cast(from_simple_type: Type, to_simple_type: Type, ir_val: ir.values.Value) -> ir.values.Value:
      assert not isinstance(from_simple_type, UnionType)
      assert not isinstance(to_simple_type, UnionType)

      if from_simple_type == to_simple_type:
        return ir_val
      assert isinstance(from_simple_type, ReferenceType)
      assert isinstance(to_simple_type, ReferenceType)
      from_pointee_type, to_pointee_type = from_simple_type.pointee_type, to_simple_type.pointee_type
      assert from_pointee_type != to_pointee_type  # we handled this above already
      assert isinstance(from_pointee_type, UnionType)
      if isinstance(to_pointee_type, UnionType):
        simple_subset = all(
          to_pointee_type.contains(a_type) and to_pointee_type.get_variant_num(a_type) == a_num
          for a_num, a_type in zip(from_pointee_type.possible_type_nums, from_pointee_type.possible_types))
        assert simple_subset

        # Note: if the size of to_type is strictly smaller than from_type, we need to truncate the value
        assert max(a_type.size for a_type in from_pointee_type.possible_types) <= to_pointee_type.val_size
        raw_ir_val = self.ir_val
      else:
        assert not isinstance(to_pointee_type, UnionType)
        raw_ir_val = from_pointee_type.make_untagged_union_void_ptr(
          union_ir_alloca=self.ir_val, context=context, name='%s_from_val' % name)
      # maybe the size of the pointer decreased
      return context.builder.bitcast(val=raw_ir_val, typ=to_simple_type.ir_type, name='%s_to_val' % name)

    if isinstance(to_type, UnionType):
      to_ir_alloca = context.alloca_at_entry(to_type.ir_type, name='%s_ptr' % name)
      if isinstance(from_type, UnionType):
        tag_mapping_ir_type = ir.types.VectorType(to_type.tag_ir_type, max(from_type.possible_type_nums) + 1)
        tag_mapping = [-1] * (max(from_type.possible_type_nums) + 1)
        for from_variant_num, from_variant_type in zip(from_type.possible_type_nums, from_type.possible_types):
          assert to_type.contains(from_variant_type)
          tag_mapping[from_variant_num] = to_type.get_variant_num(from_variant_type)
        ir_tag_mapping = ir.values.Constant(tag_mapping_ir_type, tag_mapping_ir_type.wrap_constant_value(tag_mapping))
        ir_from_tag = from_type.make_extract_tag(self.ir_val, context=context, name='%s_from_tag' % name)
        ir_to_tag = context.builder.extract_element(ir_tag_mapping, ir_from_tag, name='%s_to_tag' % name)
        context.builder.store(
          ir_to_tag, to_type.make_tag_ptr(to_ir_alloca, context=context, name='%s_tag_ptr' % name))

        # Note: if the size of to_type is strictly smaller than from_type, we need to truncate the value
        assert max(possible_from_type.size for possible_from_type in from_type.possible_types) <= to_type.val_size
        ir_from_untagged_union = from_type.make_extract_void_val(
          self.ir_val, context=context, name='%s_from_val' % name)
        from sleepy.utilities import truncate_ir_value
        ir_from_untagged_union_truncated = truncate_ir_value(
          from_type=from_type.untagged_union_ir_type, to_type=to_type.untagged_union_ir_type,
          ir_val=ir_from_untagged_union, context=context, name=name)

        ir_to_untagged_union_ptr = to_type.make_untagged_union_void_ptr(
          to_ir_alloca, context=context, name='%s_to_val_raw' % name)
        ir_to_untagged_union_ptr_casted = context.builder.bitcast(
          ir_to_untagged_union_ptr, ir.PointerType(to_type.untagged_union_ir_type), name='%s_to_val' % name)
        context.builder.store(ir_from_untagged_union_truncated, ir_to_untagged_union_ptr_casted)
      else:
        assert not isinstance(from_type, UnionType)
        ir_to_tag = ir.Constant(to_type.tag_ir_type, to_type.get_variant_num(from_type))
        context.builder.store(
          ir_to_tag, to_type.make_tag_ptr(to_ir_alloca, context=context, name='%s_tag_ptr' % name))
        context.builder.store(
          self.ir_val,
          to_type.make_untagged_union_ptr(to_ir_alloca, from_type, context=context, name='%s_val_ptr' % name))
      new.ir_val = context.builder.load(to_ir_alloca, name=name)
    else:
      assert not isinstance(to_type, UnionType)
      if isinstance(from_type, UnionType):
        assert all(possible_from_type == to_type for possible_from_type in from_type.possible_types)
        new.ir_val = from_type.make_extract_val(self.ir_val, to_type, context=context, name=name)
      else:
        new.ir_val = do_simple_cast(from_simple_type=from_type, to_simple_type=to_type, ir_val=self.ir_val)
    return new

  def __repr__(self):
    attrs = ['type']
    if self.narrowed_type != self.type:
      attrs.append('narrowed_type')
    if self.ir_val is not None:
      attrs.append('ir_val')
    if self.num_unbindings:
      attrs.append('num_unbindings')
    return 'TypedValue(%s)' % ', '.join(['%s=%r' % (attr, getattr(self, attr)) for attr in attrs])

  def num_possible_unbindings(self) -> int:
    return self.narrowed_type.num_possible_unbindings()

  def copy_collapse(self, context: Optional[CodegenContext], name: str = 'val') -> TypedValue:
    min_ref_depth, max_ref_depth = min_max_ref_depth(self.narrowed_type)
    assert 0 <= self.num_unbindings <= min_ref_depth <= max_ref_depth
    # If possible, do not split up unions if not necessary
    if self.num_unbindings == min_ref_depth == max_ref_depth:
      return self

    # Otherwise, handle each union member individually

    def do_collapse(concrete_type: Type, caller_context: Optional[CodegenContext]) -> TypedValue:
      assert not isinstance(concrete_type, UnionType)
      concrete_min_ref_depth, concrete_max_ref_depth = min_max_ref_depth(concrete_type)
      assert 0 <= self.num_unbindings <= concrete_min_ref_depth <= concrete_max_ref_depth
      if self.num_unbindings == concrete_min_ref_depth == concrete_max_ref_depth:  # done.
        return self.copy_set_narrowed_type(concrete_type)
      assert isinstance(concrete_type, ReferenceType)  # still need to collapse, must be a reference.
      # collapse once.
      new_value = self.copy_set_narrowed_type(concrete_type)
      new_value = new_value.copy_with_implicit_cast(to_type=concrete_type, context=caller_context, name=name)
      new_value.type = concrete_type.pointee_type
      new_value.narrowed_type = concrete_type.pointee_type
      if caller_context is not None and caller_context.emits_ir:
        assert new_value.ir_val is not None
        new_value.ir_val = caller_context.builder.load(new_value.ir_val, name="%s_unbind" % name)
      else:
        new_value.ir_val = None
      return new_value.copy_collapse(context=caller_context, name=name)

    possible_types = (
      self.narrowed_type.possible_types if isinstance(self.narrowed_type, UnionType) else {self.narrowed_type})
    if context is None or not context.emits_ir:
      new = self.copy()
      new.type = get_common_type([do_collapse(typ, caller_context=context).narrowed_type for typ in possible_types])
      new.narrowed_type = new.type
      new.ir_val = None
      return new
    from functools import partial
    return make_union_switch_ir(
      case_funcs={(typ,): partial(do_collapse, typ) for typ in possible_types},
      calling_args=[self],
      name='collapse_%s' % name, context=context)

  def copy_collapse_as_mutates(self, context: CodegenContext, name: str = 'val') -> TypedValue:
    return self.copy_unbind().copy_collapse(context=context, name=name)

  def copy_unbind(self) -> TypedValue:
    assert self.num_unbindings + 1 <= self.num_possible_unbindings()
    new = self.copy()
    new.num_unbindings += 1
    return new

  def __eq__(self, other) -> bool:
    if not isinstance(other, TypedValue):
      return False
    return (self.type, self.narrowed_type, self.num_unbindings, self.ir_val) == (
      other.type, other.narrowed_type, other.num_unbindings, other.ir_val)
