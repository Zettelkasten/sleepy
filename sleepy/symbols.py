"""
Implements a symbol table.
"""
import ctypes
from typing import Dict, Optional, List, Tuple, Set

from llvmlite import ir


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
    self.base = None  # type: Optional[Symbol]


class Type:
  """
  A type of a declared variable.
  """
  def __init__(self, ir_type, pass_by_ref, c_type):
    """
    :param ir.types.Type ir_type:
    :param bool pass_by_ref:
    :param ctypes._CData|None c_type:
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
    possible_types = list(dict.fromkeys(possible_types)) # possibly remove duplicates
    possible_type_nums = list(range(len(possible_types)))
    val_size = max(ctypes.sizeof(possible_type.c_type) for possible_type in possible_types)
    return UnionType(possible_types=possible_types, possible_type_nums=possible_type_nums, val_size=val_size)


class StructType(Type):
  """
  A struct.
  """
  def __init__(self, struct_identifier, member_identifiers, member_types, member_mutables, pass_by_ref):
    """
    :param str struct_identifier:
    :param list[str] member_identifiers:
    :param list[Type] member_types:
    :param list[bool] member_mutables:
    """
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


SLEEPY_VOID = VoidType()
SLEEPY_NEVER = UnionType(possible_types=[], possible_type_nums=[], val_size=0)
SLEEPY_DOUBLE = DoubleType()
SLEEPY_BOOL = BoolType()
SLEEPY_INT = IntType()
SLEEPY_LONG = LongType()
SLEEPY_CHAR = CharType()
SLEEPY_DOUBLE_PTR = DoublePtrType()

SLEEPY_TYPES = {
  'Void': SLEEPY_VOID, 'Double': SLEEPY_DOUBLE, 'Bool': SLEEPY_BOOL, 'Int': SLEEPY_INT, 'Long': SLEEPY_LONG,
  'Char': SLEEPY_CHAR, 'DoublePtr': SLEEPY_DOUBLE_PTR}  # type: Dict[str, Type]
SLEEPY_NUMERICAL_TYPES = {SLEEPY_DOUBLE, SLEEPY_INT, SLEEPY_LONG}


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
  assert not to_type.is_pass_by_ref()
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
  def __init__(self, ir_func, return_type, return_mutable, arg_identifiers, arg_types, arg_mutables,
               arg_type_narrowings, is_inline=False):
    """
    :param ir.Function|None ir_func:
    :param Type return_type:
    :param Bool return_mutable:
    :param list[str] arg_identifiers:
    :param list[Type] arg_types:
    :param list[bool] arg_mutables:
    :param list[Type] arg_type_narrowings:
    :param bool is_inline:
    """
    assert ir_func is None or isinstance(ir_func, ir.Function)
    assert isinstance(return_type, Type)
    assert len(arg_identifiers) == len(arg_types) == len(arg_mutables) == len(arg_type_narrowings)
    self.ir_func = ir_func
    self.return_type = return_type
    self.return_mutable = return_mutable
    self.arg_identifiers = arg_identifiers
    self.arg_types = arg_types
    self.arg_mutables = arg_mutables
    self.arg_type_narrowings = arg_type_narrowings
    self.is_inline = is_inline

  def get_c_arg_types(self):
    """
    :rtype: tuple[callable]
    """
    return (self.return_type.c_type,) + tuple(arg_type.c_type for arg_type in self.arg_types)

  def make_ir_function_type(self):
    """
    :rtype: ir.FunctionType
    """
    return ir.FunctionType(self.return_type.ir_type, [arg_type.ir_type for arg_type in self.arg_types])

  def to_signature_str(self):
    """
    :rtype: str
    """
    return '(%s) -> %s' % (
        ', '.join(['%s %s' % arg_tuple for arg_tuple in zip(self.arg_types, self.arg_identifiers)]), self.return_type)

  def make_py_func(self, engine):
    """
    :param ExecutionEngine engine:
    :rtype: callable
    """
    assert not self.is_inline
    assert self.ir_func is not None
    from sleepy.jit import get_func_address
    main_func_ptr = get_func_address(engine, self.ir_func)
    py_func = ctypes.CFUNCTYPE(
      self.return_type.c_type, *[arg_type.c_type for arg_type in self.arg_types])(main_func_ptr)
    assert callable(py_func)
    return py_func

  @property
  def returns_void(self):
    """
    :rtype: bool
    """
    return self.return_type == SLEEPY_VOID

  def make_inline_func_call_ir(self, ir_func_args, caller_context):
    """
    :param list[ir.values.Value] ir_func_args:
    :param CodegenContext caller_context:
    :rtype: ir.values.Value|None
    :return: the return value of this function call, or None if it is a void function
    """
    assert self.is_inline
    raise NotImplementedError()

  def __repr__(self):
    """
    :rtype str:
    """
    return (
      'ConcreteFunction(ir_func=%r, return_type=%r, arg_identifiers=%r, arg_types=%r, arg_type_narrowings=%r, '
      'is_inline=%r)' % (
        self.ir_func, self.return_type, self.arg_identifiers, self.arg_types, self.arg_type_narrowings, self.is_inline))


class FunctionSymbol(Symbol):
  """
  A set of declared (static) functions (not a function pointer) with a fixed name.
  Can have one or multiple overloaded concrete functions accepting different parameter types.
  These functions are managed here by their expanded (i.e. simple, non union types).
  """
  def __init__(self, returns_void):
    """
    :param bool returns_void:
    """
    super().__init__()
    self.concrete_funcs = {}  # type: Dict[Tuple[Type], ConcreteFunction]
    self.returns_void = returns_void

  def has_concrete_func(self, arg_types):
    """
    :param list[Type]|tuple[Type] arg_types:
    :rtype: bool
    """
    all_expanded_arg_types = self._iter_expanded_possible_arg_types(arg_types)
    return all(expanded_arg_types in self.concrete_funcs for expanded_arg_types in all_expanded_arg_types)

  def get_concrete_funcs(self, arg_types):
    """
    :param list[Type]|tuple[Type] arg_types:
    :rtype: list[ConcreteFunction]
    """
    possible_concrete_funcs = []
    for expanded_arg_types in self._iter_expanded_possible_arg_types(arg_types):
      if expanded_arg_types in self.concrete_funcs:
        concrete_func = self.concrete_funcs[expanded_arg_types]
        if concrete_func not in possible_concrete_funcs:
          possible_concrete_funcs.append(concrete_func)
    return possible_concrete_funcs

  def add_concrete_func(self, concrete_func):
    """
    :param ConcreteFunction concrete_func:
    """
    assert not self.has_concrete_func(concrete_func.arg_types)
    assert concrete_func.returns_void == self.returns_void
    for expanded_arg_types in self._iter_expanded_possible_arg_types(concrete_func.arg_types):
     self.concrete_funcs[expanded_arg_types] = concrete_func

  def get_single_concrete_func(self):
    """
    :rtype: ConcreteFunction
    """
    assert len(set(self.concrete_funcs.values())) == 1
    return next(iter(self.concrete_funcs.values()))

  @classmethod
  def _iter_expanded_possible_arg_types(cls, arg_types):
    """
    :param list[Type]|tuple[Type] arg_types:
    :rtype: Iterator[Tuple[Type]]
    """
    import itertools
    return itertools.product(*[
      arg_type.possible_types if isinstance(arg_type, UnionType) else [arg_type] for arg_type in arg_types])


class TypeSymbol(Symbol):
  """
  A (statically) declared type.
  """
  def __init__(self, type, constructor_symbol):
    """
    :param Type type:
    :param FunctionSymbol|None constructor_symbol:
    """
    super().__init__()
    self.type = type
    self.constructor_symbol = constructor_symbol


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
    else:
      self.symbols = copy_from.symbols.copy()  # type: Dict[str, Symbol]
      if copy_new_current_func is None:
        self.current_func = copy_from.current_func  # type: Optional[ConcreteFunction]
        self.current_scope_identifiers = copy_from.current_scope_identifiers.copy()  # type: List[str]
      else:
        self.current_func = copy_new_current_func  # type: Optional[ConcreteFunction]
        self.current_scope_identifiers = []  # type: List[str]
      self.used_ir_func_names = copy_from.used_ir_func_names  # type: Set[str]

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
    else:
      ir_func_name = '_'.join([func_identifier] + [str(arg_type) for arg_type in concrete_func.arg_types])
      assert ir_func_name not in self.used_ir_func_names
    self.used_ir_func_names.add(ir_func_name)
    return ir_func_name

  def apply_symbols_from(self, other_symbol_table):
    """
    :param SymbolTable other_symbol_table:
    """
    for symbol_identifier, other_symbol in other_symbol_table.symbols.items():
      if symbol_identifier in self and self[symbol_identifier].base == other_symbol.base:
        self[symbol_identifier] = other_symbol

  def reset_narrowed_types(self):
    """
    Applies symbol.copy_reset_narrowed_type() for all variable symbols.
    """
    for symbol_identifier, symbol in self.symbols.items():
      if isinstance(symbol, VariableSymbol):
        if symbol.declared_var_type != symbol.narrowed_var_type:
          self[symbol_identifier] = symbol.copy_reset_narrowed_type()


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


SLEEPY_INBUILT_BINARY_OPS = ['+', '-', '*', '/', '==', '!=', '<', '>', '<=', '>=']  # type: List[str]
SLEEPY_INBUILT_BINARY_OPS_ARG_TYPES = [
  [
    [SLEEPY_DOUBLE, SLEEPY_DOUBLE], [SLEEPY_INT, SLEEPY_INT], [SLEEPY_LONG, SLEEPY_LONG],
    [SLEEPY_DOUBLE_PTR, SLEEPY_INT],
    [SLEEPY_DOUBLE], [SLEEPY_INT], [SLEEPY_LONG]
  ], [
    [SLEEPY_DOUBLE, SLEEPY_DOUBLE], [SLEEPY_INT, SLEEPY_INT], [SLEEPY_LONG, SLEEPY_LONG],
    [SLEEPY_DOUBLE], [SLEEPY_INT], [SLEEPY_LONG]
  ],
  [[SLEEPY_DOUBLE, SLEEPY_DOUBLE], [SLEEPY_INT, SLEEPY_INT], [SLEEPY_LONG, SLEEPY_LONG]],
  [[SLEEPY_DOUBLE, SLEEPY_DOUBLE]]] + [
    [[SLEEPY_DOUBLE, SLEEPY_DOUBLE], [SLEEPY_INT, SLEEPY_INT], [SLEEPY_LONG, SLEEPY_LONG],
    [SLEEPY_DOUBLE_PTR, SLEEPY_DOUBLE_PTR]]] * 6  # type: List[List[List[Type]]]
SLEEPY_INBUILT_BINARY_OPS_RETURN_TYPES = [
  [SLEEPY_DOUBLE, SLEEPY_INT, SLEEPY_LONG, SLEEPY_DOUBLE_PTR, SLEEPY_DOUBLE, SLEEPY_INT, SLEEPY_LONG],
  [SLEEPY_DOUBLE, SLEEPY_INT, SLEEPY_LONG, SLEEPY_DOUBLE, SLEEPY_INT, SLEEPY_LONG],
  [SLEEPY_DOUBLE, SLEEPY_INT, SLEEPY_LONG],
  [SLEEPY_DOUBLE]] + [[SLEEPY_BOOL] * 4] * 6  # type: List[List[Type]]
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
    return make_binary_op(
      {SLEEPY_DOUBLE: body_builder.fmul, SLEEPY_INT: body_builder.mul, SLEEPY_LONG: body_builder.mul}, instr_name='mul')
  if op == '/':
    return make_binary_op({SLEEPY_DOUBLE: body_builder.fdiv}, instr_name='div')
  if op == '+':
    if len(op_arg_types) == 1:
      assert len(ir_func_args) == 1
      return ir_func_args[0]
    assert len(op_arg_types) == len(ir_func_args) == 2
    left_type, right_type = op_arg_types
    left_val, right_val = ir_func_args
    if left_type == SLEEPY_DOUBLE_PTR and right_type == SLEEPY_INT:
      return body_builder.gep(left_val, (right_val,), name='add')
    return make_binary_op(
      {SLEEPY_DOUBLE: body_builder.fadd, SLEEPY_INT: body_builder.add, SLEEPY_LONG: body_builder.add}, instr_name='add')
  if op == '-':
    if len(op_arg_types) == 1:
      assert len(ir_func_args) == 1
      val_type, ir_val = op_arg_types[0], ir_func_args[0]
      constant_minus_one = ir.Constant(val_type.ir_type, -1)
      if val_type == SLEEPY_DOUBLE:
        return body_builder.fmul(constant_minus_one, ir_val, name='neg')
      if val_type in {SLEEPY_INT, SLEEPY_LONG}:
        return body_builder.mul(constant_minus_one, ir_val, name='neg')
    return make_binary_op(
      {SLEEPY_DOUBLE: body_builder.fsub, SLEEPY_INT: body_builder.sub, SLEEPY_LONG: body_builder.sub}, instr_name='sub')
  if op in {'==', '!=', '<', '>', '<=', '>='}:
    from functools import partial
    return make_binary_op({
      SLEEPY_DOUBLE: partial(body_builder.fcmp_ordered, op), SLEEPY_INT: partial(body_builder.icmp_signed, op),
      SLEEPY_LONG: partial(body_builder.icmp_signed, op), SLEEPY_DOUBLE_PTR: partial(body_builder.icmp_unsigned, op)},
      instr_name='cmp')
  assert False, 'Operator %s not handled!' % op


def build_initial_ir(symbol_table, context):
  """
  :param SymbolTable symbol_table:
  :param CodegenContext context:
  """
  assert 'free' not in symbol_table
  free_symbol = FunctionSymbol(returns_void=True)
  symbol_table['free'] = free_symbol
  for type_identifier, inbuilt_type in SLEEPY_TYPES.items():
    assert type_identifier not in symbol_table
    symbol_table[type_identifier] = TypeSymbol(inbuilt_type, constructor_symbol=None)
    if inbuilt_type == SLEEPY_VOID:
      continue

    # add destructor
    destructor = ConcreteFunction(
      ir_func=None, return_type=SLEEPY_VOID, return_mutable=False, arg_identifiers=['self'], arg_types=[inbuilt_type],
      arg_mutables=[False], arg_type_narrowings=[SLEEPY_NEVER])
    if context.emits_ir:
      ir_func_name = symbol_table.make_ir_func_name('free', extern=False, concrete_func=destructor)
      destructor.ir_func = ir.Function(context.module, destructor.make_ir_function_type(), name=ir_func_name)
      destructor_block = destructor.ir_func.append_basic_block(name='entry')
      destructor_context = context.copy_with_builder(ir.IRBuilder(destructor_block))
      destructor_context.builder.ret_void()  # destructor does not do anything for value types
    free_symbol.add_concrete_func(destructor)

  if context.emits_ir:
    module = context.builder.module
    context.ir_func_malloc = ir.Function(
      module, ir.FunctionType(LLVM_VOID_POINTER_TYPE, [LLVM_SIZE_TYPE]), name='malloc')
    context.ir_func_free = ir.Function(module, ir.FunctionType(ir.VoidType(), [LLVM_VOID_POINTER_TYPE]), name='free')

  for op, op_arg_type_list, op_return_type_list in zip(
    SLEEPY_INBUILT_BINARY_OPS, SLEEPY_INBUILT_BINARY_OPS_ARG_TYPES, SLEEPY_INBUILT_BINARY_OPS_RETURN_TYPES):
    assert op not in symbol_table
    func_symbol = FunctionSymbol(returns_void=False)
    symbol_table[op] = func_symbol
    for op_arg_types, op_return_type in zip(op_arg_type_list, op_return_type_list):
      assert not func_symbol.has_concrete_func(op_arg_types)
      op_arg_identifiers = _make_builtin_op_arg_names(op, op_arg_types)
      assert len(op_arg_types) == len(op_arg_identifiers)
      # ir_func will be set in build_initial_module_ir
      concrete_func = ConcreteFunction(
        ir_func=None, return_type=op_return_type, return_mutable=False, arg_identifiers=op_arg_identifiers,
        arg_types=op_arg_types, arg_mutables=[False] * len(op_arg_types), arg_type_narrowings=op_arg_types,
        is_inline=True)
      if context.emits_ir:
        from functools import partial
        concrete_func.make_inline_func_call_ir = partial(_make_builtin_op_ir_val, op, op_arg_types)
      func_symbol.add_concrete_func(concrete_func)
