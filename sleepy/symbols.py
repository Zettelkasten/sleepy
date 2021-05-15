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
  def __init__(self, possible_types, possible_type_nums):
    """
    :param list[Type] possible_types:
    :param list[int] possible_type_nums:
    """
    assert len(possible_types) == len(possible_type_nums)
    self.possible_types = possible_types
    self.possible_type_nums = possible_type_nums
    self.identifier = 'Union(%s)' % '_'.join(str(possible_type) for possible_type in possible_types)

    self.tag_c_type = ctypes.c_uint8
    self.tag_ir_type = ir.types.IntType(8)
    self.untagged_union_c_type = type(
      '%s_UntaggedCType' % self.identifier, (ctypes.Union,),
      {'_fields_': [('variant%s' % num, possible_type.c_type) for num, possible_type in enumerate(possible_types)]})
    # Somehow, you cannot bitcast a byte array to e.g. a double, thus we just use a very large int type here
    self.untagged_union_ir_type = ir.types.IntType(8 * ctypes.sizeof(self.untagged_union_c_type))
    c_type = type(
      '%s_CType' % self.identifier, (ctypes.Structure,),
      {'_fields_': [('tag', self.tag_c_type), ('untagged_union', self.untagged_union_c_type)]})
    ir_type = ir.types.LiteralStructType([self.tag_ir_type, self.untagged_union_ir_type])
    super().__init__(ir_type, pass_by_ref=False, c_type=c_type)

  def __repr__(self):
    return self.identifier

  def __eq__(self, other):
    """
    :param Any other:
    """
    if not isinstance(other, UnionType):
      return False
    return self.possible_types == other.possible_types and self.possible_type_nums == other.possible_type_nums

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
    return self.possible_types.index(variant_type)
  
  def make_tag_ptr(self, union_ir_alloca, context):
    """
    :param ir.instructions.AllocaInstr union_ir_alloca:
    :param CodegenContext context:
    :rtype: ir.instructions.Instruction
    """
    assert context.emits_ir
    tag_gep_indices = (ir.Constant(ir.IntType(32), 0), ir.Constant(ir.IntType(32), 0))
    return context.builder.gep(union_ir_alloca, tag_gep_indices, name='%s_tag_ptr' % self.identifier)

  def make_untagged_union_ptr(self, union_ir_alloca, variant_type, context):
    """
    :param ir.instructions.AllocaInstr union_ir_alloca:
    :param Type variant_type:
    :param CodegenContext context:
    :rtype: ir.instructions.Instruction
    """
    assert context.emits_ir
    assert variant_type in self.possible_types
    untagged_union_gep_indices = (ir.Constant(ir.IntType(32), 0), ir.Constant(ir.IntType(32), 1))
    untagged_union_ptr = context.builder.gep(
      union_ir_alloca, untagged_union_gep_indices, name='%s_untagged_union_ptr' % self.identifier)
    return context.builder.bitcast(
      untagged_union_ptr, ir.types.PointerType(variant_type.ir_type),
      name='%s_untagged_union_ptr_cast' % self.identifier)

  def make_extract_val(self, union_ir_val, variant_type, context):
    """
    :param ir.values.Value union_ir_val:
    :param Type variant_type:
    :param CodegenContext context:
    :rtype: ir.values.Value
    """
    assert context.emits_ir
    assert variant_type in self.possible_types
    untagged_union_val = context.builder.extract_value(union_ir_val, 1, name='%s_untagged_union' % self.identifier)
    return context.builder.bitcast(
      untagged_union_val, variant_type.ir_type, name='%s_untagged_union_cast' % self.identifier)


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


SLEEPY_VOID = VoidType()
SLEEPY_NEVER = UnionType(possible_types=[], possible_type_nums=[])
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
  if isinstance(from_type, UnionType):
    possible_from_types = from_type.possible_types
  else:
    possible_from_types = [from_type]
  if not isinstance(to_type, UnionType):
    to_type = UnionType(possible_types=[to_type], possible_type_nums=[0])
  return all(to_type.contains(possible_from_type) for possible_from_type in possible_from_types)


def make_implicit_cast_to_ir_val(from_type, to_type, from_ir_val, context):
  """
  :param Type from_type:
  :param Type to_type:
  :param ir.values.Value from_ir_val:
  :param CodegenContext context:
  :rtype: ir.values.Value
  """
  assert context.emits_ir
  assert can_implicit_cast_to(from_type, to_type)
  if from_type == to_type:
    return from_ir_val
  assert not to_type.is_pass_by_ref()
  if isinstance(to_type, UnionType):
    assert not isinstance(from_type, UnionType), 'not implemented yet'
    variant_num = to_type.get_variant_num(from_type)
    to_ir_alloca = context.builder.alloca(to_type.ir_type, name=str(to_type))
    context.builder.store(
      ir.Constant(to_type.tag_ir_type, variant_num), to_type.make_tag_ptr(to_ir_alloca, context=context))
    context.builder.store(from_ir_val, to_type.make_untagged_union_ptr(to_ir_alloca, from_type, context=context))
    return context.builder.load(to_ir_alloca)
  else:
    assert not isinstance(to_type, UnionType)
    # this is only possible when from_type is a single-type union
    assert isinstance(from_type, UnionType)
    assert all(possible_from_type == to_type for possible_from_type in from_type.possible_types)
    return from_type.make_extract_val(from_ir_val, to_type, context=context)


def narrow_type(from_type, narrow_to):
  """
  :param Type from_type:
  :param Type narrow_to:
  :rtype: Type
  """
  if from_type == narrow_to:
    return from_type
  if not isinstance(from_type, UnionType):
    return SLEEPY_NEVER
  if not isinstance(narrow_to, UnionType):
    narrow_to = UnionType(possible_types=[narrow_to], possible_type_nums=[0])
  possible_types = [
    possible_type for possible_type in narrow_to.possible_types if possible_type in from_type.possible_types]
  possible_type_nums = [
    type_num for type_num, possible_type in zip(narrow_to.possible_type_nums, narrow_to.possible_types)
    if possible_type in from_type.possible_types]
  return UnionType(possible_types=possible_types, possible_type_nums=possible_type_nums)


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
    return True
  if not isinstance(known_type, UnionType):
    return False
  if not known_type.contains(check_type):
    return False
  assert not isinstance(check_type, UnionType), 'not implemented yet'
  union_tag = context.builder.extract_value(ir_val, 0)
  cmp_val = context.builder.icmp_signed(
    '==', union_tag, ir.Constant(known_type.tag_ir_type, known_type.get_variant_num(check_type)))
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

  def copy_with_narrowed_type(self, asserted_var_type):
    """
    :param Type asserted_var_type:
    :rtype: VariableSymbol
    """
    new_var_symbol = VariableSymbol(self.ir_alloca, self.declared_var_type, self.mutable)
    new_var_symbol.narrowed_var_type = asserted_var_type
    return new_var_symbol

  def build_ir_alloca(self, context, identifier):
    """
    :param CodegenContext context:
    :param str identifier:
    """
    assert self.ir_alloca is None
    if not context.emits_ir:
      return
    self.ir_alloca = context.builder.alloca(self.declared_var_type.ir_type, name=identifier)


class ConcreteFunction:
  """
  An actual function implementation.
  """
  def __init__(self, ir_func, return_type, return_mutable, arg_identifiers, arg_types, arg_mutables,
               arg_type_assertions, is_inline=False):
    """
    :param ir.Function|None ir_func:
    :param Type return_type:
    :param Bool return_mutable:
    :param list[str] arg_identifiers:
    :param list[Type] arg_types:
    :param list[bool] arg_mutables:
    :param list[Type] arg_type_assertions:
    :param bool is_inline:
    """
    assert ir_func is None or isinstance(ir_func, ir.Function)
    assert isinstance(return_type, Type)
    assert len(arg_identifiers) == len(arg_types) == len(arg_mutables) == len(arg_type_assertions)
    self.ir_func = ir_func
    self.return_type = return_type
    self.return_mutable = return_mutable
    self.arg_identifiers = arg_identifiers
    self.arg_types = arg_types
    self.arg_mutables = arg_mutables
    self.arg_type_narrowings = arg_type_assertions
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
      'ConcreteFunction(ir_func=%r, return_type=%r, arg_identifiers=%r, arg_types=%r, arg_type_assertions=%r, '
      'is_inline=%r)' % (
        self.ir_func, self.return_type, self.arg_identifiers, self.arg_types, self.arg_type_assertions, self.is_inline))


class FunctionSymbol(Symbol):
  """
  A set of declared (static) functions (not a function pointer) with a fixed name.
  Can have one or multiple overloaded concrete functions accepting different parameter types.
  """
  def __init__(self):
    super().__init__()
    self.concrete_funcs = {}  # type: Dict[Tuple[Type], ConcreteFunction]

  def has_concrete_func(self, arg_types):
    """
    :param list[Type]|tuple[Type] arg_types:
    :rtype: bool
    """
    return tuple(arg_types) in self.concrete_funcs

  def get_concrete_func(self, arg_types):
    """
    :param list[Type]|tuple[Type] arg_types:
    :rtype: ConcreteFunction
    """
    return self.concrete_funcs[tuple(arg_types)]

  def add_concrete_func(self, concrete_func):
    """
    :param ConcreteFunction concrete_func:
    """
    assert not self.has_concrete_func(concrete_func.arg_types)
    self.concrete_funcs[tuple(concrete_func.arg_types)] = concrete_func

  def get_single_concrete_func(self):
    """
    :rtype: ConcreteFunction
    """
    assert len(self.concrete_funcs) == 1
    return next(iter(self.concrete_funcs.values()))


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
      self.ir_func_malloc = None  # type: Optional[ir.Function]
      self.ir_func_free = None  # type: Optional[ir.Function]
      self.used_ir_func_names = set()  # type: Set[str]
    else:
      self.symbols = copy_from.symbols.copy()  # type: Dict[str, Symbol]
      if copy_new_current_func is None:
        self.current_func = copy_from.current_func  # type: Optional[ConcreteFunction]
        self.current_scope_identifiers = copy_from.current_scope_identifiers.copy()  # type: List[str]
      else:
        self.current_func = copy_new_current_func  # type: Optional[ConcreteFunction]
        self.current_scope_identifiers = []  # type: List[str]
      self.ir_func_malloc = copy_from.ir_func_malloc  # type: Optional[ir.Function]
      self.ir_func_free = copy_from.ir_func_free  # type: Optional[ir.Function]
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
    else:
      self.emits_ir = copy_from.emits_ir
      self.is_terminated = copy_from.is_terminated
      
      self.current_func_inline_return_collect_block = copy_from.current_func_inline_return_collect_block  # type: Optional[ir.Block]  # noqa
      self.current_func_inline_return_ir_alloca = copy_from.current_func_inline_return_ir_alloca  # type: Optional[ir.instructions.AllocaInstr]  # noqa
      self.inline_func_call_stack = copy_from.inline_func_call_stack.copy()  # type: List[ConcreteFunction]

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
  for type_identifier, inbuilt_type in SLEEPY_TYPES.items():
    assert type_identifier not in symbol_table
    symbol_table[type_identifier] = TypeSymbol(inbuilt_type, constructor_symbol=None)

  for op, op_arg_type_list, op_return_type_list in zip(
    SLEEPY_INBUILT_BINARY_OPS, SLEEPY_INBUILT_BINARY_OPS_ARG_TYPES, SLEEPY_INBUILT_BINARY_OPS_RETURN_TYPES):
    assert op not in symbol_table
    func_symbol = FunctionSymbol()
    symbol_table[op] = func_symbol
    for op_arg_types, op_return_type in zip(op_arg_type_list, op_return_type_list):
      assert not func_symbol.has_concrete_func(op_arg_types)
      op_arg_identifiers = _make_builtin_op_arg_names(op, op_arg_types)
      assert len(op_arg_types) == len(op_arg_identifiers)
      # ir_func will be set in build_initial_module_ir
      concrete_func = ConcreteFunction(
        ir_func=None, return_type=op_return_type, return_mutable=False, arg_identifiers=op_arg_identifiers,
        arg_types=op_arg_types, arg_mutables=[False] * len(op_arg_types), arg_type_assertions=op_arg_types,
        is_inline=True)
      func_symbol.add_concrete_func(concrete_func)

  if context.emits_ir:
    module = context.builder.module
    symbol_table.ir_func_malloc = ir.Function(
      module, ir.FunctionType(LLVM_VOID_POINTER_TYPE, [LLVM_SIZE_TYPE]), name='malloc')
    symbol_table.ir_func_free = ir.Function(
      module, ir.FunctionType(ir.VoidType(), [LLVM_VOID_POINTER_TYPE]), name='free')

    for op, op_arg_type_list, op_return_type_list in zip(
      SLEEPY_INBUILT_BINARY_OPS, SLEEPY_INBUILT_BINARY_OPS_ARG_TYPES, SLEEPY_INBUILT_BINARY_OPS_RETURN_TYPES):
      assert op in symbol_table
      func_symbol = symbol_table[op]
      assert isinstance(func_symbol, FunctionSymbol)
      for op_arg_types, op_return_type in zip(op_arg_type_list, op_return_type_list):
        assert func_symbol.has_concrete_func(op_arg_types)
        concrete_func = func_symbol.get_concrete_func(op_arg_types)
        from functools import partial
        concrete_func.make_inline_func_call_ir = partial(_make_builtin_op_ir_val, op, op_arg_types)
