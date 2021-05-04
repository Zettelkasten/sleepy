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
  def __init__(self, size, ir_type, pass_by_ref, c_type):
    """
    :param int size:
    :param ir.types.Type ir_type:
    :param bool pass_by_ref:
    :param Callable|None c_type:
    """
    assert not pass_by_ref or isinstance(ir_type, ir.types.PointerType)
    self.size = size
    self.ir_type = ir_type
    self.pass_by_ref = pass_by_ref
    self.c_type = c_type

  def __repr__(self):
    return self.__class__.__name__

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
  Typ returned when nothing is returned.
  """
  def __init__(self):
    super().__init__(0, ir.VoidType(), pass_by_ref=False, c_type=None)


class DoubleType(Type):
  """
  A double.
  """
  def __init__(self):
    super().__init__(8, ir.DoubleType(), pass_by_ref=False, c_type=ctypes.c_double)


class BoolType(Type):
  """
  A 1-bit integer.
  """
  def __init__(self):
    super().__init__(1, ir.IntType(bits=1), pass_by_ref=False, c_type=ctypes.c_bool)


class IntType(Type):
  """
  An 32-bit integer.
  """
  def __init__(self):
    super().__init__(4, ir.IntType(bits=32), pass_by_ref=False, c_type=ctypes.c_int32)


class LongType(Type):
  """
  A 64-bit integer.
  """
  def __init__(self):
    super().__init__(8, ir.IntType(bits=64), pass_by_ref=False, c_type=ctypes.c_int64)


class CharType(Type):
  """
  An 8-bit character.
  """
  def __init__(self):
    super().__init__(1, ir.IntType(bits=8), pass_by_ref=False, c_type=ctypes.c_char)


class DoublePtrType(Type):
  """
  A double pointer.
  """
  def __init__(self):
    super().__init__(8, ir.PointerType(ir.DoubleType()), pass_by_ref=False, c_type=ctypes.POINTER(ctypes.c_double))


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
      super().__init__(8, ir.types.PointerType(ir_val_type), pass_by_ref=True, c_type=ctypes.POINTER(c_type))
    else:
      size = sum(member_type.size for member_type in member_types)
      super().__init__(size, ir_val_type, pass_by_ref=False, c_type=c_type)
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
    self.var_type = var_type
    self.mutable = mutable


class ConcreteFunction:
  """
  An actual function implementation.
  """
  def __init__(self, ir_func, return_type, return_mutable, arg_identifiers, arg_types, arg_mutables, is_inline=False):
    """
    :param ir.Function|None ir_func:
    :param Type return_type:
    :param Bool return_mutable:
    :param list[str] arg_identifiers:
    :param list[Type] arg_types:
    :param list[bool] arg_mutables:
    :param bool is_inline:
    """
    assert ir_func is None or isinstance(ir_func, ir.Function)
    assert isinstance(return_type, Type)
    assert len(arg_identifiers) == len(arg_types) == len(arg_mutables)
    self.ir_func = ir_func
    self.return_type = return_type
    self.return_mutable = return_mutable
    self.arg_identifiers = arg_identifiers
    self.arg_types = arg_types
    self.arg_mutables = arg_mutables
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

  def make_inline_func_call_ir(self, ir_func_args, body_builder):
    """
    :param list[ir.values.Value] ir_func_args:
    :param ir.IRBuilder body_builder:
    :rtype: (ir.values.Value|None, ir.IRBuilder)
    """
    assert self.is_inline
    raise NotImplementedError()

  def __repr__(self):
    """
    :rtype str:
    """
    return 'ConcreteFunction(ir_func=%r, return_type=%r, arg_identifiers=%r, arg_types=%r, is_inline=%r)' % (
      self.ir_func, self.return_type, self.arg_identifiers, self.arg_types, self.is_inline)


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
  def __init__(self, copy_from=None):
    """
    :param SymbolTable|None copy_from:
    """
    if copy_from is None:
      self.symbols = {}  # type: Dict[str, Symbol]
      self.current_func = None  # type: Optional[ConcreteFunction]
      self.current_func_inline_return_collect_block = None  # type: Optional[ir.Block]
      self.current_func_inline_return_ir_alloca = None  # type: Optional[ir.instructions.AllocaInstr]
      self.ir_func_malloc = None  # type: Optional[ir.Function]
      self.ir_func_free = None  # type: Optional[ir.Function]
      self.used_ir_func_names = set()  # type: Set[str]
    else:
      self.symbols = copy_from.symbols.copy()  # type: Dict[str, Symbol]
      self.current_func = copy_from.current_func  # type: Optional[ConcreteFunction]
      self.current_func_inline_return_collect_block = copy_from.current_func_inline_return_collect_block  # type: Optional[ir.Block]  # noqa
      self.current_func_inline_return_ir_alloca = copy_from.current_func_inline_return_ir_alloca  # type: Optional[ir.instructions.AllocaInstr]  # noqa
      self.ir_func_malloc = copy_from.ir_func_malloc  # type: Optional[ir.Function]
      self.ir_func_free = copy_from.ir_func_free  # type: Optional[ir.Function]
      self.used_ir_func_names = copy_from.used_ir_func_names  # type: Set[str]
    self.current_scope_identifiers = []  # type: List[str]
    if self.current_func is None:
      assert self.current_func_inline_return_ir_alloca is None
    else:
      assert (self.current_func_inline_return_collect_block is None) == (not self.current_func.is_inline)

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


def _make_builtin_op_ir_val(op, op_arg_types, ir_func_args, body_builder):
  """
  :param str op:
  :param list[Type] op_arg_types:
  :param list[ir.values.Value] ir_func_args:
  :param ir.IRBuilder body_builder:
  :rtype: (ir.values.Value, ir.IRBuilder)
  """
  op_arg_types = tuple(op_arg_types)
  assert len(op_arg_types) == len(ir_func_args)

  def make_binary_op(single_type_instr, instr_name):
    """
    :param Dict[Type,Callable] single_type_instr:
    :param str instr_name:
    :rtype: (ir.values.Value, ir.IRBuilder)
    """
    assert len(op_arg_types) == len(ir_func_args) == 2
    type_instr = {(arg_type, arg_type): instr for arg_type, instr in single_type_instr.items()}
    return make_op(type_instr=type_instr, instr_name=instr_name)

  def make_op(type_instr, instr_name):
    """
    :param Dict[Tuple[Type],Callable] type_instr:
    :param str instr_name:
    :rtype: (ir.values.Value, ir.IRBuilder)
    """
    assert op_arg_types in type_instr
    return type_instr[op_arg_types](*ir_func_args, name=instr_name), body_builder

  if op == '*':
    return make_binary_op(
      {SLEEPY_DOUBLE: body_builder.fmul, SLEEPY_INT: body_builder.mul, SLEEPY_LONG: body_builder.mul}, instr_name='mul')
  if op == '/':
    return make_binary_op({SLEEPY_DOUBLE: body_builder.fdiv}, instr_name='div')
  if op == '+':
    if len(op_arg_types) == 1:
      assert len(ir_func_args) == 1
      return ir_func_args[0], body_builder
    assert len(op_arg_types) == len(ir_func_args) == 2
    left_type, right_type = op_arg_types
    left_val, right_val = ir_func_args
    if left_type == SLEEPY_DOUBLE_PTR and right_type == SLEEPY_INT:
      return body_builder.gep(left_val, (right_val,), name='add'), body_builder
    return make_binary_op(
      {SLEEPY_DOUBLE: body_builder.fadd, SLEEPY_INT: body_builder.add, SLEEPY_LONG: body_builder.add}, instr_name='add')
  if op == '-':
    if len(op_arg_types) == 1:
      assert len(ir_func_args) == 1
      val_type, ir_val = op_arg_types[0], ir_func_args[0]
      constant_minus_one = ir.Constant(val_type.ir_type, -1)
      if val_type == SLEEPY_DOUBLE:
        return body_builder.fmul(constant_minus_one, ir_val, name='neg'), body_builder
      if val_type in {SLEEPY_INT, SLEEPY_LONG}:
        return body_builder.mul(constant_minus_one, ir_val, name='neg'), body_builder
    return make_binary_op(
      {SLEEPY_DOUBLE: body_builder.fsub, SLEEPY_INT: body_builder.sub, SLEEPY_LONG: body_builder.sub}, instr_name='sub')
  if op in {'==', '!=', '<', '>', '<=', '>='}:
    from functools import partial
    return make_binary_op({
      SLEEPY_DOUBLE: partial(body_builder.fcmp_ordered, op), SLEEPY_INT: partial(body_builder.icmp_signed, op),
      SLEEPY_LONG: partial(body_builder.icmp_signed, op), SLEEPY_DOUBLE_PTR: partial(body_builder.icmp_unsigned, op)},
      instr_name='cmp')
  assert False, 'Operator %s not handled!' % op


def make_initial_symbol_table():
  """
  :rtype: SymbolTable
  """
  symbol_table = SymbolTable()
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
        arg_types=op_arg_types, arg_mutables=[False] * len(op_arg_types), is_inline=True)
      func_symbol.add_concrete_func(concrete_func)
  return symbol_table


def build_initial_module_ir(module, symbol_table):
  """
  :param ir.Module module:
  :param SymbolTable symbol_table:
  """
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
