"""
Implements a symbol table.
"""
import ctypes
from typing import Dict, Optional, List, Tuple

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
    :param Callable|None c_type:
    """
    assert not pass_by_ref or isinstance(ir_type, ir.types.PointerType)
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

  def make_ir_size(self, builder):
    """
    :param ir.IRBuilder builder:
    :rtype: ir.values.Value
    """
    ir_null_ptr = ir.values.Constant(ir.types.PointerType(self.ir_type), 'null')
    ir_size_ptr = builder.gep(ir_null_ptr, [ir.values.Constant(LLVM_SIZE_TYPE, 1)], name='size_ptr')
    return builder.ptrtoint(ir_size_ptr, LLVM_SIZE_TYPE, name='size')


class VoidType(Type):
  """
  Typ returned when nothing is returned.
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
  An 32-bit character.
  """
  def __init__(self):
    super().__init__(ir.IntType(bits=32), pass_by_ref=False, c_type=ctypes.c_char)


class DoublePtrType(Type):
  """
  A double pointer.
  """
  def __init__(self):
    super().__init__(ir.PointerType(ir.DoubleType()), pass_by_ref=False, c_type=ctypes.POINTER(ctypes.c_double))


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

SLEEPY_INBUILT_BINARY_OPS = ['+', '-', '*', '/']  # type: List[str]
SLEEPY_INBUILT_BINARY_IR_BUILDERS = [
  {SLEEPY_DOUBLE: 'fadd', SLEEPY_INT: 'add', SLEEPY_LONG: 'add'},
  {SLEEPY_DOUBLE: 'fsub', SLEEPY_INT: 'sub', SLEEPY_LONG: 'sub'},
  {SLEEPY_DOUBLE: 'fmul', SLEEPY_INT: 'mul', SLEEPY_LONG: 'mul'},
  {SLEEPY_DOUBLE: 'fdiv'}]  # type: List[Dict[Type, str]]
SLEEPY_INBUILT_BINARY_OP_RETURN_TYPE = []


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
  def __init__(self, ir_func, return_type, return_mutable, arg_identifiers, arg_types, arg_mutables):
    """
    :param ir.Function|None ir_func:
    :param Type return_type:
    :param Bool return_mutable:
    :param list[str] arg_identifiers:
    :param list[Type] arg_types:
    :param list[bool] arg_mutables:
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
    assert self.ir_func is not None
    from sleepy.jit import get_func_address
    main_func_ptr = get_func_address(engine, self.ir_func)
    py_func = ctypes.CFUNCTYPE(
      self.return_type.c_type, *[arg_type.c_type for arg_type in self.arg_types])(main_func_ptr)
    assert callable(py_func)
    return py_func

  def __repr__(self):
    """
    :rtype str:
    """
    return 'ConcreteFunction(ir_func=%r, return_type=%r, arg_identifiers=%r, arg_types=%r)' % (
      self.ir_func, self.return_type, self.arg_identifiers, self.arg_types)


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
      self.ir_func_malloc = None  # type: Optional[ir.Function]
      self.ir_func_free = None  # type: Optional[ir.Function]
    else:
      self.symbols = copy_from.symbols.copy()  # type: Dict[str, Symbol]
      self.current_func = copy_from.current_func  # type: Optional[ConcreteFunction]
      self.ir_func_malloc = copy_from.ir_func_malloc  # type: Optional[ir.Function]
      self.ir_func_free = copy_from.ir_func_free  # type: Optional[ir.Function]
    self.current_scope_identifiers = []  # type: List[str]

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

  @classmethod
  def make_ir_func_name(cls, func_identifier, extern, concrete_func):
    """
    :param str func_identifier:
    :param bool extern:
    :param ConcreteFunction concrete_func:
    :rtype: str
    """
    if extern:
      return func_identifier
    else:
      return func_identifier + ''.join([str(arg_type) for arg_type in concrete_func.arg_types])


def make_initial_symbol_table():
  """
  :rtype: SymbolTable
  """
  symbol_table = SymbolTable()
  for type_identifier, inbuilt_type in SLEEPY_TYPES.items():
    assert type_identifier not in symbol_table
    symbol_table[type_identifier] = TypeSymbol(inbuilt_type, constructor_symbol=None)

  for op, op_builders in zip(SLEEPY_INBUILT_BINARY_OPS, SLEEPY_INBUILT_BINARY_IR_BUILDERS):
    assert op not in symbol_table
    func_symbol = FunctionSymbol()
    symbol_table[op] = func_symbol
    for operand_type in op_builders.keys():
      assert not func_symbol.has_concrete_func([operand_type, operand_type])
      # ir_func will be set in build_initial_module_ir
      concrete_func = ConcreteFunction(
        ir_func=None, return_type=operand_type, return_mutable=False, arg_identifiers=['left', 'right'],
        arg_types=[operand_type, operand_type], arg_mutables=[False, False])
      func_symbol.add_concrete_func(concrete_func)
  return symbol_table


def build_initial_module_ir(module, builder, symbol_table):
  """
  :param ir.Module module:
  :param ir.IRBuilder builder:
  :param SymbolTable symbol_table:
  :rtype: ir.IRBuilder
  """
  symbol_table.ir_func_malloc = ir.Function(
    module, ir.FunctionType(LLVM_VOID_POINTER_TYPE, [LLVM_SIZE_TYPE]), name='malloc')
  symbol_table.ir_func_free = ir.Function(
    module, ir.FunctionType(ir.VoidType(), [LLVM_VOID_POINTER_TYPE]), name='free')

  for op, op_builders in zip(SLEEPY_INBUILT_BINARY_OPS, SLEEPY_INBUILT_BINARY_IR_BUILDERS):
    assert op in symbol_table
    func_symbol = symbol_table[op]
    assert isinstance(func_symbol, FunctionSymbol)
    for operand_type, op_builder_name in op_builders.items():
      assert func_symbol.has_concrete_func([operand_type, operand_type])
      concrete_func = func_symbol.get_concrete_func([operand_type, operand_type])
      ir_func_type = concrete_func.make_ir_function_type()
      ir_func_name = symbol_table.make_ir_func_name(op, extern=False, concrete_func=concrete_func)
      concrete_func.ir_func = ir.Function(module, ir_func_type, name=ir_func_name)
      assert len(concrete_func.ir_func.args) == 2
      left_arg, right_arg = concrete_func.ir_func.args
      left_arg.name, right_arg.name = 'left', 'right'
      body_block = concrete_func.ir_func.append_basic_block(name='entry')
      body_builder = ir.IRBuilder(body_block)
      assert hasattr(body_builder, op_builder_name)
      builder_func = getattr(body_builder, op_builder_name)
      assert callable(builder_func)
      ir_op = builder_func(lhs=left_arg, rhs=right_arg, name=op_builder_name)
      body_builder.ret(ir_op)
  print(symbol_table.symbols)
  return builder
