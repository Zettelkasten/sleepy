"""
Implements a symbol table.
"""
import ctypes
from typing import Dict, Optional, List

from llvmlite import ir


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
  def __init__(self, ir_type, c_type):
    """
    :param ir_type:
    :param Callable|None c_type:
    """
    self.ir_type = ir_type
    self.c_type = c_type

  def __repr__(self):
    return self.__class__.__name__


class VoidType(Type):
  """
  Typ returned when nothing is returned.
  """
  def __init__(self):
    super().__init__(ir.VoidType(), None)


class DoubleType(Type):
  """
  A double.
  """
  def __init__(self):
    super().__init__(ir.DoubleType(), ctypes.c_double)


class BoolType(Type):
  """
  A 1-bit integer.
  """
  def __init__(self):
    super().__init__(ir.IntType(bits=1), ctypes.c_bool)


class IntType(Type):
  """
  An 32-bit integer.
  """
  def __init__(self):
    super().__init__(ir.IntType(bits=32), ctypes.c_int32)


class LongType(Type):
  """
  A 64-bit integer.
  """
  def __init__(self):
    super().__init__(ir.IntType(bits=64), ctypes.c_int64)


class CharType(Type):
  """
  An 32-bit character.
  """
  def __init__(self):
    super().__init__(ir.IntType(bits=32), ctypes.c_char)


class DoublePtrType(Type):
  """
  A double pointer.
  """
  def __init__(self):
    super().__init__(ir.PointerType(ir.DoubleType()), ctypes.pointer(ctypes.c_double()))


class StructType(Type):
  """
  A struct.
  """
  def __init__(self, struct_identifier, member_identifiers, member_types):
    """
    :param str struct_identifier:
    :param list[str] member_identifiers:
    :param list[Type] member_types:
    """
    assert len(member_identifiers) == len(member_types)
    member_ir_types = [member_type.ir_type for member_type in member_types]
    member_c_types = [
      (member_identifier, member_type.c_type)
      for member_identifier, member_type in zip(member_identifiers, member_types)]
    c_type = type('%s_CType' % struct_identifier, (ctypes.Structure,), {'_fields_': member_c_types})
    super().__init__(ir.LiteralStructType(member_ir_types), c_type)


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
  def __init__(self, ir_alloca, var_type):
    """
    :param ir.instructions.AllocaInstr|None ir_alloca:
    :param Type var_type:
    """
    super().__init__()
    assert ir_alloca is None or isinstance(ir_alloca, ir.instructions.AllocaInstr)
    self.ir_alloca = ir_alloca
    self.var_type = var_type


class FunctionSymbol(Symbol):
  """
  A declared (static) function (not a function pointer).
  """
  def __init__(self, ir_func, arg_identifiers, arg_types, return_type):
    """
    :param ir.Function|None ir_func:
    :param list[str] arg_identifiers:
    :param list[Type] arg_types:
    :param Type|None return_type:
    """
    super().__init__()
    assert ir_func is None or isinstance(ir_func, ir.Function)
    assert len(arg_identifiers) == len(arg_types)
    self.ir_func = ir_func
    self.arg_identifiers = arg_identifiers
    self.arg_types = arg_types
    self.return_type = return_type

  def get_c_arg_types(self):
    """
    :rtype:
    """
    return (self.return_type.c_type,) + tuple(arg_type.c_type for arg_type in self.arg_types)

  def make_ir_function_type(self):
    """
    :rtype: ir.FunctionType
    """
    return ir.FunctionType(self.return_type.ir_type, [arg_type.ir_type for arg_type in self.arg_types])


class TypeSymbol(Symbol):
  """
  A (statically) declared type.
  """
  def __init__(self, type, constructor_symbol):
    """
    :param Type type:
    :param FunctionSymbol|None constructor_symbol:
    """
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
      self.current_func = None  # type: Optional[FunctionSymbol]
    else:
      self.symbols = copy_from.symbols.copy()  # type: Dict[str, Symbol]
      self.current_func = copy_from.current_func  # type: Optional[FunctionSymbol]
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


def make_initial_symbol_table():
  """
  :rtype: SymbolTable
  """
  symbol_table = SymbolTable()
  for type_identifier, inbuilt_type in SLEEPY_TYPES.items():
    assert type_identifier not in symbol_table
    symbol_table[type_identifier] = TypeSymbol(inbuilt_type, constructor_symbol=None)
  return symbol_table
