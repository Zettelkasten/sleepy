"""
Implements a symbol table.
"""
import ctypes
from typing import Dict

from llvmlite import ir

from sleepy.grammar import SemanticError


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


SLEEPY_VOID = VoidType()
SLEEPY_DOUBLE = DoubleType()
SLEEPY_INT = IntType()
SLEEPY_LONG = LongType()
SLEEPY_CHAR = CharType()
SLEEPY_DOUBLE_PTR = DoublePtrType()

SLEEPY_TYPES = {
  'Void': SLEEPY_VOID, 'Double': SLEEPY_DOUBLE, 'Int': SLEEPY_INT, 'Long': SLEEPY_LONG, 'Char': SLEEPY_CHAR,
  'DoublePtr': SLEEPY_DOUBLE_PTR}
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
