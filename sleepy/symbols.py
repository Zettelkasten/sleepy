"""
Implements a symbol table.
"""
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
  def __init__(self, ir_type):
    """
    :param ir_type:
    """
    self.ir_type = ir_type

  def __repr__(self):
    return self.__class__.__name__


class DoubleType(Type):
  """
  A double.
  """
  def __init__(self):
    super().__init__(ir.DoubleType())


class IntType(Type):
  """
  An 32-bit integer.
  """
  def __init__(self):
    super().__init__(ir.IntType(bits=32))


class LongType(Type):
  """
  A 64-bit integer.
  """
  def __init__(self):
    super().__init__(ir.IntType(bits=64))


class CharType(Type):
  """
  An 32-bit character.
  """
  def __init__(self):
    super().__init__(ir.IntType(bits=32))


SLEEPY_DOUBLE = DoubleType()
SLEEPY_INT = IntType()
SLEEPY_LONG = LongType()
SLEEPY_CHAR = CharType()

SLEEPY_TYPES = {'Double': SLEEPY_DOUBLE, 'Int': SLEEPY_INT, 'Long': SLEEPY_LONG, 'Char': SLEEPY_CHAR}
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
    :param Type return_type:
    """
    super().__init__()
    assert ir_func is None or isinstance(ir_func, ir.Function)
    assert len(arg_identifiers) == len(arg_types)
    self.ir_func = ir_func
    self.arg_identifiers = arg_identifiers
    self.arg_types = arg_types
    self.return_type = return_type
