"""
Implements a symbol table.
"""
from typing import Dict

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


class DoubleType(Type):
  """
  A double.
  """


SLEEPY_DOUBLE = DoubleType()


class VariableSymbol(Symbol):
  """
  A declared variable.
  """
  def __init__(self, ir_alloca, var_type):
    """
    :param ir.instructions.AllocaInstr ir_alloca:
    :param Type var_type:
    """
    super().__init__()
    assert isinstance(ir_alloca, ir.instructions.AllocaInstr)
    self.ir_alloca = ir_alloca
    self.var_type = var_type


class FunctionSymbol(Symbol):
  """
  A declared (static) function (not a function pointer).
  """
  def __init__(self, ir_func):
    """
    :param ir.Function ir_func:
    """
    super().__init__()
    assert isinstance(ir_func, ir.Function)
    self.ir_func = ir_func
