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


class DoubleType(Type):
  """
  A double.
  """
  def __init__(self):
    super().__init__(ir.DoubleType())


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


def join_declared_var_types(arg_identifier_list, ast_list):
  """
  Joins declared variables .

  :param list[dict[str,Type|None]] arg_identifier_list: via ``AbstractSyntaxTree.get_declared_var_types``
  :param list[AbstractSyntaxTree] ast_list: for error messages
  :rtype: dict[str,Type]
  :raise: SemanticError
  """
  print(arg_identifier_list, ast_list)
  assert len(arg_identifier_list) == len(ast_list)
  declared_types = {}  # type: Dict[str, Type]
  for arg_identifiers, ast in zip(arg_identifier_list, ast_list):
    for var_identifier, var_type in arg_identifiers.items():
      if var_identifier not in declared_types:
        if var_type is None:
          var_type = SLEEPY_DOUBLE  # assume everything is a double for now.
        declared_types[var_identifier] = var_type
      else:
        if var_type is not None and declared_types[var_identifier] != var_type:
          raise SemanticError('%r: Cannot redefine variable %r of type %r with new type %r' % (
            ast, var_identifier, declared_types[var_identifier], var_type))
  return declared_types
