

# Operator precedence: * / stronger than + - stronger than == != < <= > >=
from typing import Union, Dict

from llvmlite import ir

from sleepy.grammar import SemanticError

SLOPPY_OP_TYPES = {'*', '/', '+', '-', '==', '!=', '<', '>', '<=', '>', '>='}


class AbstractSyntaxTree:
  """
  Abstract syntax tree of a sleepy program.
  """
  pass


class ExpressionAst(AbstractSyntaxTree):
  """
  Expr.
  """
  def __init__(self):
    super().__init__()

  def build_expr_ir(self, module, builder, symbol_table):
    """
    :param ir.Module module:
    :param ir.IRBuilder builder:
    :param dict[str, ir.FunctionType|ir.values.Value] symbol_table:
    """
    raise NotImplementedError()


class TopLevelExpressionAst(ExpressionAst):
  """
  TopLevelExpr.
  """
  def __init__(self, expr_list):
    """
    :param list[ExpressionAst] expr_list:
    """
    super().__init__()
    self.expr_list = expr_list

  def build_expr_ir(self, module, builder, symbol_table):
    """
    :param ir.Module module:
    :param ir.IRBuilder builder:
    :param dict[str, ir.FunctionType|ir.values.Value] symbol_table:
    """
    for expr in self.expr_list:
      expr.build_expr_ir(module=module, builder=builder, symbol_table=symbol_table)

  def make_module_ir(self, module_name):
    """
    :param str module_name:
    :rtype: ir.Module
    """
    module = ir.Module(name=module_name)
    io_func_type = ir.FunctionType(ir.VoidType(), ())
    ir_io_func = ir.Function(module, io_func_type, name='io')
    symbol_table = {}

    block = ir_io_func.append_basic_block(name='entry')
    body_builder = ir.IRBuilder(block)
    for expr in self.expr_list:
      expr.build_expr_ir(module=module, builder=body_builder, symbol_table=symbol_table)
    body_builder.ret_void()

    return module


class FunctionDeclarationAst(ExpressionAst):
  """
  Expr -> func identifier ( IdentifierList ) { ExprList }
  """
  def __init__(self, identifier, arg_identifiers, expr_list):
    """
    :param str identifier:
    :param list[str] arg_identifiers:
    :param list[ExpressionAst] expr_list:
    """
    super().__init__()
    self.identifier = identifier
    self.arg_identifiers = arg_identifiers
    self.expr_list = expr_list

  def build_expr_ir(self, module, builder, symbol_table):
    """
    :param ir.Module module:
    :param ir.IRBuilder builder:
    :param dict[str, ir.FunctionType|ir.values.Value] symbol_table:
    """
    if self.identifier in symbol_table:
      raise SemanticError('%r: cannot redefine function with name %r' % (self, self.identifier))
    double = ir.DoubleType()
    func_type = ir.FunctionType(double, (double,) * len(self.arg_identifiers))
    ir_func = ir.Function(module, func_type, name=self.identifier)
    symbol_table[self.identifier] = ir_func
    body_symbol_table = symbol_table.copy()
    for arg_identifier, ir_arg in zip(self.arg_identifiers, ir_func.args):
      ir_arg.name = arg_identifier
      body_symbol_table[arg_identifier] = ir_arg

    block = ir_func.append_basic_block(name='entry')
    body_builder = ir.IRBuilder(block)
    for expr in self.expr_list:
      expr.build_expr_ir(module=module, builder=body_builder, symbol_table=body_symbol_table)
    if not block.is_terminated:
      body_builder.ret(ir.Constant(double, 0.0))


class CallExpressionAst(ExpressionAst):
  """
  Expr -> identifier ( ValList )
  """
  def __init__(self, func_identifier, func_arg_vals):
    """
    :param str func_identifier:
    :param list[ValueAst] func_arg_vals:
    """
    super().__init__()
    self.func_identifier = func_identifier
    self.func_arg_vals = func_arg_vals


class ReturnExpressionAst(ExpressionAst):
  """
  Expr -> return val ;
  """
  def __init__(self, return_val):
    """
    :param ValueAst return_val:
    """
    super().__init__()
    self.return_val = return_val

  def build_expr_ir(self, module, builder, symbol_table):
    """
    :param ir.Module module:
    :param ir.IRBuilder builder:
    :param dict[str, ir.FunctionType|ir.values.Value] symbol_table:
    """
    builder.ret(self.return_val.make_ir_value(builder=builder, symbol_table=symbol_table))


class AssignExpressionAst(ExpressionAst):
  """
  Expr -> identifier = Val ;
  """
  def __init__(self, var_identifier, var_val):
    """
    :param str var_identifier:
    :param ValueAst var_val:
    """
    super().__init__()
    self.var_identifier = var_identifier
    self.var_val = var_val

  def build_expr_ir(self, module, builder, symbol_table):
    """
    :param ir.Module module:
    :param ir.IRBuilder builder:
    :param dict[str, ir.FunctionType|ir.values.Value] symbol_table:
    """
    if self.var_identifier in symbol_table and not isinstance(symbol_table[self.var_identifier], ir.values.Value):
      raise SemanticError('%r: Cannot redefine non-value %r using a value' % (self, self.var_identifier))
    symbol_table[self.var_identifier] = self.var_val.make_ir_value(builder=builder, symbol_table=symbol_table)


class IfExpressionAst(ExpressionAst):
  """
  Expr -> if Val { ExprList }
        | if Val { ExprList } else { ExprList }
  """
  def __init__(self, condition_val, true_expr_list, false_expr_list):
    """
    :param ValueAst condition_val:
    :param list[ExpressionAst] true_expr_list:
    :param list[ExpressionAst] false_expr_list:
    """
    super().__init__()
    self.condition_val = condition_val
    self.true_expr_list, self.false_expr_list = true_expr_list, false_expr_list


class ValueAst(AbstractSyntaxTree):
  """
  Val, SumVal, ProdVal, PrimaryVal
  """
  def __init__(self):
    super().__init__()

  def make_ir_value(self, builder, symbol_table):
    """
    :param ir.IRBuilder builder:
    :param dict[str, ir.Function] symbol_table:
    :rtype: ir.values.Value
    """
    raise NotImplementedError()


class OperatorValueAst(ValueAst):
  """
  Val, SumVal, ProdVal.
  """
  def __init__(self, op, left_val, right_val):
    """
    :param str op:
    :param ValueAst left_val:
    :param ValueAst right_val:
    """
    super().__init__()
    assert op in SLOPPY_OP_TYPES
    self.op = op
    self.left_val, self.right_val = left_val, right_val

  def make_ir_value(self, builder, symbol_table):
    """
    :param ir.IRBuilder builder:
    :param dict[str, ir.Function] symbol_table:
    :rtype: ir.values.Value
    """
    left_ir = self.left_val.make_ir_value(builder=builder, symbol_table=symbol_table)
    right_ir = self.right_val.make_ir_value(builder=builder, symbol_table=symbol_table)
    if self.op == '*':
      return builder.fmul(left_ir, right_ir, name='mul_tmp')
    if self.op == '/':
      return builder.fdiv(left_ir, right_ir, name='div_tmp')
    if self.op == '+':
      return builder.fadd(left_ir, right_ir, name='add_tmp')
    if self.op == '-':
      return builder.fsub(left_ir, right_ir, name='sub_tmp')
    assert False, '%r: operator %s not handled!' % (self, self.op)


class ConstantValueAst(ValueAst):
  """
  PrimaryVal -> number
  """
  def __init__(self, constant):
    """
    :param float constant:
    """
    super().__init__()
    self.constant = constant

  def make_ir_value(self, builder, symbol_table):
    """
    :param ir.IRBuilder builder:
    :param dict[str, ir.Function] symbol_table:
    :rtype: ir.values.Value
    """
    return ir.Constant(ir.DoubleType(), self.constant)


class VariableValueAst(ValueAst):
  """
  PrimaryVal -> identifier
  """
  def __init__(self, var_identifier):
    """
    :param str var_identifier:
    """
    super().__init__()
    self.var_identifier = var_identifier

  def make_ir_value(self, builder, symbol_table):
    """
    :param ir.IRBuilder builder:
    :param dict[str, ir.Function] symbol_table:
    :rtype: ir.values.Value
    """
    if self.var_identifier not in symbol_table:
      raise SemanticError('%r: Variable %r referenced before declaring' % (self, self.var_identifier))
    return symbol_table[self.var_identifier]


class CallValueAst(ValueAst):
  """
  PrimaryVal -> identifier ( ValList )
  """
  def __init__(self, func_identifier, func_arg_vals):
    """
    :param str func_identifier:
    :param list[ValueAst] func_arg_vals:
    """
    super().__init__()
    self.func_identifier = func_identifier
    self.func_arg_vals = func_arg_vals

  def make_ir_value(self, builder, symbol_table):
    """
    :param ir.IRBuilder builder:
    :param dict[str, ir.Function] symbol_table:
    :rtype: ir.values.Value
    """
    if self.func_identifier not in symbol_table:
      raise SemanticError('Function name %r referenced before declaration' % self.func_identifier)
    func = symbol_table[self.func_identifier]
    assert isinstance(func, ir.Function)
    if not len(func.args) == len(self.func_arg_vals):
      raise SemanticError('Function %r called with %r arguments %r, but expected %r arguments %r' % (
        self.func_identifier, len(self.func_arg_vals), self.func_arg_vals, len(func.args), func.args))
    func_args_ir = [val.make_ir_value(builder=builder, symbol_table=symbol_table) for val in self.func_arg_vals]
    return builder.call(func, func_args_ir, name='call')
