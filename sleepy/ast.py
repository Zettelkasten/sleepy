

class SloppyTree:
  """
  Abstract syntax tree of a sloppy program.
  """
  pass


class TopLevelExpressionAst(SloppyTree):
  """
  TopLevelExpr.
  """
  def __init__(self, expr_list):
    """
    :param list[ExpressionAst] expr_list:
    """
    super().__init__()
    self.expr_list = expr_list


class ExpressionAst(SloppyTree):
  """
  Expr.
  """
  def __init__(self):
    super().__init__()


class FunctionDeclarationAst(ExpressionAst):
  """
  Expr -> func identifier ( IdentifierList ) { ExprList }
  """
  def __init__(self, identifier, arg_identifiers, expr_list):
    """
    :param str identifier:
    :param str arg_identifiers:
    :param list[ExpressionAst] expr_list:
    """
    super().__init__()
    self.identifier = identifier
    self.arg_identifiers = arg_identifiers
    self.expr_list = expr_list


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


class ValueAst(SloppyTree):
  """
  Val, SumVal, ProdVal, PrimaryVal
  """
  def __init__(self):
    super().__init__()


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