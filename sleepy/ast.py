

# Operator precedence: * / stronger than + - stronger than == != < <= > >=
from typing import Dict

from llvmlite import ir

from sleepy.grammar import SemanticError, Grammar, Production, AttributeGrammar
from sleepy.lexer import LexerGenerator
from sleepy.parser import ParserGenerator
from sleepy.symbols import FunctionSymbol, Symbol, VariableSymbol, SLEEPY_DOUBLE, Type, join_declared_var_types, \
  SLEEPY_TYPES, SLEEPY_INT, SLEEPY_CHAR

SLOPPY_OP_TYPES = {'*', '/', '+', '-', '==', '!=', '<', '>', '<=', '>', '>='}


def make_func_call_ir(func_identifier, func_arg_vals, builder, symbol_table):
  """
  :param str func_identifier:
  :param list[ExpressionAst] func_arg_vals:
  :param IRBuilder builder:
  :param dict[str,Symbol] symbol_table:
  :rtype: ir.values.Value
  """
  if func_identifier not in symbol_table:
    raise SemanticError('Function name %r referenced before declaration' % func_identifier)
  func_symbol = symbol_table[func_identifier]
  if not isinstance(func_symbol, FunctionSymbol):
    raise SemanticError('Referenced name %r is not a function, but a %r' % (func_identifier, type(func_symbol)))
  ir_func = func_symbol.ir_func
  if not len(ir_func.args) == len(func_arg_vals):
    raise SemanticError('Function %r called with %r arguments %r, but expected %r arguments %r' % (
      func_identifier, len(func_arg_vals), func_arg_vals, len(ir_func.args), ir_func.args))
  func_args_ir = [val.make_ir_value(builder=builder, symbol_table=symbol_table) for val in func_arg_vals]
  return builder.call(ir_func, func_args_ir, name='call')


class AbstractSyntaxTree:
  """
  Abstract syntax tree of a sleepy program.
  """
  pass


class StatementAst(AbstractSyntaxTree):
  """
  Expr.
  """
  def __init__(self):
    super().__init__()

  def build_expr_ir(self, module, builder, symbol_table):
    """
    :param ir.Module module:
    :param ir.IRBuilder builder:
    :param dict[str,Symbol] symbol_table:
    :rtype: ir.IRBuilder
    """
    raise NotImplementedError()

  def get_declared_var_types(self, symbol_table):
    """
    :param dict[str,Symbol] symbol_table:
    :rtype: dict[str,Type|None]
    """
    raise NotImplementedError()

  def make_type(self, type_identifier, symbol_table):
    """
    :param str|None type_identifier:
    :param dict[str,Symbol] symbol_table:
    :rtype: Type|None
    """
    if type_identifier is None:
      return None
    if type_identifier in SLEEPY_TYPES:
      return SLEEPY_TYPES[type_identifier]
    raise SemanticError('%r: Unknown type identifier %r. Available: %r' % (
      self, type_identifier, ', '.join('%r' % type_identifier for type_identifier in SLEEPY_TYPES.keys())))


class TopLevelStatementAst(StatementAst):
  """
  TopLevelExpr.
  """
  def __init__(self, stmt_list):
    """
    :param list[StatementAst] stmt_list:
    """
    super().__init__()
    self.stmt_list = stmt_list

  def build_expr_ir(self, module, builder, symbol_table):
    """
    :param ir.Module module:
    :param ir.IRBuilder builder:
    :param dict[str,Symbol] symbol_table:
    :rtype: ir.IRBuilder
    """
    for expr in self.stmt_list:
      builder = expr.build_expr_ir(module=module, builder=builder, symbol_table=symbol_table)

  def make_module_ir(self, module_name):
    """
    :param str module_name:
    :rtype: ir.Module
    """
    module = ir.Module(name=module_name)
    io_func_type = ir.FunctionType(ir.VoidType(), ())
    ir_io_func = ir.Function(module, io_func_type, name='io')
    symbol_table = {}  # type: Dict[str, Symbol]

    block = ir_io_func.append_basic_block(name='entry')
    body_builder = ir.IRBuilder(block)
    for stmt in self.stmt_list:
      body_builder = stmt.build_expr_ir(module=module, builder=body_builder, symbol_table=symbol_table)
    assert not block.is_terminated
    body_builder.ret_void()

    return module

  def get_declared_var_types(self, symbol_table):
    """
    :param dict[str, Symbol] symbol_table:
    :rtype: dict[str,Type|None]
    """
    return {}


class FunctionDeclarationAst(StatementAst):
  """
  Stmt -> func identifier ( TypedIdentifierList ) { StmtList }
  """
  def __init__(self, identifier, arg_identifiers, arg_type_identifiers, return_type_identifier, stmt_list):
    """
    :param str identifier:
    :param list[str|None] arg_identifiers:
    :param list[str|None] arg_type_identifiers:
    :param str|None return_type_identifier:
    :param list[StatementAst]|None stmt_list: body, or None if extern function.
    """
    super().__init__()
    assert len(arg_identifiers) == len(arg_type_identifiers)
    self.identifier = identifier
    self.arg_identifiers = arg_identifiers
    self.arg_type_identifiers = arg_type_identifiers
    self.return_type_identifier = return_type_identifier
    self.stmt_list = stmt_list

  @property
  def is_extern(self):
    """
    :rtype: bool
    """
    return self.stmt_list is None

  def make_arg_types(self, symbol_table):
    """
    :param dict[str,Symbol] symbol_table:
    :rtype: list[Type]
    """
    arg_types = [self.make_type(identifier, symbol_table=symbol_table) for identifier in self.arg_type_identifiers]
    if any(arg_type is None for arg_type in arg_types):
      raise SemanticError('%r: need to specify all parameter types of function %r' % (self, self.identifier))
    return arg_types

  def build_expr_ir(self, module, builder, symbol_table):
    """
    :param ir.Module module:
    :param ir.IRBuilder builder:
    :param dict[str,Symbol] symbol_table:
    :rtype: ir.IRBuilder
    """
    if self.identifier in symbol_table:
      raise SemanticError('%r: cannot redefine function with name %r' % (self, self.identifier))
    arg_types = self.make_arg_types(symbol_table=symbol_table)
    return_type = self.make_type(self.return_type_identifier, symbol_table=symbol_table)
    if return_type is None:
      raise SemanticError('%r: need to specify return type of function %r' % (self, self.identifier))
    ir_func_type = ir.FunctionType(
      return_type.ir_type, [arg_type.ir_type for arg_type in arg_types])
    ir_func = ir.Function(module, ir_func_type, name=self.identifier)
    symbol_table[self.identifier] = FunctionSymbol(ir_func)

    if not self.is_extern:
      body_symbol_table = symbol_table.copy()  # type: Dict[str, Symbol]
      block = ir_func.append_basic_block(name='entry')
      body_builder = ir.IRBuilder(block)

      for identifier_name, identifier_type in self.get_body_var_types(
          symbol_table=symbol_table, body_symbol_table=body_symbol_table).items():
        ir_alloca = body_builder.alloca(identifier_type.ir_type, name=identifier_name)
        body_symbol_table[identifier_name] = VariableSymbol(ir_alloca, identifier_type)

      for arg_identifier, ir_arg in zip(self.arg_identifiers, ir_func.args):
        arg_symbol = body_symbol_table[arg_identifier]
        assert isinstance(arg_symbol, VariableSymbol)
        ir_arg.name = arg_identifier
        body_builder.store(ir_arg, arg_symbol.ir_alloca)

      for stmt in self.stmt_list:
        body_builder = stmt.build_expr_ir(module=module, builder=body_builder, symbol_table=body_symbol_table)
      if body_builder is not None and not body_builder.block.is_terminated:
        ReturnStatementAst([]).build_expr_ir(
          module=module, builder=body_builder, symbol_table=body_symbol_table)
    return builder

  def get_declared_var_types(self, symbol_table):
    """
    :param dict[str, Symbol] symbol_table:
    :rtype: dict[str,Type|None]
    """
    return {}

  def get_body_var_types(self, symbol_table, body_symbol_table):
    """
    :param dict[str,Symbol] symbol_table:
    :param dict[str,Symbol] body_symbol_table:
    :rtype: dict[str,Type]
    """
    declared_arg_var_types = dict(zip(self.arg_identifiers, self.make_arg_types(symbol_table=symbol_table)))
    if self.is_extern:
      return join_declared_var_types([declared_arg_var_types], [self])
    else:
      return join_declared_var_types(
        [declared_arg_var_types]
        + [stmt.get_declared_var_types(symbol_table=body_symbol_table) for stmt in self.stmt_list],
        [self] + self.stmt_list)


class CallStatementAst(StatementAst):
  """
  Stmt -> identifier ( ExprList )
  """
  def __init__(self, func_identifier, func_arg_exprs):
    """
    :param str func_identifier:
    :param list[ExpressionAst] func_arg_exprs:
    """
    super().__init__()
    self.func_identifier = func_identifier
    self.func_arg_exprs = func_arg_exprs

  def build_expr_ir(self, module, builder, symbol_table):
    """
    :param ir.Module module:
    :param ir.IRBuilder builder:
    :param dict[str,Symbol] symbol_table:
    :rtype: ir.IRBuilder
    """
    make_func_call_ir(
      func_identifier=self.func_identifier, func_arg_vals=self.func_arg_exprs, builder=builder,
      symbol_table=symbol_table)
    return builder

  def get_declared_var_types(self, symbol_table):
    """
    :param dict[str, Symbol] symbol_table:
    :rtype: dict[str,Type|None]
    """
    return {}


class ReturnStatementAst(StatementAst):
  """
  Stmt -> return ExprList ;
  """
  def __init__(self, return_exprs):
    """
    :param list[ExpressionAst] return_exprs:
    """
    super().__init__()
    self.return_exprs = return_exprs
    assert len(return_exprs) <= 1, 'returning of multiple values is not support yet'

  def build_expr_ir(self, module, builder, symbol_table):
    """
    :param ir.Module module:
    :param ir.IRBuilder builder:
    :param dict[str,Symbol] symbol_table:
    :rtype: ir.IRBuilder
    """
    if len(self.return_exprs) == 1:
      builder.ret(self.return_exprs[0].make_ir_value(builder=builder, symbol_table=symbol_table))
    else:
      assert len(self.return_exprs) == 0
      builder.ret(ir.Constant(ir.DoubleType(), 0.0))
    return builder

  def get_declared_var_types(self, symbol_table):
    """
    :param dict[str, Symbol] symbol_table:
    :rtype: dict[str,Type|None]
    """
    return {}


class AssignStatementAst(StatementAst):
  """
  Stmt -> identifier = Expr ;
  """
  def __init__(self, var_identifier, var_val, var_type_identifier):
    """
    :param str var_identifier:
    :param ExpressionAst var_val:
    :param str|None var_type_identifier:
    """
    super().__init__()
    self.var_identifier = var_identifier
    self.var_val = var_val
    self.var_type_identifier = var_type_identifier

  def make_var_type(self, symbol_table):
    """
    :param dict[str, Symbol] symbol_table:
    :rtype: Type
    """
    if self.var_identifier in symbol_table:
      existing_symbol = symbol_table[self.var_identifier]
      assert isinstance(existing_symbol, VariableSymbol)
      return existing_symbol.var_type
    if self.var_type_identifier is None:
      return SLEEPY_DOUBLE  # assume double for now
    return self.make_type(self.var_type_identifier, symbol_table=symbol_table)

  def build_expr_ir(self, module, builder, symbol_table):
    """
    :param ir.Module module:
    :param ir.IRBuilder builder:
    :param dict[str,Symbol] symbol_table:
    :rtype: ir.IRBuilder
    """
    assert self.var_identifier in symbol_table
    symbol = symbol_table[self.var_identifier]
    if not isinstance(symbol, VariableSymbol):
      raise SemanticError('%r: Cannot assign non-variable %r to a variable' % (self, self.var_identifier))
    var_type = self.make_var_type(symbol_table=symbol_table)
    if self.var_type_identifier is not None and symbol.var_type != var_type:
      raise SemanticError('%r: Cannot redefine variable %r of type %r with different type %r' % (
        self, self.var_identifier, symbol.var_type, var_type))
    ir_value = self.var_val.make_ir_value(builder=builder, symbol_table=symbol_table)
    assert isinstance(ir_value, ir.values.Value)
    # TODO: Check that the type actually matches the type of ir_value.
    builder.store(ir_value, symbol.ir_alloca)
    return builder

  def get_declared_var_types(self, symbol_table):
    """
    :param dict[str, Symbol] symbol_table:
    :rtype: dict[str,Type|None]
    """
    return {self.var_identifier: self.make_var_type(symbol_table=symbol_table)}


class IfStatementAst(StatementAst):
  """
  Stmt -> if Expr { StmtList }
        | if Expr { StmtList } else { StmtList }
  """
  def __init__(self, condition_val, true_stmt_list, false_stmt_list):
    """
    :param ExpressionAst condition_val:
    :param list[StatementAst] true_stmt_list:
    :param list[StatementAst] false_stmt_list:
    """
    super().__init__()
    self.condition_val = condition_val
    self.true_stmt_list, self.false_stmt_list = true_stmt_list, false_stmt_list

  @property
  def has_true_branch(self):
    """
    :rtype: bool
    """
    return len(self.true_stmt_list) >= 1

  @property
  def has_false_branch(self):
    """
    :rtype: bool
    """
    return len(self.false_stmt_list) >= 1

  def build_expr_ir(self, module, builder, symbol_table):
    """
    :param ir.Module module:
    :param ir.IRBuilder builder:
    :param dict[str,Symbol] symbol_table:
    :rtype: ir.IRBuilder|None
    """
    cond_ir = self.condition_val.make_ir_value(builder=builder, symbol_table=symbol_table)
    constant_one = ir.Constant(ir.DoubleType(), 0.0)
    cond_ir = builder.fcmp_ordered('!=', cond_ir, constant_one, 'ifcond')

    true_block = builder.append_basic_block('true_branch')  # type: ir.Block
    false_block = builder.append_basic_block('false_branch')  # type: ir.Block
    builder.cbranch(cond_ir, true_block, false_block)
    true_builder, false_builder = ir.IRBuilder(true_block), ir.IRBuilder(false_block)

    true_symbol_table = symbol_table.copy()  # type: Dict[str, Symbol]
    false_symbol_table = symbol_table.copy()  # type: Dict[str, Symbol]

    for expr in self.true_stmt_list:
      expr.build_expr_ir(module, builder=true_builder, symbol_table=true_symbol_table)
    for expr in self.false_stmt_list:
      expr.build_expr_ir(module, builder=false_builder, symbol_table=false_symbol_table)

    if not true_block.is_terminated or not false_block.is_terminated:
      continue_block = builder.append_basic_block('continue_branch')  # type: ir.Block
      continue_builder = ir.IRBuilder(continue_block)
      if not true_block.is_terminated:
        true_builder.branch(continue_block)
      if not false_block.is_terminated:
        false_builder.branch(continue_block)
      return continue_builder
    else:
      return None

  def get_declared_var_types(self, symbol_table):
    """
    :param dict[str, Symbol] symbol_table:
    :rtype: dict[str,Type|None]
    """
    return join_declared_var_types(
      [stmt.get_declared_var_types(symbol_table=symbol_table) for stmt in self.true_stmt_list + self.false_stmt_list],
      self.true_stmt_list + self.false_stmt_list)


class WhileStatementAst(StatementAst):
  """
  Stmt -> while Expr { StmtList }
  """
  def __init__(self, condition_val, stmt_list):
    """
    :param ExpressionAst condition_val:
    :param list[StatementAst] stmt_list:
    """
    super().__init__()
    self.condition_val = condition_val
    self.stmt_list = stmt_list

  def build_expr_ir(self, module, builder, symbol_table):
    """
    :param ir.Module module:
    :param ir.IRBuilder builder:
    :param dict[str,Symbol] symbol_table:
    :rtype: ir.IRBuilder|None
    """
    def make_condition_ir(builder_, symbol_table_):
      """
      :param ir.IRBuilder builder_:
      :param dict[str,Symbol] symbol_table_:
      :rtype: FCMPInstr
      """
      cond_ir = self.condition_val.make_ir_value(builder=builder_, symbol_table=symbol_table_)
      return builder_.fcmp_ordered('!=', cond_ir, ir.Constant(ir.DoubleType(), 0.0), 'whilecond')

    cond_ir = make_condition_ir(builder_=builder, symbol_table_=symbol_table)
    body_block = builder.append_basic_block('while_body')  # type: ir.Block
    continue_block = builder.append_basic_block('continue_branch')  # type: ir.Block
    builder.cbranch(cond_ir, body_block, continue_block)
    body_builder = ir.IRBuilder(body_block)

    body_symbol_table = symbol_table.copy()  # type: Dict[str, Symbol]

    for stmt in self.stmt_list:
      body_builder = stmt.build_expr_ir(module, builder=body_builder, symbol_table=body_symbol_table)
    if not body_block.is_terminated:
      assert body_builder is not None
      body_cond_ir = make_condition_ir(builder_=body_builder, symbol_table_=body_symbol_table)
      body_builder.cbranch(body_cond_ir, body_block, continue_block)

    continue_builder = ir.IRBuilder(continue_block)
    return continue_builder

  def get_declared_var_types(self, symbol_table):
    """
    :param dict[str, Symbol] symbol_table:
    :rtype: dict[str,Type|None]
    """
    return join_declared_var_types(
      [stmt.get_declared_var_types(symbol_table=symbol_table) for stmt in self.stmt_list], self.stmt_list)


class ExpressionAst(AbstractSyntaxTree):
  """
  Val, SumVal, ProdVal, PrimaryExpr
  """
  def __init__(self):
    super().__init__()

  def make_ir_value(self, builder, symbol_table):
    """
    :param ir.IRBuilder builder:
    :param dict[str,Symbol] symbol_table:
    :rtype: ir.values.Value
    """
    raise NotImplementedError()


class OperatorValueAst(ExpressionAst):
  """
  Val, SumVal, ProdVal.
  """
  def __init__(self, op, left_expr, right_expr):
    """
    :param str op:
    :param ExpressionAst left_expr:
    :param ExpressionAst right_expr:
    """
    super().__init__()
    assert op in SLOPPY_OP_TYPES
    self.op = op
    self.left_expr, self.right_expr = left_expr, right_expr

  def make_ir_value(self, builder, symbol_table):
    """
    :param ir.IRBuilder builder:
    :param dict[str,Symbol] symbol_table:
    :rtype: ir.values.Value
    """
    left_ir = self.left_expr.make_ir_value(builder=builder, symbol_table=symbol_table)
    right_ir = self.right_expr.make_ir_value(builder=builder, symbol_table=symbol_table)
    if self.op == '*':
      return builder.fmul(left_ir, right_ir, name='mul_tmp')
    if self.op == '/':
      return builder.fdiv(left_ir, right_ir, name='div_tmp')
    if self.op == '+':
      return builder.fadd(left_ir, right_ir, name='add_tmp')
    if self.op == '-':
      return builder.fsub(left_ir, right_ir, name='sub_tmp')
    if self.op in {'==', '!=', '<', '>', '<=', '>', '>='}:
      ir_bool = builder.fcmp_ordered(self.op, left_ir, right_ir, name='cmp_tmp')
      return builder.uitofp(ir_bool, ir.DoubleType())
    assert False, '%r: operator %s not handled!' % (self, self.op)


class UnaryOperatorValueAst(ExpressionAst):
  """
  NegVal.
  """
  def __init__(self, op, expr):
    """
    :param str op:
    :param ExpressionAst expr:
    """
    super().__init__()
    assert op in {'+', '-'}
    self.op = op
    self.expr = expr

  def make_ir_value(self, builder, symbol_table):
    """
    :param ir.IRBuilder builder:
    :param dict[str,Symbol] symbol_table:
    :rtype: ir.values.Value
    """
    expr_ir = self.expr.make_ir_value(builder=builder, symbol_table=symbol_table)
    if self.op == '+':
      return expr_ir
    if self.op == '-':
      constant_minus_one = ir.Constant(ir.DoubleType(), -1.0)
      return builder.fmul(constant_minus_one, expr_ir, name='neg_tmp')
    assert False, '%r: operator %s not handled!' % (self, self.op)


class ConstantValueAst(ExpressionAst):
  """
  PrimaryExpr -> double | int | char
  """
  def __init__(self, constant_val, constant_type):
    """
    :param Any constant_val:
    :param Type constant_type:
    """
    super().__init__()
    self.constant_val = constant_val
    self.constant_type = constant_type

  def make_ir_value(self, builder, symbol_table):
    """
    :param ir.IRBuilder builder:
    :param dict[str,Symbol] symbol_table:
    :rtype: ir.values.Value
    """
    return ir.Constant(self.constant_type.ir_type, self.constant_val)


class VariableValueAst(ExpressionAst):
  """
  PrimaryExpr -> identifier
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
    :param dict[str,Symbol] symbol_table:
    :rtype: ir.values.Value
    """
    if self.var_identifier not in symbol_table:
      raise SemanticError('%r: Variable %r referenced before declaring' % (self, self.var_identifier))
    symbol = symbol_table[self.var_identifier]
    if not isinstance(symbol, VariableSymbol):
      raise SemanticError('%r: Cannot reference a non-variable %r' % (self, self.var_identifier))
    return builder.load(symbol.ir_alloca, self.var_identifier)


class CallValueAst(ExpressionAst):
  """
  PrimaryExpr -> identifier ( ExprList )
  """
  def __init__(self, func_identifier, func_arg_vals):
    """
    :param str func_identifier:
    :param list[ExpressionAst] func_arg_vals:
    """
    super().__init__()
    self.func_identifier = func_identifier
    self.func_arg_vals = func_arg_vals

  def make_ir_value(self, builder, symbol_table):
    """
    :param ir.IRBuilder builder:
    :param dict[str,Symbol] symbol_table:
    :rtype: ir.values.Value
    """
    return make_func_call_ir(
      func_identifier=self.func_identifier, func_arg_vals=self.func_arg_vals, builder=builder,
      symbol_table=symbol_table)


def parse_char(value):
  """
  :param str value: e.g. 'a', '\n', ...
  :rtype: str
  """
  assert 3 <= len(value) <= 4
  assert value[0] == value[-1] == "'"
  value = value[1:-1]
  if len(value) == 1:
    return value
  assert value[0] == '\\'
  return {'n': '\n', 'r': '\r', 't': '\t', "'": "'", '"': '"'}[value[1]]


SLEEPY_LEXER = LexerGenerator(
  [
    'func', 'extern_func', 'if', 'else', 'return', 'while', '{', '}', ';', ',', '(', ')',
    'bool_op', 'sum_op', 'prod_op', '=', 'identifier',
    'int', 'double', 'char',
    None, None
  ], [
    'func', 'extern_func', 'if', 'else', 'return', 'while', '{', '}', ';', ',', '\\(', '\\)',
    '==|!=|<=?|>=?', '\\+|\\-', '\\*|/', '=', '([A-Z]|[a-z]|_)([A-Z]|[a-z]|[0-9]|_)*',
    '(0|[1-9][0-9]*)', '(0|[1-9][0-9]*)\\.[0-9]+', "'([^\']|\\\\[nrt'\"])'",
    '#[^\n]*\n', '[ \n]+'
  ])
SLEEPY_GRAMMAR = Grammar(
  Production('TopLevelStmt', 'StmtList'),
  Production('StmtList'),
  Production('StmtList', 'Stmt', 'StmtList'),
  Production('Stmt', 'func', 'identifier', '(', 'IdentifierList', ')', '{', 'StmtList', '}'),
  Production('Stmt', 'extern_func', 'identifier', '(', 'IdentifierList', ')', ';'),
  Production('Stmt', 'identifier', '(', 'ExprList', ')', ';'),
  Production('Stmt', 'return', 'ExprList', ';'),
  Production('Stmt', 'Type', 'identifier', '=', 'Expr', ';'),
  Production('Stmt', 'identifier', '=', 'Expr', ';'),
  Production('Stmt', 'if', 'Expr', '{', 'StmtList', '}'),
  Production('Stmt', 'if', 'Expr', '{', 'StmtList', '}', 'else', '{', 'StmtList', '}'),
  Production('Stmt', 'while', 'Expr', '{', 'StmtList', '}'),
  Production('Expr', 'Expr', 'bool_op', 'SumExpr'),
  Production('Expr', 'SumExpr'),
  Production('SumExpr', 'SumExpr', 'sum_op', 'ProdExpr'),
  Production('SumExpr', 'ProdExpr'),
  Production('ProdExpr', 'ProdExpr', 'prod_op', 'NegExpr'),
  Production('ProdExpr', 'NegExpr'),
  Production('NegExpr', 'sum_op', 'PrimaryExpr'),
  Production('NegExpr', 'PrimaryExpr'),
  Production('PrimaryExpr', 'int'),
  Production('PrimaryExpr', 'double'),
  Production('PrimaryExpr', 'char'),
  Production('PrimaryExpr', 'identifier'),
  Production('PrimaryExpr', 'identifier', '(', 'ExprList', ')'),
  Production('PrimaryExpr', '(', 'Expr', ')'),
  Production('IdentifierList'),
  Production('IdentifierList', 'IdentifierList+'),
  Production('IdentifierList+', 'identifier'),
  Production('IdentifierList+', 'identifier', ',', 'IdentifierList+'),
  Production('ExprList'),
  Production('ExprList', 'ExprList+'),
  Production('ExprList+', 'Expr'),
  Production('ExprList+', 'Expr', ',', 'ExprList+'),
  Production('Type', 'identifier')
)
SLEEPY_ATTR_GRAMMAR = AttributeGrammar(
  SLEEPY_GRAMMAR,
  syn_attrs={'ast', 'stmt_list', 'identifier_list', 'val_list', 'identifier', 'type_identifier', 'op', 'number'},
  prod_attr_rules=[
    {'ast': lambda stmt_list: TopLevelStatementAst(stmt_list(1))},
    {'stmt_list': []},
    {'stmt_list': lambda ast, stmt_list: [ast(1)] + stmt_list(2)},
    {'ast': lambda identifier, identifier_list, stmt_list: (
      FunctionDeclarationAst(identifier(2), identifier_list(4), ['Double'] * len(identifier_list(4)), 'Double', stmt_list(7)))},  # noqa
    {'ast': lambda identifier, identifier_list: (
      FunctionDeclarationAst(identifier(2), identifier_list(4), ['Double'] * len(identifier_list(4)), 'Double', None))},
    {'ast': lambda identifier, val_list: CallStatementAst(identifier(1), val_list(3))},
    {'ast': lambda val_list: ReturnStatementAst(val_list(2))},
    {'ast': lambda identifier, ast, type_identifier: AssignStatementAst(identifier(2), ast(4), type_identifier(1))},
    {'ast': lambda identifier, ast: AssignStatementAst(identifier(1), ast(3), None)},
    {'ast': lambda ast, stmt_list: IfStatementAst(ast(2), stmt_list(4), [])},
    {'ast': lambda ast, stmt_list: IfStatementAst(ast(2), stmt_list(4), stmt_list(8))},
    {'ast': lambda ast, stmt_list: WhileStatementAst(ast(2), stmt_list(4))}] + [
    {'ast': lambda ast, op: OperatorValueAst(op(2), ast(1), ast(3))},
    {'ast': 'ast.1'}] * 3 + [
    {'ast': lambda ast, op: UnaryOperatorValueAst(op(1), ast(2))},
    {'ast': 'ast.1'},
    {'ast': lambda number: ConstantValueAst(number(1), SLEEPY_INT)},
    {'ast': lambda number: ConstantValueAst(number(1), SLEEPY_DOUBLE)},
    {'ast': lambda number: ConstantValueAst(number(1), SLEEPY_CHAR)},
    {'ast': lambda identifier: VariableValueAst(identifier(1))},
    {'ast': lambda identifier, val_list: CallValueAst(identifier(1), val_list(3))},
    {'ast': 'ast.2'},
    {'identifier_list': []},
    {'identifier_list': 'identifier_list.1'},
    {'identifier_list': lambda identifier: [identifier(1)]},
    {'identifier_list': lambda identifier, identifier_list: [identifier(1)] + identifier_list(3)},
    {'val_list': []},
    {'val_list': 'val_list.1'},
    {'val_list': lambda ast: [ast(1)]},
    {'val_list': lambda ast, val_list: [ast(1)] + val_list(3)},
    {'type_identifier': 'identifier.1'}
  ],
  terminal_attr_rules={
    'bool_op': {'op': lambda value: value},
    'sum_op': {'op': lambda value: value},
    'prod_op': {'op': lambda value: value},
    'identifier': {'identifier': lambda value: value},
    'int': {'number': lambda value: int(value)},
    'double': {'number': lambda value: float(value)},
    'char': {'number': lambda value: float(ord(parse_char(value)))}
  }
)
SLEEPY_PARSER = ParserGenerator(SLEEPY_GRAMMAR)


def make_program_ast(program, add_preamble=True):
  """
  :param str program:
  :param bool add_preamble:
  :rtype: TopLevelStatementAst
  """
  tokens, tokens_pos = SLEEPY_LEXER.tokenize(program)
  _, root_eval = SLEEPY_PARSER.parse_syn_attr_analysis(SLEEPY_ATTR_GRAMMAR, program, tokens, tokens_pos)
  program_ast = root_eval['ast']
  assert isinstance(program_ast, TopLevelStatementAst)
  if add_preamble:
    program_ast = add_preamble_to_ast(program_ast)
  return program_ast


def make_preamble_ast():
  """
  :rtype: TopLevelStatementAst
  """
  std_func_identifiers = [
    ('print_char', 1), ('print_double', 1), ('allocate', 1), ('deallocate', 1), ('load', 1), ('store', 2),
    ('assert', 1)]
  preamble_program = ''.join([
    'extern_func %s(%s);\n' % (identifier, ', '.join(['var%s' % num for num in range(num_args)]))
    for identifier, num_args in std_func_identifiers]) + """
  func or(a, b) { if a { return a; } else { return b; } }
  func and(a, b) { if a { return b; } else { return 0.0; } }
  func not(a) { if (a) { return 0.0; } else { return 1.0; } }
  """
  return make_program_ast(preamble_program, add_preamble=False)


def add_preamble_to_ast(program_ast):
  """
  :param TopLevelStatementAst program_ast:
  :rtype: TopLevelStatementAst
  """
  preamble_ast = make_preamble_ast()
  assert isinstance(preamble_ast, TopLevelStatementAst)
  return TopLevelStatementAst(preamble_ast.stmt_list + program_ast.stmt_list)
