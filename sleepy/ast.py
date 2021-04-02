

# Operator precedence: * / stronger than + - stronger than == != < <= > >=
from typing import Union, Dict

from llvmlite import ir

from sleepy.grammar import SemanticError, Grammar, Production, AttributeGrammar
from sleepy.lexer import LexerGenerator
from sleepy.parser import ParserGenerator

SLOPPY_OP_TYPES = {'*', '/', '+', '-', '==', '!=', '<', '>', '<=', '>', '>='}


def make_func_call_ir(func_identifier, func_arg_vals, builder, symbol_table):
  """
  :param str func_identifier:
  :param list[ExpressionAst] func_arg_vals:
  :param IRBuilder builder:
  :param dict[str, ir.Function|ir.AllocaInstr] symbol_table:
  :rtype: ir.values.Value
  """
  if func_identifier not in symbol_table:
    raise SemanticError('Function name %r referenced before declaration' % func_identifier)
  func = symbol_table[func_identifier]
  assert isinstance(func, ir.Function)
  if not len(func.args) == len(func_arg_vals):
    raise SemanticError('Function %r called with %r arguments %r, but expected %r arguments %r' % (
      func_identifier, len(func_arg_vals), func_arg_vals, len(func.args), func.args))
  func_args_ir = [val.make_ir_value(builder=builder, symbol_table=symbol_table) for val in func_arg_vals]
  return builder.call(func, func_args_ir, name='call')


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
    :param dict[str, ir.Function|ir.AllocaInstr] symbol_table:
    :rtype: ir.IRBuilder
    """
    raise NotImplementedError()

  def get_declared_identifiers(self):
    """
    :rtype: list[str]
    """
    raise NotImplementedError()


class TopLevelStatementAst(StatementAst):
  """
  TopLevelExpr.
  """
  def __init__(self, expr_list):
    """
    :param list[StatementAst] expr_list:
    """
    super().__init__()
    self.expr_list = expr_list

  def build_expr_ir(self, module, builder, symbol_table):
    """
    :param ir.Module module:
    :param ir.IRBuilder builder:
    :param dict[str, ir.Function|ir.AllocaInstr] symbol_table:
    :rtype: ir.IRBuilder
    """
    for expr in self.expr_list:
      builder = expr.build_expr_ir(module=module, builder=builder, symbol_table=symbol_table)

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
      body_builder = expr.build_expr_ir(module=module, builder=body_builder, symbol_table=symbol_table)
    body_builder.ret_void()

    return module

  def get_declared_identifiers(self):
    """
    :rtype: list[str]
    """
    return []


class FunctionDeclarationAst(StatementAst):
  """
  Stmt -> func identifier ( IdentifierList ) { StmtList }
  """
  def __init__(self, identifier, arg_identifiers, expr_list):
    """
    :param str identifier:
    :param list[str] arg_identifiers:
    :param list[StatementAst] expr_list:
    """
    super().__init__()
    self.identifier = identifier
    self.arg_identifiers = arg_identifiers
    self.expr_list = expr_list

  def build_expr_ir(self, module, builder, symbol_table):
    """
    :param ir.Module module:
    :param ir.IRBuilder builder:
    :param dict[str, ir.Function|ir.AllocaInstr] symbol_table:
    :rtype: ir.IRBuilder
    """
    if self.identifier in symbol_table:
      raise SemanticError('%r: cannot redefine function with name %r' % (self, self.identifier))
    double = ir.DoubleType()
    func_type = ir.FunctionType(double, (double,) * len(self.arg_identifiers))
    ir_func = ir.Function(module, func_type, name=self.identifier)
    symbol_table[self.identifier] = ir_func
    body_symbol_table = symbol_table.copy()
    block = ir_func.append_basic_block(name='entry')
    body_builder = ir.IRBuilder(block)

    for local_identifier in self.get_body_declared_identifiers():
      ir_alloca = body_builder.alloca(double, name=local_identifier)
      body_symbol_table[local_identifier] = ir_alloca

    for arg_identifier, ir_arg in zip(self.arg_identifiers, ir_func.args):
      ir_alloca = body_symbol_table[arg_identifier]
      ir_arg.name = arg_identifier
      body_builder.store(ir_arg, ir_alloca)

    for expr in self.expr_list:
      body_builder = expr.build_expr_ir(module=module, builder=body_builder, symbol_table=body_symbol_table)
    if body_builder is not None and not body_builder.block.is_terminated:
      body_builder.ret(ir.Constant(double, 0.0))
    return builder

  def get_declared_identifiers(self):
    """
    :rtype: list[str]
    """
    return []

  def get_body_declared_identifiers(self):
    """
    :rtype: list[str]
    """
    local_identifiers = [identifier for expr in self.expr_list for identifier in expr.get_declared_identifiers()]
    return self.arg_identifiers + [
      identifier for identifier in local_identifiers if identifier not in self.arg_identifiers]


class ExternFunctionDeclarationAst(StatementAst):
  """
  Stmt -> extern_func identifier ( IdentifierList ) ;
  """
  def __init__(self, identifier, arg_identifiers):
    """
    :param str identifier:
    :param list[str] arg_identifiers:
    """
    super().__init__()
    self.identifier = identifier
    self.arg_identifiers = arg_identifiers

  def build_expr_ir(self, module, builder, symbol_table):
    """
    :param ir.Module module:
    :param ir.IRBuilder builder:
    :param dict[str, ir.Function|ir.AllocaInstr] symbol_table:
    :rtype: ir.IRBuilder
    """
    if self.identifier in symbol_table:
      raise SemanticError('%r: cannot redefine extern function with name %r' % (self, self.identifier))
    double = ir.DoubleType()
    func_type = ir.FunctionType(double, (double,) * len(self.arg_identifiers))
    ir_func = ir.Function(module, func_type, name=self.identifier)
    symbol_table[self.identifier] = ir_func
    for arg_identifier, ir_arg in zip(self.arg_identifiers, ir_func.args):
      ir_arg.name = arg_identifier
    return builder

  def get_declared_identifiers(self):
    """
    :rtype: list[str]
    """
    return []


class CallStatementAst(StatementAst):
  """
  Stmt -> identifier ( ExprList )
  """
  def __init__(self, func_identifier, func_arg_vals):
    """
    :param str func_identifier:
    :param list[ExpressionAst] func_arg_vals:
    """
    super().__init__()
    self.func_identifier = func_identifier
    self.func_arg_vals = func_arg_vals

  def build_expr_ir(self, module, builder, symbol_table):
    """
    :param ir.Module module:
    :param ir.IRBuilder builder:
    :param dict[str, ir.Function|ir.AllocaInstr] symbol_table:
    :rtype: ir.IRBuilder
    """
    make_func_call_ir(
      func_identifier=self.func_identifier, func_arg_vals=self.func_arg_vals, builder=builder,
      symbol_table=symbol_table)
    return builder

  def get_declared_identifiers(self):
    """
    :rtype: list[str]
    """
    return []


class ReturnStatementAst(StatementAst):
  """
  Stmt -> return ExprList ;
  """
  def __init__(self, return_expr_list):
    """
    :param list[ExpressionAst] return_expr_list:
    """
    super().__init__()
    self.return_expr_list = return_expr_list
    assert len(return_expr_list) <= 1, 'returning of multiple values is not support yet'

  def build_expr_ir(self, module, builder, symbol_table):
    """
    :param ir.Module module:
    :param ir.IRBuilder builder:
    :param dict[str, ir.Function|ir.AllocaInstr] symbol_table:
    :rtype: ir.IRBuilder
    """
    if len(self.return_expr_list) == 1:
      builder.ret(self.return_expr_list[0].make_ir_value(builder=builder, symbol_table=symbol_table))
    else:
      assert len(self.return_expr_list) == 0
      builder.ret(ir.Constant(ir.DoubleType(), 0.0))
    return builder

  def get_declared_identifiers(self):
    """
    :rtype: list[str]
    """
    return []


class AssignStatementAst(StatementAst):
  """
  Stmt -> identifier = Expr ;
  """
  def __init__(self, var_identifier, var_val):
    """
    :param str var_identifier:
    :param ExpressionAst var_val:
    """
    super().__init__()
    self.var_identifier = var_identifier
    self.var_val = var_val

  def build_expr_ir(self, module, builder, symbol_table):
    """
    :param ir.Module module:
    :param ir.IRBuilder builder:
    :param dict[str, ir.Function|ir.AllocaInstr] symbol_table:
    :rtype: ir.IRBuilder
    """
    assert self.var_identifier in symbol_table
    ir_alloca = symbol_table[self.var_identifier]
    if not isinstance(ir_alloca, ir.AllocaInstr):
      raise SemanticError('%r: Cannot redefine non-value %r using a value' % (self, self.var_identifier))
    symbol_table[self.var_identifier] = ir_alloca
    ir_value = self.var_val.make_ir_value(builder=builder, symbol_table=symbol_table)
    assert isinstance(ir_value, ir.values.Value)
    builder.store(ir_value, ir_alloca)
    return builder

  def get_declared_identifiers(self):
    """
    :rtype: list[str]
    """
    return [self.var_identifier]


class IfStatementAst(StatementAst):
  """
  Stmt -> if Expr { StmtList }
        | if Expr { StmtList } else { StmtList }
  """
  def __init__(self, condition_val, true_expr_list, false_expr_list):
    """
    :param ExpressionAst condition_val:
    :param list[StatementAst] true_expr_list:
    :param list[StatementAst] false_expr_list:
    """
    super().__init__()
    self.condition_val = condition_val
    self.true_expr_list, self.false_expr_list = true_expr_list, false_expr_list

  @property
  def has_true_branch(self):
    """
    :rtype: bool
    """
    return len(self.true_expr_list) >= 1

  @property
  def has_false_branch(self):
    """
    :rtype: bool
    """
    return len(self.false_expr_list) >= 1

  def build_expr_ir(self, module, builder, symbol_table):
    """
    :param ir.Module module:
    :param ir.IRBuilder builder:
    :param dict[str, ir.Function|ir.AllocaInstr] symbol_table:
    :rtype: ir.IRBuilder|None
    """
    cond_ir = self.condition_val.make_ir_value(builder=builder, symbol_table=symbol_table)
    constant_one = ir.Constant(ir.DoubleType(), 0.0)
    cond_ir = builder.fcmp_ordered('!=', cond_ir, constant_one, 'ifcond')

    true_block = builder.append_basic_block('true_branch')  # type: ir.Block
    false_block = builder.append_basic_block('false_branch')  # type: ir.Block
    builder.cbranch(cond_ir, true_block, false_block)
    true_builder, false_builder = ir.IRBuilder(true_block), ir.IRBuilder(false_block)

    true_symbol_table, false_symbol_table = symbol_table.copy(), symbol_table.copy()

    for expr in self.true_expr_list:
      expr.build_expr_ir(module, builder=true_builder, symbol_table=true_symbol_table)
    for expr in self.false_expr_list:
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

  def get_declared_identifiers(self):
    """
    :rtype: list[str]
    """
    true_declared = [identifier for expr in self.true_expr_list for identifier in expr.get_declared_identifiers()]
    false_declared = [identifier for expr in self.true_expr_list for identifier in expr.get_declared_identifiers()]
    return true_declared + [identifier for identifier in false_declared if identifier not in true_declared]


class WhileStatementAst(StatementAst):
  """
  Stmt -> while Expr { StmtList }
  """
  def __init__(self, condition_val, expr_list):
    """
    :param ExpressionAst condition_val:
    :param list[StatementAst] expr_list:
    """
    super().__init__()
    self.condition_val = condition_val
    self.expr_list = expr_list

  def build_expr_ir(self, module, builder, symbol_table):
    """
    :param ir.Module module:
    :param ir.IRBuilder builder:
    :param dict[str, ir.Function|ir.AllocaInstr] symbol_table:
    :rtype: ir.IRBuilder|None
    """
    def make_condition_ir(builder_, symbol_table_):
      """
      :param ir.IRBuilder builder_:
      :param dict[str, ir.Function|ir.AllocaInstr] symbol_table:
      :rtype: FCMPInstr
      """
      cond_ir = self.condition_val.make_ir_value(builder=builder_, symbol_table=symbol_table_)
      return builder_.fcmp_ordered('!=', cond_ir, ir.Constant(ir.DoubleType(), 0.0), 'whilecond')

    cond_ir = make_condition_ir(builder_=builder, symbol_table_=symbol_table)
    body_block = builder.append_basic_block('while_body')  # type: ir.Block
    continue_block = builder.append_basic_block('continue_branch')  # type: ir.Block
    builder.cbranch(cond_ir, body_block, continue_block)
    body_builder = ir.IRBuilder(body_block)

    body_symbol_table = symbol_table.copy()

    for expr in self.expr_list:
      body_builder = expr.build_expr_ir(module, builder=body_builder, symbol_table=body_symbol_table)
    if not body_block.is_terminated:
      assert body_builder is not None
      body_cond_ir = make_condition_ir(builder_=body_builder, symbol_table_=body_symbol_table)
      body_builder.cbranch(body_cond_ir, body_block, continue_block)

    continue_builder = ir.IRBuilder(continue_block)
    return continue_builder

  def get_declared_identifiers(self):
    """
    :rtype: list[str]
    """
    return [identifier for expr in self.expr_list for identifier in expr.get_declared_identifiers()]


class ExpressionAst(AbstractSyntaxTree):
  """
  Val, SumVal, ProdVal, PrimaryExpr
  """
  def __init__(self):
    super().__init__()

  def make_ir_value(self, builder, symbol_table):
    """
    :param ir.IRBuilder builder:
    :param dict[str, ir.Function|ir.AllocaInstr] symbol_table:
    :rtype: ir.values.Value
    """
    raise NotImplementedError()


class OperatorValueAst(ExpressionAst):
  """
  Val, SumVal, ProdVal.
  """
  def __init__(self, op, left_val, right_val):
    """
    :param str op:
    :param ExpressionAst left_val:
    :param ExpressionAst right_val:
    """
    super().__init__()
    assert op in SLOPPY_OP_TYPES
    self.op = op
    self.left_val, self.right_val = left_val, right_val

  def make_ir_value(self, builder, symbol_table):
    """
    :param ir.IRBuilder builder:
    :param dict[str, ir.Function|ir.AllocaInstr] symbol_table:
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
    if self.op in {'==', '!=', '<', '>', '<=', '>', '>='}:
      ir_bool = builder.fcmp_ordered(self.op, left_ir, right_ir, name='cmp_tmp')
      return builder.uitofp(ir_bool, ir.DoubleType())
    assert False, '%r: operator %s not handled!' % (self, self.op)


class UnaryOperatorValueAst(ExpressionAst):
  """
  NegVal.
  """
  def __init__(self, op, val):
    """
    :param str op:
    :param ExpressionAst val:
    """
    super().__init__()
    assert op in {'+', '-'}
    self.op = op
    self.val = val

  def make_ir_value(self, builder, symbol_table):
    """
    :param ir.IRBuilder builder:
    :param dict[str, ir.Function|ir.AllocaInstr] symbol_table:
    :rtype: ir.values.Value
    """
    val_ir = self.val.make_ir_value(builder=builder, symbol_table=symbol_table)
    if self.op == '+':
      return val_ir
    if self.op == '-':
      constant_minus_one = ir.Constant(ir.DoubleType(), -1.0)
      return builder.fmul(constant_minus_one, val_ir, name='neg_tmp')
    assert False, '%r: operator %s not handled!' % (self, self.op)


class ConstantValueAst(ExpressionAst):
  """
  PrimaryExpr -> number
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
    :param dict[str, ir.Function|ir.AllocaInstr] symbol_table:
    :rtype: ir.values.Value
    """
    return ir.Constant(ir.DoubleType(), self.constant)


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
    :param dict[str, ir.Function|ir.AllocaInstr] symbol_table:
    :rtype: ir.values.Value
    """
    if self.var_identifier not in symbol_table:
      raise SemanticError('%r: Variable %r referenced before declaring' % (self, self.var_identifier))
    return builder.load(symbol_table[self.var_identifier], self.var_identifier)


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
    :param dict[str, ir.Function|ir.AllocaInstr] symbol_table:
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
    'number', 'char',
    None, None
  ], [
    'func', 'extern_func', 'if', 'else', 'return', 'while', '{', '}', ';', ',', '\\(', '\\)',
    '==|!=|<=?|>=?', '\\+|\\-', '\\*|/', '=', '([A-Z]|[a-z]|_)([A-Z]|[a-z]|[0-9]|_)*',
    '(0|[1-9][0-9]*)(\\.[0-9]+)?', "'([^\']|\\\\[nrt'\"])'",
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
  Production('PrimaryExpr', 'number'),
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
  Production('ExprList+', 'Expr', ',', 'ExprList+')
)
SLEEPY_ATTR_GRAMMAR = AttributeGrammar(
  SLEEPY_GRAMMAR,
  syn_attrs={'ast', 'expr_list', 'identifier_list', 'val_list', 'identifier', 'op', 'number'},
  prod_attr_rules=[
    {'ast': lambda expr_list: TopLevelStatementAst(expr_list(1))},
    {'expr_list': []},
    {'expr_list': lambda ast, expr_list: [ast(1)] + expr_list(2)},
    {'ast': lambda identifier, identifier_list, expr_list: (
      FunctionDeclarationAst(identifier(2), identifier_list(4), expr_list(7)))},
    {'ast': lambda identifier, identifier_list: (
      ExternFunctionDeclarationAst(identifier(2), identifier_list(4)))},
    {'ast': lambda identifier, val_list: CallStatementAst(identifier(1), val_list(3))},
    {'ast': lambda val_list: ReturnStatementAst(val_list(2))},
    {'ast': lambda identifier, ast: AssignStatementAst(identifier(1), ast(3))},
    {'ast': lambda ast, expr_list: IfStatementAst(ast(2), expr_list(4), [])},
    {'ast': lambda ast, expr_list: IfStatementAst(ast(2), expr_list(4), expr_list(8))},
    {'ast': lambda ast, expr_list: WhileStatementAst(ast(2), expr_list(4))}] + [
    {'ast': lambda ast, op: OperatorValueAst(op(2), ast(1), ast(3))},
    {'ast': 'ast.1'}] * 3 + [
    {'ast': lambda ast, op: UnaryOperatorValueAst(op(1), ast(2))},
    {'ast': 'ast.1'},
    {'ast': lambda number: ConstantValueAst(number(1))},
    {'ast': lambda number: ConstantValueAst(number(1))},
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
    {'val_list': lambda ast, val_list: [ast(1)] + val_list(3)}
  ],
  terminal_attr_rules={
    'bool_op': {'op': lambda value: value},
    'sum_op': {'op': lambda value: value},
    'prod_op': {'op': lambda value: value},
    'identifier': {'identifier': lambda value: value},
    'number': {'number': lambda value: float(value)},
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
  func and(a, b) { if a { return b; } else { return 0; } }
  func not(a) { if (a) { return 0; } else { return 1; } }
  """
  return make_program_ast(preamble_program, add_preamble=False)


def add_preamble_to_ast(program_ast):
  """
  :param TopLevelStatementAst program_ast:
  :rtype: TopLevelStatementAst
  """
  preamble_ast = make_preamble_ast()
  assert isinstance(preamble_ast, TopLevelStatementAst)
  return TopLevelStatementAst(preamble_ast.expr_list + program_ast.expr_list)
