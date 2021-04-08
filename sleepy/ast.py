

# Operator precedence: * / stronger than + - stronger than == != < <= > >=
from typing import Dict, List

from llvmlite import ir

from sleepy.grammar import SemanticError, Grammar, Production, AttributeGrammar, TreePosition
from sleepy.lexer import LexerGenerator
from sleepy.parser import ParserGenerator
from sleepy.symbols import FunctionSymbol, VariableSymbol, SLEEPY_DOUBLE, Type, SLEEPY_INT, \
  SLEEPY_LONG, SLEEPY_VOID, SLEEPY_DOUBLE_PTR, SLEEPY_BOOL, SLEEPY_CHAR, SymbolTable, TypeSymbol, \
  make_initial_symbol_table, StructType

SLOPPY_OP_TYPES = {'*', '/', '+', '-', '==', '!=', '<', '>', '<=', '>', '>='}


class AbstractSyntaxTree:
  """
  Abstract syntax tree of a sleepy program.
  """
  def __init__(self, pos):
    """
    :param TreePosition pos: position where this AST starts
    """
    self.pos = pos

  def __repr__(self):
    """
    :rtype: str
    """
    return 'AbstractSyntaxTree'

  def raise_error(self, message):
    """
    :param str message:
    """
    raise SemanticError(self.pos.word, self.pos.from_pos, self.pos.to_pos, message)

  def _check_func_call_symbol_table(self, func_identifier, func_arg_exprs, symbol_table):
    """
    :param str func_identifier:
    :param list[ExpressionAst func_arg_exprs:
    :param SymbolTable symbol_table:
    """
    if func_identifier not in symbol_table:
      self.raise_error('Function %r called before declared' % func_identifier)
    symbol = symbol_table[func_identifier]
    if not isinstance(symbol, FunctionSymbol):
      self.raise_error('Cannot call non-function %r' % func_identifier)
    if len(func_arg_exprs) != len(symbol.arg_identifiers):
      self.raise_error('Cannot call function %r with %r arguments, expected %r arguments %r' % (
        func_identifier, len(func_arg_exprs), len(symbol.arg_identifiers), symbol.arg_identifiers))
    called_types = [arg_expr.make_val_type(symbol_table=symbol_table) for arg_expr in func_arg_exprs]
    for arg_identifier, called_type, declared_type in zip(symbol.arg_identifiers, called_types, symbol.arg_types):
      if called_type != declared_type:
        self.raise_error('Cannot call function %r with parameter %r of type %r, expected %r' % (
          func_identifier, arg_identifier, called_type, declared_type))

  def _make_func_call_ir(self, func_identifier, func_arg_exprs, builder, symbol_table):
    """
    :param str func_identifier:
    :param list[ExpressionAst] func_arg_exprs:
    :param IRBuilder builder:
    :param SymbolTable symbol_table:
    :rtype: ir.values.Value
    """
    if func_identifier not in symbol_table:
      self.raise_error('Function name %r referenced before declaration' % func_identifier)
    func_symbol = symbol_table[func_identifier]
    if not isinstance(func_symbol, FunctionSymbol):
      self.raise_error('Referenced name %r is not a function, but a %r' % (func_identifier, type(func_symbol)))
    ir_func = func_symbol.ir_func
    if not len(ir_func.args) == len(func_arg_exprs):
      self.raise_error('Function %r called with %r arguments %r, but expected %r arguments %r' % (
        func_identifier, len(func_arg_exprs), func_arg_exprs, len(ir_func.args), ir_func.args))
    ir_func_args = [val.make_ir_val(builder=builder, symbol_table=symbol_table) for val in func_arg_exprs]
    return builder.call(ir_func, ir_func_args, name='call')


class StatementAst(AbstractSyntaxTree):
  """
  Expr.
  """
  def __init__(self, pos):
    """
    :param TreePosition pos:
    """
    super().__init__(pos)

  def build_symbol_table(self, symbol_table):
    """
    :param SymbolTable symbol_table:
    """
    raise NotImplementedError()

  def build_expr_ir(self, module, builder, symbol_table):
    """
    :param ir.Module module:
    :param ir.IRBuilder builder:
    :param SymbolTable symbol_table:
    :rtype: ir.IRBuilder
    """
    raise NotImplementedError()

  def make_type(self, type_identifier, symbol_table):
    """
    :param str type_identifier:
    :param SymbolTable symbol_table:
    :rtype: Type
    """
    if type_identifier not in symbol_table:
      self.raise_error('Unknown type identifier %r' % type_identifier)
    type_symbol = symbol_table[type_identifier]
    if not isinstance(type_symbol, TypeSymbol):
      self.raise_error('%r is not a type, but a %r' % (type_identifier, type(type_symbol)))
    return type_symbol.type

  def __repr__(self):
    """
    :rtype: str
    """
    return 'StatementAst'


class TopLevelStatementAst(StatementAst):
  """
  TopLevelExpr.
  """
  def __init__(self, pos, stmt_list):
    """
    :param TreePosition pos:
    :param list[StatementAst] stmt_list:
    """
    super().__init__(pos)
    self.stmt_list = stmt_list

  def build_symbol_table(self, symbol_table):
    """
    :param SymbolTable symbol_table:
    """
    pass

  def build_expr_ir(self, module, builder, symbol_table):
    """
    :param ir.Module module:
    :param ir.IRBuilder builder:
    :param SymbolTable symbol_table:
    :rtype: ir.IRBuilder
    """
    for expr in self.stmt_list:
      builder = expr.build_expr_ir(module=module, builder=builder, symbol_table=symbol_table)

  def make_module_ir_and_symbol_table(self, module_name):
    """
    :param str module_name:
    :rtype: (ir.Module,SymbolTable)
    """
    module = ir.Module(name=module_name)
    io_func_type = ir.FunctionType(ir.VoidType(), ())
    ir_io_func = ir.Function(module, io_func_type, name='io')
    symbol_table = make_initial_symbol_table()
    for stmt in self.stmt_list:
      stmt.build_symbol_table(symbol_table=symbol_table)
    for scope_symbol_identifier in symbol_table.current_scope_identifiers:
      assert scope_symbol_identifier in symbol_table
      scope_symbol = symbol_table[scope_symbol_identifier]
      if isinstance(scope_symbol, VariableSymbol):
        self.raise_error('Defining top-level variables is not supported')

    block = ir_io_func.append_basic_block(name='entry')
    body_builder = ir.IRBuilder(block)
    for stmt in self.stmt_list:
      body_builder = stmt.build_expr_ir(module=module, builder=body_builder, symbol_table=symbol_table)
    assert not block.is_terminated
    body_builder.ret_void()

    return module, symbol_table

  def __repr__(self):
    """
    :rtype: str
    """
    return 'TopLevelStatementAst(%s)' % ', '.join([repr(stmt) for stmt in self.stmt_list])


class FunctionDeclarationAst(StatementAst):
  """
  Stmt -> func identifier ( TypedIdentifierList ) { StmtList }
  """
  def __init__(self, pos, identifier, arg_identifiers, arg_type_identifiers, return_type_identifier, stmt_list):
    """
    :param TreePosition pos:
    :param str identifier:
    :param list[str|None] arg_identifiers:
    :param list[str|None] arg_type_identifiers:
    :param str|None return_type_identifier:
    :param list[StatementAst]|None stmt_list: body, or None if extern function.
    """
    super().__init__(pos)
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
    :param SymbolTable symbol_table:
    :rtype: list[Type]
    """
    arg_types = [self.make_type(identifier, symbol_table=symbol_table) for identifier in self.arg_type_identifiers]
    if any(arg_type is None for arg_type in arg_types):
      self.raise_error('need to specify all parameter types of function %r' % self.identifier)
    return arg_types

  def build_symbol_table(self, symbol_table):
    """
    :param SymbolTable symbol_table:
    """
    if self.identifier in symbol_table:
      self.raise_error('Cannot redefine function with name %r' % self.identifier)
    arg_types = self.make_arg_types(symbol_table=symbol_table)
    if self.return_type_identifier is None:
      return_type = SLEEPY_VOID
    else:
      return_type = self.make_type(self.return_type_identifier, symbol_table=symbol_table)
    if return_type is None:
      self.raise_error('Need to specify return type of function %r' % self.identifier)
    symbol_table[self.identifier] = FunctionSymbol(
      None, arg_identifiers=self.arg_identifiers, arg_types=arg_types, return_type=return_type)

  def build_expr_ir(self, module, builder, symbol_table):
    """
    :param ir.Module module:
    :param ir.IRBuilder builder:
    :param SymbolTable symbol_table:
    :rtype: ir.IRBuilder
    """
    assert self.identifier in symbol_table
    symbol = symbol_table[self.identifier]
    assert isinstance(symbol, FunctionSymbol)
    ir_func_type = symbol.make_ir_function_type()
    symbol.ir_func = ir.Function(module, ir_func_type, name=self.identifier)

    if not self.is_extern:
      block = symbol.ir_func.append_basic_block(name='entry')
      body_builder = ir.IRBuilder(block)

      body_symbol_table = symbol_table.copy()  # type: SymbolTable
      body_symbol_table.current_func = symbol
      body_symbol_table.current_scope_identifiers = []
      for arg_identifier, arg_type in zip(self.arg_identifiers, self.make_arg_types(symbol_table=symbol_table)):
        body_symbol_table[arg_identifier] = VariableSymbol(None, arg_type)
        body_symbol_table.current_scope_identifiers.append(arg_identifier)
      for stmt in self.stmt_list:
        stmt.build_symbol_table(symbol_table=body_symbol_table)

      for identifier_name in body_symbol_table.current_scope_identifiers:
        assert identifier_name in body_symbol_table
        var_symbol = body_symbol_table[identifier_name]
        if not isinstance(var_symbol, VariableSymbol):
          continue
        var_symbol.ir_alloca = body_builder.alloca(var_symbol.var_type.ir_type, name=identifier_name)

      for arg_identifier, ir_arg in zip(self.arg_identifiers, symbol.ir_func.args):
        arg_symbol = body_symbol_table[arg_identifier]
        assert isinstance(arg_symbol, VariableSymbol)
        ir_arg.name = arg_identifier
        body_builder.store(ir_arg, arg_symbol.ir_alloca)

      for stmt in self.stmt_list:
        body_builder = stmt.build_expr_ir(module=module, builder=body_builder, symbol_table=body_symbol_table)
      if body_builder is not None and not body_builder.block.is_terminated:
        return_pos = TreePosition(self.pos.word, self.pos.to_pos, self.pos.to_pos)
        ReturnStatementAst(return_pos, []).build_expr_ir(
          module=module, builder=body_builder, symbol_table=body_symbol_table)
    return builder

  def __repr__(self):
    """
    :rtype: str
    """
    return (
        'FunctionDeclarationAst(identifier=%r, arg_identifiers=%r, arg_type_identifiers=%r, '
        'return_type_identifier=%r, %s)' % (self.identifier, self.arg_identifiers, self.arg_type_identifiers,
    self.return_type_identifier, 'extern' if self.is_extern else ', '.join([repr(stmt) for stmt in self.stmt_list])))


class CallStatementAst(StatementAst):
  """
  Stmt -> identifier ( ExprList )
  """
  def __init__(self, pos, func_identifier, func_arg_exprs):
    """
    :param TreePosition pos:
    :param str func_identifier:
    :param list[ExpressionAst] func_arg_exprs:
    """
    super().__init__(pos)
    self.func_identifier = func_identifier
    self.func_arg_exprs = func_arg_exprs

  def build_symbol_table(self, symbol_table):
    """
    :param SymbolTable symbol_table:
    """
    # just verify that the argument types are correctly specified, but do not alter symbol_table
    self._check_func_call_symbol_table(
      func_identifier=self.func_identifier, func_arg_exprs=self.func_arg_exprs, symbol_table=symbol_table)

  def build_expr_ir(self, module, builder, symbol_table):
    """
    :param ir.Module module:
    :param ir.IRBuilder builder:
    :param SymbolTable symbol_table:
    :rtype: ir.IRBuilder
    """
    self._make_func_call_ir(
      func_identifier=self.func_identifier, func_arg_exprs=self.func_arg_exprs, builder=builder,
      symbol_table=symbol_table)
    return builder

  def __repr__(self):
    """
    :rtype: str
    """
    return 'CallStatementAst(func_identifier=%r, func_arg_exprs=%r)' % (self.func_identifier, self.func_arg_exprs)


class ReturnStatementAst(StatementAst):
  """
  Stmt -> return ExprList ;
  """
  def __init__(self, pos, return_exprs):
    """
    :param TreePosition pos:
    :param list[ExpressionAst] return_exprs:
    """
    super().__init__(pos)
    self.return_exprs = return_exprs
    assert len(return_exprs) <= 1, 'returning of multiple values is not support yet'

  def build_symbol_table(self, symbol_table):
    """
    :param SymbolTable symbol_table:
    """
    # Check that return typ matches.
    if symbol_table.current_func is None:
      self.raise_error('Can only use return inside a function declaration')
    if len(self.return_exprs) == 1:
      return_val_type = self.return_exprs[0].make_val_type(symbol_table=symbol_table)
      if return_val_type != symbol_table.current_func.return_type:
        if symbol_table.current_func.return_type == SLEEPY_VOID:
          self.raise_error('Function declared to return void, but return value is of type %r' % (
            return_val_type))
        else:
          self.raise_error('Function declared to return type %r, but return value is of type %r' % (
            return_val_type, symbol_table.current_func.return_type))
    else:
      assert len(self.return_exprs) == 0
      if symbol_table.current_func.return_type != SLEEPY_VOID:
        self.raise_error('Function declared to return a value of type %r, but returned void' % (
          symbol_table.current_func.return_type))

  def build_expr_ir(self, module, builder, symbol_table):
    """
    :param ir.Module module:
    :param ir.IRBuilder builder:
    :param SymbolTable symbol_table:
    :rtype: ir.IRBuilder
    """
    if len(self.return_exprs) == 1:
      builder.ret(self.return_exprs[0].make_ir_val(builder=builder, symbol_table=symbol_table))
    else:
      assert len(self.return_exprs) == 0
      builder.ret_void()
    return builder

  def __repr__(self):
    """
    :rtype: str
    """
    return 'ReturnStatementAst(return_exprs=%r)' % self.return_exprs


class StructDeclarationAst(StatementAst):
  """
  Stmt -> struct identifier { StmtList }
  """
  def __init__(self, pos, struct_identifier, stmt_list):
    """
    :param TreePosition pos:
    :param str struct_identifier:
    :param List[StatementAst] stmt_list:
    """
    self.pos = pos
    self.struct_identifier = struct_identifier
    self.stmt_list = stmt_list

  def build_symbol_table(self, symbol_table):
    """
    :param SymbolTable symbol_table:
    """
    if self.struct_identifier in symbol_table.current_scope_identifiers:
      self.raise_error('Cannot refined struct with name %r' % self.struct_identifier)
    body_symbol_table = symbol_table.copy()
    for member_num, stmt in enumerate(self.stmt_list):
      if not isinstance(stmt, AssignStatementAst):
        stmt.raise_error('Can only use declare statements within a struct declaration')
      stmt.build_symbol_table(symbol_table=body_symbol_table)
      if len(body_symbol_table.current_scope_identifiers) != member_num + 1:
        stmt.raise_error('Cannot declare member %r multiple times in struct declaration' % stmt.var_identifier)
    assert len(body_symbol_table.current_scope_identifiers) == len(self.stmt_list)
    member_identifiers = []
    member_types = []
    for declared_identifier in body_symbol_table.current_scope_identifiers:
      assert declared_identifier in body_symbol_table
      declared_symbol = body_symbol_table[declared_identifier]
      assert isinstance(declared_symbol, VariableSymbol)
      member_identifiers.append(declared_identifier)
      member_types.append(declared_symbol.var_type)
    assert len(member_identifiers) == len(member_types) == len(self.stmt_list)

    struct_type = StructType(self.struct_identifier, member_identifiers, member_types)
    # ir_func will be set in build_expr_ir
    constructor = FunctionSymbol(ir_func=None, arg_identifiers=[], arg_types=[], return_type=struct_type)
    symbol_table[self.struct_identifier] = TypeSymbol(struct_type, constructor_symbol=constructor)
    symbol_table.current_scope_identifiers.append(self.struct_identifier)

  def _make_constructor_body_ir(self, constructor, symbol_table):
    """
    :param constructor: FunctionSymbol
    :param SymbolTable symbol_table:
    """
    # TODO: populate the member variables of the struct
    constructor_symbol_table = symbol_table.copy()
    constructor_block = constructor.ir_func.append_basic_block(name='entry')
    constructor_builder = ir.IRBuilder(constructor_block)

    self_ir_alloca = constructor_builder.alloca(constructor.return_type.ir_type, name='self')
    for member_num, stmt in enumerate(self.stmt_list):
      assert isinstance(stmt, AssignStatementAst)
      member_identifier = stmt.var_identifier
      ir_val = stmt.var_val.make_ir_val(builder=constructor_builder, symbol_table=constructor_symbol_table)
      gep_indices = (ir.Constant(ir.IntType(32), 0), ir.Constant(ir.IntType(32), member_num))
      member_ptr = constructor_builder.gep(self_ir_alloca, gep_indices, '%s_ptr' % member_identifier)
      constructor_builder.store(ir_val, member_ptr)

    constructor_builder.ret(constructor_builder.load(self_ir_alloca, 'self'))
    assert constructor_block.is_terminated

  def build_expr_ir(self, module, builder, symbol_table):
    """
    :param ir.Module module:
    :param ir.IRBuilder builder:
    :param SymbolTable symbol_table:
    :rtype: ir.IRBuilder
    """
    assert self.struct_identifier in symbol_table
    struct_symbol = symbol_table[self.struct_identifier]
    assert isinstance(struct_symbol, TypeSymbol)
    constructor = struct_symbol.constructor_symbol
    assert constructor is not None
    assert struct_symbol.type == constructor.return_type
    ir_func_type = constructor.make_ir_function_type()
    constructor.ir_func = ir.Function(module, ir_func_type, name='construct_%s' % self.struct_identifier)
    self._make_constructor_body_ir(constructor, symbol_table=symbol_table)
    return builder

  def __repr__(self):
    """
    :rtype: str
    """
    return 'StructDeclarationAst(struct_identifier=%r, stmt_list=%r)' % (self.struct_identifier, self.stmt_list)


class AssignStatementAst(StatementAst):
  """
  Stmt -> identifier = Expr ;
  """
  def __init__(self, pos, var_identifier, var_val, var_type_identifier):
    """
    :param TreePosition pos:
    :param str var_identifier:
    :param ExpressionAst var_val:
    :param str|None var_type_identifier:
    """
    super().__init__(pos)
    self.var_identifier = var_identifier
    self.var_val = var_val
    self.var_type_identifier = var_type_identifier

  def build_symbol_table(self, symbol_table):
    """
    :param SymbolTable symbol_table:
    """
    if self.var_type_identifier is not None:
      declared_type = self.make_type(self.var_type_identifier, symbol_table=symbol_table)
    else:
      declared_type = None
    val_type = self.var_val.make_val_type(symbol_table=symbol_table)
    if declared_type is not None and declared_type != val_type:
      self.raise_error('Cannot assign variable %r with declared type %r a value of type %r' % (
        self.var_identifier, declared_type, val_type))
    if self.var_identifier in symbol_table.current_scope_identifiers:
      # variable name in this scope already declared. just check that types match, but do not change symbol_table.
      assert self.var_identifier in symbol_table
      symbol = symbol_table[self.var_identifier]
      if not isinstance(symbol, VariableSymbol):
        self.raise_error('Cannot assign non-variable %r to a variable' % self.var_identifier)
      if symbol.var_type != val_type:
        self.raise_error('Cannot redefine variable %r of type %r with new type %r' % (
          self.var_identifier, symbol.var_type, val_type))
    else:
      assert self.var_identifier not in symbol_table.current_scope_identifiers
      # declare new variable, override entry in symbol_table (maybe it was defined in an outer scope before).
      symbol = VariableSymbol(None, var_type=val_type)
      symbol_table[self.var_identifier] = symbol
      symbol_table.current_scope_identifiers.append(self.var_identifier)

  def build_expr_ir(self, module, builder, symbol_table):
    """
    :param ir.Module module:
    :param ir.IRBuilder builder:
    :param SymbolTable symbol_table:
    :rtype: ir.IRBuilder
    """
    assert self.var_identifier in symbol_table
    symbol = symbol_table[self.var_identifier]
    assert isinstance(symbol, VariableSymbol)
    assert symbol.ir_alloca is not None  # ir_alloca is set in FunctionDeclarationAst
    ir_val = self.var_val.make_ir_val(builder=builder, symbol_table=symbol_table)
    builder.store(ir_val, symbol.ir_alloca)
    return builder

  def __repr__(self):
    """
    :rtype: str
    """
    return 'AssignStatementAst(var_identifier=%r, var_val=%r, var_type_identifier=%r)' % (
      self.var_identifier, self.var_val, self.var_type_identifier)


class IfStatementAst(StatementAst):
  """
  Stmt -> if Expr { StmtList }
        | if Expr { StmtList } else { StmtList }
  """
  def __init__(self, pos, condition_val, true_stmt_list, false_stmt_list):
    """
    :param TreePosition pos:
    :param ExpressionAst condition_val:
    :param list[StatementAst] true_stmt_list:
    :param list[StatementAst] false_stmt_list:
    """
    super().__init__(pos)
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

  def build_symbol_table(self, symbol_table):
    """
    :param SymbolTable symbol_table:
    """
    # TODO: Make this a separate scope.
    # It is probably easiest to add this by making every scope it's own Statement (essentially a Statement list),
    # and then not having true/false_stmt_list but just a single statement.
    cond_type = self.condition_val.make_val_type(symbol_table=symbol_table)
    if not cond_type == SLEEPY_BOOL:
      self.raise_error('Condition use expression of type %r as if-condition' % cond_type)
    for stmt in self.true_stmt_list + self.false_stmt_list:
      stmt.build_symbol_table(symbol_table=symbol_table)

  def build_expr_ir(self, module, builder, symbol_table):
    """
    :param ir.Module module:
    :param ir.IRBuilder builder:
    :param SymbolTable symbol_table:
    :rtype: ir.IRBuilder|None
    """
    ir_cond = self.condition_val.make_ir_val(builder=builder, symbol_table=symbol_table)
    true_block = builder.append_basic_block('true_branch')  # type: ir.Block
    false_block = builder.append_basic_block('false_branch')  # type: ir.Block
    builder.cbranch(ir_cond, true_block, false_block)
    true_builder, false_builder = ir.IRBuilder(true_block), ir.IRBuilder(false_block)

    true_symbol_table = symbol_table.copy()  # type: SymbolTable
    false_symbol_table = symbol_table.copy()  # type: SymbolTable

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

  def __repr__(self):
    """
    :rtype: str
    """
    return 'IfStatementAst(condition_val=%r, true_stmt_list=%r, false_stmt_list=%r)' % (
      self.condition_val, self.true_stmt_list, self.false_stmt_list)


class WhileStatementAst(StatementAst):
  """
  Stmt -> while Expr { StmtList }
  """
  def __init__(self, pos, condition_val, stmt_list):
    """
    :param TreePosition pos:
    :param ExpressionAst condition_val:
    :param list[StatementAst] stmt_list:
    """
    super().__init__(pos)
    self.condition_val = condition_val
    self.stmt_list = stmt_list

  def build_symbol_table(self, symbol_table):
    """
    :param SymbolTable symbol_table:
    """
    # TODO: Make this a separate scope. Also see IfExpressionAst.
    cond_type = self.condition_val.make_val_type(symbol_table=symbol_table)
    if not cond_type == SLEEPY_BOOL:
      self.raise_error('Condition use expression of type %r as while-condition' % cond_type)
    for stmt in self.stmt_list:
      stmt.build_symbol_table(symbol_table=symbol_table)

  def build_expr_ir(self, module, builder, symbol_table):
    """
    :param ir.Module module:
    :param ir.IRBuilder builder:
    :param SymbolTable symbol_table:
    :rtype: ir.IRBuilder|None
    """
    def make_condition_ir(builder_, symbol_table_):
      """
      :param ir.IRBuilder builder_:
      :param SymbolTable symbol_table_:
      :rtype: FCMPInstr
      """
      return self.condition_val.make_ir_val(builder=builder_, symbol_table=symbol_table_)

    cond_ir = make_condition_ir(builder_=builder, symbol_table_=symbol_table)
    body_block = builder.append_basic_block('while_body')  # type: ir.Block
    continue_block = builder.append_basic_block('continue_branch')  # type: ir.Block
    builder.cbranch(cond_ir, body_block, continue_block)
    body_builder = ir.IRBuilder(body_block)

    body_symbol_table = symbol_table.copy()  # type: SymbolTable

    for stmt in self.stmt_list:
      body_builder = stmt.build_expr_ir(module, builder=body_builder, symbol_table=body_symbol_table)
    if not body_block.is_terminated:
      assert body_builder is not None
      body_cond_ir = make_condition_ir(builder_=body_builder, symbol_table_=body_symbol_table)
      body_builder.cbranch(body_cond_ir, body_block, continue_block)

    continue_builder = ir.IRBuilder(continue_block)
    return continue_builder

  def __repr__(self):
    """
    :rtype: str
    """
    return 'WhileStatementAst(condition_val=%r, stmt_list=%r)' % (self.condition_val, self.stmt_list)


class ExpressionAst(AbstractSyntaxTree):
  """
  Val, SumVal, ProdVal, PrimaryExpr
  """
  def __init__(self, pos):
    """
    :param TreePosition pos:
    """
    super().__init__(pos)

  def make_val_type(self, symbol_table):
    """
    :param SymbolTable symbol_table:
    :rtype: Type
    """
    raise NotImplementedError()

  def make_ir_val(self, builder, symbol_table):
    """
    :param ir.IRBuilder builder:
    :param SymbolTable symbol_table:
    :rtype: ir.values.Value
    """
    raise NotImplementedError()

  def __repr__(self):
    """
    :rtype: str
    """
    return 'ExpressionAst'


class BinaryOperatorExpressionAst(ExpressionAst):
  """
  Val, SumVal, ProdVal.
  """
  def __init__(self, pos, op, left_expr, right_expr):
    """
    :param TreePosition pos:
    :param str op:
    :param ExpressionAst left_expr:
    :param ExpressionAst right_expr:
    """
    super().__init__(pos)
    assert op in SLOPPY_OP_TYPES
    self.op = op
    self.left_expr, self.right_expr = left_expr, right_expr

  def make_val_type(self, symbol_table):
    """
    :param SymbolTable symbol_table:
    :rtype: Type
    """
    left_type = self.left_expr.make_val_type(symbol_table=symbol_table)
    right_type = self.right_expr.make_val_type(symbol_table=symbol_table)
    if left_type == SLEEPY_DOUBLE_PTR:
      if self.op == '+' and right_type == SLEEPY_INT:
        return SLEEPY_DOUBLE_PTR
      if right_type == SLEEPY_DOUBLE_PTR:
        if self.op not in {'==', '!=', '<', '>', '<=', '>', '>='}:
          self.raise_error('Cannot apply binary operator %r on types %r and %r' % (
            self.op, left_type, right_type))
        return SLEEPY_BOOL
      self.raise_error('Cannot apply binary operator %r on types %r and %r' % (
        self.op, left_type, right_type))
    if left_type != right_type:
      self.raise_error('Cannot apply binary operator %r on different types %r and %r' % (
        self.op, left_type, right_type))
    if self.op in {'*', '/', '+', '-'}:
      return left_type
    if self.op in {'==', '!=', '<', '>', '<=', '>', '>='}:
      return SLEEPY_BOOL
    self.raise_error('Unknown binary operator %r' % self.op)

  def make_ir_val(self, builder, symbol_table):
    """
    :param ir.IRBuilder builder:
    :param SymbolTable symbol_table:
    :rtype: ir.values.Value
    """
    left_type = self.left_expr.make_val_type(symbol_table=symbol_table)
    right_type = self.right_expr.make_val_type(symbol_table=symbol_table)
    if left_type != right_type and not (left_type == SLEEPY_DOUBLE_PTR and right_type == SLEEPY_INT):
      self.raise_error('Cannot apply binary operator %r on different types %r and %r' % (
        self.op, left_type, right_type))
    var_type = left_type
    left_val = self.left_expr.make_ir_val(builder=builder, symbol_table=symbol_table)
    right_val = self.right_expr.make_ir_val(builder=builder, symbol_table=symbol_table)

    def make_op(type_instr, instr_name):
      """
      :param Dict[Type,Callable] type_instr:
      :param str instr_name:
      :rtype: ir.values.Value
      """
      if var_type not in type_instr:
        self.raise_error('Cannot apply binary operator %r on types %r' % (self.op, var_type))
      return type_instr[var_type](left_val, right_val, name=instr_name)

    if self.op == '*':
      return make_op(
        {SLEEPY_DOUBLE: builder.fmul, SLEEPY_INT: builder.mul, SLEEPY_LONG: builder.mul}, instr_name='mul_tmp')
    if self.op == '/':
      return make_op({SLEEPY_DOUBLE: builder.fdiv}, instr_name='div_tmp')
    if self.op == '+':
      if left_type == SLEEPY_DOUBLE_PTR and right_type == SLEEPY_INT:
        return builder.gep(left_val, (right_val,), name='incr_ptr_tmp')
      return make_op(
        {SLEEPY_DOUBLE: builder.fadd, SLEEPY_INT: builder.add, SLEEPY_LONG: builder.mul}, instr_name='add_tmp')
    if self.op == '-':
      return make_op(
        {SLEEPY_DOUBLE: builder.fsub, SLEEPY_INT: builder.sub, SLEEPY_LONG: builder.mul}, instr_name='sub_tmp')
    if self.op in {'==', '!=', '<', '>', '<=', '>', '>='}:
      from functools import partial
      return make_op({
        SLEEPY_DOUBLE: partial(builder.fcmp_ordered, self.op), SLEEPY_INT: partial(builder.icmp_signed, self.op),
        SLEEPY_LONG: partial(builder.icmp_signed, self.op), SLEEPY_DOUBLE_PTR: partial(builder.icmp_unsigned, self.op)},
        instr_name='cmp_tmp')
    assert False, 'Operator %s not handled!' % self.op

  def __repr__(self):
    """
    :rtype: str
    """
    return 'BinaryOperatorExpressionAst(op=%r, left_expr=%r, right_expr=%r)' % (
      self.op, self.left_expr, self.right_expr)


class UnaryOperatorExpressionAst(ExpressionAst):
  """
  NegVal.
  """
  def __init__(self, pos, op, expr):
    """
    :param TreePosition pos:
    :param str op:
    :param ExpressionAst expr:
    """
    super().__init__(pos)
    assert op in {'+', '-'}
    self.op = op
    self.expr = expr

  def make_val_type(self, symbol_table):
    """
    :param SymbolTable symbol_table:
    :rtype: Type
    """
    return self.expr.make_val_type(symbol_table=symbol_table)

  def make_ir_val(self, builder, symbol_table):
    """
    :param ir.IRBuilder builder:
    :param SymbolTable symbol_table:
    :rtype: ir.values.Value
    """
    val_type = self.expr.make_val_type(symbol_table=symbol_table)
    ir_val = self.expr.make_ir_val(builder=builder, symbol_table=symbol_table)
    if self.op == '+':
      return ir_val
    if self.op == '-':
      constant_minus_one = ir.Constant(val_type.ir_type, -1)
      if val_type == SLEEPY_DOUBLE:
        return builder.fmul(constant_minus_one, ir_val, name='neg_tmp')
      if val_type in {SLEEPY_INT, SLEEPY_LONG}:
        return builder.mul(constant_minus_one, ir_val, name='neg_tmp')
    self.raise_error('Cannot apply unary operator %r to type %r' % (self.op, val_type))

  def __repr__(self):
    """
    :rtype: str
    """
    return 'UnaryOperatorExpressionAst(op=%r, expr=%r)' % (self.op, self.expr)


class ConstantExpressionAst(ExpressionAst):
  """
  PrimaryExpr -> double | int | char
  """
  def __init__(self, pos, constant_val, constant_type):
    """
    :param TreePosition pos:
    :param Any constant_val:
    :param Type constant_type:
    """
    super().__init__(pos)
    self.constant_val = constant_val
    self.constant_type = constant_type

  def make_val_type(self, symbol_table):
    """
    :param SymbolTable symbol_table:
    :rtype: Type
    """
    return self.constant_type

  def make_ir_val(self, builder, symbol_table):
    """
    :param ir.IRBuilder builder:
    :param SymbolTable symbol_table:
    :rtype: ir.values.Value
    """
    return ir.Constant(self.constant_type.ir_type, self.constant_val)

  def __repr__(self):
    """
    :rtype: str
    """
    return 'ConstantExpressionAst(constant_val=%r, constant_type=%r)' % (self.constant_val, self.constant_type)


class VariableExpressionAst(ExpressionAst):
  """
  PrimaryExpr -> identifier
  """
  def __init__(self, pos, var_identifier):
    """
    :param TreePosition pos:
    :param str var_identifier:
    """
    super().__init__(pos)
    self.var_identifier = var_identifier

  def get_var_symbol(self, symbol_table):
    """
    :param SymbolTable symbol_table:
    :rtype: VariableSymbol
    """
    if self.var_identifier not in symbol_table:
      self.raise_error('Variable %r referenced before declaring' % self.var_identifier)
    symbol = symbol_table[self.var_identifier]
    if not isinstance(symbol, VariableSymbol):
      self.raise_error('Cannot reference a non-variable %r' % self.var_identifier)
    return symbol

  def make_val_type(self, symbol_table):
    """
    :param SymbolTable symbol_table:
    :rtype: Type
    """
    return self.get_var_symbol(symbol_table=symbol_table).var_type

  def make_ir_val(self, builder, symbol_table):
    """
    :param ir.IRBuilder builder:
    :param SymbolTable symbol_table:
    :rtype: ir.values.Value
    """
    symbol = self.get_var_symbol(symbol_table=symbol_table)
    return builder.load(symbol.ir_alloca, self.var_identifier)

  def __repr__(self):
    """
    :rtype: str
    """
    return 'VariableExpressionAst(var_identifier=%r)' % self.var_identifier


class CallExpressionAst(ExpressionAst):
  """
  PrimaryExpr -> identifier ( ExprList )
  """
  def __init__(self, pos, func_identifier, func_arg_exprs):
    """
    :param TreePosition pos:
    :param str func_identifier:
    :param list[ExpressionAst] func_arg_exprs:
    """
    super().__init__(pos)
    self.func_identifier = func_identifier
    self.func_arg_exprs = func_arg_exprs

  def get_func_symbol(self, symbol_table):
    """
    :param SymbolTable symbol_table:
    :rtype: FunctionSymbol
    """
    if self.func_identifier not in symbol_table:
      self.raise_error('Cannot call function %r before its declaration' % self.func_identifier)
    symbol = symbol_table[self.func_identifier]
    if not isinstance(symbol, FunctionSymbol):
      self.raise_error('Cannot call non-function %r' % self.func_identifier)
    return symbol

  def make_val_type(self, symbol_table):
    """
    :param SymbolTable symbol_table:
    :rtype: Type
    """
    self._check_func_call_symbol_table(
      func_identifier=self.func_identifier, func_arg_exprs=self.func_arg_exprs, symbol_table=symbol_table)
    return self.get_func_symbol(symbol_table=symbol_table).return_type

  def make_ir_val(self, builder, symbol_table):
    """
    :param ir.IRBuilder builder:
    :param SymbolTable symbol_table:
    :rtype: ir.values.Value
    """
    return self._make_func_call_ir(
      func_identifier=self.func_identifier, func_arg_exprs=self.func_arg_exprs, builder=builder,
      symbol_table=symbol_table)

  def __repr__(self):
    """
    :rtype: str
    """
    return 'CallExpressionAst(func_identifier=%r, func_arg_exprs=%r)' % (self.func_identifier, self.func_arg_exprs)


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
    'func', 'extern_func', 'struct', 'if', 'else', 'return', 'while', '{', '}', ';', ',', '(', ')', '->',
    'bool_op', 'sum_op', 'prod_op', '=', 'identifier',
    'int', 'double', 'char',
    None, None
  ], [
    'func', 'extern_func', 'struct', 'if', 'else', 'return', 'while', '{', '}', ';', ',', '\\(', '\\)', '\\->',
    '==|!=|<=?|>=?', '\\+|\\-', '\\*|/', '=', '([A-Z]|[a-z]|_)([A-Z]|[a-z]|[0-9]|_)*',
    '(0|[1-9][0-9]*)', '(0|[1-9][0-9]*)\\.[0-9]+', "'([^\']|\\\\[nrt'\"])'",
    '#[^\n]*\n', '[ \n]+'
  ])
SLEEPY_GRAMMAR = Grammar(
  Production('TopLevelStmt', 'StmtList'),
  Production('StmtList'),
  Production('StmtList', 'Stmt', 'StmtList'),
  Production('Stmt', 'func', 'identifier', '(', 'TypedIdentifierList', ')', 'ReturnType', '{', 'StmtList', '}'),
  Production('Stmt', 'extern_func', 'identifier', '(', 'TypedIdentifierList', ')', 'ReturnType', ';'),
  Production('Stmt', 'struct', 'identifier', '{', 'StmtList', '}'),
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
  Production('TypedIdentifierList'),
  Production('TypedIdentifierList', 'TypedIdentifierList+'),
  Production('TypedIdentifierList+', 'Type', 'identifier'),
  Production('TypedIdentifierList+', 'Type', 'identifier', ',', 'TypedIdentifierList+'),
  Production('ExprList'),
  Production('ExprList', 'ExprList+'),
  Production('ExprList+', 'Expr'),
  Production('ExprList+', 'Expr', ',', 'ExprList+'),
  Production('Type', 'identifier'),
  Production('ReturnType'),
  Production('ReturnType', '->', 'Type')
)
SLEEPY_ATTR_GRAMMAR = AttributeGrammar(
  SLEEPY_GRAMMAR,
  syn_attrs={
    'ast', 'stmt_list', 'identifier_list', 'type_list', 'val_list', 'identifier', 'type_identifier', 'op',
    'number'},
  prod_attr_rules=[
    {'ast': lambda _pos, stmt_list: TopLevelStatementAst(_pos, stmt_list(1))},
    {'stmt_list': []},
    {'stmt_list': lambda ast, stmt_list: [ast(1)] + stmt_list(2)},
    {'ast': lambda _pos, identifier, identifier_list, type_list, type_identifier, stmt_list: (
      FunctionDeclarationAst(_pos, identifier(2), identifier_list(4), type_list(4), type_identifier(6), stmt_list(8)))},  # noqa
    {'ast': lambda _pos, identifier, identifier_list, type_list, type_identifier: (
      FunctionDeclarationAst(_pos, identifier(2), identifier_list(4), type_list(4), type_identifier(6), None))},
    {'ast': lambda _pos, identifier, stmt_list: StructDeclarationAst(_pos, identifier(2), stmt_list(4))},
    {'ast': lambda _pos, identifier, val_list: CallStatementAst(_pos, identifier(1), val_list(3))},
    {'ast': lambda _pos, val_list: ReturnStatementAst(_pos, val_list(2))},
    {'ast': lambda _pos, identifier, ast, type_identifier: AssignStatementAst(_pos, identifier(2), ast(4), type_identifier(1))},  # noqa
    {'ast': lambda _pos, identifier, ast: AssignStatementAst(_pos, identifier(1), ast(3), None)},
    {'ast': lambda _pos, ast, stmt_list: IfStatementAst(_pos, ast(2), stmt_list(4), [])},
    {'ast': lambda _pos, ast, stmt_list: IfStatementAst(_pos, ast(2), stmt_list(4), stmt_list(8))},
    {'ast': lambda _pos, ast, stmt_list: WhileStatementAst(_pos, ast(2), stmt_list(4))}] + [
    {'ast': lambda _pos, ast, op: BinaryOperatorExpressionAst(_pos, op(2), ast(1), ast(3))},
    {'ast': 'ast.1'}] * 3 + [
    {'ast': lambda _pos, ast, op: UnaryOperatorExpressionAst(_pos, op(1), ast(2))},
    {'ast': 'ast.1'},
    {'ast': lambda _pos, number: ConstantExpressionAst(_pos, number(1), SLEEPY_INT)},
    {'ast': lambda _pos, number: ConstantExpressionAst(_pos, number(1), SLEEPY_DOUBLE)},
    {'ast': lambda _pos, number: ConstantExpressionAst(_pos, number(1), SLEEPY_CHAR)},
    {'ast': lambda _pos, identifier: VariableExpressionAst(_pos, identifier(1))},
    {'ast': lambda _pos, identifier, val_list: CallExpressionAst(_pos, identifier(1), val_list(3))},
    {'ast': 'ast.2'},
    {'identifier_list': []},
    {'identifier_list': 'identifier_list.1'},
    {'identifier_list': lambda identifier: [identifier(1)]},
    {'identifier_list': lambda identifier, identifier_list: [identifier(1)] + identifier_list(3)},
    {'identifier_list': [], 'type_list': []},
    {'identifier_list': 'identifier_list.1', 'type_list': 'type_list.1'},
    {'identifier_list': lambda identifier: [identifier(2)], 'type_list': lambda type_identifier: [type_identifier(1)]},
    {
      'identifier_list': lambda identifier, identifier_list: [identifier(2)] + identifier_list(4),
      'type_list': lambda type_identifier, type_list: [type_identifier(1)] + type_list(4)
    },
    {'val_list': []},
    {'val_list': 'val_list.1'},
    {'val_list': lambda ast: [ast(1)]},
    {'val_list': lambda ast, val_list: [ast(1)] + val_list(3)},
    {'type_identifier': 'identifier.1'},
    {'type_identifier': None},
    {'type_identifier': 'type_identifier.2'}
  ],
  terminal_attr_rules={
    'bool_op': {'op': lambda value: value},
    'sum_op': {'op': lambda value: value},
    'prod_op': {'op': lambda value: value},
    'identifier': {'identifier': lambda value: value},
    'int': {'number': lambda value: int(value)},
    'double': {'number': lambda value: float(value)},
    'char': {'number': lambda value: ord(parse_char(value))}
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
  preamble_program = """\
  extern_func print_char(Char char);
  extern_func print_double(Double d);
  extern_func print_int(Int i);
  extern_func allocate(Int size) -> DoublePtr;
  extern_func deallocate(DoublePtr ptr);
  extern_func load(DoublePtr ptr) -> Double;
  extern_func store(DoublePtr prt, Double value);
  extern_func assert(Bool condition);
  func True() -> Bool { return 0 == 0; }
  func False() -> Bool { return 0 != 0; }
  func or(Bool a, Bool b) -> Bool { if a { return a; } else { return b; } }
  func and(Bool a, Bool b) -> Bool { if a { return b; } else { return False(); } }
  func not(Bool a) -> Bool { if (a) { return False(); } else { return True(); } }
  """
  return make_program_ast(preamble_program, add_preamble=False)


def add_preamble_to_ast(program_ast):
  """
  :param TopLevelStatementAst program_ast:
  :rtype: TopLevelStatementAst
  """
  preamble_ast = make_preamble_ast()
  assert isinstance(preamble_ast, TopLevelStatementAst)
  preamble_pos = TreePosition(program_ast.pos.word, 0, 0)
  return TopLevelStatementAst(preamble_pos, preamble_ast.stmt_list + program_ast.stmt_list)
