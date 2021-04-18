

# Operator precedence: * / stronger than + - stronger than == != < <= > >=
from typing import Dict, List

from llvmlite import ir

from sleepy.grammar import SemanticError, Grammar, Production, AttributeGrammar, TreePosition
from sleepy.lexer import LexerGenerator
from sleepy.parser import ParserGenerator
from sleepy.symbols import FunctionSymbol, VariableSymbol, SLEEPY_DOUBLE, Type, SLEEPY_INT, \
  SLEEPY_LONG, SLEEPY_VOID, SLEEPY_DOUBLE_PTR, SLEEPY_BOOL, SLEEPY_CHAR, SymbolTable, TypeSymbol, \
  make_initial_symbol_table, StructType, ConcreteFunction, build_initial_module_ir

SLOPPY_OP_TYPES = {'*', '/', '+', '-', '==', '!=', '<', '>', '<=', '>', '>='}


class AbstractSyntaxTree:
  """
  Abstract syntax tree of a sleepy program.
  """

  allowed_annotation_identifiers = frozenset()

  def __init__(self, pos):
    """
    :param TreePosition pos: position where this AST starts
    """
    self.pos = pos
    self.annotations = []  # type: List[AnnotationAst]

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
    :rtype: ConcreteFunction
    """
    if func_identifier not in symbol_table:
      self.raise_error('Function %r called before declared' % func_identifier)
    symbol = symbol_table[func_identifier]
    if isinstance(symbol, TypeSymbol):
      if symbol.constructor_symbol is None:
        self.raise_error(
          'Cannot construct an instance of type %r that does not specify a constructor' % func_identifier)
      symbol = symbol.constructor_symbol
      assert isinstance(symbol, FunctionSymbol)
    if not isinstance(symbol, FunctionSymbol):
      self.raise_error('Cannot call non-function %r' % func_identifier)
    called_types = [arg_expr.make_val_type(symbol_table=symbol_table) for arg_expr in func_arg_exprs]
    if not symbol.has_concrete_func(called_types):
      self.raise_error('Cannot call function %r with arguments of types %r, only declared for parameter types: %r' % (
        func_identifier, ', '.join([str(called_type) for called_type in called_types]),
        ', '.join([concrete_func.to_signature_str() for concrete_func in symbol.concrete_funcs.values()])))
    concrete_func = symbol.get_concrete_func(called_types)
    called_mutables = [arg_expr.is_val_mutable(symbol_table=symbol_table) for arg_expr in func_arg_exprs]
    for arg_identifier, arg_mutable, called_mutable in zip(
        concrete_func.arg_identifiers, concrete_func.arg_mutables, called_mutables):
      if not called_mutable and arg_mutable:
        self.raise_error('Cannot call function %r declared with mutable parameter %r with immutable argument' % (
          func_identifier, arg_identifier))
    return concrete_func

  def _make_func_call_ir(self, func_identifier, func_arg_exprs, builder, symbol_table):
    """
    :param str func_identifier:
    :param list[ExpressionAst] func_arg_exprs:
    :param IRBuilder builder:
    :param SymbolTable symbol_table:
    :rtype: tuple[ir.values.Value,IRBuilder]
    """
    assert func_identifier in symbol_table
    func_symbol = symbol_table[func_identifier]
    if isinstance(func_symbol, TypeSymbol):
      func_symbol = func_symbol.constructor_symbol
    assert isinstance(func_symbol, FunctionSymbol)
    called_types = [arg_expr.make_val_type(symbol_table=symbol_table) for arg_expr in func_arg_exprs]
    assert func_symbol.has_concrete_func(called_types)
    concrete_func = func_symbol.get_concrete_func(called_types)
    ir_func = concrete_func.ir_func
    assert ir_func is not None
    assert len(concrete_func.arg_types) == len(func_arg_exprs) == len(ir_func.args)
    ir_func_args = []  # type: List[ir.values.Value]
    for func_arg_expr in func_arg_exprs:
      ir_func_arg, builder = func_arg_expr.make_ir_val(builder=builder, symbol_table=symbol_table)
      ir_func_args.append(ir_func_arg)
    assert len(ir_func_args) == len(func_arg_exprs)
    if concrete_func.is_inline:
      assert callable(concrete_func.make_inline_func_call_ir)
      return concrete_func.make_inline_func_call_ir(ir_func_args=ir_func_args, body_builder=builder)
    else:
      return builder.call(ir_func, ir_func_args, name='call_%s' % func_identifier), builder

  def _make_member_val_type(self, parent_type, member_identifier, symbol_table):
    """
    :param Type parent_type:
    :param str member_identifier:
    :param SymbolTable symbol_table:
    :rtype: Type
    """
    if not isinstance(parent_type, StructType):
      self.raise_error(
        'Cannot access a member variable %r of the non-struct type %r' % (member_identifier, parent_type))
    if member_identifier not in parent_type.member_identifiers:
      self.raise_error('Struct type %r has no member variable %r, only available: %r' % (
        parent_type, member_identifier, ', '.join(parent_type.member_identifiers)))
    member_num = parent_type.get_member_num(member_identifier)
    return parent_type.member_types[member_num]


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

  def make_var_is_mutable(self, arg_name, arg_type, arg_annotation_list, default):
    """
    :param str arg_name:
    :param Type arg_type:
    :param list[AnnotationAst] arg_annotation_list:
    :param bool|None default:
    :rtype: bool|None
    """
    assert isinstance(arg_annotation_list, (tuple, list))
    has_mutable = any(annotation.identifier == 'Mutable' for annotation in arg_annotation_list)
    has_const = any(annotation.identifier == 'Const' for annotation in arg_annotation_list)
    if has_mutable and has_const:
      self.raise_error('Cannot annotate %r with both %r and %r' % (arg_name, 'Mutable', 'Const'))
    mutable = default if (not has_mutable and not has_const) else has_mutable
    if mutable and not arg_type.pass_by_ref:
      self.raise_error(
        'Type %r of mutable %s needs to have pass-by-reference semantics (annotatated by @RefType)' % (
          arg_type, arg_name))
    return mutable

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
    build_initial_module_ir(module=module, symbol_table=symbol_table)
    assert symbol_table.ir_func_malloc is not None and symbol_table.ir_func_free is not None

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

  allowed_annotation_identifiers = {'Inline'}
  allowed_arg_annotation_identifiers = {'Const', 'Mutable'}

  def __init__(self, pos, identifier, arg_identifiers, arg_type_identifiers, arg_annotations, return_type_identifier,
               return_annotation_list, stmt_list):
    """
    :param TreePosition pos:
    :param str identifier:
    :param list[str] arg_identifiers:
    :param list[str|None] arg_type_identifiers:
    :param list[list[AnnotationAst]] arg_annotations:
    :param str|None return_type_identifier:
    :param list[AnnotationAst]|None return_annotation_list:
    :param list[StatementAst]|None stmt_list: body, or None if extern function.
    """
    super().__init__(pos)
    assert len(arg_identifiers) == len(arg_type_identifiers) == len(arg_annotations)
    assert (return_type_identifier is None) == (return_annotation_list is None)
    self.identifier = identifier
    self.arg_identifiers = arg_identifiers
    self.arg_type_identifiers = arg_type_identifiers
    self.arg_annotations = arg_annotations
    self.return_type_identifier = return_type_identifier
    self.return_annotation_list = return_annotation_list
    self.stmt_list = stmt_list

  @property
  def is_extern(self):
    """
    :rtype: bool
    """
    return self.stmt_list is None

  @property
  def is_inline(self):
    """
    :rtype: bool
    """
    return any(annotation.identifier == 'Inline' for annotation in self.annotations)

  def make_arg_types(self, symbol_table):
    """
    :param SymbolTable symbol_table:
    :rtype: list[Type]
    """
    arg_types = [self.make_type(identifier, symbol_table=symbol_table) for identifier in self.arg_type_identifiers]
    if any(arg_type is None for arg_type in arg_types):
      self.raise_error('Need to specify all parameter types of function %r' % self.identifier)
    all_annotation_list = (
      self.arg_annotations + ([self.return_annotation_list] if self.return_annotation_list is not None else []))
    for arg_annotation_list in all_annotation_list:
      for arg_annotation_num, arg_annotation in enumerate(arg_annotation_list):
        if arg_annotation.identifier in arg_annotation_list[:arg_annotation_num]:
          arg_annotation.raise_error('Cannot apply annotation with identifier %r twice' % arg_annotation.identifier)
        if arg_annotation.identifier not in self.allowed_arg_annotation_identifiers:
          arg_annotation.raise_error('Cannot apply annotation with identifier %r, allowed: %r' % (
            arg_annotation.identifier, ', '.join(self.allowed_arg_annotation_identifiers)))
    return arg_types

  def build_symbol_table(self, symbol_table):
    """
    :param SymbolTable symbol_table:
    """
    arg_types = self.make_arg_types(symbol_table=symbol_table)
    if self.return_type_identifier is None:
      return_type = SLEEPY_VOID
    else:
      return_type = self.make_type(self.return_type_identifier, symbol_table=symbol_table)
    if return_type is None:
      self.raise_error('Need to specify return type of function %r' % self.identifier)
    if return_type == SLEEPY_VOID:
      return_mutable = False
    else:
      return_mutable = self.make_var_is_mutable('return type', return_type, self.return_annotation_list, default=False)
    if self.identifier in symbol_table:
      func_symbol = symbol_table[self.identifier]
      if not isinstance(func_symbol, FunctionSymbol):
        self.raise_error('Cannot redefine previously declared non-function %r with a function' % self.identifier)
    else:
      func_symbol = FunctionSymbol()
      symbol_table[self.identifier] = func_symbol
    if func_symbol.has_concrete_func(arg_types):
      self.raise_error('Cannot override definition of function %r with parameter types %r' % (
        self.identifier, ', '.join([str(arg_type) for arg_type in arg_types])))
    arg_mutables = [
      self.make_var_is_mutable('parameter %r' % arg_identifier, arg_type, arg_annotation_list, default=False)
      for arg_identifier, arg_type, arg_annotation_list in zip(self.arg_identifiers, arg_types, self.arg_annotations)]
    if self.is_inline:
      if self.is_extern:
        self.raise_error('Extern function %r cannot be inlined' % self.identifier)
      for arg_identifier, arg_mutable in zip(self.arg_identifiers, arg_mutables):
        if arg_mutable:
          self.raise_error('Cannot inline function %r with a mutable parameter %r' % (self.identifier, arg_identifier))
    func_symbol.add_concrete_func(
      ConcreteFunction(
        None, return_type=return_type, return_mutable=return_mutable, arg_identifiers=self.arg_identifiers,
        arg_types=arg_types, arg_mutables=arg_mutables, is_inline=self.is_inline))

  def _get_concrete_func(self, symbol_table):
    """
    :param SymbolTable symbol_table:
    :rtype: ConcreteFunction
    """
    arg_types = self.make_arg_types(symbol_table=symbol_table)
    assert self.identifier in symbol_table
    func_symbol = symbol_table[self.identifier]
    assert isinstance(func_symbol, FunctionSymbol)
    assert func_symbol.has_concrete_func(arg_types)
    return func_symbol.get_concrete_func(arg_types)

  def build_expr_ir(self, module, builder, symbol_table):
    """
    :param ir.Module module:
    :param ir.IRBuilder builder:
    :param SymbolTable symbol_table:
    :rtype: ir.IRBuilder
    """
    concrete_func = self._get_concrete_func(symbol_table=symbol_table)
    ir_func_type = concrete_func.make_ir_function_type()
    ir_func_name = symbol_table.make_ir_func_name(self.identifier, self.is_extern, concrete_func)
    concrete_func.ir_func = ir.Function(module, ir_func_type, name=ir_func_name)

    if not self.is_extern:
      if self.is_inline:
        def make_inline_func_call_ir(ir_func_args, body_builder):
          """
          :param list[ir.values.Value] ir_func_args:
          :param ir.IRBuilder body_builder:
          :rtype: (ir.values.Value, ir.IRBuilder)
          """
          assert self.is_inline
          assert len(ir_func_args) == len(self.arg_identifiers)
          return_val_ir_alloca = body_builder.alloca(
            concrete_func.return_type.ir_type, name='return_%s_alloca' % self.identifier)
          collect_block = body_builder.append_basic_block('collect_return_%s_block' % self.identifier)
          self._build_body_ir(
            ir_func_args=ir_func_args, module=module, body_builder=body_builder, symbol_table=symbol_table,
            inline_return_ir_alloca=return_val_ir_alloca, inline_return_collect_block=collect_block)
          collect_builder = ir.IRBuilder(collect_block)
          return_val = collect_builder.load(return_val_ir_alloca, name='return_%s' % self.identifier)
          return return_val, collect_builder

        concrete_func.make_inline_func_call_ir = make_inline_func_call_ir
      else:
        assert not self.is_inline
        block = concrete_func.ir_func.append_basic_block(name='entry')
        concrete_func = self._get_concrete_func(symbol_table=symbol_table)
        body_builder = ir.IRBuilder(block)
        self._build_body_ir(
          ir_func_args=concrete_func.ir_func.args, module=module, body_builder=body_builder, symbol_table=symbol_table)

    return builder

  def _build_body_ir(self, ir_func_args, module, body_builder, symbol_table, inline_return_ir_alloca=None,
                     inline_return_collect_block=None):
    """
    :param list[ir.values.Value] ir_func_args:
    :param ir.Module module:
    :param ir.IRBuilder body_builder:
    :param SymbolTable symbol_table:
    :param None|ir.instructions.AllocaInstr inline_return_ir_alloca:
    :param None|ir.Block inline_return_collect_block:
    :rtype: (ir.values.Value, ir.IRBuilder)
    """
    assert not self.is_extern
    assert (inline_return_ir_alloca is None) == (inline_return_collect_block is None) == (not self.is_inline)
    arg_types = self.make_arg_types(symbol_table=symbol_table)
    concrete_func = self._get_concrete_func(symbol_table=symbol_table)
    assert len(ir_func_args) == len(concrete_func.arg_identifiers)
    body_symbol_table = symbol_table.copy()  # type: SymbolTable
    body_symbol_table.current_func = concrete_func
    body_symbol_table.current_scope_identifiers = []
    body_symbol_table.current_func_inline_return_ir_alloca = inline_return_ir_alloca
    for arg_identifier, arg_type, arg_mutable in zip(self.arg_identifiers, arg_types, concrete_func.arg_mutables):
      body_symbol_table[arg_identifier] = VariableSymbol(None, arg_type, arg_mutable)
      body_symbol_table.current_scope_identifiers.append(arg_identifier)
    for stmt in self.stmt_list:
      stmt.build_symbol_table(symbol_table=body_symbol_table)

    for identifier_name in body_symbol_table.current_scope_identifiers:
      assert identifier_name in body_symbol_table
      var_symbol = body_symbol_table[identifier_name]
      if not isinstance(var_symbol, VariableSymbol):
        continue
      var_symbol.ir_alloca = body_builder.alloca(var_symbol.var_type.ir_type, name=identifier_name)

    for arg_identifier, ir_arg in zip(self.arg_identifiers, ir_func_args):
      arg_symbol = body_symbol_table[arg_identifier]
      assert isinstance(arg_symbol, VariableSymbol)
      ir_arg.name = arg_identifier
      body_builder.store(ir_arg, arg_symbol.ir_alloca)

    for stmt in self.stmt_list:
      body_builder = stmt.build_expr_ir(module=module, builder=body_builder, symbol_table=body_symbol_table)
    if body_builder is not None and not body_builder.block.is_terminated:
      return_pos = TreePosition(
        self.pos.word,
        self.pos.from_pos if len(self.stmt_list) == 0 else self.stmt_list[-1].pos.to_pos,
        self.pos.to_pos)
      return_ast = ReturnStatementAst(return_pos, [])
      if concrete_func.return_type != SLEEPY_VOID:
        return_ast.raise_error(
          'Not all branches within function declaration of %r return something' % self.identifier)
      return_ast.build_symbol_table(symbol_table=body_symbol_table)  # for error checking
      return_ast.build_expr_ir(module=module, builder=body_builder, symbol_table=body_symbol_table)

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
      if return_val_type == SLEEPY_VOID:
        self.raise_error('Cannot use void return value')
      if return_val_type != symbol_table.current_func.return_type:
        if symbol_table.current_func.return_type == SLEEPY_VOID:
          self.raise_error('Function declared to return void, but return value is of type %r' % (
            return_val_type))
        else:
          self.raise_error('Function declared to return type %r, but return value is of type %r' % (
            symbol_table.current_func.return_type, return_val_type))
      return_val_mutable = self.return_exprs[0].is_val_mutable(symbol_table=symbol_table)
      if not return_val_mutable and symbol_table.current_func.return_mutable:
        self.raise_error(
          'Function declared to return a mutable type %r, but return value is not mutable' % return_val_type)
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
      ir_val, builder = self.return_exprs[0].make_ir_val(builder=builder, symbol_table=symbol_table)
      builder.ret(ir_val)
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

  allowed_annotation_identifiers = frozenset({'ValType', 'RefType'})

  def __init__(self, pos, struct_identifier, stmt_list):
    """
    :param TreePosition pos:
    :param str struct_identifier:
    :param List[StatementAst] stmt_list:
    """
    super().__init__(pos)
    self.struct_identifier = struct_identifier
    self.stmt_list = stmt_list

  def is_pass_by_ref(self):
    """
    :rtype: bool
    """
    val_type = any(annotation.identifier == 'ValType' for annotation in self.annotations)
    ref_type = any(annotation.identifier == 'RefType' for annotation in self.annotations)
    if val_type and ref_type:
      self.raise_error('Cannot apply annotation %r and %r at the same time' % ('ValType', 'RefType'))
    return ref_type  # fall back to pass by value.

  def build_symbol_table(self, symbol_table):
    """
    :param SymbolTable symbol_table:
    """
    if self.struct_identifier in symbol_table.current_scope_identifiers:
      self.raise_error('Cannot redefine struct with name %r' % self.struct_identifier)
    body_symbol_table = symbol_table.copy()
    for member_num, stmt in enumerate(self.stmt_list):
      if not isinstance(stmt, AssignStatementAst):
        stmt.raise_error('Can only use declare statements within a struct declaration')
      if not isinstance(stmt.var_target, VariableTargetAst):
        stmt.raise_error('Can only declare variables within a struct declaration')
      stmt.build_symbol_table(symbol_table=body_symbol_table)
      if len(body_symbol_table.current_scope_identifiers) != member_num + 1:
        stmt.raise_error(
          'Cannot declare member %r multiple times in struct declaration' % stmt.var_target.var_identifier)
    assert len(self.stmt_list) == len(body_symbol_table.current_scope_identifiers)
    member_identifiers, member_types = [], []
    for stmt, declared_identifier in zip(self.stmt_list, body_symbol_table.current_scope_identifiers):
      assert declared_identifier in body_symbol_table
      declared_symbol = body_symbol_table[declared_identifier]
      assert isinstance(declared_symbol, VariableSymbol)
      member_identifiers.append(declared_identifier)
      member_types.append(declared_symbol.var_type)
    member_mutables = [False] * len(member_identifiers)
    assert len(member_identifiers) == len(member_types) == len(member_mutables) == len(self.stmt_list)

    struct_type = StructType(
      self.struct_identifier, member_identifiers, member_types, member_mutables, pass_by_ref=self.is_pass_by_ref())
    constructor = FunctionSymbol()
    # ir_func will be set in build_expr_ir
    # notice that we explicitly set return_mutable=False here, even if the constructor mutated the struct.
    constructor.add_concrete_func(ConcreteFunction(
      ir_func=None, return_type=struct_type, return_mutable=True,
      arg_types=member_types, arg_identifiers=member_identifiers, arg_mutables=member_mutables))
    symbol_table[self.struct_identifier] = TypeSymbol(struct_type, constructor_symbol=constructor)
    symbol_table.current_scope_identifiers.append(self.struct_identifier)

  def _make_constructor_body_ir(self, constructor, symbol_table):
    """
    :param ConcreteFunction constructor:
    :param SymbolTable symbol_table:
    """
    struct_type = constructor.return_type
    constructor_symbol_table = symbol_table.copy()
    constructor_block = constructor.ir_func.append_basic_block(name='entry')
    constructor_builder = ir.IRBuilder(constructor_block)
    if self.is_pass_by_ref():  # use malloc
      assert symbol_table.ir_func_malloc is not None
      self_ir_alloca_raw = constructor_builder.call(
        symbol_table.ir_func_malloc, [struct_type.make_ir_size(builder=constructor_builder)], name='self_raw_ptr')
      self_ir_alloca = constructor_builder.bitcast(self_ir_alloca_raw, struct_type.ir_type, name='self')
      # TODO: eventually free memory again
    else:  # pass by value, use alloca
      self_ir_alloca = constructor_builder.alloca(struct_type.ir_type, name='self')

    for member_num, (stmt, ir_func_arg) in enumerate(zip(self.stmt_list, constructor.ir_func.args)):
      assert isinstance(stmt, AssignStatementAst)
      assert isinstance(stmt.var_target, VariableTargetAst)
      member_identifier = stmt.var_target.var_identifier
      ir_func_arg.name = member_identifier
      gep_indices = (ir.Constant(ir.IntType(32), 0), ir.Constant(ir.IntType(32), member_num))
      member_ptr = constructor_builder.gep(self_ir_alloca, gep_indices, '%s_ptr' % member_identifier)
      constructor_builder.store(ir_func_arg, member_ptr)

    if self.is_pass_by_ref():
      constructor_builder.ret(self_ir_alloca)
    else:  # pass by value
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
    assert isinstance(struct_symbol.type, StructType)
    constructor = struct_symbol.constructor_symbol.get_concrete_func(arg_types=struct_symbol.type.member_types)
    assert constructor is not None
    assert struct_symbol.type == constructor.return_type
    ir_func_type = constructor.make_ir_function_type()
    ir_func_name = symbol_table.make_ir_func_name(self.struct_identifier, extern=False, concrete_func=constructor)
    constructor.ir_func = ir.Function(module, ir_func_type, name=ir_func_name)
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
  allowed_annotation_identifiers = frozenset({'Const', 'Mutable'})

  def __init__(self, pos, var_target, var_val, var_type_identifier):
    """
    :param TreePosition pos:
    :param TargetAst var_target:
    :param ExpressionAst var_val:
    :param str|None var_type_identifier:
    """
    super().__init__(pos)
    assert isinstance(var_target, TargetAst)
    self.var_target = var_target
    self.var_val = var_val
    self.var_type_identifier = var_type_identifier

  def is_declaration(self, symbol_table):
    """
    :param SymbolTable symbol_table:
    :rtype: bool
    """
    if not isinstance(self.var_target, VariableTargetAst):
      return False
    var_identifier = self.var_target.var_identifier
    if var_identifier not in symbol_table.current_scope_identifiers:
      return True
    assert var_identifier in symbol_table
    symbol = symbol_table[var_identifier]
    if not isinstance(symbol, VariableSymbol):
      self.raise_error('Cannot assign non-variable %r to a variable' % var_identifier)
    ptr_type = self.var_target.make_ptr_type(symbol_table=symbol_table)
    if symbol.var_type != ptr_type:
      self.raise_error('Cannot redefine variable %r of type %r with new type %r' % (
        var_identifier, symbol.var_type, ptr_type))

  def build_symbol_table(self, symbol_table):
    """
    :param SymbolTable symbol_table:
    """
    if self.var_type_identifier is not None:
      declared_type = self.make_type(self.var_type_identifier, symbol_table=symbol_table)
    else:
      declared_type = None
    val_type = self.var_val.make_val_type(symbol_table=symbol_table)
    if val_type == SLEEPY_VOID:
      self.raise_error('Cannot assign void to variable')
    if declared_type is not None and declared_type != val_type:
      self.raise_error('Cannot assign variable with declared type %r a value of type %r' % (declared_type, val_type))
    declared_mutable = self.make_var_is_mutable('left-hand-side', val_type, self.annotations, default=None)
    val_mutable = self.var_val.is_val_mutable(symbol_table=symbol_table)

    if self.is_declaration(symbol_table=symbol_table):
      assert isinstance(self.var_target, VariableTargetAst)
      var_identifier = self.var_target.var_identifier
      assert var_identifier not in symbol_table.current_scope_identifiers
      if declared_mutable is None:
        declared_mutable = False
      # declare new variable, override entry in symbol_table (maybe it was defined in an outer scope before).
      symbol = VariableSymbol(None, var_type=val_type, mutable=declared_mutable)
      symbol_table[var_identifier] = symbol
      symbol_table.current_scope_identifiers.append(var_identifier)
    else:
      # variable name in this scope already declared. just check that types match, but do not change symbol_table.
      ptr_type = self.var_target.make_ptr_type(symbol_table=symbol_table)
      if ptr_type != val_type:
        self.raise_error('Cannot redefine variable of type %r with new type %r' % (ptr_type, val_type))
      if not self.var_target.is_ptr_reassignable(symbol_table=symbol_table):
        self.raise_error('Cannot reassign member of a non-mutable variable')
      if declared_mutable is None:
        declared_mutable = val_mutable
      if declared_mutable != val_mutable:
        if declared_mutable:
          self.raise_error('Cannot redefine a variable declared as non-mutable to mutable')
        else:
          self.raise_error('Cannot redefine a variable declared as mutable to non-mutable')
    if declared_mutable and not val_mutable:
      self.raise_error('Cannot assign a non-mutable variable a mutable value of type %r' % val_type)

  def build_expr_ir(self, module, builder, symbol_table):
    """
    :param ir.Module module:
    :param ir.IRBuilder builder:
    :param SymbolTable symbol_table:
    :rtype: ir.IRBuilder
    """
    ir_val, builder = self.var_val.make_ir_val(builder=builder, symbol_table=symbol_table)
    ir_ptr = self.var_target.make_ir_ptr(builder, symbol_table=symbol_table)
    builder.store(ir_val, ir_ptr)
    return builder

  def __repr__(self):
    """
    :rtype: str
    """
    return 'AssignStatementAst(var_target=%r, var_val=%r, var_type_identifier=%r)' % (
      self.var_target, self.var_val, self.var_type_identifier)


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
    ir_cond, builder = self.condition_val.make_ir_val(builder=builder, symbol_table=symbol_table)
    true_block = builder.append_basic_block('true_branch')  # type: ir.Block
    false_block = builder.append_basic_block('false_branch')  # type: ir.Block
    builder.cbranch(ir_cond, true_block, false_block)
    true_builder, false_builder = ir.IRBuilder(true_block), ir.IRBuilder(false_block)

    true_symbol_table = symbol_table.copy()  # type: SymbolTable
    false_symbol_table = symbol_table.copy()  # type: SymbolTable

    for expr in self.true_stmt_list:
      true_builder = expr.build_expr_ir(module, builder=true_builder, symbol_table=true_symbol_table)
    for expr in self.false_stmt_list:
      false_builder = expr.build_expr_ir(module, builder=false_builder, symbol_table=false_symbol_table)

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
      :rtype: (ir.values.Value, ir.IRBuilder)
      """
      return self.condition_val.make_ir_val(builder=builder_, symbol_table=symbol_table_)

    cond_ir, builder = make_condition_ir(builder_=builder, symbol_table_=symbol_table)
    body_block = builder.append_basic_block('while_body')  # type: ir.Block
    continue_block = builder.append_basic_block('continue_branch')  # type: ir.Block
    builder.cbranch(cond_ir, body_block, continue_block)
    body_builder = ir.IRBuilder(body_block)

    body_symbol_table = symbol_table.copy()  # type: SymbolTable

    for stmt in self.stmt_list:
      body_builder = stmt.build_expr_ir(module, builder=body_builder, symbol_table=body_symbol_table)
    if not body_builder.block.is_terminated:
      assert body_builder is not None
      body_cond_ir, body_builder = make_condition_ir(builder_=body_builder, symbol_table_=body_symbol_table)
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

  def is_val_mutable(self, symbol_table):
    """
    :param SymbolTable symbol_table:
    :rtype: Type
    """
    raise NotImplementedError()

  def make_ir_val(self, builder, symbol_table):
    """
    :param ir.IRBuilder builder:
    :param SymbolTable symbol_table:
    :rtype: (ir.values.Value, ir.IRBuilder)
    :return: The value this expression is evaluated to + the builder of the context after this is executed
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
    operand_exprs = [self.left_expr, self.right_expr]
    concrete_func = self._check_func_call_symbol_table(
      func_identifier=self.op, func_arg_exprs=operand_exprs, symbol_table=symbol_table)
    return concrete_func.return_type

  def is_val_mutable(self, symbol_table):
    """
    :param SymbolTable symbol_table:
    :rtype: Type
    """
    return False

  def make_ir_val(self, builder, symbol_table):
    """
    :param ir.IRBuilder builder:
    :param SymbolTable symbol_table:
    :rtype: (ir.values.Value, ir.IRBuilder)
    """
    operand_exprs = [self.left_expr, self.right_expr]
    return self._make_func_call_ir(
      func_identifier=self.op, func_arg_exprs=operand_exprs, builder=builder, symbol_table=symbol_table)

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
    self.op = op
    self.expr = expr

  def make_val_type(self, symbol_table):
    """
    :param SymbolTable symbol_table:
    :rtype: Type
    """
    operand_exprs = [self.expr]
    concrete_func = self._check_func_call_symbol_table(
      func_identifier=self.op, func_arg_exprs=operand_exprs, symbol_table=symbol_table)
    return concrete_func.return_type

  def is_val_mutable(self, symbol_table):
    """
    :param SymbolTable symbol_table:
    :rtype: Type
    """
    return False

  def make_ir_val(self, builder, symbol_table):
    """
    :param ir.IRBuilder builder:
    :param SymbolTable symbol_table:
    :rtype: (ir.values.Value, ir.IRBuilder)
    """
    operand_exprs = [self.expr]
    return self._make_func_call_ir(
      func_identifier=self.op, func_arg_exprs=operand_exprs, builder=builder, symbol_table=symbol_table)

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

  def is_val_mutable(self, symbol_table):
    """
    :param SymbolTable symbol_table:
    :rtype: Type
    """
    return False

  def make_ir_val(self, builder, symbol_table):
    """
    :param ir.IRBuilder builder:
    :param SymbolTable symbol_table:
    :rtype: (ir.values.Value, ir.IRBuilder)
    """
    return ir.Constant(self.constant_type.ir_type, self.constant_val), builder

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

  def is_val_mutable(self, symbol_table):
    """
    :param SymbolTable symbol_table:
    :rtype: Type
    """
    return self.get_var_symbol(symbol_table=symbol_table).mutable

  def make_ir_val(self, builder, symbol_table):
    """
    :param ir.IRBuilder builder:
    :param SymbolTable symbol_table:
    :rtype: (ir.values.Value, ir.IRBuilder)
    """
    symbol = self.get_var_symbol(symbol_table=symbol_table)
    return builder.load(symbol.ir_alloca, name=self.var_identifier), builder

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
    assert self.func_identifier in symbol_table
    symbol = symbol_table[self.func_identifier]
    if isinstance(symbol, TypeSymbol):
      symbol = symbol.constructor_symbol
    assert isinstance(symbol, FunctionSymbol)
    assert symbol is not None
    return symbol

  def make_val_type(self, symbol_table):
    """
    :param SymbolTable symbol_table:
    :rtype: Type
    """
    concrete_func = self._check_func_call_symbol_table(
      func_identifier=self.func_identifier, func_arg_exprs=self.func_arg_exprs, symbol_table=symbol_table)
    return concrete_func.return_type

  def is_val_mutable(self, symbol_table):
    """
    :param SymbolTable symbol_table:
    :rtype: Type
    """
    concrete_func = self._check_func_call_symbol_table(
      func_identifier=self.func_identifier, func_arg_exprs=self.func_arg_exprs, symbol_table=symbol_table)
    return concrete_func.return_mutable

  def make_ir_val(self, builder, symbol_table):
    """
    :param ir.IRBuilder builder:
    :param SymbolTable symbol_table:
    :rtype: (ir.values.Value, ir.IRBuilder)
    """
    return self._make_func_call_ir(
      func_identifier=self.func_identifier, func_arg_exprs=self.func_arg_exprs, builder=builder,
      symbol_table=symbol_table)

  def __repr__(self):
    """
    :rtype: str
    """
    return 'CallExpressionAst(func_identifier=%r, func_arg_exprs=%r)' % (self.func_identifier, self.func_arg_exprs)


class MemberExpressionAst(ExpressionAst):
  """
  MemberExpr -> MemberExpr . identifier
  """
  def __init__(self, pos, parent_val_expr, member_identifier):
    """
    :param TreePosition pos:
    :param ExpressionAst parent_val_expr:
    :param str member_identifier:
    """
    super().__init__(pos)
    self.parent_val_expr = parent_val_expr
    self.member_identifier = member_identifier

  def make_val_type(self, symbol_table):
    """
    :param SymbolTable symbol_table:
    :rtype: Type
    """
    parent_type = self.parent_val_expr.make_val_type(symbol_table=symbol_table)
    return self._make_member_val_type(parent_type, self.member_identifier, symbol_table)

  def make_ir_val(self, builder, symbol_table):
    """
    :param ir.IRBuilder builder:
    :param SymbolTable symbol_table:
    :rtype: (ir.values.Value, ir.IRBuilder)
    """
    parent_type = self.parent_val_expr.make_val_type(symbol_table=symbol_table)
    assert isinstance(parent_type, StructType)
    parent_ir_val, builder = self.parent_val_expr.make_ir_val(builder=builder, symbol_table=symbol_table)
    if parent_type.is_pass_by_ref():
      member_num = parent_type.get_member_num(self.member_identifier)
      gep_indices = (ir.Constant(ir.IntType(32), 0), ir.Constant(ir.IntType(32), member_num))
      member_ptr = builder.gep(parent_ir_val, gep_indices, name='member_ptr_%s' % self.member_identifier)
      return builder.load(member_ptr, name='member_%s' % self.member_identifier), builder
    else:  # pass by value
      return (
        builder.extract_value(
          parent_ir_val, parent_type.get_member_num(self.member_identifier), name='member_%s' % self.member_identifier),
        builder)

  def is_val_mutable(self, symbol_table):
    """
    :param SymbolTable symbol_table:
    :rtype: Type
    """
    parent_type = self.parent_val_expr.make_val_type(symbol_table=symbol_table)
    assert isinstance(parent_type, StructType)
    member_num = parent_type.get_member_num(self.member_identifier)
    member_mutable = parent_type.member_mutables[member_num]
    return member_mutable

  def __repr__(self):
    """
    :rtype: str
    """
    return 'MemberExpressionAst(parent_val_expr=%r, member_identifier=%r)' % (
      self.parent_val_expr, self.member_identifier)


class TargetAst(AbstractSyntaxTree):
  """
  Target.
  """
  def __init__(self, pos):
    """
    :param TreePosition pos:
    """
    super().__init__(pos)

  def make_ptr_type(self, symbol_table):
    """
    :param SymbolTable symbol_table:
    :rtype: Type
    """
    raise NotImplementedError()

  def is_ptr_mutable(self, symbol_table):
    """
    :param SymbolTable symbol_table:
    :rtype: bool
    """
    raise NotImplementedError()

  def is_ptr_reassignable(self, symbol_table):
    """
    :param SymbolTable symbol_table:
    :rtype bool:
    """
    raise NotImplementedError()

  def make_ir_ptr(self, builder, symbol_table):
    """
    :param ir.IRBuilder builder:
    :param SymbolTable symbol_table:
    :rtype: ir.instructions.Instruction
    """
    raise NotImplementedError()

  def __repr__(self):
    """
    :rtype: str
    """
    return 'TargetAst'


class VariableTargetAst(TargetAst):
  """
  Target -> identifier
  """
  def __init__(self, pos, var_identifier):
    """
    :param TreePosition pos:
    :param str var_identifier:
    """
    super().__init__(pos)
    self.var_identifier = var_identifier

  def make_ptr_type(self, symbol_table):
    """
    :param SymbolTable symbol_table:
    :rtype: Type
    """
    if self.var_identifier not in symbol_table:
      self.raise_error('Cannot reference variable %r before declaration' % self.var_identifier)
    symbol = symbol_table[self.var_identifier]
    if not isinstance(symbol, VariableSymbol):
      self.raise_error('Cannot assign to non-variable %r' % self.var_identifier)
    return symbol.var_type

  def is_ptr_mutable(self, symbol_table):
    """
    :param SymbolTable symbol_table:
    :rtype: bool
    """
    assert self.var_identifier in symbol_table
    symbol = symbol_table[self.var_identifier]
    assert isinstance(symbol, VariableSymbol)
    return symbol.mutable

  def is_ptr_reassignable(self, symbol_table):
    """
    :param SymbolTable symbol_table:
    :rtype bool:
    """
    return True

  def make_ir_ptr(self, builder, symbol_table):
    """
    :param ir.IRBuilder builder:
    :param SymbolTable symbol_table:
    :rtype: ir.instructions.Instruction
    """
    assert self.var_identifier in symbol_table
    symbol = symbol_table[self.var_identifier]
    assert isinstance(symbol, VariableSymbol)
    assert symbol.ir_alloca is not None  # ir_alloca is set in FunctionDeclarationAst
    return symbol.ir_alloca

  def __repr__(self):
    """
    :rtype: str
    """
    return 'VariableTargetAst(var_identifier=%r)' % self.var_identifier


class MemberTargetAst(TargetAst):
  """
  Target -> Target . identifier
  """
  def __init__(self, pos, parent_target, member_identifier):
    """
    :param TreePosition pos:
    :param TargetAst parent_target:
    :param str member_identifier:
    """
    super().__init__(pos)
    self.parent_target = parent_target
    self.member_identifier = member_identifier

  def make_ptr_type(self, symbol_table):
    """
    :param SymbolTable symbol_table:
    :rtype: Type
    """
    parent_type = self.parent_target.make_ptr_type(symbol_table=symbol_table)
    return self._make_member_val_type(parent_type, member_identifier=self.member_identifier, symbol_table=symbol_table)

  def is_ptr_mutable(self, symbol_table):
    """
    :param SymbolTable symbol_table:
    :rtype: bool
    """
    parent_type = self.parent_target.make_ptr_type(symbol_table=symbol_table)
    assert isinstance(parent_type, StructType)
    assert self.member_identifier in parent_type.member_identifiers
    member_num = parent_type.get_member_num(self.member_identifier)
    member_mutable = parent_type.member_mutables[member_num]
    return self.parent_target.is_ptr_mutable(symbol_table=symbol_table) and member_mutable

  def is_ptr_reassignable(self, symbol_table):
    """
    :param SymbolTable symbol_table:
    :rtype bool:
    """
    return self.parent_target.is_ptr_mutable(symbol_table=symbol_table)

  def make_ir_ptr(self, builder, symbol_table):
    """
    :param ir.IRBuilder builder:
    :param SymbolTable symbol_table:
    :rtype: ir.instructions.Instruction
    """
    parent_type = self.parent_target.make_ptr_type(symbol_table=symbol_table)
    assert isinstance(parent_type, StructType)
    member_num = parent_type.get_member_num(self.member_identifier)
    parent_ptr = self.parent_target.make_ir_ptr(builder=builder, symbol_table=symbol_table)
    if parent_type.is_pass_by_ref():  # parent_ptr has type struct**
      # dereference to get struct*.
      parent_ptr = builder.load(parent_ptr, 'load_struct')
    gep_indices = [ir.Constant(ir.IntType(32), 0), ir.Constant(ir.IntType(32), member_num)]
    return builder.gep(parent_ptr, gep_indices, name='member_ptr_%s' % self.member_identifier)

  def __repr__(self):
    """
    :rtype: str
    """
    return 'MemberTargetAst(parent_target=%r, member_identifier=%r)' % (self.parent_target, self.member_identifier)


class AnnotationAst(AbstractSyntaxTree):
  """
  Annotation.
  """
  def __init__(self, pos, identifier):
    """
    :param TreePosition pos:
    :param str identifier:
    """
    super().__init__(pos)
    self.identifier = identifier
    # TODO: Add type checking for annotation identifiers.

  def __repr__(self):
    """
    :rtype: str
    """
    return 'AnnotationAst(identifier=%r)' % self.identifier


def annotate_ast(ast, annotation_list):
  """
  :param AbstractSyntaxTree ast:
  :param list[AnnotationAst] annotation_list:
  :rtype: AbstractSyntaxTree
  """
  assert len(ast.annotations) == 0
  for annotation in annotation_list:
    if annotation.identifier not in ast.allowed_annotation_identifiers:
      annotation.raise_error('Annotations with name %r not allowed here, only allowed: %s' % (
        annotation.identifier, ', '.join(ast.allowed_annotation_identifiers)))
    if any(annotation.identifier == other.identifier for other in ast.annotations):
      annotation.raise_error('Cannot add annotation with name %r multiple times' % annotation.identifier)
    ast.annotations.append(annotation)
  return ast


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
    'func', 'extern_func', 'struct', 'if', 'else', 'return', 'while', '{', '}', ';', ',', '.', '(', ')',
    '->', '@', 'bool_op', 'sum_op', 'prod_op', '=', 'identifier',
    'int', 'double', 'char',
    None, None
  ], [
    'func', 'extern_func', 'struct', 'if', 'else', 'return', 'while', '{', '}', ';', ',', '\\.', '\\(', '\\)',
    '\\->', '@', '==|!=|<=?|>=?', '\\+|\\-', '\\*|/', '=', '([A-Z]|[a-z]|_)([A-Z]|[a-z]|[0-9]|_)*',
    '(0|[1-9][0-9]*)', '(0|[1-9][0-9]*)\\.[0-9]+', "'([^\']|\\\\[nrt'\"])'",
    '#[^\n]*\n', '[ \n\t]+'
  ])
SLEEPY_GRAMMAR = Grammar(
  Production('TopLevelStmt', 'StmtList'),
  Production('StmtList'),
  Production('StmtList', 'AnnotationList', 'Stmt', 'StmtList'),
  Production('Stmt', 'func', 'identifier', '(', 'TypedIdentifierList', ')', 'ReturnType', '{', 'StmtList', '}'),
  Production('Stmt', 'func', 'Op', '(', 'TypedIdentifierList', ')', 'ReturnType', '{', 'StmtList', '}'),
  Production('Stmt', 'extern_func', 'identifier', '(', 'TypedIdentifierList', ')', 'ReturnType', ';'),
  Production('Stmt', 'struct', 'identifier', '{', 'StmtList', '}'),
  Production('Stmt', 'identifier', '(', 'ExprList', ')', ';'),
  Production('Stmt', 'return', 'ExprList', ';'),
  Production('Stmt', 'Type', 'Target', '=', 'Expr', ';'),
  Production('Stmt', 'Target', '=', 'Expr', ';'),
  Production('Stmt', 'if', 'Expr', '{', 'StmtList', '}'),
  Production('Stmt', 'if', 'Expr', '{', 'StmtList', '}', 'else', '{', 'StmtList', '}'),
  Production('Stmt', 'while', 'Expr', '{', 'StmtList', '}'),
  Production('Expr', 'Expr', 'bool_op', 'SumExpr'),
  Production('Expr', 'SumExpr'),
  Production('SumExpr', 'SumExpr', 'sum_op', 'ProdExpr'),
  Production('SumExpr', 'ProdExpr'),
  Production('ProdExpr', 'ProdExpr', 'prod_op', 'MemberExpr'),
  Production('ProdExpr', 'MemberExpr'),
  Production('MemberExpr', 'MemberExpr', '.', 'identifier'),
  Production('MemberExpr', 'NegExpr'),
  Production('NegExpr', 'sum_op', 'PrimaryExpr'),
  Production('NegExpr', 'PrimaryExpr'),
  Production('PrimaryExpr', 'int'),
  Production('PrimaryExpr', 'double'),
  Production('PrimaryExpr', 'char'),
  Production('PrimaryExpr', 'identifier'),
  Production('PrimaryExpr', 'identifier', '(', 'ExprList', ')'),
  Production('PrimaryExpr', '(', 'Expr', ')'),
  Production('Target', 'identifier'),
  Production('Target', 'Target', '.', 'identifier'),
  Production('AnnotationList'),
  Production('AnnotationList', 'Annotation', 'AnnotationList'),
  Production('Annotation', '@', 'identifier'),
  Production('IdentifierList'),
  Production('IdentifierList', 'IdentifierList+'),
  Production('IdentifierList+', 'identifier'),
  Production('IdentifierList+', 'identifier', ',', 'IdentifierList+'),
  Production('TypedIdentifierList'),
  Production('TypedIdentifierList', 'TypedIdentifierList+'),
  Production('TypedIdentifierList+', 'AnnotationList', 'Type', 'identifier'),
  Production('TypedIdentifierList+', 'AnnotationList', 'Type', 'identifier', ',', 'TypedIdentifierList+'),
  Production('ExprList'),
  Production('ExprList', 'ExprList+'),
  Production('ExprList+', 'Expr'),
  Production('ExprList+', 'Expr', ',', 'ExprList+'),
  Production('Type', 'identifier'),
  Production('ReturnType'),
  Production('ReturnType', '->', 'AnnotationList', 'Type'),
  Production('Op', 'bool_op'),
  Production('Op', 'sum_op'),
  Production('Op', 'prod_op')
)
SLEEPY_ATTR_GRAMMAR = AttributeGrammar(
  SLEEPY_GRAMMAR,
  syn_attrs={
    'ast', 'stmt_list', 'identifier_list', 'type_list', 'val_list', 'identifier', 'type_identifier', 'annotation_list',
    'op', 'number'},
  prod_attr_rules=[
    {'ast': lambda _pos, stmt_list: TopLevelStatementAst(_pos, stmt_list(1))},
    {'stmt_list': []},
    {'stmt_list': lambda ast, annotation_list, stmt_list: [annotate_ast(ast(2), annotation_list(1))] + stmt_list(3)},
    {'ast': lambda _pos, identifier, identifier_list, type_list, annotation_list, type_identifier, stmt_list: (
      FunctionDeclarationAst(_pos, identifier(2), identifier_list(4), type_list(4), annotation_list(4),
        type_identifier(6), annotation_list(6), stmt_list(8)))},
    {'ast': lambda _pos, op, identifier_list, type_list, annotation_list, type_identifier, stmt_list: (
      FunctionDeclarationAst(_pos, op(2), identifier_list(4), type_list(4), annotation_list(4), type_identifier(6),
        annotation_list(6), stmt_list(8)))},
    {'ast': lambda _pos, identifier, identifier_list, type_list, annotation_list, type_identifier: (
      FunctionDeclarationAst(_pos, identifier(2), identifier_list(4), type_list(4), annotation_list(4),
        type_identifier(6), annotation_list(6), None))},
    {'ast': lambda _pos, identifier, stmt_list: StructDeclarationAst(_pos, identifier(2), stmt_list(4))},
    {'ast': lambda _pos, identifier, val_list: CallStatementAst(_pos, identifier(1), val_list(3))},
    {'ast': lambda _pos, val_list: ReturnStatementAst(_pos, val_list(2))},
    {'ast': lambda _pos, ast, type_identifier: AssignStatementAst(_pos, ast(2), ast(4), type_identifier(1))},
    {'ast': lambda _pos, ast: AssignStatementAst(_pos, ast(1), ast(3), None)},
    {'ast': lambda _pos, ast, stmt_list: IfStatementAst(_pos, ast(2), stmt_list(4), [])},
    {'ast': lambda _pos, ast, stmt_list: IfStatementAst(_pos, ast(2), stmt_list(4), stmt_list(8))},
    {'ast': lambda _pos, ast, stmt_list: WhileStatementAst(_pos, ast(2), stmt_list(4))}] + [
    {'ast': lambda _pos, ast, op: BinaryOperatorExpressionAst(_pos, op(2), ast(1), ast(3))},
    {'ast': 'ast.1'}] * 3 + [
    {'ast': lambda _pos, ast, identifier: MemberExpressionAst(_pos, ast(1), identifier(3))},
    {'ast': 'ast.1'},
    {'ast': lambda _pos, ast, op: UnaryOperatorExpressionAst(_pos, op(1), ast(2))},
    {'ast': 'ast.1'},
    {'ast': lambda _pos, number: ConstantExpressionAst(_pos, number(1), SLEEPY_INT)},
    {'ast': lambda _pos, number: ConstantExpressionAst(_pos, number(1), SLEEPY_DOUBLE)},
    {'ast': lambda _pos, number: ConstantExpressionAst(_pos, number(1), SLEEPY_CHAR)},
    {'ast': lambda _pos, identifier: VariableExpressionAst(_pos, identifier(1))},
    {'ast': lambda _pos, identifier, val_list: CallExpressionAst(_pos, identifier(1), val_list(3))},
    {'ast': 'ast.2'},
    {'ast': lambda _pos, identifier: VariableTargetAst(_pos, identifier(1))},
    {'ast': lambda _pos, ast, identifier: MemberTargetAst(_pos, ast(1), identifier(3))},
    {'annotation_list': []},
    {'annotation_list': lambda ast, annotation_list: [ast(1)] + annotation_list(2)},
    {'ast': lambda _pos, identifier: AnnotationAst(_pos, identifier(2))},
    {'identifier_list': []},
    {'identifier_list': 'identifier_list.1'},
    {'identifier_list': lambda identifier: [identifier(1)]},
    {'identifier_list': lambda identifier, identifier_list: [identifier(1)] + identifier_list(3)},
    {'identifier_list': [], 'type_list': [], 'annotation_list': []},
    {'identifier_list': 'identifier_list.1', 'type_list': 'type_list.1', 'annotation_list': 'annotation_list.1'},
    {
      'identifier_list': lambda identifier: [identifier(3)],
      'type_list': lambda type_identifier: [type_identifier(2)],
      'annotation_list': lambda annotation_list: [annotation_list(1)]
    },
    {
      'identifier_list': lambda identifier, identifier_list: [identifier(3)] + identifier_list(5),
      'type_list': lambda type_identifier, type_list: [type_identifier(2)] + type_list(5),
      'annotation_list': lambda annotation_list: [annotation_list(1)] + annotation_list(5)
    },
    {'val_list': []},
    {'val_list': 'val_list.1'},
    {'val_list': lambda ast: [ast(1)]},
    {'val_list': lambda ast, val_list: [ast(1)] + val_list(3)},
    {'type_identifier': 'identifier.1'},
    {'type_identifier': None, 'annotation_list': None},
    {'type_identifier': 'type_identifier.3', 'annotation_list': 'annotation_list.2'},
    {'op': 'op.1'},
    {'op': 'op.1'},
    {'op': 'op.1'}
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
  import os
  preamble_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'std/preamble.slp')
  with open(preamble_path) as preamble_file:
    preamble_program = preamble_file.read()
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
