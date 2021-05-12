

# Operator precedence: * / stronger than + - stronger than == != < <= > >=
from typing import Dict, List

from llvmlite import ir

from sleepy.grammar import SemanticError, Grammar, Production, AttributeGrammar, TreePosition
from sleepy.lexer import LexerGenerator
from sleepy.parser import ParserGenerator
from sleepy.symbols import FunctionSymbol, VariableSymbol, SLEEPY_DOUBLE, Type, SLEEPY_INT, \
  SLEEPY_LONG, SLEEPY_VOID, SLEEPY_DOUBLE_PTR, SLEEPY_BOOL, SLEEPY_CHAR, SymbolTable, TypeSymbol, \
  StructType, ConcreteFunction, UnionType, can_implicit_cast_to, \
  make_implicit_cast_to_ir_val, make_ir_val_is_type, build_initial_ir, CodegenContext

SLOPPY_OP_TYPES = {'*', '/', '+', '-', '==', '!=', '<', '>', '<=', '>', '>=', 'is'}


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
    :param list[ExpressionAst] func_arg_exprs:
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

    for func_arg_expr, arg_type_assertion in zip(func_arg_exprs, concrete_func.arg_type_assertions):
      if isinstance(func_arg_expr, VariableExpressionAst):
        var_symbol = symbol_table[func_arg_expr.var_identifier]
        assert isinstance(var_symbol, VariableSymbol)
        symbol_table[func_arg_expr.var_identifier] = var_symbol.copy_with_asserted_var_type(arg_type_assertion)

    return concrete_func

  def _make_func_call_ir(self, func_identifier, func_arg_exprs, symbol_table, context):
    """
    :param str func_identifier:
    :param list[ExpressionAst] func_arg_exprs:
    :param SymbolTable symbol_table:
    :param CodegenContext context:
    :rtype: ir.values.Value|None
    """
    assert context.emits_ir
    assert func_identifier in symbol_table
    func_symbol = symbol_table[func_identifier]
    if isinstance(func_symbol, TypeSymbol):
      func_symbol = func_symbol.constructor_symbol
    assert isinstance(func_symbol, FunctionSymbol)
    called_types = [arg_expr.make_val_type(symbol_table=symbol_table) for arg_expr in func_arg_exprs]
    assert func_symbol.has_concrete_func(called_types)
    concrete_func = func_symbol.get_concrete_func(called_types)
    assert len(concrete_func.arg_types) == len(func_arg_exprs)
    ir_func_args = []  # type: List[ir.values.Value]
    for func_arg_expr in func_arg_exprs:
      ir_func_arg = func_arg_expr.make_ir_val(symbol_table=symbol_table, context=context)
      ir_func_args.append(ir_func_arg)
    assert len(ir_func_args) == len(func_arg_exprs)
    if concrete_func.is_inline:
      assert callable(concrete_func.make_inline_func_call_ir)
      return concrete_func.make_inline_func_call_ir(ir_func_args=ir_func_args, caller_context=context)
    else:
      ir_func = concrete_func.ir_func
      assert ir_func is not None
      assert len(ir_func.args) == len(func_arg_exprs)
      return context.builder.call(ir_func, ir_func_args, name='call_%s' % func_identifier)

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

  def build_ir(self, symbol_table, context):
    """
    :param SymbolTable symbol_table:
    :param CodegenContext context:
    """
    raise NotImplementedError()

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


class AbstractScopeAst(AbstractSyntaxTree):
  """
  Used to group multiple statements, forming a scope.
  """
  def __init__(self, pos, stmt_list):
    """
    :param TreePosition pos:
    :param list[StatementAst] stmt_list:
    """
    super().__init__(pos)
    self.stmt_list = stmt_list

  def build_scope_ir(self, scope_symbol_table, scope_context):
    """
    :param SymbolTable scope_symbol_table:
    :param CodegenContext scope_context:
    """
    for stmt in self.stmt_list:
      stmt.build_ir(symbol_table=scope_symbol_table, context=scope_context)

  def __repr__(self):
    """
    :rtype: str
    """
    return 'AbstractScopeAst(%s)' % ', '.join([repr(stmt) for stmt in self.stmt_list])


class TopLevelAst(AbstractSyntaxTree):
  """
  TopLevelExpr.
  """

  def __init__(self, pos, root_scope):
    """
    :param TreePosition pos:
    :param AbstractScopeAst root_scope:
    """
    super().__init__(pos)
    self.root_scope = root_scope

  def make_module_ir_and_symbol_table(self, module_name):
    """
    :param str module_name:
    :rtype: (ir.Module,SymbolTable)
    """
    module = ir.Module(name=module_name)
    io_func_type = ir.FunctionType(ir.VoidType(), ())
    ir_io_func = ir.Function(module, io_func_type, name='io')
    root_block = ir_io_func.append_basic_block(name='entry')
    root_builder = ir.IRBuilder(root_block)
    symbol_table = SymbolTable()
    context = CodegenContext(builder=root_builder)

    build_initial_ir(symbol_table=symbol_table, context=context)
    assert symbol_table.ir_func_malloc is not None and symbol_table.ir_func_free is not None
    self.root_scope.build_scope_ir(scope_symbol_table=symbol_table, scope_context=context)

    assert not context.is_terminated
    root_builder.ret_void()
    context.is_terminated = True

    return module, symbol_table

  def __repr__(self):
    """
    :rtype: str
    """
    return 'TopLevelAst(%s)' % self.root_scope


class FunctionDeclarationAst(StatementAst):
  """
  Stmt -> func identifier ( TypedIdentifierList ) Scope
  """

  allowed_annotation_identifiers = {'Inline'}
  allowed_arg_annotation_identifiers = {'Const', 'Mutable'}

  def __init__(self, pos, identifier, arg_identifiers, arg_types, arg_annotations, return_type,
               return_annotation_list, body_scope):
    """
    :param TreePosition pos:
    :param str identifier:
    :param list[str] arg_identifiers:
    :param list[TypeAst] arg_types:
    :param list[list[AnnotationAst]] arg_annotations:
    :param TypeAst|None return_type:
    :param list[AnnotationAst]|None return_annotation_list:
    :param AbstractScopeAst|None body_scope: body, or None if extern function.
    """
    super().__init__(pos)
    assert len(arg_identifiers) == len(arg_types) == len(arg_annotations)
    assert (return_type is None) == (return_annotation_list is None)
    assert body_scope is None or isinstance(body_scope, AbstractScopeAst)
    self.identifier = identifier
    self.arg_identifiers = arg_identifiers
    self.arg_types = arg_types
    self.arg_annotations = arg_annotations
    self.return_type = return_type
    self.return_annotation_list = return_annotation_list
    self.body_scope = body_scope

  @property
  def is_extern(self):
    """
    :rtype: bool
    """
    return self.body_scope is None

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
    arg_types = [arg_type.make_type(symbol_table=symbol_table) for arg_type in self.arg_types]
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

  def build_ir(self, symbol_table, context):
    """
    :param SymbolTable symbol_table:
    :param CodegenContext context:
    """
    arg_types = self.make_arg_types(symbol_table=symbol_table)
    if self.return_type is None:
      return_type = SLEEPY_VOID
    else:
      return_type = self.return_type.make_type(symbol_table=symbol_table)
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
    if self.is_inline and self.is_extern:
      self.raise_error('Extern function %r cannot be inlined' % self.identifier)
    concrete_func = ConcreteFunction(
      None, return_type=return_type, return_mutable=return_mutable, arg_identifiers=self.arg_identifiers,
      arg_types=arg_types, arg_mutables=arg_mutables, arg_type_assertions=arg_types, is_inline=self.is_inline)
    if context.emits_ir and not self.is_inline:
      ir_func_type = concrete_func.make_ir_function_type()
      ir_func_name = symbol_table.make_ir_func_name(self.identifier, self.is_extern, concrete_func)
      concrete_func.ir_func = ir.Function(context.module, ir_func_type, name=ir_func_name)

    func_symbol.add_concrete_func(concrete_func)
    if self.is_extern:
      return

    if self.is_inline:
      def make_inline_func_call_ir(ir_func_args, caller_context):
        """
        :param list[ir.values.Value] ir_func_args:
        :param CodegenContext caller_context:
        :rtype: ir.values.Value|None
        """
        assert len(ir_func_args) == len(self.arg_identifiers)
        assert caller_context.emits_ir
        assert not caller_context.is_terminated
        inline_context = CodegenContext(caller_context.builder)
        if concrete_func.return_type == SLEEPY_VOID:
          return_val_ir_alloca = None
        else:
          return_val_ir_alloca = inline_context.builder.alloca(
            concrete_func.return_type.ir_type, name='return_%s_alloca' % self.identifier)
        collect_block = inline_context.builder.append_basic_block('collect_return_%s_block' % self.identifier)
        self._build_body_ir(
          parent_symbol_table=symbol_table, body_context=inline_context, ir_func_args=ir_func_args,
          inline_return_collect_block=collect_block, inline_return_ir_alloca=return_val_ir_alloca)
        assert inline_context.is_terminated
        assert not collect_block.is_terminated
        collect_context = CodegenContext(ir.IRBuilder(collect_block))
        if concrete_func.return_type == SLEEPY_VOID:
          return_val = None
        else:
          return_val = collect_context.builder.load(return_val_ir_alloca, name='return_%s' % self.identifier)
        caller_context.builder = collect_context.builder
        assert not caller_context.is_terminated
        return return_val

      concrete_func.make_inline_func_call_ir = make_inline_func_call_ir
      # check symbol tables without emitting ir
      self._build_body_ir(parent_symbol_table=symbol_table, body_context=CodegenContext(None))
    else:
      assert not self.is_inline
      if context.emits_ir:
        body_block = concrete_func.ir_func.append_basic_block(name='entry')
        body_context = CodegenContext(ir.IRBuilder(body_block))
      else:
        body_context = CodegenContext(None)
      self._build_body_ir(
        parent_symbol_table=symbol_table, body_context=body_context, ir_func_args=concrete_func.ir_func.args)

  def _get_concrete_func(self, parent_symbol_table):
    """
    :param SymbolTable parent_symbol_table:
    :rtype: ConcreteFunction
    """
    arg_types = self.make_arg_types(symbol_table=parent_symbol_table)
    assert self.identifier in parent_symbol_table
    func_symbol = parent_symbol_table[self.identifier]
    assert isinstance(func_symbol, FunctionSymbol)
    assert func_symbol.has_concrete_func(arg_types)
    return func_symbol.get_concrete_func(arg_types)

  def _build_body_ir(self, parent_symbol_table, body_context, ir_func_args=None, inline_return_collect_block=None,
                     inline_return_ir_alloca=None):
    """
    :param SymbolTable parent_symbol_table: of the parent function, NOT of the caller.
    :param CodegenContext body_context:
    :param list[ir.values.Value]|None ir_func_args:
    :param None|ir.Block inline_return_collect_block:
    :param None|ir.instructions.AllocaInstr inline_return_ir_alloca:
    """
    assert not self.is_extern
    concrete_func = self._get_concrete_func(parent_symbol_table=parent_symbol_table)
    assert self.is_inline == concrete_func.is_inline
    body_symbol_table = parent_symbol_table.copy_with_new_current_func(concrete_func)
    body_symbol_table.current_func_inline_return_collect_block = inline_return_collect_block
    body_symbol_table.current_func_inline_return_ir_alloca = inline_return_ir_alloca

    # add arguments as variables
    for arg_identifier, arg_type, arg_mutable in zip(
        concrete_func.arg_identifiers, concrete_func.arg_types, concrete_func.arg_mutables):
      var_symbol = VariableSymbol(None, arg_type, arg_mutable)
      var_symbol.build_ir_alloca(context=body_context, identifier=arg_identifier)
      assert arg_identifier not in body_symbol_table.current_scope_identifiers
      body_symbol_table.current_scope_identifiers.append(arg_identifier)
      body_symbol_table[arg_identifier] = var_symbol
    # set function argument values
    if body_context.emits_ir:
      assert ir_func_args is not None
      assert len(ir_func_args) == len(concrete_func.arg_identifiers)
      for arg_identifier, ir_arg in zip(concrete_func.arg_identifiers, ir_func_args):
        arg_symbol = body_symbol_table[arg_identifier]
        assert isinstance(arg_symbol, VariableSymbol)
        ir_arg.name = arg_identifier
        assert arg_symbol.ir_alloca is not None
        body_context.builder.store(ir_arg, arg_symbol.ir_alloca)
    # build function body
    self.body_scope.build_scope_ir(scope_symbol_table=body_symbol_table, scope_context=body_context)
    # maybe add implicit return
    if not body_context.is_terminated:
      return_pos = TreePosition(
        self.pos.word,
        self.pos.from_pos if len(self.body_scope.stmt_list) == 0 else self.body_scope.stmt_list[-1].pos.to_pos,
        self.pos.to_pos)
      return_ast = ReturnStatementAst(return_pos, [])
      if concrete_func.return_type != SLEEPY_VOID:
        return_ast.raise_error(
          'Not all branches within function declaration of %r return something' % self.identifier)
      return_ast.build_ir(symbol_table=body_symbol_table, context=body_context)
    assert body_context.is_terminated

  def __repr__(self):
    """
    :rtype: str
    """
    return (
      'FunctionDeclarationAst(identifier=%r, arg_identifiers=%r, arg_types=%r, '
      'return_type=%r, %s)' % (self.identifier, self.arg_identifiers, self.arg_types,
      self.return_type, 'extern' if self.is_extern else self.body_scope))


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

  def build_ir(self, symbol_table, context):
    """
    :param SymbolTable symbol_table:
    :param CodegenContext context:
    """
    if context.emits_ir:
      self._make_func_call_ir(
        func_identifier=self.func_identifier, func_arg_exprs=self.func_arg_exprs,
        symbol_table=symbol_table, context=context)
    else:
      self._check_func_call_symbol_table(
        func_identifier=self.func_identifier, func_arg_exprs=self.func_arg_exprs, symbol_table=symbol_table)

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

  def build_ir(self, symbol_table, context):
    """
    :param SymbolTable symbol_table:
    :param CodegenContext context:
    """
    if symbol_table.current_func is None:
      self.raise_error('Can only use return inside a function declaration')
    if context.is_terminated:
      self.raise_error('Cannot return from function after having returned already')

    if len(self.return_exprs) == 1:
      return_expr = self.return_exprs[0]
      return_val_type = return_expr.make_val_type(symbol_table=symbol_table)
      if return_val_type == SLEEPY_VOID:
        self.raise_error('Cannot use void return value')
      if not can_implicit_cast_to(return_val_type, symbol_table.current_func.return_type):
        if symbol_table.current_func.return_type == SLEEPY_VOID:
          self.raise_error('Function declared to return void, but return value is of type %r' % (
            return_val_type))
        else:
          self.raise_error('Function declared to return type %r, but return value is of type %r' % (
            symbol_table.current_func.return_type, return_val_type))
      return_val_mutable = return_expr.is_val_mutable(symbol_table=symbol_table)
      if not return_val_mutable and symbol_table.current_func.return_mutable:
        self.raise_error(
          'Function declared to return a mutable type %r, but return value is not mutable' % return_val_type)

      if context.emits_ir:
        ir_val = return_expr.make_ir_val(symbol_table=symbol_table, context=context)
        ir_val = make_implicit_cast_to_ir_val(
          return_val_type, symbol_table.current_func.return_type, ir_val, context=context)
        if symbol_table.current_func.is_inline:
          assert symbol_table.current_func_inline_return_ir_alloca is not None
          context.builder.store(ir_val, symbol_table.current_func_inline_return_ir_alloca)
        else:
          context.builder.ret(ir_val)
    else:
      assert len(self.return_exprs) == 0
      if symbol_table.current_func.return_type != SLEEPY_VOID:
        self.raise_error('Function declared to return a value of type %r, but returned void' % (
          symbol_table.current_func.return_type))
      if context.emits_ir:
        if symbol_table.current_func.is_inline:
          assert symbol_table.current_func_inline_return_ir_alloca is None
        else:
          context.builder.ret_void()

    if context.emits_ir and symbol_table.current_func.is_inline:
      collect_block = symbol_table.current_func_inline_return_collect_block
      assert collect_block is not None
      context.builder.branch(collect_block)
      context.builder = ir.IRBuilder(collect_block)
    context.is_terminated = True

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

  def build_ir(self, symbol_table, context):
    """
    :param SymbolTable symbol_table:
    :param CodegenContext context:
    """
    if self.struct_identifier in symbol_table.current_scope_identifiers:
      self.raise_error('Cannot redefine struct with name %r' % self.struct_identifier)
    struct_symbol_table = symbol_table.copy()
    struct_symbol_table.current_scope_identifiers = []
    struct_context = CodegenContext(None)
    for member_num, stmt in enumerate(self.stmt_list):
      if not isinstance(stmt, AssignStatementAst):
        stmt.raise_error('Can only use declare statements within a struct declaration')
      if not isinstance(stmt.var_target, VariableTargetAst):
        stmt.raise_error('Can only declare variables within a struct declaration')
      stmt.build_ir(symbol_table=struct_symbol_table, context=struct_context)
      if len(struct_symbol_table.current_scope_identifiers) != member_num + 1:
        stmt.raise_error(
          'Cannot declare member %r multiple times in struct declaration' % stmt.var_target.var_identifier)
    assert len(self.stmt_list) == len(struct_symbol_table.current_scope_identifiers)
    member_identifiers, member_types = [], []
    for stmt, declared_identifier in zip(self.stmt_list, struct_symbol_table.current_scope_identifiers):
      assert declared_identifier in struct_symbol_table
      declared_symbol = struct_symbol_table[declared_identifier]
      assert isinstance(declared_symbol, VariableSymbol)
      member_identifiers.append(declared_identifier)
      member_types.append(declared_symbol.declared_var_type)
    member_mutables = [False] * len(member_identifiers)
    assert len(member_identifiers) == len(member_types) == len(member_mutables) == len(self.stmt_list)

    struct_type = StructType(
      self.struct_identifier, member_identifiers, member_types, member_mutables, pass_by_ref=self.is_pass_by_ref())
    constructor_symbol = FunctionSymbol()
    constructor = ConcreteFunction(
      ir_func=None, return_type=struct_type, return_mutable=True,
      arg_types=member_types, arg_identifiers=member_identifiers, arg_type_assertions=member_types,
      arg_mutables=member_mutables)
    if context.emits_ir:
      ir_func_type = constructor.make_ir_function_type()
      ir_func_name = symbol_table.make_ir_func_name(self.struct_identifier, extern=False, concrete_func=constructor)
      constructor.ir_func = ir.Function(context.module, ir_func_type, name=ir_func_name)
      self._make_constructor_body_ir(constructor, symbol_table=symbol_table)
    # notice that we explicitly set return_mutable=False here, even if the constructor mutated the struct.
    constructor_symbol.add_concrete_func(constructor)
    symbol_table[self.struct_identifier] = TypeSymbol(struct_type, constructor_symbol=constructor_symbol)
    symbol_table.current_scope_identifiers.append(self.struct_identifier)

  def _make_constructor_body_ir(self, constructor, symbol_table):
    """
    :param ConcreteFunction constructor:
    :param SymbolTable symbol_table:
    """
    # TODO: Make this a new scope.
    struct_type = constructor.return_type
    constructor_block = constructor.ir_func.append_basic_block(name='entry')
    constructor_builder = ir.IRBuilder(constructor_block)
    if self.is_pass_by_ref():  # use malloc
      assert symbol_table.ir_func_malloc is not None
      self_ir_alloca_raw = constructor_builder.call(
        symbol_table.ir_func_malloc, [struct_type.make_ir_size()], name='self_raw_ptr')
      self_ir_alloca = constructor_builder.bitcast(self_ir_alloca_raw, struct_type.ir_type, name='self')
      # TODO: eventually free memory again
    else:  # pass by value, use alloca
      self_ir_alloca = constructor_builder.alloca(struct_type.ir_type, name='self')

    for member_num, (stmt, ir_func_arg) in enumerate(zip(self.stmt_list, constructor.ir_func.args)):
      assert isinstance(stmt, AssignStatementAst)
      assert isinstance(stmt.var_target, VariableTargetAst)
      member_identifier = stmt.var_target.var_identifier
      ir_func_arg.identifier = member_identifier
      gep_indices = (ir.Constant(ir.IntType(32), 0), ir.Constant(ir.IntType(32), member_num))
      member_ptr = constructor_builder.gep(self_ir_alloca, gep_indices, name='%s_ptr' % member_identifier)
      constructor_builder.store(ir_func_arg, member_ptr)

    if self.is_pass_by_ref():
      constructor_builder.ret(self_ir_alloca)
    else:  # pass by value
      constructor_builder.ret(constructor_builder.load(self_ir_alloca, name='self'))

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

  def __init__(self, pos, var_target, var_val, declared_var_type):
    """
    :param TreePosition pos:
    :param TargetAst var_target:
    :param ExpressionAst var_val:
    :param TypeAst|None declared_var_type:
    """
    super().__init__(pos)
    assert isinstance(var_target, TargetAst)
    self.var_target = var_target
    self.var_val = var_val
    self.declared_var_type = declared_var_type

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
    if symbol.declared_var_type != ptr_type:
      self.raise_error('Cannot redefine variable %r of type %r with new type %r' % (
        var_identifier, symbol.declared_var_type, ptr_type))

  def build_ir(self, symbol_table, context):
    """
    :param SymbolTable symbol_table:
    :param CodegenContext context:
    """
    if self.declared_var_type is not None:
      declared_type = self.declared_var_type.make_type(symbol_table=symbol_table)
    else:
      declared_type = None
    val_type = self.var_val.make_val_type(symbol_table=symbol_table)
    if val_type == SLEEPY_VOID:
      self.raise_error('Cannot assign void to variable')
    if declared_type is not None:
      if not can_implicit_cast_to(val_type, declared_type):
        self.raise_error('Cannot assign variable with declared type %r a value of type %r' % (declared_type, val_type))
      val_type = declared_type  # implicitly cast to the declared type
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
      symbol.build_ir_alloca(context=context, identifier=var_identifier)
      symbol_table[var_identifier] = symbol
      symbol_table.current_scope_identifiers.append(var_identifier)
      target_type = val_type
    else:
      # variable name in this scope already declared. just check that types match, but do not change symbol_table.
      existing_type = self.var_target.make_ptr_type(symbol_table=symbol_table)
      assert existing_type is not None
      if not can_implicit_cast_to(val_type, existing_type):
        self.raise_error('Cannot redefine variable of type %r with new type %r' % (existing_type, val_type))
      if declared_type is not None and declared_type != existing_type:
        self.raise_error('Cannot redeclare variable of type %r with new type %r' % (existing_type, declared_type))
      if not self.var_target.is_ptr_reassignable(symbol_table=symbol_table):
        self.raise_error('Cannot reassign member of a non-mutable variable')
      if declared_mutable is None:
        declared_mutable = val_mutable
      if declared_mutable != val_mutable:
        if declared_mutable:
          self.raise_error('Cannot redefine a variable declared as non-mutable to mutable')
        else:
          self.raise_error('Cannot redefine a variable declared as mutable to non-mutable')
      target_type = existing_type
    if declared_mutable and not val_mutable:
      self.raise_error('Cannot assign a non-mutable variable a mutable value of type %r' % declared_type)

    if context.emits_ir:
      ir_val = self.var_val.make_ir_val(symbol_table=symbol_table, context=context)
      ir_val = make_implicit_cast_to_ir_val(val_type, target_type, ir_val, context=context)
      ir_ptr = self.var_target.make_ir_ptr(symbol_table=symbol_table, context=context)
      context.builder.store(ir_val, ir_ptr)

  def __repr__(self):
    """
    :rtype: str
    """
    return 'AssignStatementAst(var_target=%r, var_val=%r, var_type=%r)' % (
      self.var_target, self.var_val, self.declared_var_type)


class IfStatementAst(StatementAst):
  """
  Stmt -> if Expr Scope
        | if Expr Scope else Scope
  """
  def __init__(self, pos, condition_val, true_scope, false_scope):
    """
    :param TreePosition pos:
    :param ExpressionAst condition_val:
    :param AbstractScopeAst true_scope:
    :param AbstractScopeAst|None false_scope:
    """
    super().__init__(pos)
    self.condition_val = condition_val
    if false_scope is None:
      false_scope = AbstractScopeAst(TreePosition(pos.word, pos.to_pos, pos.to_pos), [])
    self.true_scope, self.false_scope = true_scope, false_scope

  def build_ir(self, symbol_table, context):
    """
    :param SymbolTable symbol_table:
    :param CodegenContext context:
    """
    cond_type = self.condition_val.make_val_type(symbol_table=symbol_table)
    if not cond_type == SLEEPY_BOOL:
      self.raise_error('Condition use expression of type %r as if-condition' % cond_type)

    true_symbol_table, false_symbol_table = symbol_table.copy(), symbol_table.copy()
    if context.emits_ir:
      ir_cond = self.condition_val.make_ir_val(symbol_table=symbol_table, context=context)
      true_block = context.builder.append_basic_block('true_branch')  # type: ir.Block
      false_block = context.builder.append_basic_block('false_branch')  # type: ir.Block
      context.builder.cbranch(ir_cond, true_block, false_block)
      true_context = CodegenContext(ir.IRBuilder(true_block))
      false_context = CodegenContext(ir.IRBuilder(false_block))
    else:
      true_context, false_context = CodegenContext(None), CodegenContext(None)

    self.true_scope.build_scope_ir(scope_symbol_table=true_symbol_table, scope_context=true_context)
    self.false_scope.build_scope_ir(scope_symbol_table=false_symbol_table, scope_context=false_context)

    if true_context.is_terminated and false_context.is_terminated:
      context.is_terminated = True
      if context.emits_ir:
        context.builder = None
    else:
      assert not true_context.is_terminated or not false_context.is_terminated
      if context.emits_ir:
        continue_block = context.builder.append_basic_block('continue_branch')  # type: ir.Block
        if not true_context.is_terminated:
          true_context.builder.branch(continue_block)
        if not false_context.is_terminated:
          false_context.builder.branch(continue_block)
        context.builder = ir.IRBuilder(continue_block)

  def __repr__(self):
    """
    :rtype: str
    """
    return 'IfStatementAst(condition_val=%r, true_scope=%r, false_scope=%r)' % (
      self.condition_val, self.true_scope, self.false_scope)


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

  def build_ir(self, symbol_table, context):
    """
    :param SymbolTable symbol_table:
    :param CodegenContext context:
    """
    # TODO: Make this a separate scope. Also see IfExpressionAst.
    cond_type = self.condition_val.make_val_type(symbol_table=symbol_table)
    if not cond_type == SLEEPY_BOOL:
      self.raise_error('Condition use expression of type %r as while-condition' % cond_type)

    if context.emits_ir:
      cond_ir = self.condition_val.make_ir_val(symbol_table=symbol_table, context=context)
      body_block = context.builder.append_basic_block('while_body')  # type: ir.Block
      continue_block = context.builder.append_basic_block('continue_branch')  # type: ir.Block
      context.builder.cbranch(cond_ir, body_block, continue_block)
      body_context = CodegenContext(ir.IRBuilder(body_block))
      context.builder = ir.IRBuilder(continue_block)
    else:
      body_context = CodegenContext(None)
    assert context.emits_ir == body_context.emits_ir

    body_symbol_table = symbol_table.copy()
    for stmt in self.stmt_list:
      stmt.build_ir(symbol_table=body_symbol_table, context=body_context)
    if not body_context.is_terminated and body_context.emits_ir:
      body_cond_ir = self.condition_val.make_ir_val(symbol_table=symbol_table, context=body_context)
      body_context.builder.cbranch(body_cond_ir, body_block, continue_block)

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

  def make_ir_val(self, symbol_table, context):
    """
    :param SymbolTable symbol_table:
    :param CodegenContext context:
    :rtype: ir.values.Value
    :return: The value this expression is evaluated to
    """
    assert context.emits_ir
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
    if self.op == 'is':
      assert isinstance(self.right_expr, VariableExpressionAst)

  def make_val_type(self, symbol_table):
    """
    :param SymbolTable symbol_table:
    :rtype: Type
    """
    if self.op == 'is':
      assert isinstance(self.right_expr, VariableExpressionAst)
      type_expr = IdentifierTypeAst(self.right_expr.pos, self.right_expr.var_identifier)
      type_expr.make_type(symbol_table=symbol_table)  # just check that type exists
      return SLEEPY_BOOL
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

  def make_ir_val(self, symbol_table, context):
    """
    :param SymbolTable symbol_table:
    :param CodegenContext context:
    :rtype: ir.values.Value
    """
    assert context.emits_ir
    if self.op == 'is':
      assert isinstance(self.right_expr, VariableExpressionAst)
      check_type_expr = IdentifierTypeAst(self.right_expr.pos, self.right_expr.var_identifier)
      check_type = check_type_expr.make_type(symbol_table=symbol_table)
      val_type = self.left_expr.make_val_type(symbol_table=symbol_table)
      ir_val = self.left_expr.make_ir_val(symbol_table=symbol_table, context=context)
      return make_ir_val_is_type(ir_val, val_type, check_type, context=context)
    operand_exprs = [self.left_expr, self.right_expr]
    return_val = self._make_func_call_ir(
      func_identifier=self.op, func_arg_exprs=operand_exprs, symbol_table=symbol_table, context=context)
    assert return_val is not None
    return return_val

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

  def make_ir_val(self, symbol_table, context):
    """
    :param SymbolTable symbol_table:
    :param CodegenContext context:
    :rtype: ir.values.Value
    """
    assert context.emits_ir
    operand_exprs = [self.expr]
    return self._make_func_call_ir(
      func_identifier=self.op, func_arg_exprs=operand_exprs, symbol_table=symbol_table, context=context)

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

  def make_ir_val(self, symbol_table, context):
    """
    :param SymbolTable symbol_table:
    :param CodegenContext context:
    :rtype: ir.values.Value
    """
    assert context.emits_ir
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
    return self.get_var_symbol(symbol_table=symbol_table).declared_var_type

  def is_val_mutable(self, symbol_table):
    """
    :param SymbolTable symbol_table:
    :rtype: Type
    """
    return self.get_var_symbol(symbol_table=symbol_table).mutable

  def make_ir_val(self, symbol_table, context):
    """
    :param SymbolTable symbol_table:
    :param CodegenContext context:
    :rtype: ir.values.Value
    """
    assert context.emits_ir
    symbol = self.get_var_symbol(symbol_table=symbol_table)
    return context.builder.load(symbol.ir_alloca, name=self.var_identifier)

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

  def make_ir_val(self, symbol_table, context):
    """
    :param SymbolTable symbol_table:
    :param CodegenContext context:
    :rtype: ir.values.Value
    """
    assert context.emits_ir
    return self._make_func_call_ir(
      func_identifier=self.func_identifier, func_arg_exprs=self.func_arg_exprs,
      symbol_table=symbol_table, context=context)

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

  def make_ir_val(self, symbol_table, context):
    """
    :param SymbolTable symbol_table:
    :param CodegenContext context:
    :rtype: ir.values.Value
    """
    assert context.emits_ir
    parent_type = self.parent_val_expr.make_val_type(symbol_table=symbol_table)
    assert isinstance(parent_type, StructType)
    parent_ir_val = self.parent_val_expr.make_ir_val(symbol_table=symbol_table, context=context)
    if parent_type.is_pass_by_ref():
      member_num = parent_type.get_member_num(self.member_identifier)
      gep_indices = (ir.Constant(ir.IntType(32), 0), ir.Constant(ir.IntType(32), member_num))
      member_ptr = context.builder.gep(parent_ir_val, gep_indices, name='member_ptr_%s' % self.member_identifier)
      return context.builder.load(member_ptr, name='member_%s' % self.member_identifier)
    else:  # pass by value
      return context.builder.extract_value(
          parent_ir_val, parent_type.get_member_num(self.member_identifier), name='member_%s' % self.member_identifier)

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

  def make_ir_ptr(self, symbol_table, context):
    """
    :param SymbolTable symbol_table:
    :param CodegenContext context:
    :rtype: ir.instructions.Instruction
    """
    assert context.emits_ir
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
    return symbol.asserted_var_type

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

  def make_ir_ptr(self, symbol_table, context):
    """
    :param SymbolTable symbol_table:
    :param CodegenContext context:
    :rtype: ir.instructions.Instruction
    """
    assert context.emits_ir
    assert self.var_identifier in symbol_table
    symbol = symbol_table[self.var_identifier]
    assert isinstance(symbol, VariableSymbol)
    assert symbol.ir_alloca is not None
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

  def make_ir_ptr(self, symbol_table, context):
    """
    :param SymbolTable symbol_table:
    :param CodegenContext context:
    :rtype: ir.instructions.Instruction
    """
    assert context.emits_ir
    parent_type = self.parent_target.make_ptr_type(symbol_table=symbol_table)
    assert isinstance(parent_type, StructType)
    member_num = parent_type.get_member_num(self.member_identifier)
    parent_ptr = self.parent_target.make_ir_ptr(symbol_table=symbol_table, context=context)
    if parent_type.is_pass_by_ref():  # parent_ptr has type struct**
      # dereference to get struct*.
      parent_ptr = context.builder.load(parent_ptr, 'load_struct')
    gep_indices = [ir.Constant(ir.IntType(32), 0), ir.Constant(ir.IntType(32), member_num)]
    return context.builder.gep(parent_ptr, gep_indices, name='member_ptr_%s' % self.member_identifier)

  def __repr__(self):
    """
    :rtype: str
    """
    return 'MemberTargetAst(parent_target=%r, member_identifier=%r)' % (self.parent_target, self.member_identifier)


class TypeAst(AbstractSyntaxTree):
  """
  Type.
  """
  def __init__(self, pos):
    """
    :param TreePosition pos:
    """
    super().__init__(pos)

  def make_type(self, symbol_table):
    """
    :param SymbolTable symbol_table:
    :rtype: Type
    """
    raise NotImplementedError()

  def __repr__(self):
    return 'TypeAst()'


class IdentifierTypeAst(TypeAst):
  """
  IdentifierType -> identifier.
  """
  def __init__(self, pos, type_identifier):
    """
    :param TreePosition pos:
    :param str type_identifier:
    """
    super().__init__(pos)
    self.type_identifier = type_identifier

  def make_type(self, symbol_table):
    """
    :param SymbolTable symbol_table:
    :rtype: Type
    """
    if self.type_identifier not in symbol_table:
      self.raise_error('Unknown type identifier %r' % self.type_identifier)
    type_symbol = symbol_table[self.type_identifier]
    if not isinstance(type_symbol, TypeSymbol):
      self.raise_error('%r is not a type, but a %r' % (self.type_identifier, type(type_symbol)))
    return type_symbol.type

  def __repr__(self):
    """
    :rtype: str
    """
    return 'IdentifierType(type_identifier=%r)' % self.type_identifier


class UnionTypeAst(TypeAst):
  """
  IdentifierType -> identifier.
  """
  def __init__(self, pos, variant_types):
    """
    :param TreePosition pos:
    :param list[TypeAst] variant_types:
    """
    super().__init__(pos)
    self.variant_types = variant_types

  def make_type(self, symbol_table):
    """
    :param SymbolTable symbol_table:
    :rtype: Type
    """
    if not all(isinstance(variant_type, IdentifierTypeAst) for variant_type in self.variant_types):
      self.raise_error('Union types cannot be nested')
    concrete_variant_types = [variant_type.make_type(symbol_table=symbol_table) for variant_type in self.variant_types]
    return UnionType(concrete_variant_types, list(range(len(concrete_variant_types))))

  def __repr__(self):
    """
    :rtype: str
    """
    return 'UnionTypeAst(variant_types=%r)' % self.variant_types


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
    'func', 'extern_func', 'struct', 'if', 'else', 'return', 'while', '{', '}', ';', ',', '.', '(', ')', '|',
    '->', '@', 'cmp_op', 'sum_op', 'prod_op', '=', 'identifier',
    'int', 'double', 'char',
    None, None
  ], [
    'func', 'extern_func', 'struct', 'if', 'else', 'return', 'while', '{', '}', ';', ',', '\\.', '\\(', '\\)', '\\|',
    '\\->', '@', '==|!=|<=?|>=?|is', '\\+|\\-', '\\*|/', '=', '([A-Z]|[a-z]|_)([A-Z]|[a-z]|[0-9]|_)*',
    '(0|[1-9][0-9]*)', '(0|[1-9][0-9]*)\\.[0-9]+', "'([^\']|\\\\[nrt'\"])'",
    '#[^\n]*\n', '[ \n\t]+'
  ])
SLEEPY_GRAMMAR = Grammar(
  Production('TopLevelStmt', 'StmtList'),
  Production('Scope', '{', 'StmtList', '}'),
  Production('StmtList'),
  Production('StmtList', 'AnnotationList', 'Stmt', 'StmtList'),
  Production('Stmt', 'func', 'identifier', '(', 'TypedIdentifierList', ')', 'ReturnType', 'Scope'),
  Production('Stmt', 'func', 'Op', '(', 'TypedIdentifierList', ')', 'ReturnType', 'Scope'),
  Production('Stmt', 'extern_func', 'identifier', '(', 'TypedIdentifierList', ')', 'ReturnType', ';'),
  Production('Stmt', 'struct', 'identifier', '{', 'StmtList', '}'),
  Production('Stmt', 'identifier', '(', 'ExprList', ')', ';'),
  Production('Stmt', 'return', 'ExprList', ';'),
  Production('Stmt', 'Type', 'Target', '=', 'Expr', ';'),
  Production('Stmt', 'Target', '=', 'Expr', ';'),
  Production('Stmt', 'if', 'Expr', 'Scope'),
  Production('Stmt', 'if', 'Expr', 'Scope', 'else', 'Scope'),
  Production('Stmt', 'while', 'Expr', '{', 'StmtList', '}'),
  Production('Expr', 'Expr', 'cmp_op', 'SumExpr'),
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
  Production('Type', 'Type', '|', 'IdentifierType'),
  Production('Type', 'IdentifierType'),
  Production('IdentifierType', 'identifier'),
  Production('IdentifierType', '(', 'Type', ')'),
  Production('ReturnType'),
  Production('ReturnType', '->', 'AnnotationList', 'Type'),
  Production('Op', 'cmp_op'),
  Production('Op', 'sum_op'),
  Production('Op', 'prod_op')
)
SLEEPY_ATTR_GRAMMAR = AttributeGrammar(
  SLEEPY_GRAMMAR,
  syn_attrs={
    'ast', 'stmt_list', 'identifier_list', 'type_list', 'val_list', 'identifier', 'annotation_list',
    'op', 'number'},
  prod_attr_rules=[
    {'ast': lambda _pos, stmt_list: TopLevelAst(_pos, AbstractScopeAst(_pos, stmt_list(1)))},
    {'ast': lambda _pos, stmt_list: AbstractScopeAst(_pos, stmt_list(2))},
    {'stmt_list': []},
    {'stmt_list': lambda ast, annotation_list, stmt_list: [annotate_ast(ast(2), annotation_list(1))] + stmt_list(3)},
    {'ast': lambda _pos, identifier, identifier_list, type_list, annotation_list, ast: (
      FunctionDeclarationAst(
        _pos, identifier(2), identifier_list(4), type_list(4), annotation_list(4), ast(6), annotation_list(6),
        ast(7)))},
    {'ast': lambda _pos, op, identifier_list, type_list, annotation_list, ast: (
      FunctionDeclarationAst(
        _pos, op(2), identifier_list(4), type_list(4), annotation_list(4), ast(6), annotation_list(6), ast(7)))},
    {'ast': lambda _pos, identifier, identifier_list, type_list, annotation_list, ast: (
      FunctionDeclarationAst(_pos, identifier(2), identifier_list(4), type_list(4), annotation_list(4),
        ast(6), annotation_list(6), None))},
    {'ast': lambda _pos, identifier, stmt_list: StructDeclarationAst(_pos, identifier(2), stmt_list(4))},
    {'ast': lambda _pos, identifier, val_list: CallStatementAst(_pos, identifier(1), val_list(3))},
    {'ast': lambda _pos, val_list: ReturnStatementAst(_pos, val_list(2))},
    {'ast': lambda _pos, ast: AssignStatementAst(_pos, ast(2), ast(4), ast(1))},
    {'ast': lambda _pos, ast: AssignStatementAst(_pos, ast(1), ast(3), None)},
    {'ast': lambda _pos, ast: IfStatementAst(_pos, ast(2), ast(3), None)},
    {'ast': lambda _pos, ast: IfStatementAst(_pos, ast(2), ast(3), ast(5))},
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
      'type_list': lambda ast: [ast(2)],
      'annotation_list': lambda annotation_list: [annotation_list(1)]
    },
    {
      'identifier_list': lambda identifier, identifier_list: [identifier(3)] + identifier_list(5),
      'type_list': lambda ast, type_list: [ast(2)] + type_list(5),
      'annotation_list': lambda annotation_list: [annotation_list(1)] + annotation_list(5)
    },
    {'val_list': []},
    {'val_list': 'val_list.1'},
    {'val_list': lambda ast: [ast(1)]},
    {'val_list': lambda ast, val_list: [ast(1)] + val_list(3)},
    {'ast': lambda _pos, ast: UnionTypeAst(_pos, [ast(1), ast(3)])},
    {'ast': 'ast.1'},
    {'ast': lambda _pos, identifier: IdentifierTypeAst(_pos, identifier(1))},
    {'ast': 'ast.2'},
    {'ast': None, 'annotation_list': None},
    {'ast': 'ast.3', 'annotation_list': 'annotation_list.2'},
    {'op': 'op.1'},
    {'op': 'op.1'},
    {'op': 'op.1'}
  ],
  terminal_attr_rules={
    'cmp_op': {'op': lambda value: value},
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
  :rtype: TopLevelAst
  """
  tokens, tokens_pos = SLEEPY_LEXER.tokenize(program)
  _, root_eval = SLEEPY_PARSER.parse_syn_attr_analysis(SLEEPY_ATTR_GRAMMAR, program, tokens, tokens_pos)
  program_ast = root_eval['ast']
  assert isinstance(program_ast, TopLevelAst)
  if add_preamble:
    program_ast = add_preamble_to_ast(program_ast)
  return program_ast


def make_preamble_ast():
  """
  :rtype: TopLevelAst
  """
  import os
  preamble_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'std/preamble.slp')
  with open(preamble_path) as preamble_file:
    preamble_program = preamble_file.read()
  return make_program_ast(preamble_program, add_preamble=False)


def add_preamble_to_ast(program_ast):
  """
  :param TopLevelAst program_ast:
  :rtype: TopLevelAst
  """
  preamble_ast = make_preamble_ast()
  assert isinstance(preamble_ast, TopLevelAst)
  return TopLevelAst(program_ast.pos, AbstractScopeAst(
    preamble_ast.pos, preamble_ast.root_scope.stmt_list + program_ast.root_scope.stmt_list))

