from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Any
from abc import ABC, abstractmethod

from llvmlite import ir

from sleepy.errors import SemanticError
from sleepy.grammar import TreePosition
from sleepy.symbols import FunctionSymbol, VariableSymbol, Type, SymbolTable, \
  TypeTemplateSymbol, StructType, ConcreteFunction, UnionType, can_implicit_cast_to, \
  make_ir_val_is_type, CodegenContext, get_common_type, \
  PlaceholderTemplateType, try_infer_templ_types, Symbol, FunctionSymbolCaller, SLEEPY_VOID, TypedValue, ReferenceType
from sleepy.builtin_symbols import SLEEPY_BOOL, SLEEPY_LONG, SLEEPY_CHAR, SLEEPY_CHAR_PTR, build_initial_ir

# Operator precedence: * / stronger than + - stronger than == != < <= > >=
SLOPPY_OP_TYPES = {'*', '/', '+', '-', '==', '!=', '<', '>', '<=', '>', '>=', 'is', '='}


class AbstractSyntaxTree(ABC):
  """
  Abstract syntax tree of a sleepy program.
  """

  allowed_annotation_identifiers = frozenset()

  def __init__(self, pos):
    """
    :param TreePosition pos: position where this AST starts
    """
    self.pos = pos
    self.annotations: List[AnnotationAst] = []

  def __repr__(self) -> str:
    return 'AbstractSyntaxTree'

  def raise_error(self, message: str):
    raise SemanticError(self.pos.word, self.pos.from_pos, self.pos.to_pos, message)

  def _resolve_possible_concrete_funcs(self,
                                       func_caller: FunctionSymbolCaller,
                                       calling_types: List[Type]) -> List[ConcreteFunction]:
    func, templ_types = func_caller.func, func_caller.templ_types
    if templ_types is None:
      templ_types = self._infer_templ_args(func=func, calling_types=calling_types)
    assert templ_types is not None
    assert all(not templ_type.has_templ_placeholder() for templ_type in templ_types)

    if not func.can_call_with_arg_types(concrete_templ_types=templ_types, arg_types=calling_types):
      self.raise_error(
        'Cannot call function %r with arguments of types %r and template parameters %r, '
        'only declared for parameter types:\n%s' % (
          func.identifier, ', '.join([str(calling_type) for calling_type in calling_types]),
          ', '.join([str(templ_type) for templ_type in templ_types]),
          func.make_signature_list_str()))
    if not all(called_type.is_realizable() for called_type in calling_types):
      self.raise_error('Cannot call function %r with argument of types %r which are unrealizable' % (
        func.identifier, ', '.join([str(called_type) for called_type in calling_types])))

    possible_concrete_funcs = func.get_concrete_funcs(templ_types=templ_types, arg_types=calling_types)
    assert len(possible_concrete_funcs) >= 1
    return possible_concrete_funcs

  def _infer_templ_args(self, func: FunctionSymbol, calling_types: List[Type]) -> List[Type]:
    # TODO: We currently require that the template types must be statically determinable.
    # e.g. assume that our function takes a union argument.
    # If we have overloaded signatures each with template arguments, it can happen that you should use different
    # template arguments depending on which union variant we are called with.
    # Currently, this would fail. But we could support this if we wanted to.
    # Note that that would make function calls with inferred template types more than just an auto-evaluated type arg.
    assert all(calling_type.is_realizable() for calling_type in calling_types)
    signature_templ_types = None
    for expanded_calling_types in func.iter_expanded_possible_arg_types(calling_types):
      infers = [
        try_infer_templ_types(
          calling_types=expanded_calling_types, signature_types=signature.arg_types,
          placeholder_templ_types=signature.placeholder_templ_types)
        for signature in func.signatures]
      possible_infers = [idx for idx, infer in enumerate(infers) if infer is not None]
      if len(possible_infers) == 0:
        self.raise_error(
          'Cannot infer template types for function %r from arguments of types %r, '
          'is declared for parameter types:\n%s\n\nSpecify the template types explicitly.' % (
            func.identifier, ', '.join([str(calling_type) for calling_type in calling_types]),
            func.make_signature_list_str()))
      if len(possible_infers) > 1:
        self.raise_error(
          'Cannot uniquely infer template types for function %r from arguments of types %r, '
          'is declared for parameter types:\n%s' % (
            func.identifier, ', '.join([str(calling_type) for calling_type in calling_types]),
            func.make_signature_list_str()))
      assert len(possible_infers) == 1
      expanded_signature_templ_types = infers[possible_infers[0]]
      assert expanded_signature_templ_types is not None
      if signature_templ_types is not None and signature_templ_types != expanded_signature_templ_types:
        self.raise_error(
          'Cannot uniquely statically infer template types for function %r from arguments of types %r '
          'because different expanded union types would require different template types. '
          'Function is declared for parameter types:\n%s' % (
            func.identifier, ', '.join([str(calling_type) for calling_type in calling_types]),
            func.make_signature_list_str()))
      signature_templ_types = expanded_signature_templ_types
    assert signature_templ_types is not None
    return signature_templ_types

  def build_func_call(self,
                      func_caller: FunctionSymbolCaller,
                      func_arg_exprs: List[ExpressionAst],
                      symbol_table: SymbolTable,
                      context: CodegenContext) -> TypedValue:
    # TODO: func_arg_exprs should be a List[TypedValue]. But then, type narrowing and assert(...) for types wouldn't
    # work right now.
    func, templ_types = func_caller.func, func_caller.templ_types
    func_args = [arg_expr.make_as_val(symbol_table=symbol_table, context=context) for arg_expr in func_arg_exprs]
    collapsed_func_args = [arg.copy_collapse(context=None) for num, arg in enumerate(func_args)]
    calling_types = [arg.narrowed_type for arg in collapsed_func_args]
    possible_concrete_funcs = self._resolve_possible_concrete_funcs(
      func_caller=func_caller, calling_types=calling_types)
    return_type = get_common_type([concrete_func.return_type for concrete_func in possible_concrete_funcs])

    if templ_types is None:
      templ_types = self._infer_templ_args(func=func, calling_types=calling_types)
    assert templ_types is not None

    for concrete_func in possible_concrete_funcs:
      if concrete_func in context.inline_func_call_stack:
        self.raise_error(
          'An inlined function can not call itself (indirectly), but got inline call stack: %s -> %s' % (
            ' -> '.join(str(inline_func) for inline_func in context.inline_func_call_stack), concrete_func))
      for arg_identifier, arg_mutates, arg in zip(concrete_func.arg_identifiers, concrete_func.arg_mutates, func_args):
        if arg_mutates and not arg.is_referenceable():
          self.raise_error('Cannot call function %s%s mutating parameter %r with non-referencable argument' % (
            func.identifier, concrete_func.signature.to_signature_str(), arg_identifier))

    if context.emits_ir:
      from sleepy.symbols import make_func_call_ir
      return_ir_val = make_func_call_ir(
        func=func, templ_types=templ_types, func_args=func_args, context=context)
    else:
      return_ir_val = None

    # apply type narrowings
    for arg_num, func_arg_expr in enumerate(func_arg_exprs):
      original_unbound_arg_type = func_args[arg_num].narrowed_type
      original_bound_arg_type = collapsed_func_args[arg_num].narrowed_type
      narrowed_arg_types = [concrete_func.arg_type_narrowings[arg_num] for concrete_func in possible_concrete_funcs]
      narrowed_arg_type = get_common_type(narrowed_arg_types)
      bound_narrowed_arg_type = original_unbound_arg_type.replace_types({original_bound_arg_type: narrowed_arg_type})
      if isinstance(func_arg_expr, IdentifierExpressionAst):
        var_symbol = symbol_table[func_arg_expr.identifier]
        assert isinstance(var_symbol, VariableSymbol)
        symbol_table[func_arg_expr.identifier] = var_symbol.copy_narrow_type(bound_narrowed_arg_type)

    # special handling of 'assert' call
    if func in {symbol_table.inbuilt_symbols.get(identifier) for identifier in {'assert', 'unchecked_assert'}}:
      assert len(func_arg_exprs) >= 1
      condition_expr = func_arg_exprs[0]
      make_narrow_type_from_valid_cond_ast(condition_expr, cond_holds=True, symbol_table=symbol_table)
    return TypedValue(typ=return_type, ir_val=return_ir_val)

  def build_func_call_by_identifier(self,
                                    func_identifier: str,
                                    templ_types: Optional[List[Type]],
                                    func_arg_exprs: List[ExpressionAst],
                                    symbol_table: SymbolTable,
                                    context: CodegenContext) -> TypedValue:
    if func_identifier not in symbol_table:
      self.raise_error('Function %r called before declared' % func_identifier)
    symbol = symbol_table[func_identifier]
    if not isinstance(symbol, FunctionSymbol):
      self.raise_error('Cannot call non-function %r' % func_identifier)
    func_caller = FunctionSymbolCaller(func=symbol, templ_types=templ_types)
    return self.build_func_call(
      func_caller=func_caller, func_arg_exprs=func_arg_exprs, symbol_table=symbol_table, context=context)

  @abstractmethod
  def children(self) -> List[AbstractSyntaxTree]:
    pass

  def _collect_placeholder_templ_types(self, templ_identifiers: List[str],
                                       symbol_table: SymbolTable) -> List[PlaceholderTemplateType]:
    templ_types = []
    for templ_type_identifier in templ_identifiers:
      if templ_type_identifier in symbol_table.current_scope_identifiers:
        self.raise_error('Cannot declare template variable %r multiple times' % templ_type_identifier)
      template_type = PlaceholderTemplateType(templ_type_identifier)
      templ_types.append(template_type)
      template_type_symbol = TypeTemplateSymbol(template_parameters=[], signature_type=template_type)
      symbol_table[templ_type_identifier] = template_type_symbol
    return templ_types

  def check_pos_validity(self) -> bool:
    children = self.children()
    if len(children) == 0:
      return True
    if children[0].pos.from_pos < self.pos.from_pos or children[-1].pos.to_pos > self.pos.to_pos:
      return False

    for pre, post in [(x, x + 1) for x in range(len(children) - 1)]:
      if children[pre].pos.to_pos > children[post].pos.from_pos:
        return False

    return all(child.check_pos_validity() for child in children)


class StatementAst(AbstractSyntaxTree, ABC):
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

  def __repr__(self) -> str:
    return 'StatementAst'


class AbstractScopeAst(AbstractSyntaxTree):
  """
  Used to group multiple statements, forming a scope.
  """
  def __init__(self, pos: TreePosition, stmt_list: List[StatementAst]):
    """
    :param TreePosition pos:
    :param list[StatementAst] stmt_list:
    """
    super().__init__(pos)
    self.stmt_list = stmt_list

  def build_scope_ir(self, scope_symbol_table: SymbolTable, scope_context: CodegenContext):
    with scope_context.use_pos(self.pos):
      for stmt in self.stmt_list:
        if scope_context.is_terminated:
          stmt.raise_error('Code is unreachable')
        stmt.build_ir(symbol_table=scope_symbol_table, context=scope_context)

  def __repr__(self) -> str:
    return 'AbstractScopeAst(%s)' % ', '.join([repr(stmt) for stmt in self.stmt_list])

  def children(self):
    return self.stmt_list


class FileAst(AbstractSyntaxTree):
  def __init__(self, pos: TreePosition,
               stmt_list: List[StatementAst],
               imports_ast: ImportsAst,
               file_path: Optional[Path] = None):
    super().__init__(pos)
    self.imports_ast = imports_ast
    self.stmt_list = stmt_list
    self.file_path = file_path

  def children(self) -> List[AbstractSyntaxTree]:
    return [self.imports_ast] + self.stmt_list

  def __repr__(self) -> str:
    return 'TopLevelAst(%s)' % self.stmt_list


class ImportsAst(AbstractSyntaxTree):
  def __init__(self, pos: TreePosition, import_asts: List[ImportAst]):
    super().__init__(pos)
    self.import_asts = import_asts

  @property
  def imports(self) -> List[str]:
    return [a.path for a in self.import_asts]

  def children(self) -> List[AbstractSyntaxTree]:
    return self.import_asts


class ImportAst(AbstractSyntaxTree):
  def __init__(self, pos: TreePosition, path: str):
    super().__init__(pos)
    self.path = path

  def children(self) -> List[AbstractSyntaxTree]:
    return []


class TranslationUnitAst(AbstractSyntaxTree):

  def __init__(self, pos: TreePosition, file_asts: List[FileAst]):
    self.file_asts = file_asts
    super().__init__(pos)

  @staticmethod
  def from_file_asts(asts: List[FileAst]) -> TranslationUnitAst:
    assert len(asts) > 0, "Must have at least one child ast"
    return TranslationUnitAst(
      TreePosition(
        "".join((ast.pos.word for ast in asts)),
        asts[0].pos.from_pos,
        sum(ast.pos.to_pos for ast in asts)
      ),
      asts
    )

  def make_module_ir_and_symbol_table(self, module_name: str,
                                      emit_debug: bool,
                                      main_file_path: Optional[Path] = None) -> (ir.Module, SymbolTable):
    assert self.check_pos_validity()

    module = ir.Module(name=module_name)

    root_builder = ir.IRBuilder()
    symbol_table = SymbolTable()
    context = CodegenContext(
      builder=root_builder, module=module, emits_debug=emit_debug,
      source_path=main_file_path)

    build_initial_ir(symbol_table=symbol_table, context=context)
    assert context.ir_func_malloc is not None
    for ast in self.file_asts:
      context.change_file(ast.file_path)
      for statement in ast.stmt_list:
        assert not context.is_terminated
        statement.build_ir(symbol_table=symbol_table, context=context)
    assert not context.is_terminated
    context.is_terminated = True

    return module, symbol_table

  def children(self) -> List[AbstractSyntaxTree]:
    return self.file_asts

  def check_pos_validity(self) -> bool:
    return all(child.check_pos_validity() for child in self.children())


class ExpressionStatementAst(StatementAst):
  """
  Stmt -> Expr
  """
  def __init__(self, pos: TreePosition, expr):
    """
    :param TreePosition pos:
    :param ExpressionAst expr:
    """
    super().__init__(pos)
    assert isinstance(expr, ExpressionAst)
    self.expr = expr

  def build_ir(self, symbol_table, context):
    """
    :param SymbolTable symbol_table:
    :param CodegenContext context:
    """
    with context.use_pos(self.pos):
      self.expr.make_as_val(symbol_table=symbol_table, context=context)

  def children(self) -> List[AbstractSyntaxTree]:
    return [self.expr]

  def __repr__(self) -> str:
    return 'ExpressionStatementAst(expr=%r)' % self.expr


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
    if len(return_exprs) > 1:
      self.raise_error('Returning multiple values not support yet')

  def build_ir(self, symbol_table: SymbolTable, context: CodegenContext):
    with context.use_pos(self.pos):
      if symbol_table.current_func is None:
        self.raise_error('Can only use return inside a function declaration')
      if context.is_terminated:
        self.raise_error('Cannot return from function after having returned already')

      if len(self.return_exprs) == 1:
        return_expr = self.return_exprs[0]
        return_val = return_expr.make_as_val(symbol_table=symbol_table, context=context)
        if return_val.type == SLEEPY_VOID:
          self.raise_error('Cannot use void return value')
        return_val = return_val.copy_collapse(context=context, name='return_val')
        if not can_implicit_cast_to(return_val.narrowed_type, symbol_table.current_func.return_type):
          can_implicit_cast_to(return_val.narrowed_type, symbol_table.current_func.return_type)
          if symbol_table.current_func.return_type == SLEEPY_VOID:
            self.raise_error(
              'Function declared to return void, but return value is of type %r' % return_val.narrowed_type)
          else:
            self.raise_error('Function declared to return type %r, but return value is of type %r' % (
              symbol_table.current_func.return_type, return_val.narrowed_type))

        if context.emits_ir:
          ir_val = return_val.copy_with_implicit_cast(
            to_type=symbol_table.current_func.return_type, context=context, name='return_val_cast').ir_val
          if symbol_table.current_func.is_inline:
            assert context.current_func_inline_return_ir_alloca is not None
            context.builder.store(ir_val, context.current_func_inline_return_ir_alloca)
          else:
            assert context.current_func_inline_return_ir_alloca is None
            context.builder.ret(ir_val)
      else:
        assert len(self.return_exprs) == 0
        if symbol_table.current_func.return_type != SLEEPY_VOID:
          self.raise_error('Function declared to return a value of type %r, but returned void' % (
            symbol_table.current_func.return_type))
        if context.emits_ir:
          if symbol_table.current_func.is_inline:
            assert context.current_func_inline_return_ir_alloca is None
          else:
            context.builder.ret_void()

      if context.emits_ir and symbol_table.current_func.is_inline:
        collect_block = context.current_func_inline_return_collect_block
        assert collect_block is not None
        context.builder.branch(collect_block)
        context.builder = ir.IRBuilder(collect_block)
      context.is_terminated = True

  def children(self) -> List[AbstractSyntaxTree]:
    return self.return_exprs

  def __repr__(self) -> str:
    return 'ReturnStatementAst(return_exprs=%r)' % self.return_exprs


class StructDeclarationAst(StatementAst):
  def __init__(self, pos: TreePosition, struct_identifier: str, templ_identifiers: List[str],
               member_types: List[TypeAst], member_identifiers: List[str],
               member_annotations: List[List[AnnotationAst]]):
    super().__init__(pos)
    assert len(member_types) == len(member_identifiers) == len(member_annotations)
    self.struct_identifier = struct_identifier
    self.templ_identifiers = templ_identifiers
    self.member_types = member_types
    self.member_identifiers = member_identifiers
    self.member_annotations = member_annotations

  def build_ir(self, symbol_table: SymbolTable, context: CodegenContext):
    with context.use_pos(self.pos):
      if self.struct_identifier in symbol_table.current_scope_identifiers:
        self.raise_error('Cannot redefine struct with name %r' % self.struct_identifier)

      struct_symbol_table = symbol_table.make_child_scope(
        inherit_outer_variables=False)  # symbol table including placeholder types
      placeholder_templ_types = self._collect_placeholder_templ_types(
        self.templ_identifiers, symbol_table=struct_symbol_table)

      member_types = [type_ast.make_type(symbol_table=struct_symbol_table) for type_ast in self.member_types]
      signature_struct_type = StructType(
        struct_identifier=self.struct_identifier, templ_types=placeholder_templ_types,
        member_identifiers=self.member_identifiers, member_types=member_types)

      # make constructor / destructor
      constructor = signature_struct_type.build_constructor(parent_symbol_table=symbol_table, parent_context=context)
      signature_struct_type.constructor = constructor
      signature_struct_type.build_destructor(parent_symbol_table=symbol_table, parent_context=context)

      # assemble to complete type symbol
      struct_type_symbol = TypeTemplateSymbol(template_parameters=placeholder_templ_types, signature_type=signature_struct_type)
      symbol_table[self.struct_identifier] = struct_type_symbol

  def children(self) -> List[AbstractSyntaxTree]:
    return []

  def __repr__(self) -> str:
    return (
      'StructDeclarationAst(struct_identifier=%r, templ_identifiers=%r, member_identifiers=%r, member_types=%r)' % (
        self.struct_identifier, self.templ_identifiers, self.member_identifiers, self.member_types))


class AssignStatementAst(StatementAst):
  """
  Stmt -> identifier = Expr ;
  """
  allowed_annotation_identifiers = frozenset({})

  def __init__(self, pos: TreePosition, var_target: ExpressionAst, var_val: ExpressionAst,
               declared_var_type: Optional[TypeAst]):
    super().__init__(pos)
    assert isinstance(var_target, ExpressionAst)
    self.var_target = var_target
    self.var_val = var_val
    self.declared_var_type = declared_var_type

  def get_var_identifier(self) -> Optional[str]:
    target = self.var_target
    while isinstance(target, UnbindExpressionAst):
      target = target.arg_expr
    if isinstance(target, IdentifierExpressionAst):
      return target.identifier
    else:
      return None

  def is_declaration(self, symbol_table: SymbolTable) -> bool:
    var_identifier = self.get_var_identifier()
    if var_identifier is None:
      return False
    if var_identifier not in symbol_table.current_scope_identifiers:
      return True
    assert var_identifier in symbol_table
    symbol = symbol_table[var_identifier]
    if not isinstance(symbol, VariableSymbol):
      self.raise_error('Cannot assign non-variable %r to a variable' % var_identifier)
    return False

  def build_ir(self, symbol_table: SymbolTable, context: CodegenContext):
    with context.use_pos(self.pos):
      if self.declared_var_type is not None:
        stated_type: Optional[Type] = self.declared_var_type.make_type(symbol_table=symbol_table)
      else:
        stated_type: Optional[Type] = None
      if not self.var_val.make_symbol_kind(symbol_table=symbol_table) == Symbol.Kind.VARIABLE:
        self.raise_error('Can only reassign variables')
      val = self.var_val.make_as_val(symbol_table=symbol_table, context=context)
      if val.type == SLEEPY_VOID:
        self.raise_error('Cannot assign void to variable')
      val = val.copy_collapse(context=context, name='store')
      if stated_type is not None:
        if not can_implicit_cast_to(val.narrowed_type, stated_type):
          self.raise_error(
            'Cannot assign variable with stated type %r a value of type %r' % (stated_type, val.narrowed_type))

      if self.is_declaration(symbol_table=symbol_table):
        var_identifier = self.get_var_identifier()
        assert var_identifier not in symbol_table.current_scope_identifiers
        # variables are always (implicitly) references
        declared_type = ReferenceType(val.type if stated_type is None else stated_type)
        # declare new variable, override entry in symbol_table (maybe it was defined in an outer scope before).
        symbol = VariableSymbol(None, var_type=declared_type)
        symbol.build_ir_alloca(context=context, identifier=var_identifier)
        symbol_table[var_identifier] = symbol

      # check that declared type matches assigned type
      uncollapsed_target_val = self.var_target.make_as_val(symbol_table=symbol_table, context=context)
      if not uncollapsed_target_val.is_referenceable():
        self.raise_error('Cannot reassign non-referencable type %s' % uncollapsed_target_val.type)
      target_val = uncollapsed_target_val.copy_collapse_as_mutates(context=context, name='assign_val')
      assert isinstance(target_val.type, ReferenceType)
      declared_type = target_val.type.pointee_type
      if stated_type is not None and not can_implicit_cast_to(stated_type, declared_type):
          self.raise_error('Cannot redefine variable of type %r with new type %r' % (declared_type, stated_type))
      if not can_implicit_cast_to(val.narrowed_type, declared_type):
        self.raise_error(
          'Cannot redefine variable of type %r with variable of type %r' % (declared_type, val.narrowed_type))

      # if we assign to a variable, narrow type to val_type
      if isinstance(self.var_target, IdentifierExpressionAst):
        assert self.var_target.identifier in symbol_table
        symbol = symbol_table[self.var_target.identifier]
        assert isinstance(symbol, VariableSymbol)
        narrowed_uncollapsed_val_type = uncollapsed_target_val.type.replace_types(
          {target_val.type.pointee_type: val.narrowed_type})
        narrowed_symbol = symbol.copy_with_narrowed_type(narrowed_uncollapsed_val_type)
        assert not isinstance(narrowed_symbol, UnionType) or len(narrowed_symbol.possible_types) > 0
        symbol_table[self.var_target.identifier] = narrowed_symbol

      if context.emits_ir:
        ir_val = val.copy_with_implicit_cast(declared_type, context=context, name='assign_cast').ir_val
        assert ir_val is not None
        assert target_val.ir_val is not None
        context.builder.store(value=ir_val, ptr=target_val.ir_val)

  def children(self) -> List[AbstractSyntaxTree]:
    return [self.var_target, self.var_val, self.declared_var_type]

  def __repr__(self) -> str:
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
    with context.use_pos(self.pos):
      cond_val = self.condition_val.make_as_val(symbol_table=symbol_table, context=context)
      cond_val = cond_val.copy_collapse(context=context, name='if_cond')
      if not cond_val.narrowed_type == SLEEPY_BOOL:
        self.raise_error('Cannot use expression of type %r as if-condition' % cond_val.type)

      true_symbol_table, false_symbol_table = symbol_table.make_child_scope(
        inherit_outer_variables=True), symbol_table.make_child_scope(inherit_outer_variables=True)
      make_narrow_type_from_valid_cond_ast(self.condition_val, cond_holds=True, symbol_table=true_symbol_table)
      make_narrow_type_from_valid_cond_ast(self.condition_val, cond_holds=False, symbol_table=false_symbol_table)

      if context.emits_ir:
        ir_cond = cond_val.ir_val
        true_block: ir.Block = context.builder.append_basic_block('true_branch')
        false_block: ir.Block = context.builder.append_basic_block('false_branch')
        context.builder.cbranch(ir_cond, true_block, false_block)
        true_context = context.copy_with_builder(ir.IRBuilder(true_block))
        false_context = context.copy_with_builder(ir.IRBuilder(false_block))
      else:
        true_context, false_context = context.copy_without_builder(), context.copy_without_builder()

      self.true_scope.build_scope_ir(scope_symbol_table=true_symbol_table, scope_context=true_context)
      self.false_scope.build_scope_ir(scope_symbol_table=false_symbol_table, scope_context=false_context)

      if true_context.is_terminated and false_context.is_terminated:
        context.is_terminated = True
        if context.emits_ir:
          context.builder = None
      else:
        assert not true_context.is_terminated or not false_context.is_terminated
        if true_context.is_terminated:
          symbol_table.apply_type_narrowings_from(false_symbol_table)
        elif false_context.is_terminated:
          symbol_table.apply_type_narrowings_from(true_symbol_table)
        else:  # default case
          assert not true_context.is_terminated and not false_context.is_terminated
          symbol_table.apply_type_narrowings_from(true_symbol_table, false_symbol_table)
        if context.emits_ir:
          continue_block: ir.Block = context.builder.append_basic_block('continue_branch')
          if not true_context.is_terminated:
            true_context.builder.branch(continue_block)
          if not false_context.is_terminated:
            false_context.builder.branch(continue_block)
          context.builder = ir.IRBuilder(continue_block)

  def children(self) -> List[AbstractSyntaxTree]:
    return [self.condition_val, self.true_scope, self.false_scope]

  def __repr__(self) -> str:
    return 'IfStatementAst(condition_val=%r, true_scope=%r, false_scope=%r)' % (
      self.condition_val, self.true_scope, self.false_scope)


class WhileStatementAst(StatementAst):
  """
  Stmt -> while Expr { StmtList }
  """
  def __init__(self, pos, condition_val, body_scope):
    """
    :param TreePosition pos:
    :param ExpressionAst condition_val:
    :param AbstractScopeAst body_scope:
    """
    super().__init__(pos)
    self.condition_val = condition_val
    self.body_scope = body_scope

  def build_ir(self, symbol_table, context):
    """
    :param SymbolTable symbol_table:
    :param CodegenContext context:
    """
    with context.use_pos(self.pos):
      cond_val = self.condition_val.make_as_val(symbol_table=symbol_table, context=context)
      cond_val = cond_val.copy_collapse(context=context, name='while_cond')
      if not cond_val.narrowed_type == SLEEPY_BOOL:
        self.raise_error('Cannot use expression of type %r as while-condition' % cond_val.type)

      body_symbol_table = symbol_table.make_child_scope(inherit_outer_variables=True)
      make_narrow_type_from_valid_cond_ast(self.condition_val, cond_holds=True, symbol_table=body_symbol_table)

      if context.emits_ir:
        cond_ir = cond_val.ir_val
        loop_block: ir.Block = context.builder.append_basic_block('while_body')
        continue_block: ir.Block = context.builder.append_basic_block('continue_branch')
        context.builder.cbranch(cond_ir, loop_block, continue_block)
        context.builder = ir.IRBuilder(continue_block)

        body_context = context.copy_with_builder(ir.IRBuilder(loop_block))
        self.body_scope.build_scope_ir(scope_symbol_table=body_symbol_table, scope_context=body_context)
        if not body_context.is_terminated:
          body_cond_ir = self.condition_val.make_as_val(symbol_table=symbol_table, context=body_context).ir_val
          body_context.builder.cbranch(body_cond_ir, loop_block, continue_block)
      else:
        body_context = context.copy_without_builder()
        self.body_scope.build_scope_ir(scope_symbol_table=body_symbol_table, scope_context=body_context)

      # TODO: Do a fix-point iteration over the body and wait until the most general type no longer changes.
      symbol_table.reset_narrowed_types()
      make_narrow_type_from_valid_cond_ast(self.condition_val, cond_holds=False, symbol_table=symbol_table)

  def children(self) -> List[AbstractSyntaxTree]:
    return [self.condition_val, self.body_scope]

  def __repr__(self) -> str:
    return 'WhileStatementAst(condition_val=%r, body_scope=%r)' % (self.condition_val, self.body_scope)


class ExpressionAst(AbstractSyntaxTree, ABC):
  """
  Val, SumVal, ProdVal, PrimaryExpr
  """
  def __init__(self, pos: TreePosition):
    super().__init__(pos)

  @abstractmethod
  def make_symbol_kind(self, symbol_table: SymbolTable) -> Symbol.Kind:
    raise NotImplementedError()

  @abstractmethod
  def make_as_val(self, symbol_table: SymbolTable, context: CodegenContext) -> TypedValue:
    raise NotImplementedError()

  @abstractmethod
  def make_as_func_caller(self, symbol_table: SymbolTable) -> FunctionSymbolCaller:
    """
    Try to get the statically declared function that this expression represents.
    Note that this is not a variable (e.g. a simple function pointer),
    because we want to support overloading here.
    The FunctionSymbolCaller object also keeps track of the template parameters.
    """
    raise NotImplementedError()

  @abstractmethod
  def make_as_type(self, symbol_table: SymbolTable) -> Type:
    raise NotImplementedError()

  def __repr__(self) -> str:
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
      # TODO: Then it should be a TypeExpressionAst, not a VariableExpressionAst.
      # Probably it's nicer to make an entire new ExpressionAst for `is` Expressions anyway.
      if not isinstance(self.right_expr, IdentifierExpressionAst):
        raise self.raise_error("'is' operator must be applied to a union type and a type.")

  def make_symbol_kind(self, symbol_table: SymbolTable) -> Symbol.Kind:
    return Symbol.Kind.VARIABLE

  def make_as_val(self, symbol_table: SymbolTable, context: CodegenContext) -> TypedValue:
    with context.use_pos(self.pos):
      if self.op == 'is':
        assert isinstance(self.right_expr, IdentifierExpressionAst)
        check_type_expr = IdentifierTypeAst(
          self.right_expr.pos, type_identifier=self.right_expr.identifier, templ_types=[])
        check_type = check_type_expr.make_type(symbol_table=symbol_table)
        check_value = self.left_expr.make_as_val(symbol_table=symbol_table, context=context)
        check_value = check_value.copy_collapse(context=context, name='check_val')
        if context.emits_ir:
          ir_val = make_ir_val_is_type(check_value.ir_val, check_value.narrowed_type, check_type, context=context)
        else:
          ir_val = None
        return TypedValue(typ=SLEEPY_BOOL, ir_val=ir_val)

      operand_exprs = [self.left_expr, self.right_expr]
      return self.build_func_call_by_identifier(
        func_identifier=self.op, templ_types=None, func_arg_exprs=operand_exprs, symbol_table=symbol_table,
        context=context)

  def make_as_func_caller(self, symbol_table: SymbolTable):
    self.raise_error('Cannot use result of operator %r as function' % self.op)

  def make_as_type(self, symbol_table: SymbolTable):
    self.raise_error('Cannot use result of operator %r as type' % self.op)

  def children(self) -> List[AbstractSyntaxTree]:
    return [self.left_expr, self.right_expr]

  def __repr__(self) -> str:
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

  def make_symbol_kind(self, symbol_table: SymbolTable) -> Symbol.Kind:
    return Symbol.Kind.VARIABLE

  def make_as_val(self, symbol_table: SymbolTable, context: CodegenContext) -> TypedValue:
    with context.use_pos(self.pos):
      operand_exprs = [self.expr]
      return self.build_func_call_by_identifier(
        func_identifier=self.op, templ_types=None, func_arg_exprs=operand_exprs, symbol_table=symbol_table,
        context=context)

  def children(self) -> List[AbstractSyntaxTree]:
    return [self.expr]

  def make_as_func_caller(self, symbol_table: SymbolTable):
    self.raise_error('Cannot use result of operator %r as function' % self.op)

  def make_as_type(self, symbol_table: SymbolTable):
    self.raise_error('Cannot use result of operator %r as type' % self.op)

  def __repr__(self) -> str:
    return 'UnaryOperatorExpressionAst(op=%r, expr=%r)' % (self.op, self.expr)


class ConstantExpressionAst(ExpressionAst):
  """
  PrimaryExpr -> double | int | char
  """
  def __init__(self, pos: TreePosition, constant_val: Any, constant_type: Type):
    super().__init__(pos)
    self.constant_val = constant_val
    self.constant_type = constant_type

  def make_symbol_kind(self, symbol_table: SymbolTable) -> Symbol.Kind:
    return Symbol.Kind.VARIABLE

  def make_as_val(self, symbol_table: SymbolTable, context: CodegenContext) -> TypedValue:
    with context.use_pos(self.pos):
      ir_val = ir.Constant(self.constant_type.ir_type, self.constant_val) if context.emits_ir else None
      return TypedValue(typ=self.constant_type, ir_val=ir_val)

  def make_as_func_caller(self, symbol_table: SymbolTable):
    self.raise_error('Cannot use constant expression as function')

  def make_as_type(self, symbol_table: SymbolTable):
    self.raise_error('Cannot use constant expression as type')

  def children(self) -> List[AbstractSyntaxTree]:
    return []

  def __repr__(self) -> str:
    return 'ConstantExpressionAst(constant_val=%r, constant_type=%r)' % (self.constant_val, self.constant_type)


class StringLiteralExpressionAst(ExpressionAst):
  """
  PrimaryExpr -> str
  """
  def __init__(self, pos, constant_str):
    """
    :param TreePosition pos:
    :param str constant_str:
    """
    super().__init__(pos)
    self.constant_str = constant_str

  def make_symbol_kind(self, symbol_table: SymbolTable) -> Symbol.Kind:
    return Symbol.Kind.VARIABLE

  def make_as_val(self, symbol_table: SymbolTable, context: CodegenContext) -> TypedValue:
    with context.use_pos(self.pos):
      assert 'Str' in symbol_table.inbuilt_symbols is not None
      str_symbol = symbol_table.inbuilt_symbols['Str']
      assert isinstance(str_symbol, TypeTemplateSymbol)
      str_type = str_symbol.get_type(template_arguments=[])
      assert isinstance(str_type, StructType)
      assert str_type.member_identifiers == ['start', 'length', 'alloc_length']

      if context.emits_ir:
        str_val = tuple(self.constant_str.encode())
        assert context.ir_func_malloc is not None
        from sleepy.symbols import LLVM_SIZE_TYPE
        ir_start_raw = context.builder.call(
          context.ir_func_malloc, args=[ir.Constant(LLVM_SIZE_TYPE, len(str_val))], name='str_literal_start_raw')
        str_ir_type = ir.ArrayType(SLEEPY_CHAR.ir_type, len(str_val))
        ir_start_array = context.builder.bitcast(
          ir_start_raw, ir.PointerType(str_ir_type), name='str_literal_start_array')
        context.builder.store(ir.Constant(str_ir_type, str_val), ir_start_array)
        ir_start = context.builder.bitcast(ir_start_array, SLEEPY_CHAR_PTR.ir_type, name='str_literal_start')
        length_ir_type = str_type.member_types[1].ir_type
        ir_length = ir.Constant(length_ir_type, len(str_val))

        ir_val = ir.Constant(str_type.ir_type, (
          ir.FormattedConstant(context.ir_func_malloc.function_type.return_type, constant='undef'),
          ir_length, ir_length))
        completed_string_value = context.builder.insert_value(agg=ir_val, value=ir_start, idx=0, name='str_literal_store_start')
      else:
        completed_string_value = None
      return TypedValue(typ=str_type, ir_val=completed_string_value)

  def make_as_func_caller(self, symbol_table: SymbolTable):
    self.raise_error('Cannot use string literal as function')

  def make_as_type(self, symbol_table: SymbolTable):
    self.raise_error('Cannot use string literal as type')

  def children(self) -> List[AbstractSyntaxTree]:
    return []

  def __repr__(self) -> str:
    return 'StringLiteralExpressionAst(constant_str=%r)' % self.constant_str


class IdentifierExpressionAst(ExpressionAst):
  """
  PrimaryExpr -> identifier
  """
  def __init__(self, pos, identifier):
    """
    :param TreePosition pos:
    :param str identifier:
    """
    super().__init__(pos)
    self.identifier = identifier

  def make_symbol_kind(self, symbol_table: SymbolTable) -> Symbol.Kind:
    return self.get_symbol(symbol_table=symbol_table).kind

  def get_symbol(self, symbol_table: SymbolTable) -> Symbol:
    if self.identifier not in symbol_table:
      self.raise_error('Identifier %r referenced before declaring' % self.identifier)
    symbol = symbol_table[self.identifier]
    return symbol

  def get_var_symbol(self, symbol_table: SymbolTable) -> VariableSymbol:
    symbol = self.get_symbol(symbol_table=symbol_table)
    if not isinstance(symbol, VariableSymbol):
      self.raise_error('Cannot reference a non-variable %r, got a %s' % (self.identifier, symbol.kind))
    if self.identifier not in symbol_table.current_scope_identifiers:
      # TODO add variable captures
      self.raise_error('Cannot capture variable %r from outer scope' % self.identifier)
    return symbol

  def get_func_symbol(self, symbol_table: SymbolTable) -> FunctionSymbol:
    symbol = self.get_symbol(symbol_table=symbol_table)
    if isinstance(symbol, TypeTemplateSymbol):
      symbol_type = self.make_as_type(symbol_table=symbol_table)
      if symbol_type.constructor is None:
        self.raise_error('Cannot call non-existing constructor of type %r' % symbol_type)
      symbol = symbol_type.constructor
    if not isinstance(symbol, FunctionSymbol):
      self.raise_error('Cannot reference a non-function %r, got a %s' % (self.identifier, symbol.kind))
    return symbol

  def make_as_val(self, symbol_table: SymbolTable, context: CodegenContext) -> TypedValue:
    with context.use_pos(self.pos):
      symbol = self.get_var_symbol(symbol_table=symbol_table)
      if context.emits_ir:
        assert symbol.ir_alloca is not None
      return TypedValue(
        typ=symbol.declared_var_type, narrowed_type=symbol.narrowed_var_type, ir_val=symbol.ir_alloca)

  def make_as_func_caller(self, symbol_table: SymbolTable) -> FunctionSymbolCaller:
    return FunctionSymbolCaller(func=self.get_func_symbol(symbol_table=symbol_table))

  def make_as_type(self, symbol_table: SymbolTable) -> Type:
    type_symbol = self.get_symbol(symbol_table=symbol_table)
    if not isinstance(type_symbol, TypeTemplateSymbol):
      self.raise_error('%r is not a type, but a %r' % (self.identifier, type_symbol.kind))
    return type_symbol.signature_type

  def children(self) -> List[AbstractSyntaxTree]:
    return []

  def __repr__(self) -> str:
    return 'IdentifierExpressionAst(var_identifier=%r)' % self.identifier


class CallExpressionAst(ExpressionAst):
  """
  PrimaryExpr -> PrimaryExpr ( ExprList )
  """
  def __init__(self, pos: TreePosition, func_expr: ExpressionAst, func_arg_exprs: List[ExpressionAst]):
    super().__init__(pos)
    self.func_expr = func_expr
    self.func_arg_exprs = func_arg_exprs

  def _maybe_get_specified_templ_types(self, symbol_table: SymbolTable) -> Optional[List[Type]]:
    """
    There is some special logic currently because we do not support functions that operate on types
    and return new functions yet.
    This function handles the special case that this is foo[Types...],
    in which case we assume that `foo` is a function or a template type,
    and we specify some template types with concrete types by setting the template parameters.
    """
    if not isinstance(self.func_expr, IdentifierExpressionAst) or self.func_expr.identifier != 'index':
      return None
    if not len(self.func_arg_exprs) >= 1:
      return None
    signature_func_expr, *templ_arg_exprs = self.func_arg_exprs
    if signature_func_expr.make_symbol_kind(symbol_table=symbol_table) not in {Symbol.Kind.FUNCTION, Symbol.Kind.TYPE}:
      return None
    return [templ_arg.make_as_type(symbol_table=symbol_table) for templ_arg in templ_arg_exprs]

  def make_symbol_kind(self, symbol_table: SymbolTable) -> Symbol.Kind:
    concrete_templ_types = self._maybe_get_specified_templ_types(symbol_table=symbol_table)
    func_kind = self.func_expr.make_symbol_kind(symbol_table=symbol_table)

    if concrete_templ_types is not None:  # is a templ type specialization, keeps kind.
      assert func_kind == Symbol.Kind.FUNCTION  # func is get/set
      func_caller_kind = self.func_arg_exprs[0].make_symbol_kind(symbol_table=symbol_table)
      if func_caller_kind not in {Symbol.Kind.FUNCTION, Symbol.Kind.TYPE}:
        self.raise_error('Cannot specify template parameters for symbol of kind %r' % func_kind)
      return func_caller_kind
    if func_kind == Symbol.Kind.FUNCTION:
      return Symbol.Kind.VARIABLE
    if func_kind == Symbol.Kind.TYPE:
      called_type = self.func_expr.make_as_type(symbol_table=symbol_table)
      if called_type.constructor is None:
        self.raise_error('Cannot call constructor of type %r because it does not exist' % called_type)
      return Symbol.Kind.VARIABLE
    self.raise_error('Cannot call non-function symbol, is a %r' % func_kind)

  def _make_func_expr_as_func_caller(self, symbol_table: SymbolTable) -> FunctionSymbolCaller:
    assert self.make_symbol_kind(symbol_table=symbol_table) == Symbol.Kind.VARIABLE
    func_kind = self.func_expr.make_symbol_kind(symbol_table=symbol_table)
    if func_kind == Symbol.Kind.FUNCTION:
      return self.func_expr.make_as_func_caller(symbol_table=symbol_table)
    elif func_kind == Symbol.Kind.TYPE:
      called_type = self.func_expr.make_as_type(symbol_table=symbol_table)
      if called_type.constructor is None:
        self.raise_error('Cannot call constructor of type %r because it does not exist' % called_type)
      return called_type.constructor_caller
    else:
      self.raise_error('Cannot call an expression of kind %r' % (
        self.func_expr.make_symbol_kind(symbol_table=symbol_table)))

  def _is_special_call(self, inbuilt_func_identifier: str, symbol_table: SymbolTable):
    if not isinstance(self.func_expr, IdentifierExpressionAst):
      return False
    inbuilt_func = symbol_table.inbuilt_symbols[inbuilt_func_identifier]
    assert isinstance(inbuilt_func, FunctionSymbol)
    if self.func_expr.make_as_func_caller(symbol_table=symbol_table).func != inbuilt_func:
      return False
    return True

  def _is_size_call(self, symbol_table: SymbolTable):
    # TODO: make this a normal compile time function operating on types
    return self._is_special_call('size', symbol_table=symbol_table)

  def make_as_val(self, symbol_table: SymbolTable, context: CodegenContext) -> TypedValue:
    assert self.make_symbol_kind(symbol_table=symbol_table) == Symbol.Kind.VARIABLE
    with context.use_pos(self.pos):
      if self._is_size_call(symbol_table=symbol_table):
        if len(self.func_arg_exprs) != 1:
          self.raise_error('Must call size(type: Type) with exactly one argument')
        size_of_type = self.func_arg_exprs[0].make_as_type(symbol_table=symbol_table)
        from sleepy.symbols import LLVM_SIZE_TYPE
        ir_val = ir.Constant(LLVM_SIZE_TYPE, size_of_type.size) if context.emits_ir else None
        return TypedValue(typ=SLEEPY_LONG, ir_val=ir_val)

      # default case
      func_caller = self._make_func_expr_as_func_caller(symbol_table=symbol_table)
      return self.build_func_call(
        func_caller=func_caller, func_arg_exprs=self.func_arg_exprs, symbol_table=symbol_table, context=context)

  def make_as_func_caller(self, symbol_table: SymbolTable) -> FunctionSymbolCaller:
    assert self.make_symbol_kind(symbol_table=symbol_table) == Symbol.Kind.FUNCTION
    concrete_templ_types = self._maybe_get_specified_templ_types(symbol_table=symbol_table)
    if concrete_templ_types is None:
      return self.func_expr.make_as_func_caller(symbol_table=symbol_table)
    else:
      func_caller = self.func_arg_exprs[0].make_as_func_caller(symbol_table=symbol_table)
      if func_caller.templ_types is not None:
        self.raise_error('Cannot specify template types multiple times')
      return func_caller.copy_with_templ_types(concrete_templ_types)

  def make_as_type(self, symbol_table: SymbolTable) -> Type:
    assert self.make_symbol_kind(symbol_table=symbol_table) == Symbol.Kind.TYPE
    concrete_templ_types = self._maybe_get_specified_templ_types(symbol_table=symbol_table)
    assert concrete_templ_types is not None
    signature_type = self.func_arg_exprs[0].make_as_type(symbol_table=symbol_table)
    if len(concrete_templ_types) != len(signature_type.templ_types):
      if len(concrete_templ_types) == 0:
        self.raise_error('Type %r needs to be constructed with template arguments for template parameters %r' % (
          signature_type, signature_type.templ_types))
      else:
        self.raise_error(
          'Type %r with placeholder template parameters %r cannot be constructed with template arguments %r' % (
            signature_type, signature_type.templ_types, concrete_templ_types))
    replacements = dict(zip(signature_type.templ_types, concrete_templ_types))
    return signature_type.replace_types(replacements=replacements)

  def children(self) -> List[AbstractSyntaxTree]:
    return self.func_arg_exprs

  def __repr__(self) -> str:
    return 'CallExpressionAst(func_expr=%r, func_arg_exprs=%r)' % (self.func_expr, self.func_arg_exprs)


class ReferenceExpressionAst(ExpressionAst):
  """
  PrimaryExpr -> ref ( Expr )
  """
  def __init__(self, pos: TreePosition, arg_expr: ExpressionAst):
    super().__init__(pos)
    self.arg_expr = arg_expr

  def make_symbol_kind(self, symbol_table: SymbolTable) -> Symbol.Kind:
    arg_kind = self.arg_expr.make_symbol_kind(symbol_table=symbol_table)
    if arg_kind != Symbol.Kind.VARIABLE:
      self.raise_error('Cannot take reference to non-variable, got a %r' % arg_kind)
    return Symbol.Kind.VARIABLE

  def make_as_val(self, symbol_table: SymbolTable, context: CodegenContext) -> TypedValue:
    with context.use_pos(self.pos):
      arg_val = self.arg_expr.make_as_val(symbol_table=symbol_table, context=context)
      if not arg_val.is_referenceable():
        self.raise_error('Cannot take reference to non-assignable variable')
      assert isinstance(arg_val.type, ReferenceType)
      assert isinstance(arg_val.narrowed_type, ReferenceType)
      from sleepy.symbols import PointerType
      return TypedValue(
        typ=PointerType(arg_val.type.pointee_type), narrowed_type=PointerType(arg_val.narrowed_type.pointee_type),
        ir_val=arg_val.ir_val)

  def make_as_func_caller(self, symbol_table: SymbolTable):
    self.raise_error('Cannot use reference as function')

  def make_as_type(self, symbol_table: SymbolTable):
    self.raise_error('Cannot use reference as type')

  def children(self) -> List[AbstractSyntaxTree]:
    return [self.arg_expr]

  def __repr__(self) -> str:
    return 'ReferenceExpressionAst(arg_expr=%r)' % self.arg_expr


class MemberExpressionAst(ExpressionAst):
  """
  MemberExpr -> MemberExpr . identifier

  TODO: In the future, make this a special function attr(a: Struct, member_identifier: str) or so
  This is a little magic, because it behaves differently depending on whether a is actually a reference or not.
  Not sure how to make this better.
  """
  def __init__(self, pos: TreePosition, parent_val_expr: ExpressionAst, member_identifier: str):
    super().__init__(pos)
    self.parent_val_expr = parent_val_expr
    self.member_identifier = member_identifier

  def make_symbol_kind(self, symbol_table: SymbolTable) -> Symbol.Kind:
    return Symbol.Kind.VARIABLE

  def make_as_val(self, symbol_table: SymbolTable, context: CodegenContext) -> TypedValue:
    with context.use_pos(self.pos):
      arg_val = self.parent_val_expr.make_as_val(symbol_table=symbol_table, context=context)
      struct_type = arg_val.copy_collapse(context=None, name='struct').narrowed_type
      if not isinstance(struct_type, StructType):
        self.raise_error(
          'Cannot access a member variable %r of the non-struct type %r' % (self.member_identifier, struct_type))
      if self.member_identifier not in struct_type.member_identifiers:
        self.raise_error('Struct type %r has no member variable %r, only available: %r' % (
          struct_type, self.member_identifier, ', '.join(struct_type.member_identifiers)))
      member_num = struct_type.get_member_num(self.member_identifier)
      member_type = struct_type.member_types[member_num]

      if arg_val.is_referenceable():
        if context.emits_ir:
          parent_ptr = arg_val.copy_collapse_as_mutates(context=context, name='parent').ir_val
          gep_indices = [ir.Constant(ir.IntType(32), 0), ir.Constant(ir.IntType(32), member_num)]
          self_ptr = context.builder.gep(parent_ptr, gep_indices, name='member_ptr_%s' % self.member_identifier)
        else:
          self_ptr = None
        return TypedValue(typ=ReferenceType(member_type), ir_val=self_ptr)
      else:
        assert arg_val.num_possible_binds() == 0
        ir_val = struct_type.make_extract_member_val_ir(
          self.member_identifier, struct_ir_val=arg_val.ir_val, context=context) if context.emits_ir else None
        return TypedValue(typ=member_type, ir_val=ir_val)

  def make_as_func_caller(self, symbol_table: SymbolTable):
    self.raise_error('Cannot use member %r as function' % self.member_identifier)

  def make_as_type(self, symbol_table: SymbolTable):
    self.raise_error('Cannot use member %r as type' % self.member_identifier)

  def children(self) -> List[AbstractSyntaxTree]:
    return [self.parent_val_expr]

  def __repr__(self) -> str:
    return 'MemberExpressionAst(parent_val_expr=%r, member_identifier=%r)' % (
      self.parent_val_expr, self.member_identifier)


class UnbindExpressionAst(ExpressionAst):
  """
  NegExpr -> ! NegExpr
  """
  def __init__(self, pos: TreePosition, arg_expr: ExpressionAst):
    super().__init__(pos)
    self.arg_expr = arg_expr

  def make_symbol_kind(self, symbol_table: SymbolTable) -> Symbol.Kind:
    return Symbol.Kind.VARIABLE

  def make_as_val(self, symbol_table: SymbolTable, context: CodegenContext) -> TypedValue:
    with context.use_pos(self.pos):
      arg_val = self.arg_expr.make_as_val(symbol_table=symbol_table, context=context)
      return arg_val.copy_unbind()

  def make_as_func_caller(self, symbol_table: SymbolTable):
    self.raise_error('Cannot use unbind expression as function')

  def make_as_type(self, symbol_table: SymbolTable):
    self.raise_error('Cannot use unbind expression as type')

  def children(self) -> List[AbstractSyntaxTree]:
    return [self.arg_expr]

  def __repr__(self) -> str:
    return 'UnbindExpressionAst(arg_expr=%r)' % self.arg_expr


class TypeAst(AbstractSyntaxTree, ABC):
  """
  Type.

  TODO: We do not want this anymore: Make this IdentifierExpressionAst.
  This means that we need to overload operator | and [..] for building union types and templated types.
  These are essentially functions operating on types.
  But these functions need to be executed at compile time, so for now, handle them in special cases everywhere.
  """
  def __init__(self, pos):
    """
    :param TreePosition pos:
    """
    super().__init__(pos)

  def make_type(self, symbol_table: SymbolTable) -> Type:
    raise NotImplementedError()

  def __repr__(self) -> str:
    return 'TypeAst()'


class IdentifierTypeAst(TypeAst):
  """
  IdentifierType -> identifier.
  """
  def __init__(self, pos: TreePosition, type_identifier: str, templ_types: List[TypeAst]):
    super().__init__(pos)
    self.type_identifier = type_identifier
    self.templ_types = templ_types

  def make_type(self, symbol_table: SymbolTable) -> Type:
    if self.type_identifier not in symbol_table:
      self.raise_error('Unknown type identifier %r' % self.type_identifier)
    type_symbol = symbol_table[self.type_identifier]
    if not isinstance(type_symbol, TypeTemplateSymbol):
      self.raise_error('%r is not a type, but a %r' % (self.type_identifier, type(type_symbol)))
    templ_type_symbols = [
      template_type.make_type(symbol_table=symbol_table) for template_type in self.templ_types]
    if len(templ_type_symbols) != len(type_symbol.template_parameters):
      if len(templ_type_symbols) == 0:
        self.raise_error('Type %r needs to be constructed with template arguments for template parameters %r' % (
          self.type_identifier, type_symbol.template_parameters))
      else:
        self.raise_error(
          'Type %r with placeholder template parameters %r cannot be constructed with template arguments %r' % (
            self.type_identifier, type_symbol.template_parameters, templ_type_symbols))
    return type_symbol.get_type(template_arguments=templ_type_symbols)

  def children(self) -> List[AbstractSyntaxTree]:
    return self.templ_types

  def __repr__(self) -> str:
    return 'IdentifierType(type_identifier=%r, template_types=%r)' % (self.type_identifier, self.templ_types)


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

  def make_type(self, symbol_table: SymbolTable) -> Type:
    concrete_variant_types = [variant_type.make_type(symbol_table=symbol_table) for variant_type in self.variant_types]
    return get_common_type(concrete_variant_types)

  def children(self) -> List[AbstractSyntaxTree]:
    return self.variant_types

  def __repr__(self) -> str:
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

  def children(self) -> List[AbstractSyntaxTree]:
    return []

  def __repr__(self) -> str:
    return 'AnnotationAst(identifier=%r)' % self.identifier


def annotate_ast(ast: AbstractSyntaxTree, annotation_list: List[AnnotationAst]) -> AbstractSyntaxTree:
  assert len(ast.annotations) == 0
  for annotation in annotation_list:
    if annotation.identifier not in ast.allowed_annotation_identifiers:
      annotation.raise_error('Annotations with name %r not allowed here, only allowed: %s' % (
        annotation.identifier, ', '.join(ast.allowed_annotation_identifiers)))
    if any(annotation.identifier == other.identifier for other in ast.annotations):
      annotation.raise_error('Cannot add annotation with name %r multiple times' % annotation.identifier)
    ast.annotations.append(annotation)
  return ast


def make_narrow_type_from_valid_cond_ast(cond_expr_ast: ExpressionAst,
                                         cond_holds: bool,
                                         symbol_table: SymbolTable):
  # TODO: This is super limited currently: Will only work for if(local_var is Type), nothing more.
  if isinstance(cond_expr_ast, BinaryOperatorExpressionAst) and cond_expr_ast.op == 'is':
    var_expr = cond_expr_ast.left_expr
    if not isinstance(var_expr, IdentifierExpressionAst):
      return
    var_symbol = var_expr.get_var_symbol(symbol_table=symbol_table)
    collapsed_var = var_symbol.as_typed_var(ir_val=None).copy_collapse(context=None)
    assert isinstance(cond_expr_ast.right_expr, IdentifierExpressionAst)
    check_type_expr = IdentifierTypeAst(
      cond_expr_ast.right_expr.pos, cond_expr_ast.right_expr.identifier, templ_types=[])
    check_type = check_type_expr.make_type(symbol_table=symbol_table)
    uncollapsed_check_type = var_symbol.declared_var_type.replace_types({collapsed_var.type: check_type})
    if cond_holds:
      symbol_table[var_expr.identifier] = var_symbol.copy_narrow_type(uncollapsed_check_type)
    else:
      symbol_table[var_expr.identifier] = var_symbol.copy_exclude_type(uncollapsed_check_type)
