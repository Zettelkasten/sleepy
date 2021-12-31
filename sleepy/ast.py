from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Any, Union

from llvmlite import ir

from sleepy.builtin_symbols import SLEEPY_BOOL, SLEEPY_LONG, SLEEPY_CHAR, SLEEPY_CHAR_PTR, build_initial_ir
from sleepy.errors import CompilerError, raise_error
from sleepy.ir_generation import make_ir_end_block_jump, make_ir_val_is_type, make_call_ir, \
  resolve_possible_concrete_funcs
from sleepy.struct_type import build_constructor, build_destructor
from sleepy.symbols import VariableSymbol, TypeTemplateSymbol, SymbolTable, Symbol, determine_kind, SymbolKind
from sleepy.syntactical_analysis.grammar import TreePosition, DummyPath
from sleepy.types import OverloadSet, Type, StructType, ConcreteFunction, UnionType, can_implicit_cast_to, \
  CodegenContext, get_common_type, \
  PlaceholderTemplateType, FunctionSymbolCaller, SLEEPY_UNIT, TypedValue, \
  ReferenceType, SLEEPY_NEVER, StructIdentity, PartialIdentifiedStructType, CleanupHandlingCG, make_union_switch_ir


def narrow_types_from_function_call(symbol_table: SymbolTable,
                                    arg_expressions: List[ExpressionAst],
                                    arg_values: List[TypedValue],
                                    collapsed_argument_values: List[TypedValue],
                                    possible_concrete_funcs: List[ConcreteFunction]):
  for arg_num, arg_expression in enumerate(arg_expressions):
    original_unbound_arg_type = arg_values[arg_num].narrowed_type
    original_bound_arg_type = collapsed_argument_values[arg_num].narrowed_type
    narrowed_arg_types = [concrete_func.arg_type_narrowings[arg_num] for concrete_func in possible_concrete_funcs]
    narrowed_arg_type = get_common_type(narrowed_arg_types)
    bound_narrowed_arg_type = original_unbound_arg_type.replace_types({original_bound_arg_type: narrowed_arg_type})
    if isinstance(arg_expression, IdentifierExpressionAst):
      var_symbol = symbol_table[arg_expression.identifier]
      assert isinstance(var_symbol, VariableSymbol)
      symbol_table[arg_expression.identifier] = var_symbol.copy_narrow_type(bound_narrowed_arg_type)


def make_call_ir_with_narrowing(pos: TreePosition,
                                caller: FunctionSymbolCaller,
                                arg_expressions: List[ExpressionAst],
                                symbol_table: SymbolTable,
                                context: CodegenContext) -> TypedValue:

  overloads, template_arguments = caller.overload_set, caller.template_parameters
  arg_values = [arg_expr.make_as_val(symbol_table=symbol_table, context=context) for arg_expr in arg_expressions]
  collapsed_argument_values = [arg.copy_collapse(context=None) for num, arg in enumerate(arg_values)]
  calling_types = [arg.narrowed_type for arg in collapsed_argument_values]

  for calling_type, arg_expression in zip(calling_types, arg_expressions):
    if not calling_type.is_realizable():
      raise_error("Cannot call function %r with argument of unrealizable type"
                  % overloads.identifier, arg_expression.pos)

  possible_concrete_functions = resolve_possible_concrete_funcs(pos, func_caller=caller, calling_types=calling_types)

  return_val = make_call_ir(
    pos,
    caller=caller,
    argument_values=arg_values,
    context=context)

  # apply type narrowings
  narrow_types_from_function_call(symbol_table,
                                  arg_expressions,
                                  arg_values,
                                  collapsed_argument_values,
                                  possible_concrete_functions)

  # special handling of 'assert' call
  if overloads.identifier in {'assert', 'unchecked_assert'}:
    assert len(arg_expressions) >= 1
    condition_expr = arg_expressions[0]
    make_narrow_type_from_valid_cond_ast(condition_expr, cond_holds=True, symbol_table=symbol_table)

  return return_val


def make_call_ir_by_identifier(pos: TreePosition,
                               func_identifier: str,
                               templ_types: Optional[List[Type]],
                               func_arg_exprs: List[ExpressionAst],
                               symbol_table: SymbolTable,
                               context: CodegenContext) -> TypedValue:
  if func_identifier not in symbol_table:
    raise_error('Function %r called before declared' % func_identifier, pos)
  symbol = symbol_table[func_identifier]
  if not isinstance(symbol, OverloadSet):
    raise_error('Cannot call non-function %r' % func_identifier, pos)
  func_caller = FunctionSymbolCaller(overload_set=symbol, template_parameters=templ_types)
  return  make_call_ir_with_narrowing(
    pos,
    caller=func_caller, arg_expressions=func_arg_exprs, symbol_table=symbol_table, context=context)


class AbstractSyntaxTree(ABC):
  """
  Abstract syntax tree of a sleepy program.
  """

  allowed_annotation_identifiers = frozenset()

  def __init__(self, pos: TreePosition):
    """
    :param pos: position where this AST starts
    """
    self.pos = pos
    self.annotations: List[AnnotationAst] = []

  def __repr__(self) -> str:
    return 'AbstractSyntaxTree'

  @abstractmethod
  def children(self) -> List[AbstractSyntaxTree]:
    pass

  def _collect_placeholder_templ_types(self, templ_identifiers: List[str],
                                       symbol_table: SymbolTable) -> List[PlaceholderTemplateType]:
    templ_types = []
    for templ_type_identifier in templ_identifiers:
      if templ_type_identifier in symbol_table.current_scope_identifiers:
        raise_error('Cannot declare template variable %r multiple times' % templ_type_identifier, self.pos)
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
  def __init__(self, pos: TreePosition):
    super().__init__(pos)

  @abstractmethod
  def build_ir(self, symbol_table: SymbolTable, context: CleanupHandlingCG):
    raise NotImplementedError()

  def __repr__(self) -> str:
    return 'StatementAst'


class DeclarationAst(AbstractSyntaxTree, ABC):
  @abstractmethod
  def create_symbol(self, symbol_table: SymbolTable, context: CodegenContext) -> Symbol:
    raise NotImplementedError()

  @property
  @abstractmethod
  def identifier(self) -> str:
    raise NotImplementedError()


class AbstractScopeAst(AbstractSyntaxTree):
  """
  Used to group multiple statements, forming a scope.
  """
  Element = Union[StatementAst, DeclarationAst]

  def __init__(self, pos: TreePosition, stmt_list: List[Element]):
    super().__init__(pos)
    self.stmt_list = stmt_list

  def create_symbols(self, scope_symbol_table: SymbolTable, scope_context: CodegenContext):
    with scope_context.use_pos(self.pos):
      for declaration in [e for e in self.stmt_list if isinstance(e, DeclarationAst)]:
        scope_symbol_table[declaration.identifier] = declaration.create_symbol(scope_symbol_table, scope_context)

  def build_scope_ir(self, scope_symbol_table: SymbolTable, scope_context: CleanupHandlingCG):
    self.create_symbols(scope_symbol_table, scope_context.base)

    # enumerate before building ir because check whether assignment is declaration only works
    # before variable symbols are added
    variable_definitions = self._enumerate_variable_definitions(scope_symbol_table)

    with scope_context.base.use_pos(self.pos):
      for stmt in [e for e in self.stmt_list if isinstance(e, StatementAst)]:
        if scope_context.base.all_paths_returned: raise_error('Code is unreachable', stmt.pos)
        stmt.build_ir(scope_symbol_table, scope_context)

    if not scope_context.base.all_paths_returned:
      scope_context.jump_to_end()

    # should usually use scope_context.end_block_builder instead of switching to end_block, but function call
    # generation needs the entire context
    builder_before = scope_context.builder
    scope_context.base.switch_to_block(scope_context.scope.end_block)

    for defining_ast in reversed(variable_definitions):
      symbol = scope_symbol_table[defining_ast.get_var_identifier()]
      assert isinstance(symbol, VariableSymbol)

      if (isinstance(symbol.narrowed_var_type, ReferenceType) and
              isinstance(symbol.narrowed_var_type.pointee_type, StructType)):
        caller = FunctionSymbolCaller(overload_set=scope_symbol_table.free_overloads, template_parameters=None)
        # call destructor
        make_call_ir(
          self.pos,
          caller=caller,
          argument_values=[symbol.typed_value],
          context=scope_context.base
        )

    scope_context.base.builder = builder_before
    scope_context.end_block_builder.position_at_end(scope_context.end_block_builder.block)

  def _enumerate_variable_definitions(self, symbol_table: SymbolTable) -> List[AssignStatementAst]:
    variable_symbols = []

    for stmt in [e for e in self.stmt_list if isinstance(e, AssignStatementAst)]:
      if stmt.is_declaration(symbol_table): variable_symbols.append(stmt)

    return variable_symbols

  def __repr__(self) -> str:
    return 'AbstractScopeAst(%s)' % ', '.join([repr(stmt) for stmt in self.stmt_list])

  def children(self):
    return self.stmt_list

  def check_scope(self, symbol_table: SymbolTable):
    # TODO
    pass


class FileAst(AbstractSyntaxTree):
  def __init__(self, pos: TreePosition,
               scope: AbstractScopeAst,
               imports_ast: ImportsAst):
    super().__init__(pos)
    self.imports_ast = imports_ast
    self.scope = scope

  def build_ir(self, symbol_table: SymbolTable, context: CodegenContext):
    with context.use_pos(self.pos):
      self.scope.create_symbols(symbol_table, context)

  def children(self) -> List[AbstractSyntaxTree]:
    return [self.imports_ast, self.scope]

  def __repr__(self) -> str:
    return 'TopLevelAst(%s)' % self.scope


class ImportsAst(AbstractSyntaxTree):
  def __init__(self, pos: TreePosition, imports: List[str]):
    super().__init__(pos)
    self.imports = imports

  def children(self) -> List[AbstractSyntaxTree]:
    return []


class TranslationUnitAst:
  """
  Multiple FileAsts.
  """

  def __init__(self, file_asts: List[FileAst]):
    self.file_asts = file_asts

  @staticmethod
  def from_file_asts(file_asts: List[FileAst]) -> TranslationUnitAst:
    assert len(file_asts) > 0
    return TranslationUnitAst(file_asts)

  def make_module_ir_and_symbol_table(self, module_name: str,
                                      emit_debug: bool,
                                      main_file_path: Path | DummyPath = DummyPath("module"),
                                      implicitly_exported_functions: Set(str) = set()) -> (str, SymbolTable):  # noqa
    assert self.check_pos_validity()

    module = ir.Module(name=module_name, context=ir.Context())

    root_builder = ir.IRBuilder()
    symbol_table = SymbolTable()
    context = CodegenContext(builder=root_builder, module=module, emits_debug=emit_debug, file_path=main_file_path)

    build_initial_ir(symbol_table=symbol_table, context=context)
    assert context.ir_func_malloc is not None
    for ast in self.file_asts:
      ast.build_ir(symbol_table=symbol_table, context=context)
    assert not context.all_paths_returned

    for identifier in implicitly_exported_functions:
      if identifier not in symbol_table:
        raise CompilerError("Function %s is supposed to be exported, but does not exist." % identifier)
      symbol = symbol_table[identifier]
      if not isinstance(symbol, OverloadSet): raise CompilerError(
        "Function %s is supposed to be exported, but is a %s instead" % (identifier, determine_kind(symbol)))
      if not symbol.has_single_concrete_func(): raise CompilerError(
        "Function %s is supposed to be exported, but is not uniquely identifiable by its name."
        "Possible overloads are %s" % (identifier, repr(symbol)))
      symbol.get_single_concrete_func()

    context.all_paths_returned = True

    return context.get_patched_ir(), symbol_table

  def check_pos_validity(self) -> bool:
    return all(child.check_pos_validity() for child in self.file_asts)


class ExpressionStatementAst(StatementAst):
  """
  Stmt -> Expr
  """

  def __init__(self, pos: TreePosition, expr: ExpressionAst):
    super().__init__(pos)
    assert isinstance(expr, ExpressionAst)
    self.expr = expr

  def build_ir(self, symbol_table: SymbolTable, context: CleanupHandlingCG):
    with context.base.use_pos(self.pos):
      self.expr.make_as_val(symbol_table=symbol_table, context=context.base)

  def children(self) -> List[AbstractSyntaxTree]:
    return [self.expr]

  def __repr__(self) -> str:
    return 'ExpressionStatementAst(expr=%r)' % self.expr


class ReturnStatementAst(StatementAst):
  """
  Stmt -> return ExprList ;
  """

  def __init__(self, pos: TreePosition, return_exprs: List[ExpressionAst]):
    super().__init__(pos)
    self.return_exprs = return_exprs
    if len(return_exprs) > 1:
      raise_error('Returning multiple values not support yet', self.pos)

  def build_ir(self, symbol_table: SymbolTable, cleanup_context: CleanupHandlingCG):
    context = cleanup_context.base

    with context.use_pos(self.pos):
      if symbol_table.current_func is None:
        raise_error('Can only use return inside a function declaration', self.pos)
      if context.all_paths_returned:
        raise_error('Cannot return from function after having returned already', self.pos)

      if len(self.return_exprs) == 1:
        return_expr = self.return_exprs[0]
        return_val = return_expr.make_as_val(symbol_table=symbol_table, context=context)

        return_val = return_val.copy_collapse(context=context, name='return_val')
        if not can_implicit_cast_to(return_val.narrowed_type, symbol_table.current_func.return_type):
          can_implicit_cast_to(return_val.narrowed_type, symbol_table.current_func.return_type)

          raise_error('Function declared to return type %r, but return value is of type %r' % (
            symbol_table.current_func.return_type, return_val.narrowed_type), self.pos)

        if context.emits_ir:
          ir_val = return_val.copy_with_implicit_cast(
            to_type=symbol_table.current_func.return_type, context=context, name='return_val_cast').ir_val
          cleanup_context.return_with_cleanup(ir_val)
      else:
        assert len(self.return_exprs) == 0
        if symbol_table.current_func.return_type != SLEEPY_UNIT:
          raise_error('Function declared to return a value of type %r, but implicitly returned %s' % (
            symbol_table.current_func.return_type, SLEEPY_UNIT), self.pos)
        if context.emits_ir and not symbol_table.current_func.is_inline:
          cleanup_context.return_with_cleanup(SLEEPY_UNIT.unit_constant())

      context.all_paths_returned = True

  def children(self) -> List[AbstractSyntaxTree]:
    return self.return_exprs

  def __repr__(self) -> str:
    return 'ReturnStatementAst(return_exprs=%r)' % self.return_exprs


class AssignStatementAst(StatementAst):
  """
  Stmt -> identifier = Expr ;
  """
  allowed_annotation_identifiers = frozenset({})

  def __init__(self, pos: TreePosition,
               target_ast: ExpressionAst,
               source_ast: ExpressionAst,
               declared_type: Optional[TypeAst]):
    super().__init__(pos)
    assert isinstance(target_ast, ExpressionAst)
    self.target_ast = target_ast
    self.source_ast = source_ast
    self.declared_type = declared_type

  def get_var_identifier(self) -> Optional[str]:
    target = self.target_ast
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
      raise_error('Cannot assign non-variable %r to a variable' % var_identifier, self.pos)
    return False

  def build_ir(self, symbol_table: SymbolTable, cleanup_context: CleanupHandlingCG):
    context = cleanup_context.base
    with context.use_pos(self.pos):
      # example y: A|B = x (where x is of type A)
      if self.declared_type is not None:
        stated_type: Optional[Type] = self.declared_type.make_type(symbol_table=symbol_table)  # A|B
      else:
        stated_type: Optional[Type] = None
      if not self.source_ast.make_symbol_kind(symbol_table=symbol_table) == SymbolKind.VARIABLE:
        raise_error('Can only reassign variables', self.pos)
      source_value = self.source_ast.make_as_val(symbol_table=symbol_table, context=context)

      source_value = source_value.copy_collapse(context=context, name='store')
      if stated_type is not None and not can_implicit_cast_to(source_value.narrowed_type, stated_type):
        raise_error('Cannot assign variable with stated type %r a value of type %r' % (
          stated_type, source_value.narrowed_type), self.pos)

      if self.is_declaration(symbol_table=symbol_table):
        var_identifier = self.get_var_identifier()  # y
        assert var_identifier not in symbol_table.current_scope_identifiers
        # variables are always (implicitly) references
        declared_type = ReferenceType(source_value.type if stated_type is None else stated_type)  # Ref[A|B]
        # declare new variable, override entry in symbol_table (maybe it was defined in an outer scope before).
        symbol = VariableSymbol.make_new_variable(declared_type, var_identifier, context)
        symbol_table[var_identifier] = symbol

      # check that declared type matches assigned type
      uncollapsed_target_val = self.target_ast.make_as_val(symbol_table=symbol_table, context=context)  # Ref[A|B]
      if not uncollapsed_target_val.is_referenceable():
        raise_error('Cannot reassign non-referencable type %s' % uncollapsed_target_val.type, self.pos)
      # narrow the target type to the assigned type s.t. we can unbind properly
      # even if some unions variants are not unbindable
      uncollapsed_target_val = uncollapsed_target_val.copy_set_narrowed_collapsed_type(
        ReferenceType(source_value.narrowed_type))  # Ref[A]
      if uncollapsed_target_val.narrowed_type == SLEEPY_NEVER:
        raise_error('Cannot assign variable of type %r a value of type %r' % (
          uncollapsed_target_val.type, source_value.narrowed_type), self.pos)
      target_val = uncollapsed_target_val.copy_collapse_as_mutates(context=context, name='assign_val')  # Ref[A]
      assert isinstance(target_val.type, ReferenceType)
      declared_type = target_val.type.pointee_type  # A
      if stated_type is not None and not can_implicit_cast_to(stated_type, declared_type):
        raise_error('Cannot %s variable collapsing to type %r with new type %r' % (
          'declare' if self.is_declaration(symbol_table=symbol_table) else 'redefine', declared_type, stated_type),
                    self.pos)
      assert can_implicit_cast_to(source_value.narrowed_type, declared_type)

      # if we assign to a variable, narrow type to val_type
      if (var_identifier := self.get_var_identifier()) is not None:
        assert var_identifier in symbol_table
        symbol = symbol_table[var_identifier]
        assert isinstance(symbol, VariableSymbol)
        narrowed_symbol = symbol.copy_set_narrowed_type(uncollapsed_target_val.narrowed_type)
        assert not isinstance(narrowed_symbol, UnionType) or len(narrowed_symbol.possible_types) > 0
        symbol_table[var_identifier] = narrowed_symbol

      if context.emits_ir:
        ir_val = source_value.copy_with_implicit_cast(declared_type, context=context, name='assign_cast').ir_val
        assert ir_val is not None
        assert target_val.ir_val is not None
        context.builder.store(value=ir_val, ptr=target_val.ir_val)

  def children(self) -> List[AbstractSyntaxTree]:
    return [self.target_ast, self.source_ast, self.declared_type]

  def __repr__(self) -> str:
    return 'AssignStatementAst(var_target=%r, var_val=%r, var_type=%r)' % (
      self.target_ast, self.source_ast, self.declared_type)


class IfStatementAst(StatementAst):
  """
  Stmt -> if Expr Scope
        | if Expr Scope else Scope
  """

  def __init__(self, pos: TreePosition,
               condition_val: ExpressionAst,
               true_scope: AbstractScopeAst,
               false_scope: Optional[AbstractScopeAst]):
    super().__init__(pos)
    self.condition_val = condition_val
    if false_scope is None:
      false_scope = AbstractScopeAst(TreePosition(pos.word, pos.to_pos, pos.to_pos), [])
    self.true_scope, self.false_scope = true_scope, false_scope

  def build_ir(self, symbol_table: SymbolTable, context: CleanupHandlingCG):
    base = context.base

    with base.use_pos(self.pos):
      cond_val = self.condition_val.make_as_val(symbol_table=symbol_table, context=base)
      cond_val = cond_val.copy_collapse(context=base, name='if_cond')
      if not cond_val.narrowed_type == SLEEPY_BOOL:
        raise_error('Cannot use expression of type %r as if-condition' % cond_val.type, self.pos)

      true_symbol_table, false_symbol_table = symbol_table.make_child_scope(
        inherit_outer_variables=True), symbol_table.make_child_scope(inherit_outer_variables=True)
      make_narrow_type_from_valid_cond_ast(self.condition_val, cond_holds=True, symbol_table=true_symbol_table)
      make_narrow_type_from_valid_cond_ast(self.condition_val, cond_holds=False, symbol_table=false_symbol_table)

      if base.emits_ir:
        ir_cond = cond_val.ir_val
        true_context = context.make_child_scope(scope_name='true_branch')
        false_context = context.make_child_scope(scope_name='false_branch')
        base.builder.cbranch(ir_cond, true_context.builder.block, false_context.builder.block)
      else:
        true_context, false_context = base.copy_without_builder(), base.copy_without_builder()

      self.true_scope.build_scope_ir(scope_symbol_table=true_symbol_table, scope_context=true_context)
      self.false_scope.build_scope_ir(scope_symbol_table=false_symbol_table, scope_context=false_context)

      if true_context.base.all_paths_returned and false_context.base.all_paths_returned:  # both terminated
        base.all_paths_returned = True
        true_context.end_block_builder.branch(context.scope.end_block)
        false_context.end_block_builder.branch(context.scope.end_block)
      else:
        continue_block: ir.Block = base.builder.append_basic_block('continue_block')
        if true_context.base.all_paths_returned:  # only true terminated
          symbol_table.apply_type_narrowings_from(false_symbol_table)
          true_context.end_block_builder.branch(context.scope.end_block)
          make_ir_end_block_jump(false_context, continuation=continue_block, parent_end_block=context.scope.end_block)

        elif false_context.base.all_paths_returned:  # only false terminated
          symbol_table.apply_type_narrowings_from(true_symbol_table)
          make_ir_end_block_jump(true_context, continuation=continue_block, parent_end_block=context.scope.end_block)
          false_context.end_block_builder.branch(context.scope.end_block)

        else:  # neither terminated
          assert not true_context.base.all_paths_returned and not false_context.base.all_paths_returned
          symbol_table.apply_type_narrowings_from(true_symbol_table, false_symbol_table)
          make_ir_end_block_jump(true_context, continuation=continue_block, parent_end_block=context.scope.end_block)
          make_ir_end_block_jump(false_context, continuation=continue_block, parent_end_block=context.scope.end_block)

        base.switch_to_block(continue_block)

  def children(self) -> List[AbstractSyntaxTree]:
    return [self.condition_val, self.true_scope, self.false_scope]

  def __repr__(self) -> str:
    return 'IfStatementAst(condition_val=%r, true_scope=%r, false_scope=%r)' % (
      self.condition_val, self.true_scope, self.false_scope)


class WhileStatementAst(StatementAst):
  """
  Stmt -> while Expr { StmtList }
  """

  def __init__(self, pos: TreePosition,
               condition_val: ExpressionAst,
               body_scope: AbstractScopeAst):
    super().__init__(pos)
    self.condition_val = condition_val
    self.body_scope = body_scope

  def _make_condition_ir(self, symbol_table: SymbolTable, context: CodegenContext) -> ir.Value:
    cond_val = self.condition_val.make_as_val(symbol_table=symbol_table, context=context)
    cond_val = cond_val.copy_collapse(context=context, name='while_cond')
    if not cond_val.narrowed_type == SLEEPY_BOOL:
      raise_error('Cannot use expression of type %r as while-condition' % cond_val.type, self.pos)
    return cond_val.ir_val

  def build_ir(self, symbol_table: SymbolTable, context: CleanupHandlingCG):
    base = context.base

    with base.use_pos(self.pos):
      body_symbol_table = symbol_table.make_child_scope(inherit_outer_variables=True)
      make_narrow_type_from_valid_cond_ast(self.condition_val, cond_holds=True, symbol_table=body_symbol_table)

      if base.emits_ir:
        body_context = context.make_child_scope(scope_name='while_body')
        body_block = body_context.builder.block
        condition_check_block = context.builder.append_basic_block('condition_check')
        continue_block = base.builder.append_basic_block('continue_branch')

        context.builder.branch(condition_check_block)
        context.base.switch_to_block(condition_check_block)

        condition_ir = self._make_condition_ir(symbol_table, context=base)
        context.builder.cbranch(condition_ir, body_block, continue_block)

        self.body_scope.build_scope_ir(scope_symbol_table=body_symbol_table, scope_context=body_context)

        if body_context.base.all_paths_returned:  # all branches return, simply jump to parent cleanup
          body_context.end_block_builder.branch(context.scope.end_block)
        else:  # return or jump back to condition check
          make_ir_end_block_jump(body_context, continuation=condition_check_block,
                                 parent_end_block=context.scope.end_block)

        context.base.switch_to_block(continue_block)
      else:
        self.body_scope.check_scope(symbol_table=body_symbol_table)

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
  def make_symbol_kind(self, symbol_table: SymbolTable) -> SymbolKind:
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


class ConstantExpressionAst(ExpressionAst):
  """
  PrimaryExpr -> double | int | char
  """

  def __init__(self, pos: TreePosition, constant_val: Any, constant_type: Type):
    super().__init__(pos)
    self.constant_val = constant_val
    self.constant_type = constant_type

  def make_symbol_kind(self, symbol_table: SymbolTable) -> SymbolKind:
    return SymbolKind.VARIABLE

  def make_as_val(self, symbol_table: SymbolTable, context: CodegenContext) -> TypedValue:
    with context.use_pos(self.pos):
      ir_val = ir.Constant(self.constant_type.ir_type, self.constant_val) if context.emits_ir else None
      return TypedValue(typ=self.constant_type, ir_val=ir_val)

  def make_as_func_caller(self, symbol_table: SymbolTable):
    raise_error('Cannot use constant expression as function', self.pos)

  def make_as_type(self, symbol_table: SymbolTable):
    raise_error('Cannot use constant expression as type', self.pos)

  def children(self) -> List[AbstractSyntaxTree]:
    return []

  def __repr__(self) -> str:
    return 'ConstantExpressionAst(constant_val=%r, constant_type=%r)' % (self.constant_val, self.constant_type)


class StringLiteralExpressionAst(ExpressionAst):
  """
  PrimaryExpr -> str
  """

  def __init__(self, pos: TreePosition, constant_str: str):
    super().__init__(pos)
    self.constant_str = constant_str

  def make_symbol_kind(self, symbol_table: SymbolTable) -> SymbolKind:
    return SymbolKind.VARIABLE

  def make_as_val(self, symbol_table: SymbolTable, context: CodegenContext) -> TypedValue:
    with context.use_pos(self.pos):
      assert 'Str' in symbol_table.builtin_symbols
      str_symbol = symbol_table['Str']
      assert isinstance(str_symbol, TypeTemplateSymbol)
      str_type = str_symbol.get_type(template_arguments=[])
      assert isinstance(str_type, StructType)
      assert str_type.member_identifiers == ['start', 'length', 'alloc_length']

      if context.emits_ir:
        str_val = tuple(self.constant_str.encode())
        assert context.ir_func_malloc is not None
        from sleepy.types import LLVM_SIZE_TYPE
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
        completed_string_value = context.builder.insert_value(agg=ir_val, value=ir_start, idx=0,
                                                              name='str_literal_store_start')
      else:
        completed_string_value = None
      return TypedValue(typ=str_type, ir_val=completed_string_value)

  def make_as_func_caller(self, symbol_table: SymbolTable):
    raise_error('Cannot use string literal as function', self.pos)

  def make_as_type(self, symbol_table: SymbolTable):
    raise_error('Cannot use string literal as type', self.pos)

  def children(self) -> List[AbstractSyntaxTree]:
    return []

  def __repr__(self) -> str:
    return 'StringLiteralExpressionAst(constant_str=%r)' % self.constant_str


class IdentifierExpressionAst(ExpressionAst):
  """
  PrimaryExpr -> identifier
  """

  def __init__(self, pos: TreePosition, identifier: str):
    super().__init__(pos)
    self.identifier = identifier

  def make_symbol_kind(self, symbol_table: SymbolTable) -> SymbolKind:
    return determine_kind(self.get_symbol(symbol_table=symbol_table))

  def get_symbol(self, symbol_table: SymbolTable) -> Symbol:
    if self.identifier not in symbol_table:
      raise_error('Identifier %r referenced before declaring' % self.identifier, self.pos)
    symbol = symbol_table[self.identifier]
    return symbol

  def get_var_symbol(self, symbol_table: SymbolTable) -> VariableSymbol:
    symbol = self.get_symbol(symbol_table=symbol_table)
    if not isinstance(symbol, VariableSymbol):
      raise_error('Cannot use %r as a variable, got a %s' % (self.identifier, determine_kind(symbol)), self.pos)
    if self.identifier not in symbol_table.current_scope_identifiers:
      # TODO add variable captures
      raise_error('Cannot capture variable %r from outer scope' % self.identifier, self.pos)
    if symbol.narrowed_var_type == SLEEPY_NEVER:
      raise_error('Cannot use variable %r with narrowed type %r' % (self.identifier, symbol.narrowed_var_type),
                  self.pos)
    return symbol

  def get_func_symbol(self, symbol_table: SymbolTable) -> OverloadSet:
    symbol = self.get_symbol(symbol_table=symbol_table)
    if isinstance(symbol, TypeTemplateSymbol):
      symbol_type = self.make_as_type(symbol_table=symbol_table)
      if symbol_type.constructor is None:
        raise_error('Cannot call non-existing constructor of type %r' % symbol_type, self.pos)
      symbol = symbol_type.constructor
    if not isinstance(symbol, OverloadSet):
      raise_error('Cannot reference a non-function %r, got a %s' % (self.identifier, determine_kind(symbol)), self.pos)
    return symbol

  def make_as_val(self, symbol_table: SymbolTable, context: CodegenContext) -> TypedValue:
    with context.use_pos(self.pos):
      return self.get_var_symbol(symbol_table=symbol_table).typed_value

  def make_as_func_caller(self, symbol_table: SymbolTable) -> FunctionSymbolCaller:
    return FunctionSymbolCaller(overload_set=self.get_func_symbol(symbol_table=symbol_table))

  def make_as_type(self, symbol_table: SymbolTable) -> Type:
    type_symbol = self.get_symbol(symbol_table=symbol_table)
    if not isinstance(type_symbol, TypeTemplateSymbol):
      raise_error('%r is not a type, but a %r' % (self.identifier, determine_kind(type_symbol)), self.pos)
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

  def _get_template_arguments_or_none(self, symbol_table: SymbolTable) -> Optional[List[Type]]:
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
    if signature_func_expr.make_symbol_kind(symbol_table=symbol_table) not in {SymbolKind.FUNCTION, SymbolKind.TYPE}:
      return None
    return [templ_arg.make_as_type(symbol_table=symbol_table) for templ_arg in templ_arg_exprs]

  def make_symbol_kind(self, symbol_table: SymbolTable) -> SymbolKind:
    concrete_templ_types = self._get_template_arguments_or_none(symbol_table=symbol_table)
    func_kind = self.func_expr.make_symbol_kind(symbol_table=symbol_table)

    if concrete_templ_types is not None:  # is a templ type specialization, keeps kind.
      assert func_kind == SymbolKind.FUNCTION  # func is get/set
      func_caller_kind = self.func_arg_exprs[0].make_symbol_kind(symbol_table=symbol_table)
      if func_caller_kind not in {SymbolKind.FUNCTION, SymbolKind.TYPE}:
        raise_error('Cannot specify template parameters for symbol of kind %r' % func_kind, self.pos)
      return func_caller_kind
    if func_kind == SymbolKind.FUNCTION:
      if self._is_type_union_call(symbol_table=symbol_table):
        return SymbolKind.TYPE
      return SymbolKind.VARIABLE
    if func_kind == SymbolKind.TYPE:
      called_type = self.func_expr.make_as_type(symbol_table=symbol_table)
      if called_type.constructor is None:
        raise_error('Cannot call constructor of type %r because it does not exist' % called_type, self.pos)
      return SymbolKind.VARIABLE
    raise_error('Cannot call non-function symbol, is a %r' % func_kind, self.pos)

  def _make_func_expr_as_func_caller(self, symbol_table: SymbolTable) -> FunctionSymbolCaller:
    assert self.make_symbol_kind(symbol_table=symbol_table) == SymbolKind.VARIABLE
    func_kind = self.func_expr.make_symbol_kind(symbol_table=symbol_table)
    if func_kind == SymbolKind.FUNCTION:
      return self.func_expr.make_as_func_caller(symbol_table=symbol_table)
    elif func_kind == SymbolKind.TYPE:
      called_type = self.func_expr.make_as_type(symbol_table=symbol_table)
      if called_type.constructor is None:
        raise_error('Cannot call constructor of type %r because it does not exist' % called_type, self.pos)
      return called_type.constructor_caller
    else:
      raise_error('Cannot call an expression of kind %r' % (
        self.func_expr.make_symbol_kind(symbol_table=symbol_table)), self.pos)

  def _is_special_call(self, builtin_func_identifier: str, symbol_table: SymbolTable):
    return isinstance(self.func_expr, IdentifierExpressionAst) \
           and self.func_expr.identifier == builtin_func_identifier \
           and builtin_func_identifier in symbol_table.builtin_symbols

  def _is_size_call(self, symbol_table: SymbolTable):
    # TODO: make this a normal compile time function operating on types
    return self._is_special_call('size', symbol_table=symbol_table)

  def _is_type_union_call(self, symbol_table: SymbolTable):
    # TODO: make this a normal compile time function operating on types
    return self._is_special_call('|', symbol_table=symbol_table)

  def _is_is_call(self, symbol_table: SymbolTable):
    # TODO: make this a normal compile time function operating on types
    return self._is_special_call('is', symbol_table=symbol_table)

  def make_as_val(self, symbol_table: SymbolTable, context: CodegenContext) -> TypedValue:
    assert self.make_symbol_kind(symbol_table=symbol_table) == SymbolKind.VARIABLE
    with context.use_pos(self.pos):
      if self._is_size_call(symbol_table=symbol_table):
        if len(self.func_arg_exprs) != 1:
          raise_error('Must call size(type: Type) with exactly one argument', self.pos)
        size_of_type = self.func_arg_exprs[0].make_as_type(symbol_table=symbol_table)
        from sleepy.types import LLVM_SIZE_TYPE
        ir_val = ir.Constant(LLVM_SIZE_TYPE, size_of_type.size) if context.emits_ir else None
        return TypedValue(typ=SLEEPY_LONG, ir_val=ir_val)
      if self._is_is_call(symbol_table=symbol_table):
        if len(self.func_arg_exprs) != 2:
          raise_error('Operator "is" must be used between exactly two arguments', self.pos)
        val_arg, type_arg = self.func_arg_exprs
        if val_arg.make_symbol_kind(symbol_table=symbol_table) != SymbolKind.VARIABLE:
          raise_error('Left-side argument of operator "is" must be a value', self.pos)
        if type_arg.make_symbol_kind(symbol_table=symbol_table) != SymbolKind.TYPE:
          raise_error('Right-side argument of operator "is" must be a type', self.pos)
        check_type = type_arg.make_as_type(symbol_table=symbol_table)
        check_value = val_arg.make_as_val(symbol_table=symbol_table, context=context)
        check_value = check_value.copy_collapse(context=context, name='check_val')
        if context.emits_ir:
          ir_val = make_ir_val_is_type(check_value.ir_val, check_value.narrowed_type, check_type, context=context)
        else:
          ir_val = None
        return TypedValue(typ=SLEEPY_BOOL, ir_val=ir_val)

      # default case
      func_caller = self._make_func_expr_as_func_caller(symbol_table=symbol_table)
      return make_call_ir_with_narrowing(
        self.pos,
        caller=func_caller, arg_expressions=self.func_arg_exprs, symbol_table=symbol_table, context=context)

  def make_as_func_caller(self, symbol_table: SymbolTable) -> FunctionSymbolCaller:
    assert self.make_symbol_kind(symbol_table=symbol_table) == SymbolKind.FUNCTION
    template_arguments = self._get_template_arguments_or_none(symbol_table=symbol_table)
    if template_arguments is None:
      return self.func_expr.make_as_func_caller(symbol_table=symbol_table)
    else:
      func_caller = self.func_arg_exprs[0].make_as_func_caller(symbol_table=symbol_table)
      if func_caller.template_parameters is not None:
        raise_error('Cannot specify template types multiple times', self.pos)
      return func_caller.copy_with_template_arguments(template_arguments)

  def make_as_type(self, symbol_table: SymbolTable) -> Type:
    assert self.make_symbol_kind(symbol_table=symbol_table) == SymbolKind.TYPE
    if self._is_type_union_call(symbol_table=symbol_table):
      possible_types = [arg.make_as_type(symbol_table=symbol_table) for arg in self.func_arg_exprs]
      return UnionType.from_types(possible_types)
    template_arguments = self._get_template_arguments_or_none(symbol_table=symbol_table)
    assert template_arguments is not None
    signature_type = self.func_arg_exprs[0].make_as_type(symbol_table=symbol_table)
    if len(template_arguments) != len(signature_type.template_param_or_arg):
      if len(template_arguments) == 0:
        raise_error('Type %r needs to be constructed with template arguments for template parameters %r' % (
          signature_type, signature_type.template_param_or_arg), self.pos)
      else:
        raise_error(
          'Type %r with template parameters %r cannot be constructed with template arguments %r' % (
            signature_type, signature_type.template_param_or_arg, template_arguments), self.pos)
    replacements = dict(zip(signature_type.template_param_or_arg, template_arguments))
    return signature_type.replace_types(replacements=replacements)

  def children(self) -> List[AbstractSyntaxTree]:
    return self.func_arg_exprs

  def __repr__(self) -> str:
    return 'CallExpressionAst(func_expr=%r, func_arg_exprs=%r)' % (self.func_expr, self.func_arg_exprs)


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

  def make_symbol_kind(self, symbol_table: SymbolTable) -> SymbolKind:
    return SymbolKind.VARIABLE

  def make_as_val(self, symbol_table: SymbolTable, context: CodegenContext) -> TypedValue:
    with context.use_pos(self.pos):
      arg_val = self.parent_val_expr.make_as_val(symbol_table=symbol_table, context=context)

      def do_member_access(struct_type: Type, caller_context: CodegenContext) -> TypedValue:
        nonlocal arg_val
        assert not isinstance(struct_type, UnionType)
        if not isinstance(struct_type, StructType):
          raise_error(
            'Cannot access a member variable %r of the non-struct type %r' % (self.member_identifier, struct_type),
            self.pos)
        if self.member_identifier not in struct_type.member_identifiers:
          raise_error('Struct type %r has no member variable %r, only available: %r' % (
            struct_type, self.member_identifier, ', '.join(struct_type.member_identifiers)), self.pos)
        member_num = struct_type.get_member_num(self.member_identifier)
        member_type = struct_type.member_types[member_num]

        # TODO This is not entirely correct: It could be that the arg_val is not referenceable for all possible variant
        # types, but that for the concrete `struct_type` we can reference it.
        if arg_val.is_referenceable():
          if context.emits_ir:
            arg_val = arg_val.copy_collapse_as_mutates(context=caller_context, name='parent_ptr')
            arg_val = arg_val.copy_with_implicit_cast(
              ReferenceType(struct_type), context=caller_context, name='parent_ptr_cast')
            parent_ptr = arg_val.ir_val
            gep_indices = [ir.Constant(ir.IntType(32), 0), ir.Constant(ir.IntType(32), member_num)]
            self_ptr = caller_context.builder.gep(
              parent_ptr, gep_indices, name='member_ptr_%s' % self.member_identifier)
          else:
            self_ptr = None
          return TypedValue(typ=ReferenceType(member_type), ir_val=self_ptr)
        else:
          assert arg_val.num_possible_unbindings() == 0
          if context.emits_ir:
            arg_val = arg_val.copy_collapse(context=caller_context, name='parent')
            arg_val = arg_val.copy_with_implicit_cast(struct_type, context=caller_context, name='parent_cast')
            ir_val = struct_type.make_extract_member_val_ir(
              self.member_identifier, struct_ir_val=arg_val.ir_val, context=caller_context)
          else:
            ir_val = None
          return TypedValue(typ=member_type, ir_val=ir_val)

      from functools import partial
      collapsed_arg_val = arg_val.copy_collapse(context=context, name='struct')
      collapsed_type = collapsed_arg_val.narrowed_type
      possible_types = collapsed_type.possible_types if isinstance(collapsed_type, UnionType) else {collapsed_type}
      return make_union_switch_ir(
        case_funcs={(typ,): partial(do_member_access, typ) for typ in possible_types},
        calling_args=[collapsed_arg_val],
        name='extract_member', context=context)

  def make_as_func_caller(self, symbol_table: SymbolTable):
    raise_error('Cannot use member %r as function' % self.member_identifier, self.pos)

  def make_as_type(self, symbol_table: SymbolTable):
    raise_error('Cannot use member %r as type' % self.member_identifier, self.pos)

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

  def make_symbol_kind(self, symbol_table: SymbolTable) -> SymbolKind:
    return SymbolKind.VARIABLE

  def make_as_val(self, symbol_table: SymbolTable, context: CodegenContext) -> TypedValue:
    with context.use_pos(self.pos):
      arg_val = self.arg_expr.make_as_val(symbol_table=symbol_table, context=context)
      if arg_val.num_unbindings + 1 > arg_val.num_possible_unbindings():
        if arg_val.num_possible_unbindings() == 0:
          raise_error(
            'Cannot unbind value of narrowed type %r which is not referencable' % arg_val.narrowed_type, self.pos)
        else:
          raise_error('Cannot unbind value of narrowed type %r more than %s times' % (
            arg_val.narrowed_type, arg_val.num_possible_unbindings()), self.pos)
      return arg_val.copy_unbind()

  def make_as_func_caller(self, symbol_table: SymbolTable):
    raise_error('Cannot use unbind expression as function', self.pos)

  def make_as_type(self, symbol_table: SymbolTable):
    raise_error('Cannot use unbind expression as type', self.pos)

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

  def __init__(self, pos: TreePosition):
    super().__init__(pos)

  def make_type(self, symbol_table: SymbolTable) -> Type:
    raise NotImplementedError()

  def __repr__(self) -> str:
    return 'TypeAst()'


class IdentifierTypeAst(TypeAst):
  """
  IdentifierType -> identifier.
  """

  def __init__(self, pos: TreePosition, type_identifier: str, template_parameters: List[TypeAst]):
    super().__init__(pos)
    self.type_identifier = type_identifier
    self.template_parameters = template_parameters

  def make_type(self, symbol_table: SymbolTable) -> Type:
    if self.type_identifier not in symbol_table:
      raise_error('Unknown type identifier %r' % self.type_identifier, self.pos)
    type_symbol = symbol_table[self.type_identifier]
    if not isinstance(type_symbol, TypeTemplateSymbol):
      raise_error('%r is not a type, but a %r' % (self.type_identifier, type(type_symbol)), self.pos)
    template_arguments = [t_param.make_type(symbol_table=symbol_table) for t_param in self.template_parameters]
    if len(template_arguments) != len(type_symbol.template_parameters):
      if len(template_arguments) == 0:
        raise_error('Type %r needs to be constructed with template arguments for template parameters %r' % (
          self.type_identifier, type_symbol.template_parameters), self.pos)
      else:
        raise_error(
          'Type %r with template parameters %r cannot be constructed with template arguments %r' % (
            self.type_identifier, type_symbol.template_parameters, template_arguments), self.pos)
    return type_symbol.get_type(template_arguments=template_arguments)

  def children(self) -> List[AbstractSyntaxTree]:
    return self.template_parameters

  def __repr__(self) -> str:
    return 'IdentifierType(type_identifier=%r, template_types=%r)' % (self.type_identifier, self.template_parameters)


class UnionTypeAst(TypeAst):
  """
  IdentifierType -> identifier.
  """

  def __init__(self, pos: TreePosition, variant_types: List[TypeAst]):
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

  def __init__(self, pos: TreePosition, identifier: str):
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
      raise_error('Annotations with name %r not allowed here, only allowed: %s' % (
        annotation.identifier, ', '.join(ast.allowed_annotation_identifiers)), annotation.pos)
    if any(annotation.identifier == other.identifier for other in ast.annotations):
      raise_error('Cannot add annotation with name %r multiple times' % annotation.identifier, annotation.pos)
    ast.annotations.append(annotation)
  return ast


def make_narrow_type_from_valid_cond_ast(cond_expr_ast: ExpressionAst,
                                         cond_holds: bool,
                                         symbol_table: SymbolTable):
  # TODO: This is super limited currently: Will only work for "local_var is Type", and not(...), nothing more.
  # noinspection PyProtectedMember
  if isinstance(cond_expr_ast, CallExpressionAst) and cond_expr_ast._is_is_call(symbol_table=symbol_table):
    assert len(cond_expr_ast.func_arg_exprs) == 2
    var_expr = cond_expr_ast.func_arg_exprs[0]
    if not isinstance(var_expr, IdentifierExpressionAst):
      return
    var_symbol = var_expr.get_var_symbol(symbol_table=symbol_table)
    check_type = cond_expr_ast.func_arg_exprs[1].make_as_type(symbol_table=symbol_table)
    if cond_holds:
      symbol_table[var_expr.identifier] = var_symbol.copy_narrow_collapsed_type(collapsed_type=check_type)
    else:
      symbol_table[var_expr.identifier] = var_symbol.copy_exclude_collapsed_type(collapsed_type=check_type)
  elif isinstance(cond_expr_ast, CallExpressionAst):
    func_expr = cond_expr_ast.func_expr
    if not isinstance(func_expr, IdentifierExpressionAst):
      return
    if func_expr.identifier == 'not':
      if not len(cond_expr_ast.func_arg_exprs) == 1:
        return
      arg_ast = cond_expr_ast.func_arg_exprs[0]
      make_narrow_type_from_valid_cond_ast(cond_expr_ast=arg_ast, cond_holds=not cond_holds, symbol_table=symbol_table)


class StructDeclarationAst(DeclarationAst):
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

  @property
  def identifier(self) -> str:
    return self.struct_identifier

  def create_symbol(self, symbol_table: SymbolTable, context: CodegenContext) -> Symbol:
    with context.use_pos(self.pos):
      if self.struct_identifier in symbol_table.current_scope_identifiers:
        raise_error('Cannot redefine struct with name %r' % self.struct_identifier, self.pos)

      # symbol table including placeholder types
      struct_symbol_table = symbol_table.make_child_scope(inherit_outer_variables=False)

      placeholder_templ_types = self._collect_placeholder_templ_types(
        self.templ_identifiers, symbol_table=struct_symbol_table)

      # Struct members might reference this struct itself (indirectly).
      # We temporarily add a placeholder to the symbol table so it is defined here.
      struct_identity = StructIdentity(struct_identifier=self.struct_identifier, context=context)
      partial_struct_type = PartialIdentifiedStructType(identity=struct_identity,
                                                        template_param_or_arg=placeholder_templ_types)
      struct_identity.partial_struct_type = partial_struct_type
      struct_symbol_table[self.struct_identifier] = TypeTemplateSymbol(
        template_parameters=placeholder_templ_types, signature_type=partial_struct_type)

      member_types = [type_ast.make_type(symbol_table=struct_symbol_table) for type_ast in self.member_types]
      signature_struct_type = StructType(
        identity=struct_identity, template_param_or_arg=placeholder_templ_types,
        member_identifiers=self.member_identifiers, member_types=member_types, partial_struct_type=partial_struct_type)

      # make constructor / destructor
      signature_struct_type.constructor = build_constructor(struct_type=signature_struct_type,
                                                            parent_symbol_table=struct_symbol_table,
                                                            parent_context=context)
      build_destructor(struct_type=signature_struct_type,
                       parent_symbol_table=symbol_table,
                       parent_context=context)

      # assemble to complete type symbol
      struct_type_symbol = TypeTemplateSymbol(
        template_parameters=placeholder_templ_types, signature_type=signature_struct_type)
      return struct_type_symbol

  def children(self) -> List[AbstractSyntaxTree]:
    return []

  def __repr__(self) -> str:
    return (
            'StructDeclarationAst(struct_identifier=%r, templ_identifiers=%r, member_identifiers=%r, member_types=%r)' % (
      self.struct_identifier, self.templ_identifiers, self.member_identifiers, self.member_types))
