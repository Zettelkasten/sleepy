from __future__ import annotations

# Operator precedence: * / stronger than + - stronger than == != < <= > >=
from typing import List, Optional, Union, Tuple, cast

from llvmlite import ir

from sleepy.errors import SemanticError
from sleepy.grammar import TreePosition
from sleepy.symbols import FunctionSymbol, VariableSymbol, Type, SLEEPY_VOID, SLEEPY_BOOL, SLEEPY_CHAR, SymbolTable, \
  TypeSymbol, \
  StructType, ConcreteFunction, UnionType, can_implicit_cast_to, \
  make_implicit_cast_to_ir_val, make_ir_val_is_type, build_initial_ir, CodegenContext, get_common_type, \
  SLEEPY_CHAR_PTR, FunctionTemplate, TemplateType, ConcreteFunctionFactory, TypeFactory, try_infer_templ_types, Symbol, \
  FunctionSymbolCaller

SLOPPY_OP_TYPES = {'*', '/', '+', '-', '==', '!=', '<', '>', '<=', '>', '>=', 'is', '='}
from abc import ABC, abstractmethod



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

  def resolve_func_call_by_identifier(self, func_identifier: str, templ_types: Optional[List[Type]],
                                      func_arg_exprs: List[ExpressionAst],
                                      symbol_table: SymbolTable) -> List[ConcreteFunction]:
    if func_identifier not in symbol_table:
      self.raise_error('Function %r called before declared' % func_identifier)
    symbol = symbol_table[func_identifier]
    if not isinstance(symbol, FunctionSymbol):
      self.raise_error('Cannot call non-function %r' % func_identifier)
    func_caller = FunctionSymbolCaller(func=symbol, templ_types=templ_types)
    return self.resolve_func_call(
      func_caller=func_caller, func_arg_exprs=func_arg_exprs, symbol_table=symbol_table)

  def resolve_func_call(self, func_caller: FunctionSymbolCaller, func_arg_exprs: List[ExpressionAst],
                        symbol_table: SymbolTable) -> List[ConcreteFunction]:
    func, templ_types = func_caller.func, func_caller.templ_types
    calling_types = [arg_expr.make_val_type(symbol_table=symbol_table) for arg_expr in func_arg_exprs]

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
    for concrete_func in possible_concrete_funcs:
      called_mutables = [arg_expr.is_val_mutable(symbol_table=symbol_table) for arg_expr in func_arg_exprs]
      for arg_identifier, arg_mutable, called_mutable in zip(
          concrete_func.arg_identifiers, concrete_func.arg_mutables, called_mutables):
        if not called_mutable and arg_mutable:
          self.raise_error('Cannot call function %s%s declared with mutable parameter %r with immutable argument' % (
              func.identifier, concrete_func.signature.to_signature_str(), arg_identifier))
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

  def _build_func_call(self, func_caller: FunctionSymbolCaller, func_arg_exprs: List[ExpressionAst],
                       symbol_table: SymbolTable, context: CodegenContext):
    func, templ_types = func_caller.func, func_caller.templ_types
    possible_concrete_funcs = self.resolve_func_call(
      func_caller=func_caller, func_arg_exprs=func_arg_exprs, symbol_table=symbol_table)
    calling_types = [arg_expr.make_val_type(symbol_table=symbol_table) for arg_expr in func_arg_exprs]

    if templ_types is None:
      templ_types = self._infer_templ_args(func=func, calling_types=calling_types)
    assert templ_types is not None

    for concrete_func in possible_concrete_funcs:
      if concrete_func in context.inline_func_call_stack:
        self.raise_error(
          'An inlined function can not call itself (indirectly), but got inline call stack: %s -> %s' % (
            ' -> '.join(str(inline_func) for inline_func in context.inline_func_call_stack), concrete_func))

    if context.emits_ir:
      ir_func_args = [
        func_arg_expr.make_ir_val(symbol_table=symbol_table, context=context) for func_arg_expr in func_arg_exprs]
      from sleepy.symbols import make_func_call_ir
      return_ir_val = make_func_call_ir(
        func=func, templ_types=templ_types, calling_arg_types=calling_types, calling_ir_args=ir_func_args,
        context=context)
    else:
      return_ir_val = None

    # apply type narrowings
    for arg_num, func_arg_expr in enumerate(func_arg_exprs):
      narrowed_arg_types = [concrete_func.arg_type_narrowings[arg_num] for concrete_func in possible_concrete_funcs]
      narrowed_arg_type = get_common_type(narrowed_arg_types)
      if isinstance(func_arg_expr, IdentifierExpressionAst):
        var_symbol = symbol_table[func_arg_expr.identifier]
        assert isinstance(var_symbol, VariableSymbol)
        symbol_table[func_arg_expr.identifier] = var_symbol.copy_narrow_type(narrowed_arg_type)

    # special handling of 'assert' call
    if func.base in {symbol_table.inbuilt_symbols.get(identifier) for identifier in {'assert', 'unchecked_assert'}}:
      assert len(func_arg_exprs) >= 1
      condition_expr = func_arg_exprs[0]
      make_narrow_type_from_valid_cond_ast(condition_expr, cond_holds=True, symbol_table=symbol_table)
    return return_ir_val

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

  @abstractmethod
  def children(self) -> List[AbstractSyntaxTree]:
    pass

  def _collect_placeholder_templ_types(self, templ_identifiers: List[str],
                                       symbol_table: SymbolTable) -> List[TemplateType]:
    templ_types = []
    for templ_type_identifier in templ_identifiers:
      if templ_type_identifier in symbol_table.current_scope_identifiers:
        self.raise_error('Cannot declare template variable %r multiple times' % templ_type_identifier)
      template_type = TemplateType(templ_type_identifier)
      templ_types.append(template_type)
      template_type_factory = TypeFactory(placeholder_templ_types=[], signature_type=template_type)
      template_type_symbol = TypeSymbol(type_factory=template_type_factory)
      symbol_table.current_scope_identifiers.append(templ_type_identifier)
      symbol_table[templ_type_identifier] = template_type_symbol
    return templ_types


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

  def make_var_is_mutable(self, arg_name, arg_annotation_list, default):
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
    return default if (not has_mutable and not has_const) else has_mutable

  def __repr__(self):
    """
    :rtype: str
    """
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

  def build_scope_ir(self, scope_symbol_table, scope_context):
    """
    :param SymbolTable scope_symbol_table:
    :param CodegenContext scope_context:
    """
    with scope_context.use_pos(self.pos):
      for stmt in self.stmt_list:
        if scope_context.is_terminated:
          stmt.raise_error('Code is unreachable')
        stmt.build_ir(symbol_table=scope_symbol_table, context=scope_context)

  def __repr__(self):
    """
    :rtype: str
    """
    return 'AbstractScopeAst(%s)' % ', '.join([repr(stmt) for stmt in self.stmt_list])

  def children(self):
    return self.stmt_list


class TopLevelAst(AbstractSyntaxTree):
  """
  TopLevelExpr.
  """

  def __init__(self, pos: TreePosition, stmt_list: List[StatementAst]):
    super().__init__(pos)
    self.stmt_list = stmt_list

  def make_module_ir_and_symbol_table(self, module_name: str, emit_debug: bool) -> (ir.Module, SymbolTable):
    module = ir.Module(name=module_name)

    io_func_type = ir.FunctionType(ir.VoidType(), ())
    ir_io_func = ir.Function(module, io_func_type, name='io')
    root_block = ir_io_func.append_basic_block(name='entry')
    root_builder = ir.IRBuilder(root_block)
    symbol_table = SymbolTable()
    context = CodegenContext(builder=root_builder)
    context.emits_debug = emit_debug

    build_initial_ir(symbol_table=symbol_table, context=context)
    assert context.ir_func_malloc is not None and context.ir_func_free is not None
    for stmt in self.stmt_list:
      assert not context.is_terminated
      stmt.build_ir(symbol_table=symbol_table, context=context)
    assert not context.is_terminated
    root_builder.ret_void()
    context.is_terminated = True

    return module, symbol_table

  def children(self) -> List[AbstractSyntaxTree]:
    return self.stmt_list

  def __repr__(self):
    """
    :rtype: str
    """
    return 'TopLevelAst(%s)' % self.stmt_list


class FunctionDeclarationAst(StatementAst):
  """
  Stmt -> func identifier ( TypedIdentifierList ) Scope
  """

  allowed_annotation_identifiers = {'Inline'}
  allowed_arg_annotation_identifiers = {'Const', 'Mutable'}

  def __init__(self, pos: TreePosition, identifier: str, templ_identifiers: List[str], arg_identifiers: List[str],
               arg_types: List[TypeAst], arg_annotations: List[List[AnnotationAst]], return_type: Optional[TypeAst],
               return_annotation_list: Optional[List[AnnotationAst]], body_scope: Optional[AbstractScopeAst]):
    """
    :param pos:
    :param identifier:
    :param templ_identifiers:
    :param arg_identifiers:
    :param arg_types:
    :param arg_annotations:
    :param return_type:
    :param return_annotation_list:
    :param body_scope: body, or None if extern function.
    """
    super().__init__(pos)
    assert len(arg_identifiers) == len(arg_types) == len(arg_annotations)
    assert (return_type is None) == (return_annotation_list is None)
    assert body_scope is None or isinstance(body_scope, AbstractScopeAst)
    self.identifier = identifier
    self.templ_identifiers = templ_identifiers
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

  def make_arg_types(self, func_symbol_table: SymbolTable) -> List[Type]:
    arg_types = [arg_type.make_type(symbol_table=func_symbol_table) for arg_type in self.arg_types]
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

  def build_ir(self, symbol_table: SymbolTable, context: CodegenContext):
    func_symbol_table = symbol_table.copy()
    func_symbol_table.current_scope_identifiers = []

    placeholder_templ_types = self._collect_placeholder_templ_types(
      self.templ_identifiers, symbol_table=func_symbol_table)

    arg_types = self.make_arg_types(func_symbol_table=func_symbol_table)
    arg_mutables = [
      self.make_var_is_mutable('parameter %r' % arg_identifier, arg_annotation_list, default=False)
      for arg_identifier, arg_type, arg_annotation_list in zip(self.arg_identifiers, arg_types, self.arg_annotations)]
    for arg_identifier, arg_type, arg_mutable in zip(self.arg_identifiers, arg_types, arg_mutables):
     if not arg_type.is_pass_by_ref() and arg_mutable:
       self.raise_error(
         'Type %r of mutable parameter %r needs to have pass-by-reference semantics (annotated by @RefType)' % (
           arg_identifier, arg_type))
    if self.return_type is None:
      return_type = SLEEPY_VOID
    else:
      return_type = self.return_type.make_type(symbol_table=func_symbol_table)
    if return_type is None:
      self.raise_error('Need to specify return type of function %r' % self.identifier)
    if return_type == SLEEPY_VOID:
      return_mutable = False
    else:
      return_mutable = self.make_var_is_mutable('return type', self.return_annotation_list, default=False)
    if not return_type.is_pass_by_ref() and return_mutable:
      self.raise_error(
        'Type %r of return value needs to have pass-by-reference semantics (annotated by @RefType)' % (
          return_type))
    if self.identifier in symbol_table:
      func_symbol = symbol_table[self.identifier]
      if not isinstance(func_symbol, FunctionSymbol):
        self.raise_error('Cannot redefine previously declared non-function %r with a function' % self.identifier)
    else:
      func_symbol = FunctionSymbol(identifier=self.identifier, returns_void=(return_type == SLEEPY_VOID))
      symbol_table[self.identifier] = func_symbol
      func_symbol_table[self.identifier] = func_symbol
    if func_symbol in {symbol_table.inbuilt_symbols.get(name) for name in {'assert', 'unchecked_assert'}}:
      if len(arg_types) < 1 or arg_types[0] != SLEEPY_BOOL:
        self.raise_error('Inbuilt %r must be overloaded with signature(Bool condition, ...)' % self.identifier)
    if not func_symbol.is_undefined_for_arg_types(
        placeholder_templ_types=placeholder_templ_types, arg_types=arg_types):
      self.raise_error(
        'Cannot override definition of function %r with template types %r and parameter types %r, already declared:\n%s' % (  # noqa
        self.identifier, ', '.join([templ_type.identifier for templ_type in placeholder_templ_types]),
        ', '.join([str(arg_type) for arg_type in arg_types]), func_symbol.make_signature_list_str()))
    if func_symbol.returns_void != (return_type == SLEEPY_VOID):
      self.raise_error(
        'Function declared with name %r must consistently return a value or consistently return void' %
        self.identifier)
    if self.is_inline and self.is_extern:
      self.raise_error('Extern function %r cannot be inlined' % self.identifier)

    class DeclaredConcreteFunctionFactory(ConcreteFunctionFactory):
      def build_concrete_func_ir(self_, concrete_func: ConcreteFunction):
        with context.use_pos(self.pos):  # TODO: Better debug symbols for inlined functions
          if self.is_extern:
            if func_symbol_table.has_extern_func(self.identifier):
              extern_concrete_func = func_symbol_table.get_extern_func(self.identifier)
              if not extern_concrete_func.has_same_signature_as(concrete_func):
                self.raise_error('Cannot redefine extern func %r previously declared as %s with new signature %s' % (
                  self.identifier, extern_concrete_func.signature.to_signature_str(),
                  concrete_func.signature.to_signature_str()))
              # Sometimes the ir_func has not been set for a previously declared extern func,
              # e.g. because it was declared in an inlined func.
              should_declare_func = extern_concrete_func.ir_func is None
            else:
              func_symbol_table.add_extern_func(self.identifier, concrete_func)
              should_declare_func = True
          else:
            assert not self.is_extern
            should_declare_func = True
          if context.emits_ir and not self.is_inline:
            if should_declare_func:
              concrete_func.make_ir_func(
                identifier=self.identifier, extern=self.is_extern, symbol_table=func_symbol_table, context=context)
            else:
              assert not should_declare_func
              assert self.is_extern and func_symbol_table.has_extern_func(self.identifier)
              concrete_func.ir_func = func_symbol_table.get_extern_func(self.identifier).ir_func

          if self.is_extern:
            return concrete_func

          if self.is_inline:
            def make_inline_func_call_ir(ir_func_args: List[ir.values.Value],
                                         caller_context: CodegenContext) -> Optional[ir.values.Value]:
              assert len(ir_func_args) == len(self.arg_identifiers)
              assert caller_context.emits_ir
              assert not caller_context.is_terminated
              if concrete_func.return_type == SLEEPY_VOID:
                return_val_ir_alloca = None
              else:
                return_val_ir_alloca = caller_context.alloca_at_entry(
                  concrete_func.return_type.ir_type, name='return_%s_alloca' % self.identifier)
              collect_block = caller_context.builder.append_basic_block('collect_return_%s_block' % self.identifier)
              inline_context = caller_context.copy_with_inline_func(
                concrete_func, return_ir_alloca=return_val_ir_alloca, return_collect_block=collect_block)
              self._build_body_ir(
                parent_symbol_table=func_symbol_table, concrete_func=concrete_func, body_context=inline_context,
                ir_func_args=ir_func_args)
              assert inline_context.is_terminated
              assert not collect_block.is_terminated
              # use the caller_context instead of reusing the inline_context
              # because the inline_context will be terminated
              caller_context.builder = ir.IRBuilder(collect_block)
              if concrete_func.return_type == SLEEPY_VOID:
                return_val = None
              else:
                return_val = caller_context.builder.load(return_val_ir_alloca, name='return_%s' % self.identifier)
              assert not caller_context.is_terminated
              return return_val

            concrete_func.make_inline_func_call_ir = make_inline_func_call_ir
            # check symbol tables without emitting ir
            self._build_body_ir(
              parent_symbol_table=func_symbol_table, concrete_func=concrete_func,
              body_context=context.copy_without_builder())
          else:
            assert not self.is_inline
            if context.emits_ir:
              body_block = concrete_func.ir_func.append_basic_block(name='entry')
              body_context = context.copy_with_func(concrete_func, builder=ir.IRBuilder(body_block))
            else:
              body_context = context.copy_with_func(concrete_func, builder=None)  # proceed without emitting ir.
            self._build_body_ir(
              parent_symbol_table=func_symbol_table, concrete_func=concrete_func, body_context=body_context,
              ir_func_args=concrete_func.ir_func.args)

    concrete_func_factory = DeclaredConcreteFunctionFactory()
    signature_ = FunctionTemplate(
      concrete_func_factory=concrete_func_factory, placeholder_templ_types=placeholder_templ_types,
      return_type=return_type, return_mutable=return_mutable, arg_identifiers=self.arg_identifiers,
      arg_types=arg_types, arg_mutables=arg_mutables, arg_type_narrowings=arg_types, is_inline=self.is_inline)
    func_symbol.add_signature(signature_)

    # TODO: check symbol table generically: e.g. use a concrete functions with the template type arguments

    # Always generate IR for functions without template types
    if not self.is_inline and len(placeholder_templ_types) == 0:
      signature_.get_concrete_func(concrete_templ_types=[])

  def _build_body_ir(self, parent_symbol_table: SymbolTable, concrete_func: ConcreteFunction,
                     body_context: CodegenContext, ir_func_args: Optional[List[ir.values.Value]] = None):
    assert not self.is_extern
    assert self.is_inline == concrete_func.signature.is_inline
    body_symbol_table = parent_symbol_table.copy_with_new_current_func(concrete_func)

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

  def children(self) -> List[AbstractSyntaxTree]:
    return cast(List[AbstractSyntaxTree], [el for lst in self.arg_annotations for el in lst])\
           + self.arg_types + ([self.return_type] if self.return_type else [])

  def __repr__(self):
    """
    :rtype: str
    """
    return (
      'FunctionDeclarationAst(identifier=%r, arg_identifiers=%r, arg_types=%r, '
      'return_type=%r, %s)' % (self.identifier, self.arg_identifiers, self.arg_types,
      self.return_type, 'extern' if self.is_extern else self.body_scope))


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
      self.expr.make_val_type(symbol_table=symbol_table)
      if context.emits_ir:
        # ignore return value.
        _ = self.expr.make_ir_val(symbol_table=symbol_table, context=context)

  def children(self) -> List[AbstractSyntaxTree]:
    return [self.expr]

  def __repr__(self):
    """
    :rtype: str
    """
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
            return_val_type, symbol_table.current_func.return_type, ir_val, context=context, name='return_val_cast')
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

  def __repr__(self):
    """
    :rtype: str
    """
    return 'ReturnStatementAst(return_exprs=%r)' % self.return_exprs


class StructDeclarationAst(StatementAst):
  allowed_annotation_identifiers = frozenset({'ValType', 'RefType'})

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

  def is_pass_by_ref(self) -> bool:
    val_type = any(annotation.identifier == 'ValType' for annotation in self.annotations)
    ref_type = any(annotation.identifier == 'RefType' for annotation in self.annotations)
    if val_type and ref_type:
      self.raise_error('Cannot apply annotation %r and %r at the same time' % ('ValType', 'RefType'))
    return ref_type  # fall back to pass by value.

  def build_ir(self, symbol_table: SymbolTable, context: CodegenContext):
    with context.use_pos(self.pos):
      if self.struct_identifier in symbol_table.current_scope_identifiers:
        self.raise_error('Cannot redefine struct with name %r' % self.struct_identifier)

      struct_symbol_table = symbol_table.copy()  # symbol table including placeholder types
      placeholder_templ_types = self._collect_placeholder_templ_types(
        self.templ_identifiers, symbol_table=struct_symbol_table)

      member_types = [type_ast.make_type(symbol_table=struct_symbol_table) for type_ast in self.member_types]
      member_mutables = [
        self.make_var_is_mutable(member_identifier, member_annotations, default=False)
        for member_identifier, member_annotations in zip(self.member_identifiers, self.member_annotations)]

      signature_struct_type = StructType(
        struct_identifier=self.struct_identifier, templ_types=placeholder_templ_types,
        member_identifiers=self.member_identifiers, member_types=member_types, member_mutables=member_mutables,
        pass_by_ref=self.is_pass_by_ref())

      struct_type_factory = TypeFactory(
        placeholder_templ_types=placeholder_templ_types, signature_type=signature_struct_type)

      # make constructor / destructor
      constructor = signature_struct_type.build_constructor(parent_symbol_table=symbol_table, parent_context=context)
      signature_struct_type.constructor = constructor
      signature_struct_type.build_destructor(parent_symbol_table=symbol_table, parent_context=context)

      # assemble to complete type symbol
      struct_type_symbol = TypeSymbol(type_factory=struct_type_factory)
      symbol_table.add_to_current_scope(self.struct_identifier, struct_type_symbol)

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
  allowed_annotation_identifiers = frozenset({'Const', 'Mutable'})

  def __init__(self, pos: TreePosition, var_target: ExpressionAst, var_val: ExpressionAst,
               declared_var_type: Optional[TypeAst]):
    super().__init__(pos)
    assert isinstance(var_target, ExpressionAst)
    self.var_target = var_target
    self.var_val = var_val
    self.declared_var_type = declared_var_type

  def is_declaration(self, symbol_table):
    """
    :param SymbolTable symbol_table:
    :rtype: bool
    """
    if not isinstance(self.var_target, IdentifierExpressionAst):
      return False
    var_identifier = self.var_target.identifier
    if var_identifier not in symbol_table.current_scope_identifiers:
      return True
    assert var_identifier in symbol_table
    symbol = symbol_table[var_identifier]
    if not isinstance(symbol, VariableSymbol):
      self.raise_error('Cannot assign non-variable %r to a variable' % var_identifier)
    ptr_type = self.var_target.make_val_type(symbol_table=symbol_table)
    if symbol.narrowed_var_type != ptr_type:
      self.raise_error('Cannot redefine variable %r of type %r with new type %r' % (
        var_identifier, symbol.narrowed_var_type, ptr_type))

  def build_ir(self, symbol_table, context):
    """
    :param SymbolTable symbol_table:
    :param CodegenContext context:
    """
    with context.use_pos(self.pos):
      if self.declared_var_type is not None:
        stated_type = self.declared_var_type.make_type(symbol_table=symbol_table)  # type: Optional[Type]
      else:
        stated_type = None  # type: Optional[Type]
      if not self.var_val.make_symbol_kind(symbol_table=symbol_table) == Symbol.Kind.VARIABLE:
        self.raise_error('Can only reassign variables')
      val_type = self.var_val.make_val_type(symbol_table=symbol_table)
      if val_type == SLEEPY_VOID:
        self.raise_error('Cannot assign void to variable')
      if stated_type is not None:
        if not can_implicit_cast_to(val_type, stated_type):
          self.raise_error('Cannot assign variable with stated type %r a value of type %r' % (stated_type, val_type))
      declared_mutable = self.make_var_is_mutable('left-hand-side', self.annotations, default=None)
      val_mutable = self.var_val.is_val_mutable(symbol_table=symbol_table)

      if self.is_declaration(symbol_table=symbol_table):
        assert isinstance(self.var_target, IdentifierExpressionAst)
        var_identifier = self.var_target.identifier
        assert var_identifier not in symbol_table.current_scope_identifiers
        if stated_type is not None:
          declared_type = stated_type
        else:
          declared_type = val_type
        if declared_mutable is None:
          declared_mutable = False
        # declare new variable, override entry in symbol_table (maybe it was defined in an outer scope before).
        symbol = VariableSymbol(None, var_type=declared_type, mutable=declared_mutable)
        symbol.build_ir_alloca(context=context, identifier=var_identifier)
        symbol_table[var_identifier] = symbol
        symbol_table.current_scope_identifiers.append(var_identifier)
      else:
        # variable name in this scope already declared. just check that types match, but do not change symbol_table.
        declared_type = self.var_target.make_declared_val_type(symbol_table=symbol_table)
        assert declared_type is not None
        if stated_type is not None and not can_implicit_cast_to(stated_type, declared_type):
            self.raise_error('Cannot redefine variable of type %r with new type %r' % (declared_type, stated_type))
        if not can_implicit_cast_to(val_type, declared_type):
          self.raise_error('Cannot redefine variable of type %r with variable of type %r' % (declared_type, val_type))
        if not self.var_target.is_val_assignable(symbol_table=symbol_table):
          self.raise_error('Cannot reassign member of a non-mutable variable')
        if declared_mutable is None:
          declared_mutable = val_mutable
        if declared_mutable != val_mutable:
          if declared_mutable:
            self.raise_error('Cannot redefine a variable declared as non-mutable to mutable')
          else:
            self.raise_error('Cannot redefine a variable declared as mutable to non-mutable')
      assert declared_type is not None
      if declared_mutable and not val_mutable:
        self.raise_error('Cannot assign a non-mutable value of type %r to a mutable variable' % declared_type)
      assert self.var_target.is_val_assignable(symbol_table=symbol_table)

      # if we assign to a variable, narrow type to val_type
      if isinstance(self.var_target, IdentifierExpressionAst):
        assert self.var_target.identifier in symbol_table
        symbol = symbol_table[self.var_target.identifier]
        assert isinstance(symbol, VariableSymbol)
        narrowed_symbol = symbol.copy_with_narrowed_type(val_type)
        assert not isinstance(narrowed_symbol, UnionType) or len(narrowed_symbol.possible_types) > 0
        symbol_table[self.var_target.identifier] = narrowed_symbol

      if context.emits_ir:
        ir_val = self.var_val.make_ir_val(symbol_table=symbol_table, context=context)
        ir_val = make_implicit_cast_to_ir_val(val_type, declared_type, ir_val, context=context, name='assign_cast')
        ir_ptr = self.var_target.make_ir_val_ptr(symbol_table=symbol_table, context=context)
        assert ir_ptr is not None
        context.builder.store(ir_val, ir_ptr)

  def children(self) -> List[AbstractSyntaxTree]:
    return [self.var_target, self.var_val, self.declared_var_type]

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
    with context.use_pos(self.pos):
      cond_type = self.condition_val.make_val_type(symbol_table=symbol_table)
      if not cond_type == SLEEPY_BOOL:
        self.raise_error('Cannot use expression of type %r as if-condition' % cond_type)
  
      true_symbol_table, false_symbol_table = symbol_table.copy(), symbol_table.copy()
      make_narrow_type_from_valid_cond_ast(self.condition_val, cond_holds=True, symbol_table=true_symbol_table)
      make_narrow_type_from_valid_cond_ast(self.condition_val, cond_holds=False, symbol_table=false_symbol_table)
  
      if context.emits_ir:
        ir_cond = self.condition_val.make_ir_val(symbol_table=symbol_table, context=context)
        assert isinstance(ir_cond, ir.values.Value)
        true_block = context.builder.append_basic_block('true_branch')  # type: ir.Block
        false_block = context.builder.append_basic_block('false_branch')  # type: ir.Block
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
          continue_block = context.builder.append_basic_block('continue_branch')  # type: ir.Block
          if not true_context.is_terminated:
            true_context.builder.branch(continue_block)
          if not false_context.is_terminated:
            false_context.builder.branch(continue_block)
          context.builder = ir.IRBuilder(continue_block)

  def children(self) -> List[AbstractSyntaxTree]:
    return [self.condition_val, self.true_scope, self.false_scope]

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
      cond_type = self.condition_val.make_val_type(symbol_table=symbol_table)
      if not cond_type == SLEEPY_BOOL:
        self.raise_error('Cannot use expression of type %r as while-condition' % cond_type)

      body_symbol_table = symbol_table.copy()
      make_narrow_type_from_valid_cond_ast(self.condition_val, cond_holds=True, symbol_table=body_symbol_table)

      if context.emits_ir:
        cond_ir = self.condition_val.make_ir_val(symbol_table=symbol_table, context=context)
        body_block = context.builder.append_basic_block('while_body')  # type: ir.Block
        continue_block = context.builder.append_basic_block('continue_branch')  # type: ir.Block
        context.builder.cbranch(cond_ir, body_block, continue_block)
        context.builder = ir.IRBuilder(continue_block)

        body_context = context.copy_with_builder(ir.IRBuilder(body_block))
        self.body_scope.build_scope_ir(scope_symbol_table=body_symbol_table, scope_context=body_context)
        if not body_context.is_terminated:
          body_cond_ir = self.condition_val.make_ir_val(symbol_table=symbol_table, context=body_context)
          body_context.builder.cbranch(body_cond_ir, body_block, continue_block)
      else:
        body_context = context.copy_without_builder()
        self.body_scope.build_scope_ir(scope_symbol_table=body_symbol_table, scope_context=body_context)

      # TODO: Do a fix-point iteration over the body and wait until the most general type no longer changes.
      symbol_table.reset_narrowed_types()
      make_narrow_type_from_valid_cond_ast(self.condition_val, cond_holds=False, symbol_table=symbol_table)

  def children(self) -> List[AbstractSyntaxTree]:
    return [self.condition_val, self.body_scope]

  def __repr__(self):
    """
    :rtype: str
    """
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
  def make_val_type(self, symbol_table: SymbolTable) -> Type:
    raise NotImplementedError()

  def make_declared_val_type(self, symbol_table: SymbolTable) -> Type:
    if self.is_val_mutable(symbol_table=symbol_table) or self.is_val_assignable(symbol_table=symbol_table):
      raise NotImplementedError()
    return self.make_val_type(symbol_table=symbol_table)

  def is_val_mutable(self, symbol_table: SymbolTable) -> bool:
    return False

  def is_val_assignable(self, symbol_table: SymbolTable) -> bool:
    return False

  def make_ir_val(self, symbol_table: SymbolTable, context: CodegenContext) -> Optional[ir.values.Value]:
    """
    :param SymbolTable symbol_table:
    :param CodegenContext context:
    :rtype: ir.values.Value
    :return: The value this expression is evaluated to
    """
    assert context.emits_ir
    raise NotImplementedError()

  def make_ir_val_ptr(self, symbol_table: SymbolTable,
                      context: CodegenContext) -> Optional[ir.instructions.Instruction]:
    assert context.emits_ir
    if self.is_val_mutable(symbol_table=symbol_table) or self.is_val_assignable(symbol_table=symbol_table):
      raise NotImplementedError()
    return None

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

  def make_val_type(self, symbol_table):
    """
    :param SymbolTable symbol_table:
    :rtype: Type
    """
    if self.op == 'is':
      assert isinstance(self.right_expr, IdentifierExpressionAst)
      type_expr = IdentifierTypeAst(
        self.right_expr.pos, type_identifier=self.right_expr.identifier, templ_types=[])
      type_expr.make_type(symbol_table=symbol_table)  # just check that type exists
      return SLEEPY_BOOL
    operand_exprs = [self.left_expr, self.right_expr]
    possible_concrete_funcs = self.resolve_func_call_by_identifier(
      func_identifier=self.op, templ_types=None, func_arg_exprs=operand_exprs, symbol_table=symbol_table)
    return get_common_type([concrete_func.return_type for concrete_func in possible_concrete_funcs])

  def is_val_mutable(self, symbol_table):
    """
    :param SymbolTable symbol_table:
    :rtype: Type
    """
    if self.op == 'is':
      return False
    operand_exprs = [self.left_expr, self.right_expr]
    possible_concrete_funcs = self.resolve_func_call_by_identifier(
      func_identifier=self.op, templ_types=None, func_arg_exprs=operand_exprs, symbol_table=symbol_table)
    return all(concrete_func.return_mutable for concrete_func in possible_concrete_funcs)

  def make_ir_val(self, symbol_table, context):
    """
    :param SymbolTable symbol_table:
    :param CodegenContext context:
    :rtype: ir.values.Value
    """
    with context.use_pos(self.pos):
      assert context.emits_ir
      if self.op == 'is':
        assert isinstance(self.right_expr, IdentifierExpressionAst)
        check_type_expr = IdentifierTypeAst(
          self.right_expr.pos, type_identifier=self.right_expr.identifier, templ_types=[])
        check_type = check_type_expr.make_type(symbol_table=symbol_table)
        val_type = self.left_expr.make_val_type(symbol_table=symbol_table)
        ir_val = self.left_expr.make_ir_val(symbol_table=symbol_table, context=context)
        return make_ir_val_is_type(ir_val, val_type, check_type, context=context)
      func = symbol_table[self.op]
      assert isinstance(func, FunctionSymbol)
      operand_exprs = [self.left_expr, self.right_expr]
      return_val = self._build_func_call(
        func_caller=FunctionSymbolCaller(func), func_arg_exprs=operand_exprs, symbol_table=symbol_table,
        context=context)
      assert return_val is not None
      return return_val

  def make_as_func_caller(self, symbol_table: SymbolTable):
    self.raise_error('Cannot use result of operator %r as function' % self.op)

  def make_as_type(self, symbol_table: SymbolTable):
    self.raise_error('Cannot use result of operator %r as type' % self.op)

  def children(self) -> List[AbstractSyntaxTree]:
    return [self.left_expr, self.right_expr]

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

  def make_symbol_kind(self, symbol_table: SymbolTable) -> Symbol.Kind:
    return Symbol.Kind.VARIABLE

  def make_val_type(self, symbol_table):
    """
    :param SymbolTable symbol_table:
    :rtype: Type
    """
    operand_exprs = [self.expr]
    possible_concrete_funcs = self.resolve_func_call_by_identifier(
      func_identifier=self.op, templ_types=None, func_arg_exprs=operand_exprs, symbol_table=symbol_table)
    return get_common_type([concrete_func.return_type for concrete_func in possible_concrete_funcs])

  def is_val_mutable(self, symbol_table):
    """
    :param SymbolTable symbol_table:
    :rtype: Type
    """
    operand_exprs = [self.expr]
    possible_concrete_funcs = self.resolve_func_call_by_identifier(
      func_identifier=self.op, templ_types=None, func_arg_exprs=operand_exprs, symbol_table=symbol_table)
    return all(concrete_func.return_mutable for concrete_func in possible_concrete_funcs)

  def make_ir_val(self, symbol_table, context):
    """
    :param SymbolTable symbol_table:
    :param CodegenContext context:
    :rtype: ir.values.Value
    """
    with context.use_pos(self.pos):
      assert context.emits_ir
      func = symbol_table[self.op]
      assert isinstance(func, FunctionSymbol)
      operand_exprs = [self.expr]
      return self._build_func_call(
        func_caller=FunctionSymbolCaller(func), func_arg_exprs=operand_exprs, symbol_table=symbol_table,
        context=context)

  def children(self) -> List[AbstractSyntaxTree]:
    return [self.expr]

  def make_as_func_caller(self, symbol_table: SymbolTable):
    self.raise_error('Cannot use result of operator %r as function' % self.op)

  def make_as_type(self, symbol_table: SymbolTable):
    self.raise_error('Cannot use result of operator %r as type' % self.op)

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

  def make_symbol_kind(self, symbol_table: SymbolTable) -> Symbol.Kind:
    return Symbol.Kind.VARIABLE

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
    with context.use_pos(self.pos):
      assert context.emits_ir
      return ir.Constant(self.constant_type.ir_type, self.constant_val)

  def make_as_func_caller(self, symbol_table: SymbolTable):
    self.raise_error('Cannot use constant expression as function')

  def make_as_type(self, symbol_table: SymbolTable):
    self.raise_error('Cannot use constant expression as type')

  def children(self) -> List[AbstractSyntaxTree]:
    return []

  def __repr__(self):
    """
    :rtype: str
    """
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

  def make_val_type(self, symbol_table):
    """
    :param SymbolTable symbol_table:
    :rtype: Type
    """
    assert 'Str' in symbol_table.inbuilt_symbols is not None
    str_symbol = symbol_table.inbuilt_symbols['Str']
    assert isinstance(str_symbol, TypeSymbol)
    return str_symbol.get_type(concrete_templ_types=[])

  def is_val_mutable(self, symbol_table):
    """
    :param SymbolTable symbol_table:
    :rtype: Type
    """
    return True

  def make_ir_val(self, symbol_table: SymbolTable, context: CodegenContext) -> ir.values.Value:
    """
    :param SymbolTable symbol_table:
    :param CodegenContext context:
    :rtype: ir.values.Value
    """
    with context.use_pos(self.pos):
      assert context.emits_ir
      assert 'Str' in symbol_table.inbuilt_symbols is not None
      str_symbol = symbol_table.inbuilt_symbols['Str']
      assert isinstance(str_symbol, TypeSymbol)
      str_type = str_symbol.get_type(concrete_templ_types=[])
      assert isinstance(str_type, StructType)
      assert str_type.member_identifiers == ['start', 'length', 'alloc_length']

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

      str_ir_alloca = str_type.make_ir_alloca(context=context)
      str_type.make_store_members_ir(
        member_ir_vals=[ir_start, ir_length, ir_length], struct_ir_alloca=str_ir_alloca, context=context)
      return str_ir_alloca

  def make_as_func_caller(self, symbol_table: SymbolTable):
    self.raise_error('Cannot use string literal as function')

  def make_as_type(self, symbol_table: SymbolTable):
    self.raise_error('Cannot use string literal as type')

  def children(self) -> List[AbstractSyntaxTree]:
    return []

  def __repr__(self):
    """
    :rtype: str
    """
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
    if isinstance(symbol, TypeSymbol):
      type = self.make_as_type(symbol_table=symbol_table)
      if type.constructor is None:
        self.raise_error('Cannot call non-existing constructor of type %r' % type)
      symbol = type.constructor
    if not isinstance(symbol, FunctionSymbol):
      self.raise_error('Cannot reference a non-function %r, got a %s' % (self.identifier, symbol.kind))
    return symbol

  def make_val_type(self, symbol_table: SymbolTable) -> Type:
    return self.get_var_symbol(symbol_table=symbol_table).narrowed_var_type

  def make_declared_val_type(self, symbol_table: SymbolTable) -> Type:
    return self.get_var_symbol(symbol_table=symbol_table).declared_var_type

  def is_val_mutable(self, symbol_table: SymbolTable) -> bool:
    return self.get_var_symbol(symbol_table=symbol_table).mutable

  def is_val_assignable(self, symbol_table: SymbolTable) -> bool:
    return True

  def make_ir_val(self, symbol_table: SymbolTable, context: CodegenContext) -> ir.values.Value:
    with context.use_pos(self.pos):
      assert context.emits_ir
      symbol = self.get_var_symbol(symbol_table=symbol_table)
      return context.builder.load(symbol.ir_alloca, name=self.identifier)

  def make_ir_val_ptr(self, symbol_table: SymbolTable, context: CodegenContext) -> ir.instructions.Instruction:
    with context.use_pos(self.pos):
      assert context.emits_ir
      symbol = self.get_var_symbol(symbol_table=symbol_table)
      assert symbol.ir_alloca is not None
      return symbol.ir_alloca

  def make_as_func_caller(self, symbol_table: SymbolTable) -> FunctionSymbolCaller:
    return FunctionSymbolCaller(func=self.get_func_symbol(symbol_table=symbol_table))

  def make_as_type(self, symbol_table: SymbolTable) -> Type:
    type_symbol = self.get_symbol(symbol_table=symbol_table)
    if not isinstance(type_symbol, TypeSymbol):
      self.raise_error('%r is not a type, but a %r' % (self.identifier, type_symbol.kind))
    return type_symbol.type_factory.signature_type

  def children(self) -> List[AbstractSyntaxTree]:
    return []

  def __repr__(self):
    """
    :rtype: str
    """
    return 'VariableExpressionAst(var_identifier=%r)' % self.identifier


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
    if not isinstance(self.func_expr, IdentifierExpressionAst) or self.func_expr.identifier not in {'get', 'set'}:
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

  def make_val_type(self, symbol_table: SymbolTable) -> Type:
    if self._is_size_call(symbol_table=symbol_table):
      from sleepy.symbols import SLEEPY_LONG
      return SLEEPY_LONG
    func_caller = self._make_func_expr_as_func_caller(symbol_table=symbol_table)
    possible_concrete_funcs = self.resolve_func_call(
      func_caller=func_caller, func_arg_exprs=self.func_arg_exprs, symbol_table=symbol_table)
    return get_common_type([concrete_func.return_type for concrete_func in possible_concrete_funcs])

  def is_val_mutable(self, symbol_table: SymbolTable) -> bool:
    if self._is_size_call(symbol_table=symbol_table):
      return False
    func_caller = self._make_func_expr_as_func_caller(symbol_table=symbol_table)
    possible_concrete_funcs = self.resolve_func_call(
      func_caller=func_caller, func_arg_exprs=self.func_arg_exprs,
      symbol_table=symbol_table)
    return all(concrete_func.return_mutable for concrete_func in possible_concrete_funcs)

  def _is_size_call(self, symbol_table: SymbolTable):
    # TODO: make this a normal compile time function operating on types
    if not isinstance(self.func_expr, IdentifierExpressionAst):
      return False
    from sleepy.symbols import SLEEPY_SIZE_FUNC
    if self.func_expr.make_as_func_caller(symbol_table=symbol_table).func.base != SLEEPY_SIZE_FUNC.base:
      return False
    return True

  def make_ir_val(self, symbol_table: SymbolTable, context: CodegenContext) -> ir.values.Value:
    assert self.make_symbol_kind(symbol_table=symbol_table) == Symbol.Kind.VARIABLE
    with context.use_pos(self.pos):
      assert context.emits_ir
      if self._is_size_call(symbol_table=symbol_table):
        if len(self.func_arg_exprs) != 1:
          self.raise_error('Must call size(type: Type) with exactly one argument')
        size_of_type = self.func_arg_exprs[0].make_as_type(symbol_table=symbol_table)
        from sleepy.symbols import LLVM_SIZE_TYPE
        return ir.Constant(LLVM_SIZE_TYPE, size_of_type.size)

      return self._build_func_call(
        func_caller=self._make_func_expr_as_func_caller(symbol_table=symbol_table), func_arg_exprs=self.func_arg_exprs,
        symbol_table=symbol_table, context=context)

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

  def make_as_type(self, symbol_table: SymbolTable):
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

  def __repr__(self):
    """
    :rtype: str
    """
    return 'CallExpressionAst(func_expr=%r, func_arg_exprs=%r)' % (self.func_expr, self.func_arg_exprs)


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

  def make_symbol_kind(self, symbol_table: SymbolTable) -> Symbol.Kind:
    return Symbol.Kind.VARIABLE

  def make_val_type(self, symbol_table):
    """
    :param SymbolTable symbol_table:
    :rtype: Type
    """
    parent_type = self.parent_val_expr.make_val_type(symbol_table=symbol_table)
    return self._make_member_val_type(parent_type, self.member_identifier, symbol_table)

  def make_declared_val_type(self, symbol_table):
    """
    :param SymbolTable symbol_table:
    :rtype: Type
    """
    return self.make_val_type(symbol_table=symbol_table)

  def make_ir_val(self, symbol_table, context):
    """
    :param SymbolTable symbol_table:
    :param CodegenContext context:
    :rtype: ir.values.Value
    """
    with context.use_pos(self.pos):
      assert context.emits_ir
      parent_type = self.parent_val_expr.make_val_type(symbol_table=symbol_table)
      assert isinstance(parent_type, StructType)
      parent_ir_val = self.parent_val_expr.make_ir_val(symbol_table=symbol_table, context=context)
      return parent_type.make_extract_member_val_ir(
        self.member_identifier, struct_ir_val=parent_ir_val, context=context)

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

  def is_val_assignable(self, symbol_table):
    """
    :param SymbolTable symbol_table:
    :rtype bool:
    """
    return self.parent_val_expr.is_val_mutable(symbol_table=symbol_table)

  def make_ir_val_ptr(self, symbol_table, context):
    """
    :param SymbolTable symbol_table:
    :param CodegenContext context:
    :rtype: ir.instructions.Instruction
    """
    with context.use_pos(self.pos):
      assert context.emits_ir
      parent_type = self.parent_val_expr.make_val_type(symbol_table=symbol_table)
      assert isinstance(parent_type, StructType)
      member_num = parent_type.get_member_num(self.member_identifier)
      parent_ptr = self.parent_val_expr.make_ir_val_ptr(symbol_table=symbol_table, context=context)
      if parent_type.is_pass_by_ref():  # parent_ptr has type struct**
        # dereference to get struct*.
        parent_ptr = context.builder.load(parent_ptr, 'load_struct')
      gep_indices = [ir.Constant(ir.IntType(32), 0), ir.Constant(ir.IntType(32), member_num)]
      return context.builder.gep(parent_ptr, gep_indices, name='member_ptr_%s' % self.member_identifier)

  def make_as_func_caller(self, symbol_table: SymbolTable):
    self.raise_error('Cannot use member %r as function' % self.member_identifier)

  def make_as_type(self, symbol_table: SymbolTable):
    self.raise_error('Cannot use member %r as type' % self.member_identifier)

  def children(self) -> List[AbstractSyntaxTree]:
    return [self.parent_val_expr]

  def __repr__(self):
    """
    :rtype: str
    """
    return 'MemberExpressionAst(parent_val_expr=%r, member_identifier=%r)' % (
      self.parent_val_expr, self.member_identifier)


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
  def __init__(self, pos: TreePosition, type_identifier: str, templ_types: List[TypeAst]):
    super().__init__(pos)
    self.type_identifier = type_identifier
    self.templ_types = templ_types

  def make_type(self, symbol_table: SymbolTable) -> Type:
    if self.type_identifier not in symbol_table:
      self.raise_error('Unknown type identifier %r' % self.type_identifier)
    type_symbol = symbol_table[self.type_identifier]
    if not isinstance(type_symbol, TypeSymbol):
      self.raise_error('%r is not a type, but a %r' % (self.type_identifier, type(type_symbol)))
    templ_type_symbols = [
      template_type.make_type(symbol_table=symbol_table) for template_type in self.templ_types]
    if len(templ_type_symbols) != len(type_symbol.type_factory.placeholder_templ_types):
      if len(templ_type_symbols) == 0:
        self.raise_error('Type %r needs to be constructed with template arguments for template parameters %r' % (
          self.type_identifier, type_symbol.type_factory.placeholder_templ_types))
      else:
        self.raise_error(
          'Type %r with placeholder template parameters %r cannot be constructed with template arguments %r' % (
            self.type_identifier, type_symbol.type_factory.placeholder_templ_types, templ_type_symbols))
    return type_symbol.get_type(concrete_templ_types=templ_type_symbols)

  def children(self) -> List[AbstractSyntaxTree]:
    return self.templ_types

  def __repr__(self):
    """
    :rtype: str
    """
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

  def make_type(self, symbol_table):
    """
    :param SymbolTable symbol_table:
    :rtype: Type
    """
    concrete_variant_types = [variant_type.make_type(symbol_table=symbol_table) for variant_type in self.variant_types]
    return get_common_type(concrete_variant_types)

  def children(self) -> List[AbstractSyntaxTree]:
    return self.variant_types

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

  def children(self) -> List[AbstractSyntaxTree]:
    return []

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


def make_narrow_type_from_valid_cond_ast(cond_expr_ast, cond_holds, symbol_table):
  """
  :param ExpressionAst cond_expr_ast:
  :param bool cond_holds:
  :param SymbolTable symbol_table:
  """
  # TODO: This is super limited currently: Will only work for if(local_var is Type), nothing more.
  if isinstance(cond_expr_ast, BinaryOperatorExpressionAst) and cond_expr_ast.op == 'is':
    var_expr = cond_expr_ast.left_expr
    if not isinstance(var_expr, IdentifierExpressionAst):
      return
    var_symbol = var_expr.get_var_symbol(symbol_table=symbol_table)
    assert isinstance(cond_expr_ast.right_expr, IdentifierExpressionAst)
    check_type_expr = IdentifierTypeAst(
      cond_expr_ast.right_expr.pos, cond_expr_ast.right_expr.identifier, templ_types=[])
    check_type = check_type_expr.make_type(symbol_table=symbol_table)
    if cond_holds:
      symbol_table[var_expr.identifier] = var_symbol.copy_narrow_type(check_type)
    else:
      symbol_table[var_expr.identifier] = var_symbol.copy_exclude_type(check_type)
