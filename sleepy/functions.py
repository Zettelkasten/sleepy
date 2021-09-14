from typing import List, Optional, cast, Callable

from llvmlite import ir

from sleepy.ast import StatementAst, TypeAst, AnnotationAst, AbstractScopeAst, ReturnStatementAst, AbstractSyntaxTree
from sleepy.grammar import TreePosition
from sleepy.symbols import SymbolTable, Type, CodegenContext, SLEEPY_VOID, FunctionSymbol, SLEEPY_BOOL, \
  ConcreteFunctionFactory, ConcreteFunction, FunctionTemplate, VariableSymbol


class FunctionDeclarationAst(StatementAst):
  """
  Stmt -> func identifier ( TypedIdentifierList ) Scope
  """

  allowed_annotation_identifiers = {'Inline'}
  allowed_arg_annotation_identifiers = {}

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
    func_symbol_table = symbol_table.copy(new_variable_scope=True)

    placeholder_templ_types = self._collect_placeholder_templ_types(
      self.templ_identifiers, symbol_table=func_symbol_table)

    arg_types = self.make_arg_types(func_symbol_table=func_symbol_table)
    if self.return_type is None:
      return_type = SLEEPY_VOID
    else:
      return_type = self.return_type.make_type(symbol_table=func_symbol_table)
    if return_type is None:
      self.raise_error('Need to specify return type of function %r' % self.identifier)
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
      return_type=return_type, arg_identifiers=self.arg_identifiers, arg_types=arg_types, arg_type_narrowings=arg_types,
      is_inline=self.is_inline)
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
    for arg_identifier, arg_type in zip(concrete_func.arg_identifiers, concrete_func.arg_types):
      var_symbol = VariableSymbol(None, arg_type)
      var_symbol.build_ir_alloca(context=body_context, identifier=arg_identifier)
      assert arg_identifier not in body_symbol_table.current_scope_identifiers
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