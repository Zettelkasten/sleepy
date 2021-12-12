from __future__ import annotations

from typing import List, Optional, cast

from llvmlite import ir

from sleepy.ast import TypeAst, AnnotationAst, AbstractScopeAst, ReturnStatementAst, AbstractSyntaxTree, \
  raise_error, DeclarationAst
from sleepy.builtin_symbols import SLEEPY_BOOL
from sleepy.syntactical_analysis.grammar import TreePosition
from sleepy.types import Type, CodegenContext, OverloadSet, ConcreteFunction, FunctionTemplate, \
  PlaceholderTemplateType, SLEEPY_UNIT, TypedValue, ReferenceType
from sleepy.symbols import VariableSymbol, FunctionSymbol, SymbolTable, Symbol


class FunctionDeclarationAst(DeclarationAst):
  """
  Stmt -> func identifier ( TypedIdentifierList ) Scope
  """

  allowed_annotation_identifiers = {'Inline'}
  allowed_arg_annotation_identifiers = {}

  def __init__(self,
               pos: TreePosition,
               identifier: str,
               templ_identifiers: List[str],
               arg_identifiers: List[str],
               arg_types: List[TypeAst],
               arg_annotations: List[List[AnnotationAst]],
               arg_mutates: List[bool],
               return_type: Optional[TypeAst],
               return_annotation_list: Optional[List[AnnotationAst]],
               body_scope: Optional[AbstractScopeAst]):
    super().__init__(pos)
    assert len(arg_identifiers) == len(arg_types) == len(arg_annotations) == len(arg_mutates)
    assert (return_type is None) == (return_annotation_list is None)
    assert body_scope is None or isinstance(body_scope, AbstractScopeAst)
    self._identifier = identifier
    self.templ_identifiers = templ_identifiers
    self.arg_identifiers = arg_identifiers
    self.arg_types = arg_types
    self.arg_annotations = arg_annotations
    self.arg_mutates = arg_mutates
    self.return_type = return_type
    self.return_annotation_list = return_annotation_list
    self.body_scope = body_scope

  @property
  def identifier(self) -> str:
    return self._identifier

  @property
  def is_extern(self) -> bool:
    return self.body_scope is None

  @property
  def is_inline(self) -> bool:
    return any(annotation.identifier == 'Inline' for annotation in self.annotations)

  def make_arg_types(self, func_symbol_table: SymbolTable) -> List[Type]:
    arg_types = [arg_type.make_type(symbol_table=func_symbol_table) for arg_type in self.arg_types]
    if any(arg_type is None for arg_type in arg_types):
      raise_error('Need to specify all parameter types of function %r' % self.identifier, self.pos)
    all_annotation_list = (
      self.arg_annotations + ([self.return_annotation_list] if self.return_annotation_list is not None else []))
    for arg_annotation_list in all_annotation_list:
      for arg_annotation_num, arg_annotation in enumerate(arg_annotation_list):
        if arg_annotation.identifier in arg_annotation_list[:arg_annotation_num]:
          raise_error('Cannot apply annotation with identifier %r twice' % arg_annotation.identifier,
                                     arg_annotation.pos)
        if arg_annotation.identifier not in self.allowed_arg_annotation_identifiers:
          raise_error('Cannot apply annotation with identifier %r, allowed: %r' % (
            arg_annotation.identifier, ', '.join(self.allowed_arg_annotation_identifiers)), arg_annotation.pos)
    return arg_types

  def create_symbol(self, symbol_table: SymbolTable, context: CodegenContext) -> Symbol:
    func_symbol_table = symbol_table.make_child_scope(inherit_outer_variables=False)

    placeholder_templ_types = self._collect_placeholder_templ_types(
      self.templ_identifiers, symbol_table=func_symbol_table)

    arg_types = self.make_arg_types(func_symbol_table=func_symbol_table)
    if self.return_type is None:
      return_type = SLEEPY_UNIT
    else:
      return_type = self.return_type.make_type(symbol_table=func_symbol_table)

    if return_type is None:
      raise_error('Need to specify return type of function %r' % self.identifier, self.pos)

    if self.identifier in symbol_table:
      func_symbol = symbol_table[self.identifier]
      if not isinstance(func_symbol, OverloadSet):
        raise_error('Cannot redefine previously declared non-function %r with a function' % self.identifier,
                         self.pos)
      if not func_symbol.is_undefined_for_arg_types(placeholder_templ_types=placeholder_templ_types,
                                                    arg_types=arg_types):
        raise_error(
          'Cannot override definition of function %r with signature [%s](%s), already declared:\n%s' % (  # noqa
            self.identifier,
            ', '.join([templ_type.identifier for templ_type in placeholder_templ_types]),
            ', '.join(
              ['%s%s' % ('mutates ' if mutates else '', typ) for mutates, typ in zip(self.arg_mutates, arg_types)]),
            # noqa
            func_symbol.make_signature_list_str()), self.pos)

    if self.identifier in {'assert', 'unchecked_assert'}:
      if len(arg_types) < 1 or arg_types[0] != SLEEPY_BOOL:
        raise_error('Builtin %r must be overloaded with signature(Bool condition, ...)' % self.identifier,
                         self.pos)

    if self.is_inline and self.is_extern:
      raise_error('Extern function %r cannot be inlined' % self.identifier, self.pos)

    signature = DeclaredFunctionTemplate(
      placeholder_template_types=placeholder_templ_types,
      return_type=return_type, arg_identifiers=self.arg_identifiers, arg_types=arg_types, arg_type_narrowings=arg_types,
      arg_mutates=self.arg_mutates, ast=self, captured_symbol_table=func_symbol_table, captured_context=context)

    return FunctionSymbol([signature])
    # TODO: check symbol table generically: e.g. use a concrete functions with the template type arguments

  def build_body_ir(self, parent_symbol_table: SymbolTable, concrete_func: ConcreteFunction,
                    body_context: CodegenContext, ir_func_args: Optional[List[ir.Value]] = None):
    assert not self.is_extern

    template_parameter_names = [t.identifier for t in concrete_func.signature.placeholder_templ_types]
    template_arguments = concrete_func.template_arguments

    body_symbol_table = parent_symbol_table.make_child_scope(
      inherit_outer_variables=False, new_function=concrete_func,
      type_substitutions=zip(template_parameter_names, template_arguments))

    # add arguments as variables
    for arg_identifier, arg_type in zip(concrete_func.arg_identifiers, concrete_func.arg_types):
      # if arg is of type T, we create a local variable of type Ref[T]
      var_symbol = VariableSymbol(None, ReferenceType(arg_type))
      assert arg_identifier not in body_symbol_table.current_scope_identifiers
      body_symbol_table[arg_identifier] = var_symbol
    # set function argument values
    if body_context.emits_ir:
      assert ir_func_args is not None
      assert len(ir_func_args) == len(concrete_func.arg_identifiers)
      for arg_identifier, ir_arg, arg_mutates in zip(concrete_func.arg_identifiers, ir_func_args, concrete_func.arg_mutates):  # noqa
        arg_symbol = body_symbol_table[arg_identifier]
        assert isinstance(arg_symbol, VariableSymbol)
        if arg_mutates:
          arg_symbol.build_ir_alloca(context=body_context, identifier=arg_identifier, initial_ir_alloca=ir_arg)
        else:  # not arg_mutates, default case
          ir_arg.name = arg_identifier
          arg_symbol.build_ir_alloca(context=body_context, identifier=arg_identifier)
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
      if concrete_func.return_type != SLEEPY_UNIT:
        raise_error('Not all branches within function declaration of %r return something' % self.identifier,
                               return_ast.pos)
      return_ast.build_ir(symbol_table=body_symbol_table, context=body_context)
    assert body_context.is_terminated

  def children(self) -> List[AbstractSyntaxTree]:
    return cast(List[AbstractSyntaxTree], [el for lst in self.arg_annotations for el in lst])\
           + self.arg_types + ([self.return_type] if self.return_type else [])

  def __repr__(self) -> str:
    return (
      'FunctionDeclarationAst(identifier=%r, arg_identifiers=%r, arg_types=%r, '
      'return_type=%r, %s)' % (
        self.identifier, self.arg_identifiers, self.arg_types, self.return_type,
        'extern' if self.is_extern else self.body_scope))


class ConcreteDeclaredFunction(ConcreteFunction):
  def __init__(self, signature: FunctionTemplate,
               ir_func: Optional[ir.Function],
               concrete_template_types: List[Type],
               return_type: Type, arg_types: List[Type],
               arg_type_narrowings: List[Type],
               arg_mutates: List[bool],
               ast: FunctionDeclarationAst,
               captured_symbol_table: SymbolTable,
               captured_context: CodegenContext):
    super().__init__(
      signature, ir_func, concrete_template_types, return_type, arg_types, arg_type_narrowings,
      parameter_mutates=arg_mutates)
    self.ast = ast
    self.captured_symbol_table = captured_symbol_table
    self.captured_context = captured_context

  @property
  def is_inline(self) -> bool:
    return self.ast.is_inline

  def make_inline_func_call_ir(self, func_args: List[TypedValue],
                               caller_context: CodegenContext) -> Optional[ir.Value]:
    assert len(func_args) == len(self.arg_identifiers)
    assert caller_context.emits_ir
    assert not caller_context.is_terminated

    return_val_ir_alloca = caller_context.alloca_at_entry(
      self.return_type.ir_type, name='return_%s_alloca' % self.ast.identifier)

    collect_block = caller_context.builder.append_basic_block('collect_return_%s_block' % self.ast.identifier)
    inline_context = caller_context.copy_with_inline_func(
      self, return_ir_alloca=return_val_ir_alloca, return_collect_block=collect_block)
    self.ast.build_body_ir(
      parent_symbol_table=self.captured_symbol_table, concrete_func=self, body_context=inline_context,
      ir_func_args=[arg.ir_val for arg in func_args])
    assert inline_context.is_terminated
    assert not collect_block.is_terminated
    # use the caller_context instead of reusing the inline_context
    # because the inline_context will be terminated
    caller_context.builder = ir.IRBuilder(collect_block)

    return_val = caller_context.builder.load(return_val_ir_alloca, name='return_%s' % self.ast.identifier)
    assert not caller_context.is_terminated
    return return_val

  def build_ir(self):
    assert not self.captured_context.is_terminated
    with self.captured_context.use_pos(self.ast.pos):
      if self.ast.is_extern:
        if self.captured_symbol_table.has_extern_func(self.ast.identifier):
          extern_concrete_func = self.captured_symbol_table.get_extern_func(self.ast.identifier)
          if not extern_concrete_func.has_same_signature_as(self):
            raise_error('Cannot redefine extern func %r previously declared as %s with new signature %s' % (
              self.ast.identifier, extern_concrete_func.signature.to_signature_str(),
              self.signature.to_signature_str()), self.ast.pos)
          # Sometimes the ir_func has not been set for a previously declared extern func,
          # e.g. because it was declared in an inlined func.
          should_declare_func = extern_concrete_func.ir_func is None
        else:
          self.captured_symbol_table.add_extern_func(self.ast.identifier, self)
          should_declare_func = True
      else:
        assert not self.ast.is_extern
        should_declare_func = True
      if self.captured_context.emits_ir and not self.ast.is_inline:
        if should_declare_func:
          self.make_ir_func(identifier=self.ast.identifier, extern=self.ast.is_extern, context=self.captured_context)
        else:
          assert not should_declare_func
          assert self.ast.is_extern and self.captured_symbol_table.has_extern_func(self.ast.identifier)
          self.ir_func = self.captured_symbol_table.get_extern_func(self.ast.identifier).ir_func

      if self.ast.is_extern:
        return

      if self.ast.is_inline:
        # check symbol tables without emitting ir
        self.ast.build_body_ir(
          parent_symbol_table=self.captured_symbol_table, concrete_func=self,
          body_context=self.captured_context.copy_without_builder())
      else:
        assert not self.is_inline
        if self.captured_context.emits_ir:
          body_block = self.ir_func.append_basic_block(name='entry')
          body_context = self.captured_context.copy_with_func(self, builder=ir.IRBuilder(body_block))
        else:
          body_context = self.captured_context.copy_with_func(self, builder=None)  # proceed without emitting ir.
        self.ast.build_body_ir(
          parent_symbol_table=self.captured_symbol_table, concrete_func=self, body_context=body_context,
          ir_func_args=self.ir_func.args if body_context.emits_ir else None)


class DeclaredFunctionTemplate(FunctionTemplate):
  def __init__(self, placeholder_template_types: List[PlaceholderTemplateType],
               return_type: Type,
               arg_identifiers: List[str],
               arg_types: List[Type],
               arg_type_narrowings: List[Type],
               arg_mutates: List[bool],
               ast: FunctionDeclarationAst,
               captured_symbol_table: SymbolTable,
               captured_context: CodegenContext):
    super().__init__(
      placeholder_template_types=placeholder_template_types, return_type=return_type, arg_identifiers=arg_identifiers,
      arg_types=arg_types, arg_type_narrowings=arg_type_narrowings, arg_mutates=arg_mutates)
    self.ast = ast
    self.captured_symbol_table = captured_symbol_table
    self.captured_context = captured_context

  def _get_concrete_function(self, concrete_template_arguments: List[Type],
                             concrete_parameter_types: List[Type],
                             concrete_narrowed_parameter_types: List[Type],
                             concrete_return_type: Type) -> ConcreteFunction:
    concrete_function = ConcreteDeclaredFunction(
      signature=self, ir_func=None, concrete_template_types=concrete_template_arguments,
      return_type=concrete_return_type, arg_types=concrete_parameter_types,
      arg_type_narrowings=concrete_narrowed_parameter_types, arg_mutates=self.arg_mutates, ast=self.ast,
      captured_symbol_table=self.captured_symbol_table, captured_context=self.captured_context)
    self.initialized_templ_funcs[tuple(concrete_template_arguments)] = concrete_function
    concrete_function.build_ir()
    return concrete_function