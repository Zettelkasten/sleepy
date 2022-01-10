from __future__ import annotations

from typing import List, Optional, cast

from llvmlite import ir

from sleepy.ast import TypeAst, AnnotationAst, AbstractScopeAst, AbstractSyntaxTree, \
  DeclarationAst
from sleepy.builtin_symbols import SLEEPY_BOOL
from sleepy.errors import raise_error
from sleepy.ir_generation import make_ir_end_block_return
from sleepy.symbols import VariableSymbol, SymbolTable, Symbol
from sleepy.syntactical_analysis.grammar import TreePosition
from sleepy.types import Type, CodegenContext, OverloadSet, ConcreteFunction, FunctionTemplate, \
  PlaceholderTemplateType, SLEEPY_UNIT, TypedValue, ReferenceType, CleanupHandlingCG


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
        raise_error('Cannot redefine previously declared non-function %r with a function' % self.identifier, self.pos)
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
        raise_error('Builtin %r must be overloaded with signature(Bool condition, ...)' % self.identifier, self.pos)

    if self.is_inline and self.is_extern:
      raise_error('Extern function %r cannot be inlined' % self.identifier, self.pos)

    signature = DeclaredFunctionTemplate(
      placeholder_template_types=placeholder_templ_types,
      return_type=return_type, arg_identifiers=self.arg_identifiers, arg_types=arg_types, arg_type_narrowings=arg_types,
      arg_mutates=self.arg_mutates, ast=self, captured_symbol_table=func_symbol_table, captured_context=context)

    return OverloadSet(self.identifier, [signature])
    # TODO: check symbol table generically: e.g. use a concrete functions with the template type arguments

  def check_body(self, symbol_table: SymbolTable):
    # TODO
    pass

  def build_body_ir(self, parent_symbol_table: SymbolTable,
                    concrete_func: ConcreteFunction,
                    body_context: CleanupHandlingCG,
                    ir_func_args: List[ir.Value]):
    assert body_context.base.emits_ir
    assert not self.is_extern

    template_parameter_names = [t.identifier for t in concrete_func.signature.placeholder_templ_types]
    template_arguments = concrete_func.template_arguments

    argument_symbols = {}
    for identifier, ir_value, mutates, typ in zip(concrete_func.arg_identifiers,
                                                  ir_func_args,
                                                  concrete_func.arg_mutates,
                                                  concrete_func.arg_types):
      if mutates:
        argument_symbols[identifier] = VariableSymbol.make_ref_to_variable(ir_value, ReferenceType(typ),
                                                                           identifier, body_context.base)
      else:
        ir_value.name = identifier
        symbol = VariableSymbol.make_new_variable(ReferenceType(typ), identifier, body_context.base)
        body_context.base.builder.store(ir_value, symbol.ir_alloca)
        argument_symbols[identifier] = symbol

    body_symbol_table = parent_symbol_table.make_child_scope(
      inherit_outer_variables=False, new_function=concrete_func,
      type_substitutions=zip(template_parameter_names, template_arguments),
      new_symbols=argument_symbols)

    # build function body
    self.body_scope.build_scope_ir(scope_symbol_table=body_symbol_table, scope_context=body_context)

    if not body_context.base.all_paths_returned and not concrete_func.return_type is SLEEPY_UNIT:
      self.raise_missing_return_error()

    if not self.is_inline:
      make_ir_end_block_return(body_context)

    body_context.base.all_paths_returned = True

  def raise_missing_return_error(self):
    return_pos = TreePosition(
      self.pos.word,
      self.pos.from_pos if len(self.body_scope.stmt_list) == 0 else self.body_scope.stmt_list[-1].pos.to_pos,
      self.pos.to_pos)
    raise_error('Not all branches within function declaration of %r return something' % self.identifier, return_pos)

  def children(self) -> List[AbstractSyntaxTree]:
    return cast(List[AbstractSyntaxTree], [el for lst in self.arg_annotations for el in lst]) \
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
    assert not caller_context.all_paths_returned

    inline_context = caller_context.copy_with_cleanup_handling_inline_function(self,
                                                                               existing_entry_block=caller_context.block,
                                                                               identifier=self.ast.identifier)
    self.ast.build_body_ir(
      parent_symbol_table=self.captured_symbol_table,
      concrete_func=self,
      body_context=inline_context,
      ir_func_args=[arg.ir_val for arg in func_args])
    assert inline_context.base.all_paths_returned

    # continue in end_block
    caller_context.switch_to_block(inline_context.scope.end_block)

    return_val = caller_context.builder.load(inline_context.function.return_slot_ir,
                                             name='return_%s' % self.ast.identifier)
    assert not caller_context.all_paths_returned
    return return_val

  def build_ir(self):
    assert not self.captured_context.all_paths_returned
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

      if self.ast.is_inline or not self.captured_context.emits_ir:
        # check symbol tables without emitting ir
        self.ast.check_body(symbol_table=self.captured_symbol_table)
      else:
        body_context = self.captured_context.copy_with_cleanup_handling_func(concrete_function=self,
                                                                             identifier=self.ast.identifier)

        self.ast.build_body_ir(
          parent_symbol_table=self.captured_symbol_table,
          concrete_func=self,
          body_context=body_context,
          ir_func_args=self.ir_func.args)


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
