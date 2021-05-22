

# Operator precedence: * / stronger than + - stronger than == != < <= > >=
from typing import Dict, List, Optional

from llvmlite import ir

from sleepy.grammar import SemanticError, Grammar, Production, AttributeGrammar, TreePosition
from sleepy.lexer import LexerGenerator
from sleepy.parser import ParserGenerator
from sleepy.symbols import FunctionSymbol, VariableSymbol, SLEEPY_DOUBLE, Type, SLEEPY_INT, \
  SLEEPY_LONG, SLEEPY_VOID, SLEEPY_DOUBLE_PTR, SLEEPY_BOOL, SLEEPY_CHAR, SymbolTable, TypeSymbol, \
  StructType, ConcreteFunction, UnionType, can_implicit_cast_to, \
  make_implicit_cast_to_ir_val, make_ir_val_is_type, build_initial_ir, CodegenContext, narrow_type, get_common_type

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
    :rtype: list[ConcreteFunction]
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

    possible_concrete_funcs = symbol.get_concrete_funcs(called_types)
    for concrete_func in possible_concrete_funcs:
      called_mutables = [arg_expr.is_val_mutable(symbol_table=symbol_table) for arg_expr in func_arg_exprs]
      for arg_identifier, arg_mutable, called_mutable in zip(
          concrete_func.arg_identifiers, concrete_func.arg_mutables, called_mutables):
        if not called_mutable and arg_mutable:
          self.raise_error('Cannot call function %s%s declared with mutable parameter %r with immutable argument' % (
              func_identifier, concrete_func.to_signature_str(), arg_identifier))

    for arg_num, func_arg_expr in enumerate(func_arg_exprs):
      narrowed_arg_types = [concrete_func.arg_type_narrowings[arg_num] for concrete_func in possible_concrete_funcs]
      narrowed_arg_type = get_common_type(narrowed_arg_types)
      if isinstance(func_arg_expr, VariableExpressionAst):
        var_symbol = symbol_table[func_arg_expr.var_identifier]
        assert isinstance(var_symbol, VariableSymbol)
        symbol_table[func_arg_expr.var_identifier] = var_symbol.copy_with_narrowed_type(narrowed_arg_type)

    return possible_concrete_funcs

  def _make_func_call_ir(self, func_identifier, func_arg_exprs, symbol_table, context):
    """
    :param str func_identifier:
    :param list[ExpressionAst] func_arg_exprs:
    :param SymbolTable symbol_table:
    :param CodegenContext context:
    :rtype: ir.values.Value|None
    """
    assert context.emits_ir
    self._check_func_call_symbol_table(
      func_identifier=func_identifier, func_arg_exprs=func_arg_exprs, symbol_table=symbol_table)
    assert func_identifier in symbol_table
    func_symbol = symbol_table[func_identifier]
    if isinstance(func_symbol, TypeSymbol):
      func_symbol = func_symbol.constructor_symbol
    assert isinstance(func_symbol, FunctionSymbol)
    calling_arg_types = [arg_expr.make_val_type(symbol_table=symbol_table) for arg_expr in func_arg_exprs]
    ir_func_args = [
      func_arg_expr.make_ir_val(symbol_table=symbol_table, context=context) for func_arg_expr in func_arg_exprs]

    def make_call_func(concrete_func, concrete_calling_arg_types, caller_context):
      """
      :param ConcreteFunction concrete_func:
      :param list[Type] concrete_calling_arg_types:
      :param CodegenContext caller_context:
      :rtype: ir.values.Value
      """
      assert len(concrete_func.arg_types) == len(func_arg_exprs)
      casted_ir_func_args = [make_implicit_cast_to_ir_val(
        from_type=called_arg_type, to_type=declared_arg_type, from_ir_val=ir_func_arg, context=caller_context,
        name='call_arg_%s_cast' % arg_identifier)
        for arg_identifier, ir_func_arg, called_arg_type, declared_arg_type in zip(
          concrete_func.arg_identifiers, ir_func_args, concrete_calling_arg_types, concrete_func.arg_types)]
      assert len(ir_func_args) == len(casted_ir_func_args) == len(func_arg_exprs)
      if concrete_func.is_inline:
        if concrete_func in caller_context.inline_func_call_stack:
          self.raise_error(
            'An inlined function can not call itself (indirectly), but got inline call stack: %s -> %s' % (
              ' -> '.join(str(inline_func) for inline_func in caller_context.inline_func_call_stack), concrete_func))
        assert callable(concrete_func.make_inline_func_call_ir)
        return concrete_func.make_inline_func_call_ir(ir_func_args=casted_ir_func_args, caller_context=caller_context)
      else:
        ir_func = concrete_func.ir_func
        assert ir_func is not None
        assert len(ir_func.args) == len(func_arg_exprs)
        return caller_context.builder.call(ir_func, casted_ir_func_args, name='call_%s' % func_identifier)

    assert func_symbol.has_concrete_func(calling_arg_types)
    possible_concrete_funcs = func_symbol.get_concrete_funcs(calling_arg_types)
    if len(possible_concrete_funcs) == 1:
      return make_call_func(
        concrete_func=possible_concrete_funcs[0], concrete_calling_arg_types=calling_arg_types, caller_context=context)
    else:
      import numpy as np, itertools
      # The arguments which we need to look at to determine which concrete function to call
      # TODO: This could use a better heuristic, only because something is a union type does not mean that it
      # distinguishes different concrete funcs.
      distinguishing_arg_nums = [
        arg_num for arg_num, calling_arg_type in enumerate(calling_arg_types)
        if isinstance(calling_arg_type, UnionType)]
      assert len(distinguishing_arg_nums) >= 1
      # noinspection PyTypeChecker
      distinguishing_calling_arg_types = [
        calling_arg_types[arg_num] for arg_num in distinguishing_arg_nums]  # type: List[UnionType]
      assert all(isinstance(calling_arg, UnionType) for calling_arg in distinguishing_calling_arg_types)
      # To distinguish which concrete func to call, use this table
      block_addresses_distinguished_mapping = np.ndarray(
        shape=tuple(max(calling_arg.possible_type_nums) + 1 for calling_arg in distinguishing_calling_arg_types),
        dtype=ir.values.BlockAddress)

      # Go through all concrete functions, and add one block for each
      concrete_func_caller_contexts = [
        context.copy_with_builder(ir.IRBuilder(context.builder.append_basic_block("call_%s_%s" % (
          func_identifier, '_'.join(str(arg_type) for arg_type in concrete_func.arg_types)))))
        for concrete_func in possible_concrete_funcs]
      for concrete_func, concrete_caller_context in zip(possible_concrete_funcs, concrete_func_caller_contexts):
        concrete_func_block = concrete_caller_context.block
        concrete_func_block_address = ir.values.BlockAddress(context.builder.function, concrete_func_block)
        concrete_func_distinguishing_args = [concrete_func.arg_types[arg_num] for arg_num in distinguishing_arg_nums]
        concrete_func_possible_types_per_arg = [
          [possible_type
            for possible_type in (arg_type.possible_types if isinstance(arg_type, UnionType) else [arg_type])
            if possible_type in calling_arg_type.possible_types]
          for arg_type, calling_arg_type in zip(concrete_func_distinguishing_args, distinguishing_calling_arg_types)]
        assert len(distinguishing_arg_nums) == len(concrete_func_possible_types_per_arg)

        # Register the concrete function in the table
        for expanded_func_types in itertools.product(*concrete_func_possible_types_per_arg):
          assert len(expanded_func_types) == len(distinguishing_calling_arg_types)
          distinguishing_variant_nums = tuple(
            calling_arg_type.get_variant_num(concrete_arg_type)
            for calling_arg_type, concrete_arg_type in zip(distinguishing_calling_arg_types, expanded_func_types))
          assert block_addresses_distinguished_mapping[distinguishing_variant_nums] is None
          block_addresses_distinguished_mapping[distinguishing_variant_nums] = concrete_func_block_address

      # Compute the index we have to look at in the table
      from sleepy.symbols import LLVM_SIZE_TYPE, LLVM_VOID_POINTER_TYPE
      tag_ir_type = ir.types.IntType(8)
      call_block_index_ir = ir.Constant(tag_ir_type, 0)
      for arg_num, calling_arg_type in enumerate(distinguishing_calling_arg_types):
        ir_func_arg = ir_func_args[arg_num]
        base = np.prod(block_addresses_distinguished_mapping.shape[arg_num + 1:], dtype='int32')
        base_ir = ir.Constant(tag_ir_type, base)
        tag_ir = calling_arg_type.make_extract_tag(
          ir_func_arg, context=context, name='call_%s_arg%s_tag_ptr' % (func_identifier, arg_num))
        call_block_index_ir = context.builder.add(call_block_index_ir, context.builder.mul(base_ir, tag_ir))
      call_block_index_ir = context.builder.zext(
        call_block_index_ir, LLVM_SIZE_TYPE, name='call_%s_block_index' % func_identifier)

      # Look it up in the table and call the function
      ir_block_addresses_type = ir.types.VectorType(
        LLVM_VOID_POINTER_TYPE, np.prod(block_addresses_distinguished_mapping.shape))
      ir_block_addresses = ir.values.Constant(ir_block_addresses_type, ir_block_addresses_type.wrap_constant_value(
        list(block_addresses_distinguished_mapping.flatten())))
      ir_call_block_target = context.builder.extract_element(ir_block_addresses, call_block_index_ir)
      indirect_branch = context.builder.branch_indirect(ir_call_block_target)
      for concrete_caller_context in concrete_func_caller_contexts:
        indirect_branch.add_destination(concrete_caller_context.block)

      # Execute the concrete functions and collect their return value
      collect_block = context.builder.append_basic_block("collect_%s_overload" % func_identifier)
      context.builder = ir.IRBuilder(collect_block)
      concrete_func_return_ir_vals = []  # type: List[ir.values.Value]
      for concrete_func, concrete_caller_context in zip(possible_concrete_funcs, concrete_func_caller_contexts):
        concrete_calling_arg_types = [
          narrow_type(calling_arg_type, concrete_arg_type)
          for calling_arg_type, concrete_arg_type in zip(calling_arg_types, concrete_func.arg_types)]
        concrete_return_ir_val = make_call_func(
          concrete_func, concrete_calling_arg_types=concrete_calling_arg_types, caller_context=concrete_caller_context)
        concrete_func_return_ir_vals.append(concrete_return_ir_val)
        assert not concrete_caller_context.is_terminated
        concrete_caller_context.builder.branch(collect_block)
        assert concrete_func.returns_void == func_symbol.returns_void
      assert len(possible_concrete_funcs) == len(concrete_func_return_ir_vals)

      if func_symbol.returns_void:
        return None
      else:
        common_return_type = get_common_type([concrete_func.return_type for concrete_func in possible_concrete_funcs])
        collect_return_ir_phi = context.builder.phi(
          common_return_type.ir_type, name="collect_%s_overload_return" % func_identifier)
        for concrete_return_ir_val, concrete_caller_context in zip(
            concrete_func_return_ir_vals, concrete_func_caller_contexts):
          collect_return_ir_phi.add_incoming(concrete_return_ir_val, concrete_caller_context.block)
        return collect_return_ir_phi

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
    assert context.ir_func_malloc is not None and context.ir_func_free is not None
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
      func_symbol = FunctionSymbol(returns_void=(return_type == SLEEPY_VOID))
      symbol_table[self.identifier] = func_symbol
    if func_symbol.has_concrete_func(arg_types):
      self.raise_error('Cannot override definition of function %r with parameter types %r' % (
        self.identifier, ', '.join([str(arg_type) for arg_type in arg_types])))
    if func_symbol.returns_void != (return_type == SLEEPY_VOID):
      self.raise_error(
        'Function declared with name %r must consistently return a value or consistently return void' % self.identifier)
    arg_mutables = [
      self.make_var_is_mutable('parameter %r' % arg_identifier, arg_type, arg_annotation_list, default=False)
      for arg_identifier, arg_type, arg_annotation_list in zip(self.arg_identifiers, arg_types, self.arg_annotations)]
    if self.is_inline and self.is_extern:
      self.raise_error('Extern function %r cannot be inlined' % self.identifier)
    concrete_func = ConcreteFunction(
      None, return_type=return_type, return_mutable=return_mutable, arg_identifiers=self.arg_identifiers,
      arg_types=arg_types, arg_mutables=arg_mutables, arg_type_narrowings=arg_types, is_inline=self.is_inline)
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
        if concrete_func.return_type == SLEEPY_VOID:
          return_val_ir_alloca = None
        else:
          return_val_ir_alloca = caller_context.builder.alloca(
            concrete_func.return_type.ir_type, name='return_%s_alloca' % self.identifier)
        collect_block = caller_context.builder.append_basic_block('collect_return_%s_block' % self.identifier)
        inline_context = caller_context.copy_with_inline_func(
          concrete_func, return_ir_alloca=return_val_ir_alloca, return_collect_block=collect_block)
        self._build_body_ir(
          parent_symbol_table=symbol_table, concrete_func=concrete_func, body_context=inline_context,
          ir_func_args=ir_func_args)
        assert inline_context.is_terminated
        assert not collect_block.is_terminated
        # use the caller_context instead of reusing the inline_context because the inline_context will be terminated
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
        parent_symbol_table=symbol_table, concrete_func=concrete_func, body_context=context.copy_without_builder())
    else:
      assert not self.is_inline
      if context.emits_ir:
        body_block = concrete_func.ir_func.append_basic_block(name='entry')
        body_context = context.copy_with_builder(ir.IRBuilder(body_block))
      else:
        body_context = context.copy_without_builder()
      self._build_body_ir(
        parent_symbol_table=symbol_table, concrete_func=concrete_func, body_context=body_context,
        ir_func_args=concrete_func.ir_func.args)

  def _build_body_ir(self, parent_symbol_table, concrete_func, body_context, ir_func_args=None):
    """
    :param SymbolTable parent_symbol_table: of the parent function, NOT of the caller.
    :param ConcreteFunction concrete_func:
    :param CodegenContext body_context:
    :param list[ir.values.Value]|None ir_func_args:
    """
    assert not self.is_extern
    assert self.is_inline == concrete_func.is_inline
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
    struct_context = context.copy_without_builder()
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
    constructor_symbol = FunctionSymbol(returns_void=False)
    constructor = ConcreteFunction(
      ir_func=None, return_type=struct_type, return_mutable=True,
      arg_types=member_types, arg_identifiers=member_identifiers, arg_type_narrowings=member_types,
      arg_mutables=member_mutables)
    if context.emits_ir:
      ir_func_type = constructor.make_ir_function_type()
      ir_func_name = symbol_table.make_ir_func_name(self.struct_identifier, extern=False, concrete_func=constructor)
      constructor.ir_func = ir.Function(context.module, ir_func_type, name=ir_func_name)
      self._make_constructor_body_ir(constructor, symbol_table=symbol_table, context=context)
    # notice that we explicitly set return_mutable=False here, even if the constructor mutated the struct.
    constructor_symbol.add_concrete_func(constructor)
    symbol_table[self.struct_identifier] = TypeSymbol(struct_type, constructor_symbol=constructor_symbol)
    symbol_table.current_scope_identifiers.append(self.struct_identifier)

  def _make_constructor_body_ir(self, constructor, symbol_table, context):
    """
    :param ConcreteFunction constructor:
    :param SymbolTable symbol_table:
    :param CodegenContext context:
    """
    # TODO: Make this a new scope.
    struct_type = constructor.return_type
    constructor_block = constructor.ir_func.append_basic_block(name='entry')
    constructor_builder = ir.IRBuilder(constructor_block)
    if self.is_pass_by_ref():  # use malloc
      assert context.ir_func_malloc is not None
      self_ir_alloca_raw = constructor_builder.call(
        context.ir_func_malloc, [struct_type.make_ir_size()], name='self_raw_ptr')
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
    ptr_type = self.var_target.make_narrowed_ptr_type(symbol_table=symbol_table)
    if symbol.narrowed_var_type != ptr_type:
      self.raise_error('Cannot redefine variable %r of type %r with new type %r' % (
        var_identifier, symbol.narrowed_var_type, ptr_type))

  def build_ir(self, symbol_table, context):
    """
    :param SymbolTable symbol_table:
    :param CodegenContext context:
    """
    if self.declared_var_type is not None:
      stated_type = self.declared_var_type.make_type(symbol_table=symbol_table)  # type: Optional[Type]
    else:
      stated_type = None  # type: Optional[Type]
    val_type = self.var_val.make_val_type(symbol_table=symbol_table)
    if val_type == SLEEPY_VOID:
      self.raise_error('Cannot assign void to variable')
    if stated_type is not None:
      if not can_implicit_cast_to(val_type, stated_type):
        self.raise_error('Cannot assign variable with stated type %r a value of type %r' % (stated_type, val_type))
    declared_mutable = self.make_var_is_mutable('left-hand-side', val_type, self.annotations, default=None)
    val_mutable = self.var_val.is_val_mutable(symbol_table=symbol_table)

    if self.is_declaration(symbol_table=symbol_table):
      assert isinstance(self.var_target, VariableTargetAst)
      var_identifier = self.var_target.var_identifier
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
      declared_type = self.var_target.make_declared_ptr_type(symbol_table=symbol_table)
      assert declared_type is not None
      if stated_type is not None and not can_implicit_cast_to(stated_type, declared_type):
          self.raise_error('Cannot redefine variable of type %r with new type %r' % (declared_type, stated_type))
      if not can_implicit_cast_to(val_type, declared_type):
        self.raise_error('Cannot redefine variable of type %r with variable of type %r' % (declared_type, val_type))
      if not self.var_target.is_ptr_reassignable(symbol_table=symbol_table):
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
      self.raise_error('Cannot assign a non-mutable variable a mutable value of type %r' % stated_type)

    # if we assign to a variable, narrow type to val_type
    if isinstance(self.var_target, VariableTargetAst):
      assert self.var_target.var_identifier in symbol_table
      symbol = symbol_table[self.var_target.var_identifier]
      assert isinstance(symbol, VariableSymbol)
      # first reset all type assumptions because we assigned with something new
      symbol.narrowed_var_type = symbol.declared_var_type
      # then narrow it again to the actual value we assigned
      narrowed_symbol = symbol.copy_with_narrowed_type(val_type)
      assert not isinstance(narrowed_symbol, UnionType) or len(narrowed_symbol.possible_types) > 0
      symbol_table[self.var_target.var_identifier] = narrowed_symbol

    if context.emits_ir:
      ir_val = self.var_val.make_ir_val(symbol_table=symbol_table, context=context)
      ir_val = make_implicit_cast_to_ir_val(val_type, declared_type, ir_val, context=context, name='assign_cast')
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
    make_narrow_type_from_valid_cond_ast(self.condition_val, cond_holds=True, symbol_table=true_symbol_table)
    # TODO: Assert the opposite for the false branch

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
      if context.emits_ir:
        continue_block = context.builder.append_basic_block('continue_branch')  # type: ir.Block
        if not true_context.is_terminated:
          true_context.builder.branch(continue_block)
        if not false_context.is_terminated:
          false_context.builder.branch(continue_block)
        context.builder = ir.IRBuilder(continue_block)
    # TODO: Add type assertions for continue branch

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
      body_context = context.copy_with_builder(ir.IRBuilder(body_block))
      context.builder = ir.IRBuilder(continue_block)
    else:
      body_context = context.copy_without_builder()
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
      # TODO: Then it should be a TypeExpressionAst, not a VariableExpressionAst.
      # Probably it's nicer to make an entire new ExpressionAst for `is` Expressions anyway.
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
    possible_concrete_funcs = self._check_func_call_symbol_table(
      func_identifier=self.op, func_arg_exprs=operand_exprs, symbol_table=symbol_table)
    return get_common_type([concrete_func.return_type for concrete_func in possible_concrete_funcs])

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
    possible_concrete_funcs = self._check_func_call_symbol_table(
      func_identifier=self.op, func_arg_exprs=operand_exprs, symbol_table=symbol_table)
    return get_common_type([concrete_func.return_type for concrete_func in possible_concrete_funcs])

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
    return self.get_var_symbol(symbol_table=symbol_table).narrowed_var_type

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
    possible_concrete_funcs = self._check_func_call_symbol_table(
      func_identifier=self.func_identifier, func_arg_exprs=self.func_arg_exprs, symbol_table=symbol_table)
    return get_common_type([concrete_func.return_type for concrete_func in possible_concrete_funcs])

  def is_val_mutable(self, symbol_table):
    """
    :param SymbolTable symbol_table:
    :rtype: Type
    """
    possible_concrete_funcs = self._check_func_call_symbol_table(
      func_identifier=self.func_identifier, func_arg_exprs=self.func_arg_exprs, symbol_table=symbol_table)
    return all(concrete_func.return_mutable for concrete_func in possible_concrete_funcs)

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

  def make_declared_ptr_type(self, symbol_table):
    """
    :param SymbolTable symbol_table:
    :rtype: Type
    """
    raise NotImplementedError()

  def make_narrowed_ptr_type(self, symbol_table):
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

  def make_declared_ptr_type(self, symbol_table):
    """
    :param SymbolTable symbol_table:
    :rtype: Type
    """
    if self.var_identifier not in symbol_table:
      self.raise_error('Cannot reference variable %r before declaration' % self.var_identifier)
    symbol = symbol_table[self.var_identifier]
    if not isinstance(symbol, VariableSymbol):
      self.raise_error('Cannot assign to non-variable %r' % self.var_identifier)
    return symbol.declared_var_type

  def make_narrowed_ptr_type(self, symbol_table):
    """
    :param SymbolTable symbol_table:
    :rtype: Type
    """
    if self.var_identifier not in symbol_table:
      self.raise_error('Cannot reference variable %r before declaration' % self.var_identifier)
    symbol = symbol_table[self.var_identifier]
    if not isinstance(symbol, VariableSymbol):
      self.raise_error('Cannot assign to non-variable %r' % self.var_identifier)
    return symbol.narrowed_var_type

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

  def make_declared_ptr_type(self, symbol_table):
    """
    :param SymbolTable symbol_table:
    :rtype: Type
    """
    parent_type = self.parent_target.make_narrowed_ptr_type(symbol_table=symbol_table)
    return self._make_member_val_type(parent_type, member_identifier=self.member_identifier, symbol_table=symbol_table)

  def make_narrowed_ptr_type(self, symbol_table):
    """
    :param SymbolTable symbol_table:
    :rtype: Type
    """
    return self.make_declared_ptr_type(symbol_table=symbol_table)

  def is_ptr_mutable(self, symbol_table):
    """
    :param SymbolTable symbol_table:
    :rtype: bool
    """
    parent_type = self.parent_target.make_narrowed_ptr_type(symbol_table=symbol_table)
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
    parent_type = self.parent_target.make_narrowed_ptr_type(symbol_table=symbol_table)
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
    concrete_variant_types = [variant_type.make_type(symbol_table=symbol_table) for variant_type in self.variant_types]
    return get_common_type(concrete_variant_types)

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


def make_narrow_type_from_valid_cond_ast(cond_expr_ast, cond_holds, symbol_table):
  """
  :param ExpressionAst cond_expr_ast:
  :param bool cond_holds:
  :param SymbolTable symbol_table:
  """
  # TODO: This is super limited currently: Will only work for if(local_var is Type), nothing more.
  if isinstance(cond_expr_ast, BinaryOperatorExpressionAst) and cond_expr_ast.op == 'is':
    var_expr = cond_expr_ast.left_expr
    if not isinstance(var_expr, VariableExpressionAst):
      return
    var_symbol = var_expr.get_var_symbol(symbol_table=symbol_table)
    assert isinstance(cond_expr_ast.right_expr, VariableExpressionAst)
    check_type_expr = IdentifierTypeAst(cond_expr_ast.right_expr.pos, cond_expr_ast.right_expr.var_identifier)
    check_type = check_type_expr.make_type(symbol_table=symbol_table)
    assert cond_holds, 'not implemented'
    symbol_table[var_expr.var_identifier] = var_symbol.copy_with_narrowed_type(check_type)


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

