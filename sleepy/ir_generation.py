from __future__ import annotations

from typing import List

from llvmlite import ir

from sleepy.errors import raise_error
from sleepy.syntactical_analysis.grammar import TreePosition

from sleepy.types import CleanupHandlingCG, Type, CodegenContext, UnionType, LLVM_SIZE_TYPE, OverloadSet, TypedValue, \
  ConcreteFunction, make_union_switch_ir, try_infer_template_arguments, FunctionSymbolCaller, get_common_type


def make_ir_end_block_jump(context: CleanupHandlingCG, continuation: ir.Block, parent_end_block: ir.Block):
  assert context.scope.depth != 0
  builder = context.end_block_builder
  target_depth_reached = builder.icmp_unsigned('==', builder.load(context.function.unroll_count_ir),
                        ir.Constant(typ=ir.IntType(bits=64), constant=context.scope.depth))
  builder.cbranch(target_depth_reached, continuation, parent_end_block)

def make_ir_end_block_return(context: CleanupHandlingCG):
  context.end_block_builder.ret(context.end_block_builder.load(context.function.return_slot_ir))


def make_ir_val_is_type(ir_val: ir.values.Value,
                        known_type: Type,
                        check_type: Type,
                        context: CodegenContext) -> ir.values.Value:
  assert context.emits_ir
  if known_type == check_type:
    return ir.Constant(ir.IntType(1), True)
  if not isinstance(known_type, UnionType):
    return ir.Constant(ir.IntType(1), False)
  if not known_type.contains(check_type):
    return ir.Constant(ir.IntType(1), False)
  assert not isinstance(check_type, UnionType), 'not implemented yet'
  union_tag = context.builder.extract_value(ir_val, 0, name='tmp_is_val')
  cmp_val = context.builder.icmp_signed(
    '==', union_tag, ir.Constant(known_type.tag_ir_type, known_type.get_variant_num(check_type)), name='tmp_is_check')
  return cmp_val


def make_ir_size(size: int) -> ir.values.Value:
  return ir.Constant(LLVM_SIZE_TYPE, size)


def infer_template_arguments(pos: TreePosition, func: OverloadSet, calling_types: List[Type]) -> List[Type]:
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
      try_infer_template_arguments(
        calling_types=list(expanded_calling_types), signature_types=signature.arg_types,
        template_parameters=signature.placeholder_templ_types)
      for signature in func.signatures]
    possible_infers = [idx for idx, infer in enumerate(infers) if infer is not None]
    if len(possible_infers) == 0:
      raise_error('Cannot infer template types for function %r from arguments of types %r, '
                  'is declared for parameter types:\n%s\n\nSpecify the template types explicitly.' % (
                    func.identifier, ', '.join([str(calling_type) for calling_type in calling_types]),
                    func.make_signature_list_str()), pos)
    if len(possible_infers) > 1:
      raise_error('Cannot uniquely infer template types for function %r from arguments of types %r, '
                  'is declared for parameter types:\n%s' % (
                    func.identifier, ', '.join([str(calling_type) for calling_type in calling_types]),
                    func.make_signature_list_str()), pos)
    assert len(possible_infers) == 1
    expanded_signature_templ_types = infers[possible_infers[0]]
    assert expanded_signature_templ_types is not None
    if signature_templ_types is not None and signature_templ_types != expanded_signature_templ_types:
      raise_error('Cannot uniquely statically infer template types for function %r from arguments of types %r '
                  'because different expanded union types would require different template types. '
                  'Function is declared for parameter types:\n%s' % (
                    func.identifier, ', '.join([str(calling_type) for calling_type in calling_types]),
                    func.make_signature_list_str()), pos)
    signature_templ_types = expanded_signature_templ_types
  assert signature_templ_types is not None
  return signature_templ_types


def make_ir_func_call(func: OverloadSet,
                      template_arguments: List[Type],
                      func_args: List[TypedValue],
                      context: CodegenContext) -> TypedValue:
  assert not context.emits_debug or context.builder.debug_metadata is not None

  def do_call(concrete_func: ConcreteFunction,
              caller_context: CodegenContext) -> TypedValue:
    assert caller_context.emits_ir
    assert not caller_context.emits_debug or caller_context.builder.debug_metadata is not None
    assert len(concrete_func.arg_types) == len(func_args)
    assert all(
      not arg_mutates or arg.is_referenceable() for arg, arg_mutates in zip(func_args, concrete_func.arg_mutates))
    # mutates arguments are handled here as Ref[T], all others normally as T.
    collapsed_func_args = [
      (arg_val.copy_unbind() if mutates else arg_val).copy_collapse(
        context=caller_context, name='call_arg_%s_collapse' % identifier)
      for arg_val, identifier, mutates in zip(func_args, concrete_func.arg_identifiers, concrete_func.arg_mutates)]
    # Note: We first copy with the param_type as narrowed type, because at this point we already checked that
    # the param actually has the correct param_type.
    # Then, copy_with_implicit_cast only needs to handle the cases which can actually occur.
    casted_calling_args = [
      arg_val.copy_set_narrowed_type(param_type).copy_with_implicit_cast(
        to_type=param_type, context=caller_context, name='call_arg_%s_cast' % arg_identifier)
      for arg_identifier, arg_val, param_type in zip(
        concrete_func.arg_identifiers, collapsed_func_args, concrete_func.uncollapsed_arg_types)]
    if concrete_func.is_inline:
      assert concrete_func not in caller_context.inline_func_call_stack
      return_ir = concrete_func.make_inline_func_call_ir(func_args=casted_calling_args, caller_context=caller_context)
    else:
      ir_func = concrete_func.ir_func
      assert ir_func is not None and len(ir_func.args) == len(casted_calling_args)
      casted_ir_args = [arg.ir_val for arg in casted_calling_args]
      assert None not in casted_ir_args
      return_ir = caller_context.builder.call(ir_func, casted_ir_args, name='call_%s' % func.identifier)

    assert isinstance(return_ir, ir.values.Value)
    return TypedValue.create(typ=concrete_func.return_type, ir_val=return_ir)

  calling_collapsed_args = [arg.copy_collapse(context=context) for arg in func_args]
  calling_arg_types = [arg.narrowed_type for arg in calling_collapsed_args]
  assert func.can_call_with_arg_types(template_arguments=template_arguments, arg_types=calling_arg_types)
  possible_concrete_funcs = func.get_concrete_funcs(template_arguments=template_arguments, arg_types=calling_arg_types)
  from functools import partial
  return make_union_switch_ir(
    case_funcs={
      tuple(concrete_func.arg_types): partial(do_call, concrete_func) for concrete_func in possible_concrete_funcs},
    calling_args=calling_collapsed_args, name=func.identifier, context=context)


def make_call_ir(pos: TreePosition,
                 caller: FunctionSymbolCaller,
                 argument_values: List[TypedValue],
                 context: CodegenContext) -> TypedValue:

  overloads, template_arguments = caller.overload_set, caller.template_parameters
  calling_types = [arg.collapsed_type() for num, arg in enumerate(argument_values)]

  assert all(typ.is_realizable() for typ in calling_types)

  if template_arguments is None:
    template_arguments = infer_template_arguments(pos, func=overloads, calling_types=calling_types)
  assert template_arguments is not None

  possible_concrete_functions = resolve_possible_concrete_funcs(pos, func_caller=caller, calling_types=calling_types)
  for concrete_func in possible_concrete_functions:
    if concrete_func in context.inline_func_call_stack:
      raise_error('An inlined function can not call itself (indirectly), but got inline call stack: %s -> %s' % (
        ' -> '.join(str(inline_func) for inline_func in context.inline_func_call_stack), concrete_func), pos)
    for arg_identifier, arg_mutates, arg in zip(concrete_func.arg_identifiers, concrete_func.arg_mutates,
                                                argument_values):
      if arg_mutates and not arg.is_referenceable():
        raise_error('Cannot call function %s%s mutating parameter %r with non-referencable argument' % (
          overloads.identifier, concrete_func.signature.to_signature_str(), arg_identifier), pos)

  if context.emits_ir:
    return_val = make_ir_func_call(func=overloads, template_arguments=template_arguments,
                                   func_args=argument_values, context=context)
  else:
    return_type = get_common_type([concrete_func.return_type for concrete_func in possible_concrete_functions])
    return_val = TypedValue.create(typ=return_type, ir_val=None)

  return return_val


def resolve_possible_concrete_funcs(pos: TreePosition,
                                    func_caller: FunctionSymbolCaller,
                                    calling_types: List[Type]) -> List[ConcreteFunction]:
  func, templ_types = func_caller.overload_set, func_caller.template_parameters
  if templ_types is None:
    templ_types = infer_template_arguments(pos, func=func, calling_types=calling_types)
  assert templ_types is not None
  assert all(not templ_type.has_unfilled_template_parameters() for templ_type in templ_types)

  if not func.can_call_with_arg_types(template_arguments=templ_types, arg_types=calling_types):
    raise_error('Cannot call function %r with arguments of types %r and template parameters %r, '
                'only declared for parameter types:\n%s' % (
                  func.identifier, ', '.join([str(calling_type) for calling_type in calling_types]),
                  ', '.join([str(templ_type) for templ_type in templ_types]),
                  func.make_signature_list_str()), pos)
  if not all(called_type.is_realizable() for called_type in calling_types):
    raise_error('Cannot call function %r with argument of types %r which are unrealizable' % (
      func.identifier, ', '.join([str(called_type) for called_type in calling_types])), pos)

  possible_concrete_funcs = func.get_concrete_funcs(template_arguments=templ_types, arg_types=calling_types)
  assert len(possible_concrete_funcs) >= 1
  return possible_concrete_funcs