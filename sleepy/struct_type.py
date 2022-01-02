from __future__ import annotations

from typing import List

from llvmlite import ir

from sleepy.ir_generation import make_call_ir
from sleepy.symbols import SymbolTable
from sleepy.syntactical_analysis.grammar import DUMMY_POS
from sleepy.types import StructType, CodegenContext, OverloadSet, PlaceholderTemplateType, FunctionTemplate, Type, \
  ConcreteFunction, SLEEPY_UNIT, SLEEPY_NEVER, TypedValue, FunctionSymbolCaller


def build_destructor(struct_type: StructType, parent_symbol_table: SymbolTable, parent_context: CodegenContext):
  assert not parent_context.emits_debug or parent_context.builder.debug_metadata is not None

  placeholder_template_types = [
    templ_type for templ_type in struct_type.template_param_or_arg if isinstance(templ_type, PlaceholderTemplateType)]
  # TODO: Narrow type to something more meaningful then SLEEPY_NEVER
  # E.g. make a copy of the never union type and give that a name ("Freed" or sth)
  signature_ = DestructorFunctionTemplate(
    placeholder_templ_types=placeholder_template_types, struct=struct_type,
    captured_symbol_table=parent_symbol_table, captured_context=parent_context)
  parent_symbol_table.add_overload('free', signature_)


def build_constructor(struct_type: StructType, parent_symbol_table: SymbolTable,
                      parent_context: CodegenContext) -> OverloadSet:
  assert not parent_context.emits_debug or parent_context.builder.debug_metadata is not None

  placeholder_templ_types = [
    templ_type for templ_type in struct_type.template_param_or_arg if isinstance(templ_type, PlaceholderTemplateType)]
  signature = ConstructorFunctionTemplate(
    placeholder_templ_types=placeholder_templ_types, struct=struct_type,
    captured_symbol_table=parent_symbol_table, captured_context=parent_context)
  return OverloadSet(identifier=struct_type.struct_identifier, signatures=[signature])


class ConstructorFunctionTemplate(FunctionTemplate):
  def __init__(self,
               placeholder_templ_types: List[PlaceholderTemplateType],
               struct: StructType,
               captured_symbol_table: SymbolTable,
               captured_context: CodegenContext):
    super().__init__(
      placeholder_template_types=placeholder_templ_types, return_type=struct, arg_identifiers=struct.member_identifiers,
      arg_types=struct.member_types, arg_type_narrowings=struct.member_types,
      arg_mutates=[False] * len(struct.member_types))
    self.struct = struct
    self.captured_symbol_table = captured_symbol_table
    self.captured_context = captured_context

  def _get_concrete_function(self, concrete_template_arguments: List[Type],
                             concrete_parameter_types: List[Type],
                             concrete_narrowed_parameter_types: List[Type],
                             concrete_return_type: Type) -> ConcreteFunction:
    concrete_function = ConcreteFunction(
      signature=self, ir_func=None, template_arguments=concrete_template_arguments, return_type=concrete_return_type,
      parameter_types=concrete_parameter_types, narrowed_parameter_types=concrete_narrowed_parameter_types,
      parameter_mutates=self.arg_mutates)
    self.initialized_templ_funcs[tuple(concrete_template_arguments)] = concrete_function
    self.build_concrete_function_ir(concrete_function)
    return concrete_function

  def build_concrete_function_ir(self, concrete_function: ConcreteFunction):
    with self.captured_context.use_pos(self.captured_context.current_pos):
      concrete_struct_type = concrete_function.return_type
      assert isinstance(concrete_struct_type, StructType)

      if self.captured_context.emits_ir:
        concrete_function.make_ir_func(
          identifier=self.struct.struct_identifier, extern=False, context=(
            self.captured_context))
        constructor_block = concrete_function.ir_func.append_basic_block(name='entry')
        context = self.captured_context.copy_with_func(concrete_function, builder=ir.IRBuilder(constructor_block))
        self_ir_alloca = context.alloca_at_entry(concrete_struct_type.ir_type, name='self')

        for member_identifier, ir_func_arg in zip(self.struct.member_identifiers, concrete_function.ir_func.args):
          ir_func_arg.struct_identifier = member_identifier
        concrete_struct_type.make_store_members_ir(
          member_ir_vals=concrete_function.ir_func.args, struct_ir_alloca=self_ir_alloca, context=context)
        context.builder.ret(context.builder.load(self_ir_alloca, name='self'))


class DestructorFunctionTemplate(FunctionTemplate):
  def __init__(self, placeholder_templ_types: List[PlaceholderTemplateType],
               struct: StructType,
               captured_symbol_table: SymbolTable,
               captured_context: CodegenContext):
    super().__init__(
      placeholder_template_types=placeholder_templ_types, return_type=SLEEPY_UNIT, arg_types=[struct],
      arg_identifiers=['self'], arg_type_narrowings=[SLEEPY_NEVER], arg_mutates=[False])
    self.struct = struct
    self.captured_symbol_table = captured_symbol_table
    self.captured_context = captured_context

  def _get_concrete_function(self, concrete_template_arguments: List[Type],
                             concrete_parameter_types: List[Type],
                             concrete_narrowed_parameter_types: List[Type],
                             concrete_return_type: Type) -> ConcreteFunction:
    concrete_function = ConcreteFunction(
      signature=self, ir_func=None, template_arguments=concrete_template_arguments, return_type=concrete_return_type,
      parameter_types=concrete_parameter_types, narrowed_parameter_types=concrete_narrowed_parameter_types,
      parameter_mutates=self.arg_mutates)
    self.initialized_templ_funcs[tuple(concrete_template_arguments)] = concrete_function
    self.build_concrete_function_ir(concrete_function)
    return concrete_function

  def build_concrete_function_ir(self, concrete_function: ConcreteFunction):
    with self.captured_context.use_pos(self.captured_context.current_pos):
      assert len(concrete_function.arg_types) == 1
      concrete_struct_type = concrete_function.arg_types[0]
      assert isinstance(concrete_struct_type, StructType)
      if self.captured_context.emits_ir:
        concrete_function.make_ir_func(identifier='free', extern=False, context=self.captured_context)
        destructor_block = concrete_function.ir_func.append_basic_block(name='entry')
        context = self.captured_context.copy_with_func(concrete_function, builder=ir.IRBuilder(destructor_block))

        assert len(concrete_function.ir_func.args) == 1
        self_ir_alloca = concrete_function.ir_func.args[0]
        self_ir_alloca.struct_identifier = 'self_ptr'
        # Free members in reversed order
        for member_num in reversed(range(len(self.struct.member_identifiers))):
          member_identifier = self.struct.member_identifiers[member_num]
          concrete_member_type = concrete_struct_type.member_types[member_num]

          member_ir_val = self.struct.make_extract_member_val_ir(
            member_identifier, struct_ir_val=self_ir_alloca, context=context)


          make_call_ir(
            pos=DUMMY_POS,
            caller=FunctionSymbolCaller(
              overload_set=self.captured_symbol_table.free_overloads,
              template_parameters=None),  # infer
            argument_values=[TypedValue.create(
              typ=concrete_member_type,
              ir_val=member_ir_val,
              num_unbindings=concrete_member_type.num_possible_unbindings())],
              context=context
          )

          # make_func_call_ir(
          #   func=self.captured_symbol_table.free_overloads, template_arguments=template_arguments,
          #   func_args=[TypedValue(
          #     typ=concrete_member_type,
          #     ir_val=member_ir_val,
          #     num_unbindings=concrete_member_type.num_possible_unbindings())],
          #   context=context)
        context.builder.ret(SLEEPY_UNIT.unit_constant())
