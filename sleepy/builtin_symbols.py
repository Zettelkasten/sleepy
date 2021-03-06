from enum import Enum
from typing import List, Callable, Optional, Dict, Set, Union, Tuple

from llvmlite import ir
from llvmlite.ir import IRBuilder

from sleepy.grammar import TreePosition
from sleepy.symbols import FunctionTemplate, PlaceholderTemplateType, Type, ConcreteFunction, \
  ConcreteBuiltinOperationFunction, ConcreteBitcastFunction, DoubleType, FloatType, BoolType, \
  IntType, LongType, CharType, RawPointerType, PointerType, SymbolTable, CodegenContext, FunctionSymbol, \
  LLVM_VOID_POINTER_TYPE, LLVM_SIZE_TYPE, TypeSymbol, StructType, TypeFactory, SLEEPY_VOID, SLEEPY_NEVER
from sleepy.util import concat_dicts


class BuiltinOperationFunctionTemplate(FunctionTemplate):
  def __init__(self, placeholder_template_types: List[PlaceholderTemplateType],
               return_type: Type,
               arg_identifiers: List[str],
               arg_types: List[Type],
               arg_type_narrowings: List[Type],
               instruction: Callable[..., Optional[ir.values.Value]],
               emits_ir: bool):
    super().__init__(placeholder_template_types, return_type, arg_identifiers, arg_types, arg_type_narrowings)
    self.instruction = instruction
    self.emits_ir = emits_ir

  def _get_concrete_function(self, concrete_template_arguments: List[Type],
                             concrete_parameter_types: List[Type],
                             concrete_narrowed_parameter_types: List[Type],
                             concrete_return_type: Type) -> ConcreteFunction:
    concrete_function = ConcreteBuiltinOperationFunction(
      signature=self, ir_func=None, template_arguments=concrete_template_arguments, return_type=concrete_return_type,
      parameter_types=concrete_parameter_types, narrowed_parameter_types=concrete_narrowed_parameter_types,
      instruction=self.instruction, emits_ir=self.emits_ir)
    self.initialized_templ_funcs[tuple(concrete_template_arguments)] = concrete_function
    return concrete_function


class BitcastFunctionTemplate(FunctionTemplate):
  def __init__(self, placeholder_template_types: List[PlaceholderTemplateType], return_type: Type,
               arg_identifiers: List[str], arg_types: List[Type], arg_type_narrowings: List[Type]):
    super().__init__(placeholder_template_types, return_type, arg_identifiers, arg_types, arg_type_narrowings)

  def _get_concrete_function(self, concrete_template_arguments: List[Type], concrete_parameter_types: List[Type],
                             concrete_narrowed_parameter_types: List[Type],
                             concrete_return_type: Type) -> ConcreteFunction:
    concrete_function = ConcreteBitcastFunction(
      signature=self,
      ir_func=None,
      template_arguments=concrete_template_arguments,
      return_type=concrete_return_type,
      parameter_types=concrete_parameter_types,
      narrowed_parameter_types=concrete_narrowed_parameter_types
    )
    self.initialized_templ_funcs[tuple(concrete_template_arguments)] = concrete_function
    return concrete_function


SLEEPY_DOUBLE = DoubleType()
SLEEPY_FLOAT = FloatType()
SLEEPY_BOOL = BoolType()
SLEEPY_INT = IntType()
SLEEPY_LONG = LongType()
SLEEPY_CHAR = CharType()
SLEEPY_RAW_PTR = RawPointerType()
SLEEPY_CHAR_PTR = PointerType(SLEEPY_CHAR)
SLEEPY_TYPES: Dict[str, Type] = {
  'Void': SLEEPY_VOID, 'Double': SLEEPY_DOUBLE, 'Float': SLEEPY_FLOAT, 'Bool': SLEEPY_BOOL, 'Int': SLEEPY_INT,
  'Long': SLEEPY_LONG, 'Char': SLEEPY_CHAR}
INT_TYPES: Set[Type] = {SLEEPY_INT, SLEEPY_LONG}
FLOAT_TYPES: Set[Type] = {SLEEPY_FLOAT, SLEEPY_DOUBLE}
SLEEPY_NUMERICAL_TYPES: Set[Type] = INT_TYPES | FLOAT_TYPES
COMPARABLE_TYPES = SLEEPY_NUMERICAL_TYPES | {SLEEPY_BOOL}


class BuiltinBinaryOps(Enum):
  Addition = '+'
  Subtraction = '-'
  Multiplication = '*'
  Division = '/'
  Equality = '=='
  Inequality = '!='
  Less = '<'
  Greater = '>'
  LessOrEqual = '<='
  GreaterOrEqual = '>='
  BitwiseOr = 'bitwise_or'
  Mod = 'mod'


Simple_Arithmetic_Ops: List[BuiltinBinaryOps] = \
  [BuiltinBinaryOps.Addition, BuiltinBinaryOps.Subtraction, BuiltinBinaryOps.Multiplication, BuiltinBinaryOps.Division]
Simple_Comparison_Ops: List[BuiltinBinaryOps] = \
 [BuiltinBinaryOps.Equality, BuiltinBinaryOps.Inequality, BuiltinBinaryOps.Less, BuiltinBinaryOps.Greater,
  BuiltinBinaryOps.GreaterOrEqual, BuiltinBinaryOps.LessOrEqual]


def _make_str_symbol(symbol_table: SymbolTable, context: CodegenContext) -> TypeSymbol:
  str_type = StructType(
    struct_identifier='Str', member_identifiers=['start', 'length', 'alloc_length'], templ_types=[],
    member_types=[SLEEPY_CHAR_PTR, SLEEPY_INT, SLEEPY_INT],
    pass_by_ref=True)
  constructor_symbol = str_type.build_constructor(parent_symbol_table=symbol_table, parent_context=context)
  str_type.constructor = constructor_symbol
  type_factory = TypeFactory(placeholder_templ_types=[], signature_type=str_type)
  struct_symbol = TypeSymbol(type_factory=type_factory)
  str_type.build_destructor(parent_symbol_table=symbol_table, parent_context=context)
  return struct_symbol


def _make_ptr_symbol(symbol_table: SymbolTable, context: CodegenContext) -> TypeSymbol:
  pointee_type = PlaceholderTemplateType(identifier='T')
  ptr_type = PointerType(pointee_type=pointee_type)

  assert 'load' not in symbol_table
  load_symbol = FunctionSymbol(identifier='load', returns_void=False)
  load_signature = BuiltinOperationFunctionTemplate(
    placeholder_template_types=[pointee_type], return_type=pointee_type, arg_identifiers=['ptr'], arg_types=[ptr_type],
    arg_type_narrowings=[ptr_type], instruction=lambda builder, ptr: builder.load(ptr, name='load'),
    emits_ir=context.emits_ir)
  load_symbol.add_signature(signature=load_signature)
  symbol_table['load'] = load_symbol
  symbol_table.inbuilt_symbols['load'] = load_symbol

  assert 'store' not in symbol_table
  store_symbol = FunctionSymbol(identifier='store', returns_void=True)
  store_signature = BuiltinOperationFunctionTemplate(
    placeholder_template_types=[pointee_type], return_type=SLEEPY_VOID, arg_identifiers=['ptr', 'value'],
    arg_types=[ptr_type, pointee_type], arg_type_narrowings=[ptr_type, pointee_type],
    instruction=lambda builder, ptr, value: builder.store(value=value, ptr=ptr), emits_ir=context.emits_ir)
  store_symbol.add_signature(store_signature)
  symbol_table['store'] = store_symbol
  type_factory = TypeFactory(placeholder_templ_types=[pointee_type], signature_type=ptr_type)

  # cast from RawPtr -> Ptr[T]
  constructor_symbol = FunctionSymbol(identifier='Ptr', returns_void=False)
  constructor_signature = BitcastFunctionTemplate(placeholder_template_types=[pointee_type], return_type=ptr_type,
                                                  arg_identifiers=['raw_ptr'], arg_types=[SLEEPY_RAW_PTR],
                                                  arg_type_narrowings=[ptr_type])
  constructor_symbol.add_signature(signature=constructor_signature)
  ptr_type.constructor = constructor_symbol

  ptr_op_decls = [(
    BuiltinBinaryOps.Addition,
    [(lambda builder, lhs, rhs: builder.gep(lhs, (rhs,)), [ptr_type, i], ptr_type) for i in INT_TYPES] +
    [(lambda builder, lhs, rhs: builder.gep(rhs, (lhs,)), [i, ptr_type], ptr_type) for i in INT_TYPES]),
    (BuiltinBinaryOps.Subtraction,
    [(lambda builder, lhs, rhs: builder.gep(lhs, (builder.mul(ir.Constant(i.ir_type, -1), rhs),)), [ptr_type, i], ptr_type) for i in INT_TYPES])]  # noqa
  ptr_op_decls += [
    (
      op,
      [(lambda builder, lhs, rhs, op=op: builder.icmp_unsigned(op.value, lhs, rhs), [ptr_type, ptr_type], SLEEPY_BOOL)])
    for op in Simple_Comparison_Ops]
  for operator, overloads in ptr_op_decls:
    if operator.value not in symbol_table:
      symbol_table[operator.value] = FunctionSymbol(identifier=operator.value, returns_void=False)
    function_symbol = symbol_table[operator.value]
    assert isinstance(function_symbol, FunctionSymbol)

    for instruction, arg_types, return_type in overloads:
      signature = _make_func_signature(
        instruction, op_placeholder_templ_types=[pointee_type], op_arg_types=arg_types, op_return_type=return_type,
        emits_ir=context.emits_ir)
      function_symbol.add_signature(signature)

  free_signature = BuiltinOperationFunctionTemplate(
    placeholder_template_types=[pointee_type], return_type=SLEEPY_VOID, arg_identifiers=['ptr'], arg_types=[ptr_type],
    arg_type_narrowings=[SLEEPY_NEVER], instruction=lambda builder, value: None, emits_ir=context.emits_ir)
  symbol_table.free_symbol.add_signature(free_signature)

  return TypeSymbol(type_factory=type_factory)


def _make_raw_ptr_symbol(symbol_table: SymbolTable, context: CodegenContext) -> TypeSymbol:
  # add destructor
  destructor_signature = BuiltinOperationFunctionTemplate(
    placeholder_template_types=[], return_type=SLEEPY_VOID, arg_identifiers=['raw_ptr'], arg_types=[SLEEPY_RAW_PTR],
    arg_type_narrowings=[SLEEPY_NEVER], instruction=lambda builder, value: None, emits_ir=context.emits_ir)
  symbol_table.free_symbol.add_signature(destructor_signature)

  pointee_type = PlaceholderTemplateType(identifier='T')
  ptr_type = PointerType(pointee_type=pointee_type)
  constructor_symbol = FunctionSymbol(identifier='RawPtr', returns_void=False)
  # RawPtr[T](Ptr[T]) -> RawPtr
  from_specific_signature = BitcastFunctionTemplate(
    placeholder_template_types=[pointee_type], return_type=SLEEPY_RAW_PTR, arg_identifiers=['ptr'],
    arg_types=[ptr_type], arg_type_narrowings=[ptr_type])
  constructor_symbol.add_signature(signature=from_specific_signature)
  # RawPtr(Int) -> RawPtr
  for int_type in INT_TYPES:
    from_int_signature = BuiltinOperationFunctionTemplate(
      placeholder_template_types=[], return_type=SLEEPY_RAW_PTR, arg_identifiers=['int'], arg_types=[int_type],
      arg_type_narrowings=[int_type],
      instruction=lambda builder, integer: builder.inttoptr(integer, typ=SLEEPY_RAW_PTR.ir_type, name='int_to_ptr'),
      emits_ir=context.emits_ir)
    constructor_symbol.add_signature(from_int_signature)
  SLEEPY_RAW_PTR.constructor = constructor_symbol

  type_generator = TypeFactory(placeholder_templ_types=[], signature_type=SLEEPY_RAW_PTR)
  raw_ptr_symbol = TypeSymbol(type_generator)
  return raw_ptr_symbol


def _make_bitcast_symbol(symbol_table: SymbolTable, context: CodegenContext) -> FunctionSymbol:
  del symbol_table  # only to keep API consistent
  del context
  bitcast_func = FunctionSymbol(identifier='bitcast', returns_void=False)
  to_type, from_type = PlaceholderTemplateType('T'), PlaceholderTemplateType('U')
  bitcast_signature = BitcastFunctionTemplate(
    placeholder_template_types=[to_type, from_type], return_type=to_type,
    arg_identifiers=['from'], arg_types=[from_type], arg_type_narrowings=[to_type])
  bitcast_func.add_signature(bitcast_signature)
  return bitcast_func


def build_initial_ir(symbol_table: SymbolTable, context: CodegenContext):
  assert 'free' not in symbol_table
  free_symbol = FunctionSymbol(identifier='free', returns_void=True)
  symbol_table['free'] = free_symbol

  assert 'index' not in symbol_table
  symbol_table['index'] = FunctionSymbol(identifier='index', returns_void=False)
  symbol_table.inbuilt_symbols['index'] = symbol_table['index']

  assert 'size' not in symbol_table
  symbol_table['size'] = FunctionSymbol(identifier='size', returns_void=False)
  symbol_table.inbuilt_symbols['size'] = symbol_table['size']

  for type_identifier, inbuilt_type in SLEEPY_TYPES.items():
    type_generator = TypeFactory(placeholder_templ_types=[], signature_type=inbuilt_type)
    assert type_identifier not in symbol_table
    symbol_table[type_identifier] = TypeSymbol(type_generator)
    if inbuilt_type == SLEEPY_VOID:
      continue

    # add destructor
    destructor_signature = BuiltinOperationFunctionTemplate(
      placeholder_template_types=[], return_type=SLEEPY_VOID, arg_identifiers=['var'], arg_types=[inbuilt_type],
      arg_type_narrowings=[SLEEPY_NEVER], instruction=lambda builder, value: None, emits_ir=context.emits_ir)
    symbol_table.free_symbol.add_signature(destructor_signature)

  for assert_identifier in ['assert', 'unchecked_assert']:
    assert assert_identifier not in symbol_table
    assert_symbol = FunctionSymbol(identifier=assert_identifier, returns_void=True)
    symbol_table[assert_identifier] = assert_symbol
    symbol_table.inbuilt_symbols[assert_identifier] = assert_symbol

  if context.emits_ir:
    context.ir_func_malloc = ir.Function(
      context.module, ir.FunctionType(LLVM_VOID_POINTER_TYPE, [LLVM_SIZE_TYPE]), name='malloc')
    context.ir_func_free = ir.Function(
      context.module, ir.FunctionType(ir.VoidType(), [LLVM_VOID_POINTER_TYPE]), name='free')

  # TODO: currently, some inbuilt free() functions are not inlined.
  # This means that we need to add debug information to these functions, but they do not have any line numbers.
  # We use this dummy here.
  inbuilt_pos = TreePosition('', 0, 0)

  inbuilt_symbols = {
    'Str': _make_str_symbol, 'Ptr': _make_ptr_symbol, 'RawPtr': _make_raw_ptr_symbol, 'bitcast': _make_bitcast_symbol}
  with context.use_pos(inbuilt_pos):
    for symbol_identifier, setup_func in inbuilt_symbols.items():
      assert symbol_identifier not in symbol_table
      symbol = setup_func(symbol_table=symbol_table, context=context)
      symbol_table[symbol_identifier] = symbol
      symbol_table.inbuilt_symbols[symbol_identifier] = symbol

    _make_builtin_operator_functions(symbol_table, context.emits_ir)


Instructions: Dict[Tuple[BuiltinBinaryOps, Type], Callable[[CodegenContext, ir.values.Value, ir.values.Value], ir.values.Value]] = concat_dicts([  # noqa
  {(BuiltinBinaryOps.Addition, Int): IRBuilder.add for Int in INT_TYPES},
  {(BuiltinBinaryOps.Subtraction, Int): IRBuilder.sub for Int in INT_TYPES},
  {(BuiltinBinaryOps.Multiplication, Int): IRBuilder.mul for Int in INT_TYPES},
  {(BuiltinBinaryOps.Division, Int): IRBuilder.sdiv for Int in INT_TYPES},

  {(BuiltinBinaryOps.Addition, T): IRBuilder.fadd for T in FLOAT_TYPES},
  {(BuiltinBinaryOps.Subtraction, T): IRBuilder.fsub for T in FLOAT_TYPES},
  {(BuiltinBinaryOps.Multiplication, T): IRBuilder.fmul for T in FLOAT_TYPES},
  {(BuiltinBinaryOps.Division, T): IRBuilder.fdiv for T in FLOAT_TYPES},

  {
    (op, T): lambda builder, lhs, rhs, op=op: builder.icmp_signed(op.value, lhs, rhs) for op in Simple_Comparison_Ops
    for T in INT_TYPES},
  {
    (op, T): lambda builder, lhs, rhs, op=op: builder.icmp_unsigned(op.value, lhs, rhs) for op in Simple_Comparison_Ops
    for T in {SLEEPY_CHAR, SLEEPY_BOOL}},
  {
    (op, T): lambda builder, lhs, rhs, op=op: builder.fcmp_ordered(op.value, lhs, rhs)
    for op in Simple_Comparison_Ops for T in FLOAT_TYPES}
  ])
BINARY_OP_DECL: List[Tuple[BuiltinBinaryOps, List[Tuple[Callable[..., ir.values.Value], List[Type], Type]]]] = (
  # simple arithmetic on all arithmetic types
  [
    (operator, [(Instructions[(operator, arith_t)], [arith_t, arith_t], arith_t) for arith_t in SLEEPY_NUMERICAL_TYPES])
    for operator in Simple_Arithmetic_Ops] +
  # unary plus and minus on all arithmetic types
  [
    (
      BuiltinBinaryOps.Addition,
      [(lambda builder, arg: arg, [arith_t], arith_t) for arith_t in SLEEPY_NUMERICAL_TYPES] +
      [(lambda builder, lhs, rhs: builder.gep(lhs, (rhs,)), [SLEEPY_RAW_PTR, i], SLEEPY_RAW_PTR) for i in INT_TYPES] +
      [(lambda builder, lhs, rhs: builder.gep(rhs, (lhs,)), [i, SLEEPY_RAW_PTR], SLEEPY_RAW_PTR) for i in INT_TYPES]
    ),
    (
      BuiltinBinaryOps.Subtraction,
      [
        (lambda builder, arg, arith_t=arith_t: builder.mul(ir.Constant(arith_t.ir_type, -1), arg), [arith_t], arith_t)
        for arith_t in INT_TYPES]),
    (
      BuiltinBinaryOps.Subtraction,
      [
        (lambda builder, arg, arith_t=arith_t: builder.fmul(ir.Constant(arith_t.ir_type, -1), arg), [arith_t], arith_t)
        for arith_t in FLOAT_TYPES]),
    (
      BuiltinBinaryOps.Subtraction,
      [
        (
          lambda builder, lhs, rhs: builder.gep(lhs, (builder.mul(ir.Constant(arith_t.ir_type, -1), rhs),)),
          [SLEEPY_RAW_PTR, arith_t], SLEEPY_RAW_PTR)
        for arith_t in INT_TYPES])] +
  # comparisons on all types except void and char
  [
    (operator, [(Instructions[(operator, comp_t)], [comp_t, comp_t], SLEEPY_BOOL) for comp_t in COMPARABLE_TYPES])
    for operator in Simple_Comparison_Ops] +
  # equality / inequality
  [
    (
      operator, [(
        lambda builder, lhs, rhs, op=operator: builder.icmp_unsigned(op.value, lhs, rhs),
        [SLEEPY_CHAR, SLEEPY_CHAR], SLEEPY_BOOL)])
    for operator in [BuiltinBinaryOps.Equality, BuiltinBinaryOps.Inequality]] +
  # bitwise_or on integer types
  [(BuiltinBinaryOps.BitwiseOr, [(IRBuilder.or_, [int_t, int_t], int_t) for int_t in INT_TYPES])] +
  # modulo on integer types
  [(BuiltinBinaryOps.Mod, [(IRBuilder.srem, [int_t, int_t], int_t) for int_t in INT_TYPES])])


def _make_builtin_operator_functions(symbol_table: SymbolTable, emits_ir: bool):
  for operator, overloads in BINARY_OP_DECL:
    if operator.value not in symbol_table:
      symbol_table[operator.value] = FunctionSymbol(identifier=operator.value, returns_void=False)
    function_symbol = symbol_table[operator.value]
    assert isinstance(function_symbol, FunctionSymbol)

    for instruction, arg_types, return_type in overloads:
      signature = _make_func_signature(
        instruction, op_placeholder_templ_types=[], op_arg_types=arg_types, op_return_type=return_type,
        emits_ir=emits_ir)
      function_symbol.add_signature(signature)


def _make_func_signature(instruction: Callable[..., ir.values.Value],
                         op_placeholder_templ_types: Union[Tuple[PlaceholderTemplateType], List[PlaceholderTemplateType]],  # noqa
                         op_arg_types: List[Type], op_return_type: Type, emits_ir: bool) -> FunctionTemplate:
  assert len(op_arg_types) in {1, 2}
  unary: bool = len(op_arg_types) == 1
  op_arg_identifiers = ['arg'] if unary else ['lhs', 'rhs']
  assert len(op_arg_types) == len(op_arg_identifiers)
  signature = BuiltinOperationFunctionTemplate(
    placeholder_template_types=list(op_placeholder_templ_types), return_type=op_return_type,
    arg_identifiers=op_arg_identifiers, arg_types=op_arg_types, arg_type_narrowings=op_arg_types,
    instruction=instruction, emits_ir=emits_ir)

  return signature