from __future__ import annotations

from enum import Enum
from itertools import takewhile
from typing import Optional, List, Dict, Tuple, Union, Collection, Iterable, cast

from llvmlite import ir
from multimethod import multimethod

from sleepy.hierarchical_dictionary import HierarchicalDict, STUB
from sleepy.types import Type, narrow_type, narrow_with_collapsed_type, exclude_type, exclude_with_collapsed_type, \
  CodegenContext, ReferenceType, TypedValue, FunctionTemplate, PlaceholderTemplateType, ConcreteFunction, \
  get_common_type, OverloadSet


class VariableSymbol:
  """
  A declared variable.
  """

  def __init__(self, ir_alloca: Optional[ir.AllocaInstr], var_type: Type):
    super().__init__()
    assert ir_alloca is None or isinstance(ir_alloca, (ir.AllocaInstr, ir.Argument))
    self.ir_alloca = ir_alloca
    self.declared_var_type = var_type
    self.narrowed_var_type = var_type

  def copy_set_narrowed_type(self, new_narrow_type: Type) -> VariableSymbol:
    new_var_symbol = VariableSymbol(self.ir_alloca, self.declared_var_type)
    # explicitly apply narrowing from declared type here: always stay compatible to the base type
    new_var_symbol.narrowed_var_type = narrow_type(from_type=self.declared_var_type, narrow_to=new_narrow_type)
    return new_var_symbol

  def copy_narrow_type(self, narrow_to: Type) -> VariableSymbol:
    return self.copy_set_narrowed_type(new_narrow_type=narrow_type(self.narrowed_var_type, narrow_to))

  def copy_reset_narrowed_type(self) -> VariableSymbol:
    return self.copy_set_narrowed_type(new_narrow_type=self.declared_var_type)

  def copy_set_narrowed_collapsed_type(self, collapsed_type: Type) -> VariableSymbol:
    return self.copy_set_narrowed_type(narrow_with_collapsed_type(
      from_type=self.declared_var_type, collapsed_type=collapsed_type))

  def copy_narrow_collapsed_type(self, collapsed_type: Type) -> VariableSymbol:
    return self.copy_set_narrowed_type(narrow_with_collapsed_type(
      from_type=self.narrowed_var_type, collapsed_type=collapsed_type))

  def copy_exclude_type(self, excluded: Type) -> VariableSymbol:
    return self.copy_set_narrowed_type(new_narrow_type=exclude_type(self.narrowed_var_type, excluded))

  def copy_exclude_collapsed_type(self, collapsed_type: Type) -> VariableSymbol:
    return self.copy_set_narrowed_type(exclude_with_collapsed_type(
      from_type=self.narrowed_var_type, collapsed_type=collapsed_type))

  def build_ir_alloca(self, context: CodegenContext, identifier: str,
                      initial_ir_alloca: Optional[ir.values.Value] = None):
    assert self.ir_alloca is None
    assert isinstance(self.declared_var_type, ReferenceType)
    var_type = self.declared_var_type.pointee_type
    if not context.emits_ir:
      return
    if initial_ir_alloca is not None:
      self.ir_alloca = initial_ir_alloca
      initial_ir_alloca.name = '%s_ref' % identifier
    else:  # default case
      self.ir_alloca = context.alloca_at_entry(var_type.ir_type, name='%s_ptr' % identifier)
    if context.emits_debug:
      assert context.di_declare_func is not None
      di_local_var = context.module.add_debug_info(
        'DILocalVariable', {
          'name': identifier, 'scope': context.current_di_scope, 'file': context.current_di_file,
          'line': context.current_pos.get_from_line(), 'type': var_type.make_di_type(context=context)})
      di_expression = context.module.add_debug_info('DIExpression', {})
      assert context.builder.debug_metadata is not None
      context.builder.call(context.di_declare_func, args=[self.ir_alloca, di_local_var, di_expression])

  def __repr__(self) -> str:
    return 'VariableSymbol(ir_alloca=%r, declared_var_type=%r, narrowed_var_type=%r)' % (
      self.ir_alloca, self.declared_var_type, self.narrowed_var_type)

  def as_typed_var(self, ir_val: Optional[ir.Value]) -> TypedValue:
    return TypedValue(typ=self.declared_var_type, narrowed_type=self.narrowed_var_type, ir_val=ir_val)


class FunctionSymbol:
  def __init__(self, functions: Optional[List[FunctionTemplate]] = None):
    super().__init__()
    if functions is None: functions = []
    self.functions: List[FunctionTemplate] = functions


class TypeTemplateSymbol:
  """
  A (statically) declared (possibly) template type.
  Can have one or multiple template initializations that yield different concrete types.
  These are initialized lazily.
  """

  def __init__(self, template_parameters: List[PlaceholderTemplateType], signature_type: Type):
    super().__init__()
    self.template_parameters: List[PlaceholderTemplateType] = template_parameters
    self.signature_type = signature_type

    self.generated_types_cache: Dict[Tuple[Type], Type] = {}

  @staticmethod
  def make_concrete_type_symbol(concrete_type: Type) -> TypeTemplateSymbol:
    return TypeTemplateSymbol(template_parameters=[], signature_type=concrete_type)

  def get_type(self, template_arguments: List[Type]) -> Type:
    template_arguments = tuple(template_arguments)
    if template_arguments in self.generated_types_cache:
      return self.generated_types_cache[template_arguments]
    concrete_type = self.make_concrete_type(template_arguments=template_arguments)
    self.generated_types_cache[template_arguments] = concrete_type
    return concrete_type

  def make_concrete_type(self, template_arguments: List[Type] | Tuple[Type]) -> Type:
    replacements = dict(zip(self.template_parameters, template_arguments))
    return self.signature_type.replace_types(replacements)


class SymbolTableStub:
  def __init__(self):
    self.dict = STUB
    self.current_func = None
    self.known_extern_funcs = {}
    self.builtin_symbols = set()
    self.current_scope_identifiers = frozenset()


Symbol = Union[OverloadSet, TypeTemplateSymbol, VariableSymbol]


class SymbolTable:
  """
  Basically a dict mapping identifier names to symbols.
  Also contains information about the current scope.
  """

  Element = Union[FunctionSymbol, TypeTemplateSymbol, VariableSymbol]

  def __init__(self, parent: Optional[SymbolTable] = None,
               new_function: Optional[ConcreteFunction] = None,
               inherit_outer_variables: bool = False):
    self.parent = parent = SymbolTableStub() if parent is None else parent

    assert inherit_outer_variables is not None
    self.dict: HierarchicalDict[str, SymbolTable.Element] = HierarchicalDict(parent.dict)
    self.inherit_outer_variables = inherit_outer_variables
    self.current_func = parent.current_func if new_function is None else new_function
    self.known_extern_funcs = parent.known_extern_funcs
    self.builtin_symbols = parent.builtin_symbols

  @property
  def current_scope_identifiers(self) -> Collection[str]:
    if self.inherit_outer_variables:
      return self.dict.underlying_dict.keys() | self.parent.current_scope_identifiers
    else:
      return self.dict.underlying_dict.keys()

  def make_child_scope(self, *,
                       inherit_outer_variables: bool,
                       type_substitutions: Optional[Iterable[Tuple[str, Type]]] = None,
                       new_function: Optional[ConcreteFunction] = None) -> SymbolTable:
    if type_substitutions is None:
      type_substitutions = []

    new_table = SymbolTable(parent=self, inherit_outer_variables=inherit_outer_variables, new_function=new_function)
    # shadow placeholder types with their concrete substitutions
    for name, t in type_substitutions:
      existing_symbol = new_table[name]
      assert isinstance(existing_symbol, TypeTemplateSymbol)
      assert isinstance(existing_symbol.signature_type, PlaceholderTemplateType)

      new_table[name] = TypeTemplateSymbol.make_concrete_type_symbol(t)

    return new_table

  def __repr__(self) -> str:
    return 'SymbolTable%r' % self.__dict__

  def __setitem__(self, key, value):
    if isinstance(value, FunctionSymbol) and key in self.dict.underlying_dict:
      existing = self.dict.underlying_dict[key]
      assert isinstance(existing, FunctionSymbol)
      existing.functions += value.functions
    else:
      self.dict[key] = value

  def __contains__(self, item) -> bool:
    return item in self.dict

  def __getitem__(self, key: str) -> Symbol:
    value = self.dict[key]
    if isinstance(value, FunctionSymbol):
      return self.get_overloads(key)
    else:
      return value

  def apply_type_narrowings_from(self, *other_symbol_tables: SymbolTable):
    """
    For all variable symbols, copy common type of all other_symbol_tables.
    """
    for symbol_identifier, self_symbol in self.dict.items():
      if not isinstance(self_symbol, VariableSymbol): continue
      assert all(symbol_identifier in symbol_table for symbol_table in other_symbol_tables)

      other_symbols = [symbol_table[symbol_identifier] for symbol_table in other_symbol_tables]

      assert all(isinstance(other_symbol, VariableSymbol) for other_symbol in other_symbols)
      if len(other_symbols) == 0: continue

      other_symbols = cast(List[VariableSymbol], other_symbols)
      common_type = get_common_type([other_symbol.narrowed_var_type for other_symbol in other_symbols])
      self[symbol_identifier] = self_symbol.copy_set_narrowed_type(common_type)

  def reset_narrowed_types(self):
    """
    Applies symbol.copy_reset_narrowed_type() for all variable symbols.
    """
    for symbol_identifier, symbol in self.dict.items():
      if isinstance(symbol, VariableSymbol):
        if symbol.declared_var_type != symbol.narrowed_var_type:
          self[symbol_identifier] = symbol.copy_reset_narrowed_type()

  def has_extern_func(self, func_identifier: str) -> bool:
    return func_identifier in self.known_extern_funcs

  def get_extern_func(self, func_identifier: str) -> ConcreteFunction:
    assert self.has_extern_func(func_identifier)
    return self.known_extern_funcs[func_identifier]

  def add_extern_func(self, func_identifier: str, concrete_func: ConcreteFunction):
    assert not self.has_extern_func(func_identifier)
    self.known_extern_funcs[func_identifier] = concrete_func

  def get_overloads(self, key: str, include_shadowed=False) -> Optional[OverloadSet]:
    symbols = self.dict.get_all(key)

    if include_shadowed:
      functions = [sym for sym in symbols if isinstance(sym, FunctionSymbol)]
    else:
      functions = list(takewhile(lambda sym: isinstance(sym, FunctionSymbol), symbols))

    functions = cast(List[FunctionSymbol], functions)

    functions = [f for sym in functions for f in sym.functions]  # flat map

    return OverloadSet(key, functions)

  def add_overload(self, identifier: str, overload: Union[FunctionTemplate | List[FunctionTemplate]]) -> bool:
    if not isinstance(overload, List): overload = [overload]

    symbol = self.dict.underlying_dict.setdefault(identifier, FunctionSymbol())
    if not isinstance(symbol, FunctionSymbol): return False
    symbol.functions += overload
    return True

  @property
  def free_overloads(self) -> OverloadSet:
    assert 'free' in self
    return self.get_overloads('free')


class SymbolKind(Enum):
  """
  Possible symbol types.
  """
  VARIABLE = 'var'
  TYPE = 'type'
  FUNCTION = 'func'


def determine_kind(symbol: Symbol) -> SymbolKind:
  if isinstance(symbol, OverloadSet): return SymbolKind.FUNCTION
  if isinstance(symbol, TypeTemplateSymbol): return SymbolKind.TYPE
  if isinstance(symbol, VariableSymbol): return SymbolKind.VARIABLE
  raise TypeError()
