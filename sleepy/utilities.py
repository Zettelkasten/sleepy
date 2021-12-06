from typing import TypeVar, List, Dict

from llvmlite import ir

from sleepy.symbols import CodegenContext

T = TypeVar('T')
U = TypeVar('U')


def concat_dicts(dicts: List[Dict[T, U]]) -> Dict[T, U]:
  result = dicts[0]
  for d in dicts[1:]:
    result.update(d)
  return result


def truncate_ir_value(from_type: ir.Type, to_type: ir.Type, ir_val: ir.Value,
                      context: CodegenContext, name: str) -> ir.Value:
  # Note: if the size of to_type is strictly smaller than from_type, we need to truncate the value
  # There is no LLVM instruction for this, so we alloca memory and reinterpret a pointer on this
  ptr = context.alloca_at_entry(from_type, name='%s_from_ptr' % name)
  context.builder.store(value=ir_val, ptr=ptr)
  truncated_ptr = context.builder.bitcast(val=ptr, typ=ir.PointerType(to_type), name='%s_from_ptr_truncated' % name)
  return context.builder.load(truncated_ptr, name='%s_from_val_truncated' % name)
