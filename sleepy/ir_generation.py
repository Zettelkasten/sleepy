from llvmlite import ir

from sleepy.types import CleanupHandlingCG


def make_end_block_jump(context: CleanupHandlingCG, continuation: ir.Block, parent_end_block: ir.Block):
  assert context.scope.depth != 0
  builder = context.end_block_builder
  target_depth_reached = builder.icmp_unsigned('==', builder.load(context.function.unroll_count_ir),
                        ir.Constant(typ=ir.IntType(bits=64), constant=context.scope.depth))
  builder.cbranch(target_depth_reached, continuation, parent_end_block)

def make_end_block_return(context: CleanupHandlingCG):
  context.end_block_builder.ret(context.end_block_builder.load(context.function.return_slot_ir))