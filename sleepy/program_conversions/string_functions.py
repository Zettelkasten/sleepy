from typing import List, Tuple


def replace(original: str, replacements: List[Tuple[slice, str]]) -> str:
  prev = 0
  result = ''
  for at, repl in replacements:
    result += original[prev:at.start]
    result += repl
    prev = at.stop
  result += original[prev:]
  return result

def trim_whitespace(string: str, segment: slice):
  start = segment.start
  stop = segment.stop
  step = segment.step if segment.step is not None else 1
  while string[start].isspace() and start < stop: start += step
  while string[stop - 1].isspace() and stop > start: stop -= step
  return slice(start, stop, step)

