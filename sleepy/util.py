from typing import TypeVar, List, Dict

T = TypeVar('T')
U = TypeVar('U')


def concat_dicts(dicts: List[Dict[T, U]]) -> Dict[T, U]:
  result = dicts[0]
  for d in dicts[1:]:
    result.update(d)
  return result
