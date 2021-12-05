from __future__ import annotations

import os.path
from pathlib import Path
from typing import List, Optional

import networkx as nx

from sleepy.ast import FileAst, raise_error
from sleepy.grammar import DummyPath
from sleepy.parse import make_file_ast


def _process_file(path: Path | DummyPath, dag: nx.DiGraph) -> List[Path]:
  from sleepy.jit import LIBRARY_PATH
  current_ast = dag.nodes[path]["file_ast"]
  search_paths = ([path.parent] if isinstance(path, Path) else []) + [LIBRARY_PATH]

  new_nodes = []
  for child in current_ast.imports_ast.imports:
    child_path: Optional[Path] = None
    for search_path in search_paths:
      check_path = Path(search_path, child)
      if os.path.exists(check_path):
        child_path = check_path
        break
    if child_path is None:
      raise_error('Cannot resolve import %r.\nSearched in %r' % (child, search_paths), current_ast.imports_ast.pos)

    if child_path not in dag.nodes:
      dag.add_node(child_path, file_ast=make_file_ast(child_path))
      new_nodes.append(child_path)
    dag.add_edge(path, child_path)
  return new_nodes


def build_file_dependency_graph(root_ast: FileAst) -> nx.DiGraph:
  dag = nx.DiGraph()
  dag.add_node(root_ast.pos.file_path, file_ast=root_ast)
  todo = _process_file(root_ast.pos.file_path, dag)

  while len(todo) > 0:
    path = todo.pop()
    todo += _process_file(path, dag)

  return dag
