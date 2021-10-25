import os.path
from pathlib import Path
from typing import List, Tuple, Optional

import networkx as nx

from sleepy.ast import FileAst
from sleepy.errors import CompilerError
from sleepy.parse import make_program_ast


def make_ast_for_file(path: Path) -> FileAst:
  with open(path) as program_file:
    program = program_file.read()
    ast = make_program_ast(program)
    ast.file_path = path
    return ast


def resolve_import_path(import_path: Path, importing_path: Path) -> Path:
  if import_path.is_absolute():
    return import_path.resolve()
  return importing_path.parent.joinpath(import_path).resolve()


def process_file(path: Path, dag: nx.DiGraph) -> List[Path]:
  from sleepy.jit import LIBRARY_PATH
  search_paths = ['', LIBRARY_PATH]

  current_ast = make_ast_for_file(path)
  dag.nodes[path]["file_ast"] = current_ast

  new_nodes = []

  for child in current_ast.imports_ast.imports:
    child_path: Optional[Path] = None
    for search_path in search_paths:
      check_path = Path(search_path, child)
      if os.path.exists(check_path):
        child_path = check_path
        break
    if child_path is None:
      current_ast.imports_ast.raise_error('Cannot resolve import %r.\nSarched in %r' % (child, search_paths))
    child = resolve_import_path(import_path=child_path, importing_path=path)

    if child not in dag.nodes:
      dag.add_node(child, file_ast=None)
      new_nodes.append(child)

    dag.add_edge(path, child)

  return new_nodes


def build_file_dag(main_file: Path) -> Tuple[nx.DiGraph, Path]:
  if not main_file.is_absolute():
    main_file = Path.cwd().joinpath(main_file).resolve()

  dag = nx.DiGraph()

  dag.add_node(main_file, file_ast=None)
  todo = process_file(main_file, dag)

  while len(todo) > 0:
    path = todo.pop()
    todo += process_file(path, dag)

  return dag, main_file


def check_graph(graph: nx.DiGraph):
  if not nx.is_directed_acyclic_graph(graph):
    sccs = [scc for scc in nx.strongly_connected_components(graph) if len(scc) > 1]
    raise CompilerError(f"Import are cyclic. The following cycles were found:\n {sccs}")
