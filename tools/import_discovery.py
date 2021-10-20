from pathlib import Path
from typing import List, Tuple

import networkx as nx

from sleepy.ast import FileAst
from sleepy.parse import make_program_ast


def make_ast_for_file(path: Path) -> FileAst:
  with open(path) as program_file:
    program = program_file.read()
    ast = make_program_ast(program)
    return ast


def resolve_import_path(import_path: Path, importing_path: Path) -> Path:
  if import_path.is_absolute(): return importing_path.resolve()
  return importing_path.parent.joinpath(import_path).resolve()

def process_file(path: Path, dag: nx.DiGraph) -> List[Path]:
  current_ast = make_ast_for_file(path)
  dag.nodes[path]["file_ast"] = current_ast

  new_nodes = []

  for child in current_ast.imports_ast.imports:
    child = resolve_import_path(import_path=Path(child), importing_path=path)

    if child not in dag.nodes:
      dag.add_node(child, file_ast=None)
      new_nodes.append(child)

    dag.add_edge(path, child)

  return new_nodes

def build_file_dag(main_file: Path) -> Tuple[nx.DiGraph, Path]:
  if not main_file.is_absolute(): main_file = Path.cwd().joinpath(main_file).resolve()

  dag = nx.DiGraph()

  dag.add_node(main_file, file_ast=None)
  todo = process_file(main_file, dag)

  while len(todo) > 0:
    path = todo.pop()
    todo += process_file(path, dag)

  return dag, main_file