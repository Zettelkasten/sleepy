from __future__ import annotations

from pathlib import Path

from sleepy.ast import FileAst, TranslationUnitAst
from sleepy.grammar import DummyPath
from sleepy.sleepy_lexer import SLEEPY_LEXER
from sleepy.sleepy_parser import SLEEPY_ATTR_GRAMMAR, SLEEPY_PARSER


def make_file_ast(file_path: Path) -> FileAst:
  assert isinstance(file_path, Path)
  with open(file_path) as program_file:
    program = program_file.read()
  return make_file_ast_from_str(file_path, program=program)


def make_file_ast_from_str(file_path: Path | DummyPath, program: str) -> FileAst:
  tokens, tokens_pos = SLEEPY_LEXER.tokenize(program, file_path=file_path)
  _, root_eval = SLEEPY_PARSER.parse_syn_attr_analysis(
    attr_grammar=SLEEPY_ATTR_GRAMMAR, word=program, tokens=tokens, tokens_pos=tokens_pos, file_path=file_path)
  program_ast = root_eval['ast']
  assert isinstance(program_ast, FileAst)
  return program_ast


def make_preamble_ast() -> FileAst:
  preamble_path = Path(__file__).parent.joinpath("std/preamble.slp").resolve()
  file_ast = make_file_ast(preamble_path)
  return file_ast


def _make_translation_unit_ast_from_str(file_ast: FileAst, add_preamble: bool = True) -> TranslationUnitAst:
  from tools.import_discovery import build_file_dependency_graph
  from sleepy.errors import CompilerError
  import networkx as nx
  dependency_graph = build_file_dependency_graph(root_ast=file_ast)
  if nx.number_of_selfloops(dependency_graph) > 0:
    raise CompilerError(
      f"Import error: Imports are cyclic. File {nx.nodes_with_selfloops(dependency_graph)} imports itself")
  if not nx.is_directed_acyclic_graph(dependency_graph):
    sccs = [scc for scc in nx.strongly_connected_components(dependency_graph) if len(scc) > 1]
    raise CompilerError(
      f"Import error: Imports are cyclic. The following cycles were found:\n{sccs}")

  file_asts = [dependency_graph.nodes[node]["file_ast"] for node in nx.topological_sort(dependency_graph.reverse())]
  if add_preamble:
    file_asts.insert(0, make_preamble_ast())
  return TranslationUnitAst.from_file_asts(file_asts)


def make_translation_unit_ast_from_str(file_path: Path | DummyPath, program: str,
                                       add_preamble: bool = True) -> TranslationUnitAst:
  root_ast = make_file_ast_from_str(file_path=file_path, program=program)
  return _make_translation_unit_ast_from_str(file_ast=root_ast, add_preamble=add_preamble)


def make_translation_unit_ast(file_path: Path, add_preamble: bool = True) -> TranslationUnitAst:
  root_ast = make_file_ast(file_path)
  return _make_translation_unit_ast_from_str(file_ast=root_ast, add_preamble=add_preamble)
