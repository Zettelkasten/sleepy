import os
from pathlib import Path
from subprocess import CalledProcessError
from tests.run_examples_dir import find_all_example_files

from nose.tools import assert_raises

import _setup_test_env  # noqa


def _test_compile_example(code_file_name):
  assert code_file_name.endswith('.slp')
  exec_file_name = code_file_name[:-len('.slp')]
  print('\nCompiling and running example from %s.' % code_file_name)
  import subprocess
  sleepy_path = Path(__file__).parent.joinpath("../tools/sleepy.py").resolve()

  subprocess.run([str(sleepy_path), code_file_name]).check_returncode()
  subprocess.run([exec_file_name])


def _test_compile_example_failing(code_file_name):
  with assert_raises(CalledProcessError):
    _test_compile_example(code_file_name)


def test_compile_examples():
  code_file_root, code_file_names = find_all_example_files('examples')
  for code_file_name in code_file_names:
    yield _test_compile_example, os.path.join(code_file_root, code_file_name)


def test_compile_examples_import():
  code_file_root, code_file_names = find_all_example_files('examples_import')
  for code_file_name in code_file_names:
    yield _test_compile_example, os.path.join(code_file_root, code_file_name)


def test_compile_examples_cyclic_import():
  code_file_root, code_file_names = find_all_example_files('examples_cyclic_import')
  for code_file_name in code_file_names:
    yield _test_compile_example_failing, os.path.join(code_file_root, code_file_name)