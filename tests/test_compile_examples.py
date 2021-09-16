import os

import _setup_test_env  # noqa
import better_exchook
import sys
import unittest


def _test_compile_example(code_file_name):
  assert code_file_name.endswith('.slp')
  exec_file_name = code_file_name[:-len('.slp')]
  print('\nCompiling and running example from %s.' % code_file_name)
  import subprocess
  subprocess.run(['./tools/sleepy.py', code_file_name]).check_returncode()
  subprocess.run([exec_file_name])


def test_compile_examples():
  from tests.run_examples_dir import find_all_example_files
  code_file_root, code_file_names = find_all_example_files('examples')
  for code_file_name in code_file_names:
    yield _test_compile_example, os.path.join(code_file_root, code_file_name)
