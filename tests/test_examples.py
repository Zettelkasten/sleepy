import _setup_test_env  # noqa
import better_exchook
import sys
import unittest

from tests.run_examples_dir import make_example_cases


def test_examples():
  for x in make_example_cases('examples'):
    yield x


def test_exprs_as_stmts():
  for x in make_example_cases('examples_exprs_as_stmts'):
    yield x
