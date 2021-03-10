import _setup_test_env  # noqa
import sys
import unittest
import better_exchook
from nose.tools import assert_equals

from sleepy.grammar import Production, AttributeGrammar


def test_AttributeGrammar_syn():
  g = AttributeGrammar(prods=[
      Production('S', 'S', '+', 'S'),
      Production('S', 'zero'),
      Production('S', 'digit')
    ],
    inh_attrs=set(),
    syn_attrs={'res'},
    prod_attr_rules=[
      {'res.0': lambda res: res(1) + res(3)},
      {'res.0': lambda res: 0},
      {'res.0': lambda res: res(1)}
    ],
    terminal_attr_rules={'digit': lambda word: int(word)}
  )
  assert_equals(g.attrs, {'res'})
  assert_equals(g.syn_attrs, {'res'})
  assert_equals(g.inh_attrs, set())


if __name__ == "__main__":
  try:
    better_exchook.install()
    if len(sys.argv) <= 1:
      for k, v in sorted(globals().items()):
        if k.startswith("test_"):
          print("-" * 40)
          print("Executing: %s" % k)
          try:
            v()
          except unittest.SkipTest as exc:
            print("SkipTest:", exc)
          print("-" * 40)
      print("Finished all tests.")
    else:
      assert len(sys.argv) >= 2
      for arg in sys.argv[1:]:
        print("Executing: %s" % arg)
        if arg in globals():
          globals()[arg]()  # assume function and execute
        else:
          eval(arg)  # assume Python code and execute
  finally:
    pass
