import _setup_test_env  # noqa
import better_exchook
import sys
import unittest


def test_compile_script():
  import subprocess
  subprocess.run(['./tools/sleepy.py', 'tests/examples/hello.slp']).check_returncode()
  subprocess.run(['./tests/examples/hello']).check_returncode()


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
