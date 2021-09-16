import better_exchook


def setup():
  """
  Calls necessary setups.
  """
  better_exchook.install()
  better_exchook.replace_traceback_format_tb()

  import _setup_sleepy_env  # noqa


setup()
