import os
import sys

current_dir = os.path.dirname(os.path.realpath(os.path.abspath(__file__)))
root_dir = os.path.dirname(current_dir)
sys.path.insert(0, root_dir)
