# Compiler for Sleepy Script

![CI Badge](https://github.com/Zettelkasten/sleepy/actions/workflows/main.yml/badge.svg)

This is a compiler written in Python for my experimental programming language Sleepy Script.

To get started, clone the repository and run `nosetests` in the root directory.
In particular, this will compile and execute all files in `tests/examples/`, where you can also add your own programs.
To just run tests for files in this directory, run `nosetests -s tests/test_examples.py`

For example, this Sleepy program prints numbers from `-3` to `5` (very impressive!):
```c++
func main() {
  func count_from_to(Int from, Int to) {
    if from > to { return; }
    print_int(from);
    if from != to { print_char(','); print_char(' '); }
    count_from_to(from + 1, to);
  }

  count_from_to(-3, 5);
  print_char('\n');
}
```
See [the example folder](https://github.com/Zettelkasten/sleepy/tree/main/tests/examples) for more.
