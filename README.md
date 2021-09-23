# Compiler for Sleepy

![CI Badge](https://github.com/Zettelkasten/sleepy/actions/workflows/main.yml/badge.svg)

This is a compiler written in Python for our experimental programming language Sleepy.

To get started, clone the repository and run `nosetests` in the root directory.
In particular, this will compile and execute all files in `tests/examples/`, where you can also add your own programs.
To just run your own Sleepy program, simply execute `tools/sleepy.py your_program.slp`.

In case you use Nano, move the `sleepy/sleepy.nanorc` to `/usr/share/nano/sleepy.nanorc` to get syntax highlighting.

For example, this Sleepy program prints numbers from `-3` to `5` (very impressive!):
```c++
func main() {
  func count_from_to(Int from, Int to) {
    if from > to { return; }
    print(from);
    if from != to { print(','); print(' '); }
    count_from_to(from + 1, to);
  }

  count_from_to(-3, 5);
  print('\n');
}
```

Or this more complicated example implements a variable-sized list for doubles:
```c++
@RefType struct DoubleList {
  DoublePtr pointer = allocate(8);
  Int alloc_length = 8;
  Int length = 0;
}
func empty_list() -> DoubleList {
  return DoubleList(allocate(8), 8, 0);
}
func get(DoubleList l, Int idx) -> Double {
  assert(and(idx >= 0, idx < l.length));
  return load(l.pointer + idx);
}
func set(DoubleList l, Int idx, Double val) {
  assert(and(idx >= 0, idx < l.length));
  store(l.pointer + idx, val);
}
func insert(DoubleList l, Double val) {
  if l.length >= l.alloc_length {
    # no space left, resize
    new_alloc_length = 2 * l.alloc_length;
    new_pointer = allocate(new_alloc_length);
    memcpy(new_pointer, l.pointer, l.alloc_length * 8);
    deallocate(l.pointer);
    l.alloc_length = new_alloc_length;
    l.pointer = new_pointer;
  }
  new_idx = l.length;
  l.length = l.length + 1;
  set(l, new_idx, val);
}
func insert_all(DoubleList l, DoubleList other) {
  other_length = other.length;
  idx = 0;
  while idx < other_length {
    insert(l, get(other, idx));
    idx = idx + 1;
  }
}
func remove(DoubleList l, Int idx) {
  assert(and(idx >= 0, idx < l.length));
  # move all elements after idx one to front
  move_idx = idx + 1;
  while move_idx < l.length {
    set(l, move_idx - 1, get(l, move_idx));
    move_idx = move_idx + 1;
  }
  l.length = l.length - 1;
}

func print(DoubleList l) {
  print('[');
  idx = 0;
  while idx < l.length {
    print(get(l, idx));
    idx = idx + 1;
    if idx < l.length { print(','); }
  }
  print(']');
}
```
which you can then use e.g. like this:
```c++
func main() {
  list = empty_list();
  insert(list, 5.0);
  insert(list, 7.0);
  insert_all(list, list);
  remove(list, 2);
  print(list);  # will print [5.000000,7.000000,7.000000]
}
```

See [the example folder](https://github.com/Zettelkasten/sleepy/tree/main/tests/examples) for more.
