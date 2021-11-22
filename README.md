# Compiler for Sleepy

![CI Badge](https://github.com/Zettelkasten/sleepy/actions/workflows/main.yml/badge.svg)

This is a compiler written in Python for our experimental programming language Sleepy.
There is also a plugin for the intellij family of IDEs based on the compiler [here](https://github.com/leanderBehr/SleepyLangSupport).

For examples see [tests/examples](https://github.com/Zettelkasten/sleepy/tree/main/tests/examples), [usage_examples](https://github.com/Zettelkasten/sleepy/tree/main/usage_examples) as well as the [standard library source code](https://github.com/Zettelkasten/sleepy/blob/main/sleepy/std/lib/list.slp).

### Interesting Features
* Semicolons are optional, except when needed to separate statements on the same line. Similar to e.g. Kotlin.
* Built in tagged unions with `TypeA|TypeB` syntax
   - Type narrowing by control statements or explicitly via `assert(var is Type)`
* Monomorphic, i.e. C++-like, templates.
* Implicit, but fully static, type-safe variable declarations.
* Function scope is not special, anything can be done at any nesting level of functions.

### Ordinary Features
* Loops, functions, variables, pointers... a C-ish imperative language.
* Imports
* Operator overloading, slices
* Debugging with e.g. GDB

## How to use
To get started, clone the repository and run `nosetests` in the root directory.
In particular, this will compile and execute all files in `tests/examples/`, where you can also add your own programs.
To just run your own Sleepy program, simply execute `tools/sleepy.py your_program.slp`.
