# ![Sleepy Icon](https://github.com/leanderBehr/SleepyLangSupport/blob/master/src/main/resources/icons/bearded-man.png) Compiler for Sleepy

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
* Descriptive compiler error messages
* Debugging with e.g. GDB

## How to use
To get started, clone the repository and run `nosetests` in the root directory.
In particular, this will compile and execute all files in `tests/examples/`, where you can also add your own programs.
To just run your own Sleepy program, simply execute `tools/sleepy.py your_program.slp`.


## Example Program

This example program here illustrates some interesting features
of Sleepy. You can also find in `usage_examples/lists.slp` and run it yourself.

```
import "list.slp"

func main() {
  # 1. Create an empty list on the heap
  # The standard library provides an implementation for resizable continuous Lists
  # As the implementation uses a template type, we could use lists of any type,
  # here we use List[Double].
  list: List[Double] = EmptyList[Double]()

  # 2. Add some data to our list
  i = 0l  # as no type for i is given, it is implicitly inferred (i: Long = 0l)
  while i < 10l {
    add(list, 5.0 - ToDouble(i))
    i += 1l
  }
  print_line("The unsorted list:")
  print_line(list)
  assert(list.size == 10l)
  assert(not(is_sorted(list)))

  # 3. Sort the list
  quicksort(list)  # sorts the array in-place, provided by the standard library
  print_line("The sorted list:")
  print_line(list)
  assert(is_sorted(list))

  # 4. Now that we sorted the array, we can use binary search to find values quickly!
  # For this we implement a function returning either a special "NotFound" type or the
  # index of the value. The language provides a tagged union type for this: Long|NotFound
  # Further, we want it to work on lists holding any data type, so we define a
  # function with a template type "T" here.
  # Also note that functions and structs can be defined within any scope.
  
  struct NotFound {}  # struct without any members
  
  func my_binary_search[T](list: List[T], key: T) -> Long|NotFound {
    begin = 0l
    end = list.size
    middle = begin + (end - begin) / 2l

    while begin != end {
      if key < list[middle] {
        end = middle
      } else {
        if list[middle] < key {
          begin = middle + 1l
        } else {
          return middle
        }
      }
      middle = begin + (end - begin) / 2l
    }
    return NotFound()  # call an implicitly created constructor for our NotFound type
  }

  # search for 0.0 within the list
  search_result: Long|NotFound = my_binary_search(list, 0.0)
  # the return value is either a Long or a NotFound.
  # Unions are type-safe, i.e. the compiler implicitly allocates a bit to keep track of which type they are.
  if search_result is NotFound {
    print_line("Something went wrong! We did not find 0.0!")
  }  else {
    # The compiler analyses control statements and now explicitly knows search_result must be a Long.
    # Therefore, you can use it as a Long now within this scope.
    print("We found our value at index ")
    print(search_result)
    print(" with associated value ")  # will print 0.0
    print_line(list[search_result])
    # note that above we can only use list[search_result] because the compiler knows that it
    # will always be a Long here!
  }
}
```

## Thoughts on the Current State of this Project

- Sleepy currently already supports all basic C features and many more and therefore could in theory be used on an everyday basis.
- While we have created many unit tests and are confident that most features work as expected, of course there will still be some unexpected problems.
- The biggest problem with the compiler right now is that the fact that it was written in Python (including the lexer and parser) and therefore is very slow. Compiling a small program already takes more than one second (reason for this is also that the standard library will be compiled as well).
- However, as Sleepy creates LLVM bytecode the compiled executable will be very fast.
- Some language features of Sleepy are not 100% thought through yet. For example, the most recent development of the language focused on reference types (like references in C++), which are essentially pointers that can syntactically also be used as if they directly were of the type they are pointing to instead.
  We wanted to design references in Sleepy in a generalized way (e.g., allowing references to references) and experimented a lot with different semantics for this. You can find many examples using the `!` and the `Ref[T]` type in the test cases for this. However, this language feature is not at all finished yet and requires much more conceptual work to become stable.