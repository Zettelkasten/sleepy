cmake_minimum_required(VERSION 3.17)
project(std C)

set(CMAKE_C_STANDARD 11)

add_library(std SHARED std.c)
add_library(std_static STATIC std.c)
