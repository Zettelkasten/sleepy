#include <stdio.h>
#include <stdlib.h>
#include <signal.h>

void print_char(double c) {
    printf("%c", (char) c);
}

void print_double(double d) {
    printf("%f", d);
}

void print_int(int i) {
    printf("%d", i);
}

double assert(double property) {
    if (property != 1.0) {
        printf("Assertion failed!\n");
        raise(SIGABRT);
    }
    return 0;
}

double* double_to_ptr(int d) {
    return (double*) (size_t) (d * 8);
}

#pragma clang diagnostic push
#pragma ide diagnostic ignored "bugprone-integer-division"

int ptr_to_double(void* ptr) {
    assert((size_t) ptr % 8 == 0);
    return (int) ((size_t) ptr / 8);
}

#pragma clang diagnostic pop


int allocate(int size) {
    void* array = malloc(size * sizeof(double));
    return ptr_to_double(array);
}

void deallocate(int d_ptr) {
    free(double_to_ptr(d_ptr));
}

void store(int ptr, double value) {
    *double_to_ptr(ptr) = value;
}

double load(int ptr) {
    return *double_to_ptr(ptr);
}
