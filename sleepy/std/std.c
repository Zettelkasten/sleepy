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

double* allocate(int size) {
    return malloc((size_t) size * sizeof(double));
}

void deallocate(double* ptr) {
    free(ptr);
}

void store(double* ptr, double value) {
    *ptr = value;
}

double load(double* ptr) {
    return *ptr;
}
