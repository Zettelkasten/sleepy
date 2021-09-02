#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <string.h>

void print_char(char c) {
    putc(c, stdout);
}
void print_double(double d) {
    printf("%f", d);
}
void print_float(float f) {
    printf("%f", f);
}
void print_int(int i) {
    printf("%d", i);
}
void print_long(long l) {
    printf("%ld", l);
}

void flush() {
    fflush(stdout);
}

void assert(_Bool property) {
    if (!property) {
        printf("Assertion failed!\n");
        raise(SIGABRT);
    }
}

double* allocate_double(int size) {
    return malloc((size_t) size * sizeof(double));
}
float* allocate_float(int size) {
    return malloc((size_t) size * sizeof(float));
}
char* allocate_char(int size) {
    return malloc((size_t) size * sizeof(char));
}
int* allocate_int(int size) {
    return malloc((size_t) size * sizeof(int));
}
long* allocate_long(int size) {
    return malloc((size_t) size * sizeof(long));
}

void deallocate_double(double* ptr) {
    free(ptr);
}

int double_to_int(double d) {
    return d;
}
double int_to_double(int i) {
    return i;
}
int long_to_int(long l) {
    return l;
}
long int_to_long(int i) {
    return i;
}
long char_ptr_to_long(char* ptr) {
    return (long) ptr;
}
long float_ptr_to_long(float* ptr) {
    return (long) ptr;
}

double random_double() {
    return (double) rand() / RAND_MAX;
}
