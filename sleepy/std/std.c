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
void print_int(int i) {
    printf("%d", i);
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
char* allocate_char(int size) {
    return malloc((size_t) size * sizeof(char));
}

char* double_null_ptr() {
    return NULL;
}
char* char_null_ptr() {
    return NULL;
}


void deallocate_double(double* ptr) {
    free(ptr);
}
void deallocate_char(char* ptr) {
    free(ptr);
}

void store_double(double* ptr, double value) {
    *ptr = value;
}
void store_char(char* ptr, char value) {
    *ptr = value;
}

double load_double(double* ptr) {
    return *ptr;
}
char load_char(char* ptr) {
    return *ptr;
}

void memcpy_double(double* to, double* from, int len) {
    memcpy(to, from, len);
}
void memcpy_char(char* to, char* from, int len) {
    memcpy(to, from, len);
}

int double_to_int(double d) {
    return d;
}
double int_to_double(int i) {
    return i;
}

double random_double() {
    return (double) rand() / RAND_MAX;
}
