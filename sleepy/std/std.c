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

char* double_null_ptr() {
    return NULL;
}
float* float_null_ptr() {
    return NULL;
}
char* char_null_ptr() {
    return NULL;
}
int* int_null_ptr() {
    return NULL;
}
long* long_null_ptr() {
    return NULL;
}


void deallocate_double(double* ptr) {
    free(ptr);
}
void deallocate_float(float* ptr) {
    free(ptr);
}
void deallocate_char(char* ptr) {
    free(ptr);
}
void deallocate_int(int* ptr) {
    free(ptr);
}
void deallocate_long(long* ptr) {
    free(ptr);
}

void store_double(double* ptr, double value) {
    *ptr = value;
}
void store_float(float* ptr, float value) {
    *ptr = value;
}
void store_char(char* ptr, char value) {
    *ptr = value;
}
void store_int(int* ptr, int value) {
    *ptr = value;
}

double load_double(double* ptr) {
    return *ptr;
}
double load_float(float* ptr) {
    return *ptr;
}
char load_char(char* ptr) {
    return *ptr;
}
int load_int(int* ptr) {
    return *ptr;
}
long load_long(long* ptr) {
    return *ptr;
}

void memcpy_double(double* to, double* from, int len) {
    memcpy(to, from, len);
}
void memcpy_float(float* to, float* from, int len) {
    memcpy(to, from, len);
}
void memcpy_char(char* to, char* from, int len) {
    memcpy(to, from, len);
}
void memcpy_int(int* to, int* from, int len) {
    memcpy(to, from, len);
}
void memcpy_long(long* to, long* from, int len) {
    memcpy(to, from, len);
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

double random_double() {
    return (double) rand() / RAND_MAX;
}
