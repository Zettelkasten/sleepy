#include <stdio.h>
#include <stdlib.h>

double print_char(double c) {
    printf("%c", (char) c);
    return 0;
}

double print_double(double d) {
    printf("%f", d);
    return 0;
}

double* double_to_ptr(double d) {
    return (double*) (size_t) (d * 8);
}

#pragma clang diagnostic push
#pragma ide diagnostic ignored "bugprone-integer-division"
double ptr_to_double(void* ptr) {
    return (double)((size_t)ptr / 8);
}
#pragma clang diagnostic pop


double allocate(double size) {
    void* array = malloc((size_t) size * sizeof(double));
    return ptr_to_double(array);
}

double deallocate(double d_ptr) {
    free(double_to_ptr(d_ptr));
    return 0;
}

double store(double ptr, double value) {
    *double_to_ptr(ptr) = value;
    return value;
}

double load(double ptr) {
    return *double_to_ptr(ptr);
}