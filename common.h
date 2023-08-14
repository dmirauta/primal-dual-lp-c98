
#ifdef ON_CPU
#include <stdio.h>
#include <stdlib.h>
#endif

// hacky generics
typedef double FPN;
typedef size_t IDX; // unneccesary redefinition? (defined in OpenCL?)

typedef unsigned char BYTE;

#define NEG_INF -1.0 / 0.0;
