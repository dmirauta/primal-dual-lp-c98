
#ifdef ON_CPU
#include <stdio.h>
#include <stdlib.h>
#endif

// hacky generics
typedef double FPN;
typedef size_t IDX; // unneccesary redefinition? (defined in OpenCL?)

typedef unsigned char BYTE;

#define NEG_INF -1.0 / 0.0;

#define EPSILON 1e-19

#define FZERO 0.0

#define FONE 1.0

#define SETREDTEXT "\033[1;31m"
#define RESETCOLOR "\033[1;0m"
