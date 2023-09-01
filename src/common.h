#pragma once

#ifdef ON_CPU
#include <stdio.h>
#include <stdlib.h>
#endif

typedef size_t IDX;
typedef unsigned char BYTE;

#ifdef USE_FLOAT
typedef float FPN;
#define NEG_INF -1.0f / 0.0f;
#ifndef EPSILON
#define EPSILON 1e-19f
#endif
#define FZERO 0.0f
#define FONE 1.0f
#else
typedef double FPN;
#define NEG_INF -1.0 / 0.0;
#ifndef EPSILON
#define EPSILON 1e-19
#endif
#define FZERO 0.0
#define FONE 1.0
#endif // DEBUG

FPN FPN_abs(FPN a);

// clamps magnitude, i.e. prevents closeness to 0
void clamp(FPN *a);

// clamps magnitude, i.e. prevents closeness to 0
FPN clamped(FPN a);

#ifdef ON_CPU
#define SETREDTEXT "\033[1;31m"
#define RESETCOLOR "\033[1;0m"
#endif

void print_vec(FPN *vec, IDX len);
