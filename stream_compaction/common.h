#pragma once

#include <cstdio>
#include <cstring>
#include <cmath>
#include <ctime>
#include <chrono>
#include <iostream>
#include <src\sceneStructs.h>

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)

/**
 * Check for CUDA errors; print and exit if there was a problem.
 */
void checkCUDAErrorFn(const char *msg, const char *file = NULL, int line = -1);

inline int ilog2(int x) {
    int lg = 0;
    while (x >>= 1) {
        ++lg;
    }
    return lg;
}

inline int ilog2ceil(int x) {
    return ilog2(x - 1) + 1;
}


namespace StreamCompaction {
namespace Common {
	__global__ void kernMapToBoolean(int n, int *bools, const PathSegment *idata);

	__global__ void kernScatter(int n, PathSegment *odata,
		const PathSegment *idata, const int *bools, const int *indices);
}
}
