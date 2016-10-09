#include "common.h"


namespace StreamCompaction {
namespace Common {

	/**
	* Maps an array to an array of 0s and 1s for stream compaction. Elements
	* which map to 0 will be removed, and elements which map to 1 will be kept.
	*/
	__global__ void kernMapToBoolean(int n, int *bools, const PathSegment *idata) {
		int index = (blockIdx.x * blockDim.x) + threadIdx.x;
		if (index >= n)
			return;
		bools[index] = idata[index].remainingBounces <= 0 ? 0 : 1;
	}

	/**
	* Performs scatter on an array. That is, for each element in idata,
	* if bools[idx] == 1, it copies idata[idx] to odata[indices[idx]].
	*/
	__global__ void kernScatter(int n, PathSegment *odata,
		const PathSegment *idata, const int *bools, const int *indices) {
		int index = (blockIdx.x * blockDim.x) + threadIdx.x;
		if (index >= n)
			return;
		if (bools[index] == 1)
			odata[indices[index]] = idata[index];
	}

}
}
