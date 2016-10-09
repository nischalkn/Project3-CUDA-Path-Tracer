#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

#define blockSize 128

namespace StreamCompaction {
namespace Efficient {

	__global__ void upSweep(int n, int *idata, int d) {
		int k = (blockIdx.x * blockDim.x) + threadIdx.x;
		if (k >= n)
			return;
		if (k % (d * 2) == (d * 2) - 1) {
			idata[k] += idata[k - d];
		}

	}

	__global__ void downSweep(int n, int *idata, int d) {
		int k = (blockIdx.x * blockDim.x) + threadIdx.x;
		if (k >= n)
			return;
		int temp;
		if (k % (d * 2) == (d * 2) - 1) {
			//printf("kernel: %d", k);
			temp = idata[k - d];
			idata[k - d] = idata[k];  // Set left child to this node’s value
			idata[k] += temp;
		}

	}

	__global__ void makeElementZero(int *data, int index) {
		int k = (blockIdx.x * blockDim.x) + threadIdx.x;
		if (index == k) {
			data[k] = 0;
		}
	}

	__global__ void scan(int n, int D, int *odata, int *idata) {
		extern __shared__ int s_idata[];
		int k = (blockIdx.x * blockDim.x) + threadIdx.x;
		if (k >= n)
			return;
		s_idata[k] = idata[k];
		__syncthreads();
		for (int d = 0; d < D; d++) {
			if (k % ((1 << d) * 2) == ((1 << d) * 2) - 1) {
				s_idata[k] += s_idata[k - (1 << d)];
			}
		}
		__syncthreads();
		if (n-1 == k) {
			s_idata[k] = 0;
		}
		__syncthreads();
		for (int d = D - 1; d >= 0; d--) {
			int temp;
			if (k % ((1 << d) * 2) == ((1 << d) * 2) - 1) {
				//printf("kernel: %d", k);
				temp = idata[k - (1 << d)];
				idata[k - (1 << d)] = idata[k];  // Set left child to this node’s value
				idata[k] += temp;
			}
		}
		__syncthreads();
		odata[k] = s_idata[k];
	}

	__global__ void copyElements(int n, int *src, int *dest) {
		int index = (blockIdx.x * blockDim.x) + threadIdx.x;
		if (index >= n)
			return;
		dest[index] = src[index];
	}

	/**
	 * Performs stream compaction on idata, storing the result into odata.
	 * All zeroes are discarded.
	 *
	 * @param n      The number of elements in idata.
	 * @param odata  The array into which to store elements.
	 * @param idata  The array of elements to compact.
	 * @returns      The number of elements remaining after compaction.
	 */
	int compact(int n, PathSegment * dev_idata, PathSegment * dev_odata) {
		int *dev_boolean;
		int *dev_indices;
		int count;

		int paddedArraySize = 1 << ilog2ceil(n);

		dim3 fullBlocksPerGrid((n + blockSize - 1) / blockSize);
		dim3 fullBlocksPerGridPadded((paddedArraySize + blockSize - 1) / blockSize);

		cudaMalloc((void**)&dev_boolean, paddedArraySize * sizeof(int));
		checkCUDAError("Cannot allocate memory for boolean");
		cudaMalloc((void**)&dev_indices, paddedArraySize * sizeof(int));
		checkCUDAError("Cannot allocate memory for dev_indices");

		StreamCompaction::Common::kernMapToBoolean << <fullBlocksPerGrid, blockSize >> >(n, dev_boolean, dev_idata);

		copyElements << <fullBlocksPerGrid, blockSize >> >(n, dev_boolean, dev_indices);


		for (int d = 0; d < ilog2ceil(paddedArraySize); d++) {
			upSweep << <fullBlocksPerGridPadded, blockSize >> >(paddedArraySize, dev_indices, 1 << d);
		}

		makeElementZero << <fullBlocksPerGridPadded, blockSize >> >(dev_indices, paddedArraySize - 1);

		for (int d = ilog2ceil(paddedArraySize) - 1; d >= 0; d--) {
			downSweep << <fullBlocksPerGridPadded, blockSize >> >(paddedArraySize, dev_indices, 1 << d);
		}

		StreamCompaction::Common::kernScatter << <fullBlocksPerGrid, blockSize >> >(n, dev_odata, dev_idata, dev_boolean, dev_indices);

		cudaMemcpy(dev_idata, dev_odata, n*sizeof(PathSegment), cudaMemcpyDeviceToDevice);
		cudaMemcpy(&count, dev_indices + paddedArraySize - 1, sizeof(int), cudaMemcpyDeviceToHost);

		cudaFree(dev_boolean);
		cudaFree(dev_indices);
		return count;
	}

}
}
