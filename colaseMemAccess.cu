#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#include "common.cuh"

#define N 128*1024
#define MAX_OFFSET 128



__global__ void assignValue (float *a,float *b)
{
	int idx = blockDim.x * blockIdx.x + threadIdx.x;

	a[idx] = 1.0f * idx;
	b[idx] = 2.0f * idx;
}


__global__ void memcpyOffset(float *a,float *b,int offset)
{
        int idx = blockDim.x * blockIdx.x + threadIdx.x + offset;

        a[idx] = b[idx];
}


int main()
{
	cudaError_t err;
	cudaEvent_t start,end;

	size_t bytes = sizeof(float) * (N + MAX_OFFSET) ;
	
	
	float * a,*b;
	float * c;
	
	c = (float *) malloc(bytes);
	CHECK_CUDA_ERR( cudaMalloc(&a,bytes) );
	CHECK_CUDA_ERR( cudaMalloc(&b,bytes) );

	CHECK_CUDA_ERR(	cudaEventCreate(&start) );
	CHECK_CUDA_ERR( cudaEventCreate(&end)   );
	

	assignValue<<<128,1024>>> (a,b);

	CHECK_LAST_CUDA_ERR

	int i = 0;
	float ms = 0;
	for (i = 0;i<MAX_OFFSET;i++ ) {
		
		CHECK_CUDA_ERR(	cudaEventRecord(start,0));
		memcpyOffset<<<128,1024>>>(a,b,i);
		CHECK_LAST_CUDA_ERR
		CHECK_CUDA_ERR( cudaEventRecord(end,0));
		CHECK_CUDA_ERR( cudaDeviceSynchronize());

		CHECK_CUDA_ERR( cudaEventElapsedTime(&ms,start,end));

		printf("offset :%d time :%f milli second\n",i,ms);
	}
	CHECK_CUDA_ERR( cudaEventDestroy(start) );
	CHECK_CUDA_ERR( cudaEventDestroy(end) );



	CHECK_CUDA_ERR( cudaFree(a) );		
	CHECK_CUDA_ERR( cudaFree(b) );		
	free(c);

	return 0;
}
