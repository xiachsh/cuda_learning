#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <cuda.h>

#define CHECK_CUDA_ERR(x) 				\
	if ( (err = x) != cudaSuccess) 	{ 		\
	printf("%d failed with error :%s\n",__LINE__,cudaGetErrorString(err));	\
	exit(1);							\
	}

#define CHECK_LAST_ERR 					\
	if ( (err = cudaGetLastError()) != cudaSuccess) 	{ 		\
	printf("last cuda call failed with error :%s\n",cudaGetErrorString(err));	\
	exit(1);							\
	}

	



#define WIDTH 16
#define HEIGHT 16



__global__ void print_element(float * input, int pitch)
{
	int bid = blockIdx.x;
	int tid = threadIdx.x;

	float * ptr =  (float*)((char*)input + bid * pitch) + tid;


	printf("[%2d,%2d]:%10.2f\n",bid,tid,*ptr);


}

int main()
{
	cudaError_t err;
	float * buf;
	float * dBuf;

	size_t pitch;
	size_t bytes = WIDTH * HEIGHT * sizeof(float);
	size_t bytesPerRow = WIDTH * sizeof(float);
	

	CHECK_CUDA_ERR ( cudaMallocHost(&buf,bytes) );
	CHECK_CUDA_ERR ( cudaMallocPitch(&dBuf,&pitch,bytesPerRow,HEIGHT) );

	int i  = 0;
	for (;i<HEIGHT;i++) {
		int j = 0;
		for (;j<WIDTH;j++) {
			buf[i * WIDTH + j] = i * WIDTH + j;	
		}	
	}

	CHECK_CUDA_ERR ( cudaMemcpy2D(dBuf,pitch,buf,WIDTH * sizeof(float),WIDTH * sizeof(float),WIDTH,cudaMemcpyDeviceToHost) );
	
	print_element<<<WIDTH,HEIGHT>>> (dBuf,pitch);

	

	
	CHECK_CUDA_ERR( cudaFreeHost(buf) );
	CHECK_CUDA_ERR( cudaFree(dBuf) );
	return 0;
}
