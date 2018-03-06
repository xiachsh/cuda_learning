#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>



__global__ void kernel1()
{
	printf("kernel #1\n");
}

__global__ void kernel2()
{
	printf("kernel #2\n");
}
int main(int argc,char **argv)
{

	printf("Testing multiple kernel launch and show in-order execution of two kernels \n");

	int nThreadsPerBlock = 32;
	
	int blocks = 128;
	cudaStream_t s1;
	cudaStream_t s2;


	cudaStreamCreate(&s1);
	cudaStreamCreate(&s2);

	kernel1<<<blocks,nThreadsPerBlock,0,s1>>> ();	
	cudaError_t err = cudaGetLastError();               
        if (err != cudaSuccess ) printf("cuda function failure at line %d :%s \n",__LINE__,cudaGetErrorString(err));  
	kernel2<<<blocks,nThreadsPerBlock,1,s2>>> ();	
	err = cudaGetLastError();               
        if (err != cudaSuccess )  printf("cuda function failure at line %d :%s \n",__LINE__,cudaGetErrorString(err));  
	
	cudaDeviceSynchronize();
	return 0 ;
}

