#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>


#define GRID 1
#define THRDS 256

__global__ void assign(int * buf)
{
	int idx = threadIdx.x;
	buf[idx] = idx;
}

void print_buf(int * buf,int size)
{
	int i = 0;
	for (i=0;i<size;i++)
		printf("%d\t",buf[i]);
	printf("\n");
}

int main()
{
	int * buf_h;
	int * buf_d0;
	int * buf_d1;

	int size = sizeof(int) * GRID * THRDS;
	buf_h = (int*)malloc(size);
	cudaSetDevice(0);
	cudaMalloc((void **) &buf_d0,size);
	assign<<<GRID,THRDS>>> (buf_d0);
	



	cudaSetDevice(1);
	cudaMalloc((void **) &buf_d1,size);
	cudaMemcpyPeer(buf_d1,1,buf_d0,0,size);
	
	cudaMemcpy(buf_h,buf_d1,size,cudaMemcpyDeviceToHost);
	print_buf(buf_h,GRID * THRDS);
	cudaFree(buf_d1);	
	cudaFree(buf_d0);	
	free(buf_h);

	return 0;

}
