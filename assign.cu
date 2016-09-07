#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <limits.h>




__global__ void assign(float * x)
{
	int idx = (gridDim.y * blockIdx.x + blockIdx.y )*blockDim.x*blockDim.y*blockDim.z + blockDim.z*blockDim.y*threadIdx.x + blockDim.z*threadIdx.y + threadIdx.z;
	x[idx] = idx;	
	printf("%d\n",idx);
}





int main(int argc,char ** argv)
{

	float * x_h;
	float * x_d;
	int x;
	int y;
	int z;
	int gridX;
	int gridY;
	int nBytes;
	if (argc != 6) {
		printf("%s usage:\n",argv[0]);
		printf("	%s gridX gridY blockX blockY blockZ\n",argv[0]);
		exit(1);
	}
	else {
		gridX = atoi(argv[1]);
		gridY = atoi(argv[2]);
		x = atoi(argv[3]);	
		y = atoi(argv[4]);
		z = atoi(argv[5]);
	}

	nBytes = sizeof(float) * x*y*z*gridX*gridY;
	x_h = (float*) malloc(nBytes);
	cudaMalloc((void **) &x_d, nBytes);
	assign<<<dim3(gridX,gridY),dim3(x,y,z)>>>(x_d); 	
	cudaMemcpy(x_h,x_d,nBytes,cudaMemcpyDeviceToHost);

	int i = 0;
	for (i=0;i<x*y*z*gridX*gridY;i++)
	{
		printf("%f\n",x_h[i]);
	}
	cudaFree(x_d);
	free(x_h);
	return 0;
}


