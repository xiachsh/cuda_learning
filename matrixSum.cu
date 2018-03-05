#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <unistd.h>
#include <cuda.h>
#include <string.h>


#define X 1024
#define Y 1024


#define BX 4
#define BY 4



#define USEC_ELAPSED(start,end) ((end.tv_sec - start.tv_sec) * 1000 * 1000 + (end.tv_usec - start.tv_usec))

#define CHECK_RET(x)  					\ 
	ret = x;					\
	if (ret != CUDA_SUCCESS) { 			\
	cudaError_t err = cudaGetLastError(); 		\
	printf("cuda function failure at line %d :%s \n",__LINE__,cudaGetErrorString(err));	\
	exit(1); }

void print_matrix(float *buff,int x ,int y)
{

	printf("\n\n");
	int i = 0;
	for (i=0 ;i<x ;i++)
	{
		int j = 0;
		for (j=0 ;j<y ;j++)
		printf("%20.3f\t",buff[i*y + j]);
		printf("\n");
	}

}


void check_result(float * buf1, float * buf2,int x ,int y)
{
	int i = 0;
	for (i = 0 ;i<x ;i++) {
		int j = 0;
		for ( j = 0;j < y;j++) 
		if (abs(buf1[i*y+j] - buf2[i*y+j]) > 0.00000001f) {
			printf("idx :%d Host:%10.5f Device:%10.5f\n",i*y+j,buf1[i*y+j], buf2[i*y+j]);
			exit(1);
		}
	}
}

__global__ void matrixSum(float *matrixA,float *matrixB,float *matrixC)
{
	int idx = (gridDim.y * blockIdx.x + blockIdx.y )*blockDim.x*blockDim.y*blockDim.z + blockDim.z*blockDim.y*threadIdx.x + blockDim.z*threadIdx.y + threadIdx.z;	

//		printf("idx :%d grid(%d,%d,%d) block(%d,%d,%d)\n",idx,blockIdx.x,blockIdx.y,blockIdx.z,threadIdx.x,threadIdx.y,threadIdx.z);
	matrixC[idx] = matrixA[idx] + matrixB[idx];
}


void usage(int argc,char ** argv)
{
	printf("%s usage:\n",argv[0]);
	printf("	: -x dimX\n");
	printf("	: -y dimy\n");
	printf("	: -i blockX\n");
	printf("	: -j blockY\n");
	printf("exiting\n");
	exit(1);
}



int main(int argc,char **argv)
{
	int ret;
	int nx = X ;
	int ny = Y ;		
	int blockDimX = BX;
	int blockDimY = BY;
	float * matrixA,* matrixB, *matrixC, *matrixC_d;
	struct timeval start,end;

	int c ;

	while  ( (c = getopt(argc,argv,"x:y:i:j:")) != -1 ) {
	
		switch (c) 
 		{
			case	'x':
				nx = atoi(optarg);
				break;
			case 'y':
				ny = atoi(optarg);
				break;
			case 'i':
				blockDimX = atoi(optarg);
				break;
			case 'j':
				blockDimY = atoi(optarg);
				break;
			default :
				usage(argc,argv);
		}
	}
	
	int nElems = nx * ny;
	int nBytes = nElems * sizeof(float);
		
	matrixA = (float *) malloc(nBytes);
	matrixB = (float *) malloc(nBytes);
	matrixC = (float *) malloc(nBytes);
	matrixC_d = (float *) malloc(nBytes);

	float * d_matrixA,* d_matrixB, * d_matrixC;
	CHECK_RET (cudaMalloc(&d_matrixA,nBytes));
	CHECK_RET (cudaMalloc(&d_matrixB,nBytes));
	CHECK_RET (cudaMalloc(&d_matrixC,nBytes));
	
	int i = 0 ;
	for (; i < nElems; i++) 
		matrixA[i] = matrixB[i] = 0.01f * i;

	
	gettimeofday(&start,NULL);
	for (i=0; i < nElems; i++) 
		matrixC[i] = matrixA[i] +  matrixB[i] ;
	gettimeofday(&end,NULL);
	printf("%ld  usec elapsed for caculation on CPU\n",USEC_ELAPSED(start,end));

	dim3 b(blockDimX,blockDimY);
	dim3 g((nx + blockDimX - 1)/blockDimX, (ny + blockDimY - 1)/blockDimY );

	CHECK_RET(cudaMemcpy(d_matrixA,matrixA,nBytes,cudaMemcpyHostToDevice));
	CHECK_RET(cudaMemcpy(d_matrixB,matrixB,nBytes,cudaMemcpyHostToDevice));

	gettimeofday(&start,NULL);
	cudaError_t err = cudaGetLastError(); 		\
	printf("cuda function failure at line %d :%s \n",__LINE__,cudaGetErrorString(err));	\
	matrixSum<<<b,g>>>(d_matrixA,d_matrixB,d_matrixC);
	err = cudaGetLastError(); 		\
	printf("cuda function failure at line %d :%s \n",__LINE__,cudaGetErrorString(err));	\
	gettimeofday(&end,NULL);
	printf("%ld  usec elapsed for caculation on GPU\n",USEC_ELAPSED(start,end));
	

	CHECK_RET(cudaMemcpy(matrixC_d,d_matrixC,nBytes,cudaMemcpyDeviceToHost));

	check_result(matrixC,matrixC_d,nx,ny);
	cudaFree(d_matrixA);
	cudaFree(d_matrixB);
	cudaFree(d_matrixC);

#ifdef __DEBUG
	print_matrix(matrixA,nx,ny);
	print_matrix(matrixB,nx,ny);
	print_matrix(matrixC,nx,ny);
	print_matrix(matrixC_d,nx,ny);
#endif

	
	free(matrixA);
	free(matrixB);
	free(matrixC);	
	free(matrixC_d);	




	return 0;
}
