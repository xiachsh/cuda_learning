#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>

#include <curand.h>
#include <limits.h>

#define DIM_X 10
#define DIM_Y 10
#define DIM_Z 10


void usage(int argc,char ** argv)
{
	printf("%s usage:\n",argv[0]);
	printf("	:%s num\n",argv[0]);
	exit(1);
}

__global__ void inCircle(float * x ,float * y, int * pnts)
{
	int idx = (gridDim.y * blockIdx.x + blockIdx.y )*blockDim.x*blockDim.y*blockDim.z + blockDim.z*blockDim.y*threadIdx.x + blockDim.z*threadIdx.y + threadIdx.z;

	if (x[idx]*x[idx] + y[idx]*y[idx] <= 1)
		pnts[idx] = 1; 
}


#define CUDA_CALL(x) \
	if ((x)!= cudaSuccess)  { \
		printf("%s %s failed\n",__FILE__,__LINE__); \
		exit(1); }
#define CURAND_CALL(x) \
	if ((x) != CURAND_STATUS_SUCCESS) { \
		printf("%s %s failed\n",__FILE__,__LINE__); \
		exit(1); }
int main(int argc,char ** argv)
{


	int num = 0;
	if (argc != 2) 
		usage(argc,argv);
	else 
		num = atoi(argv[1]);

	
	float * aixX_d;
	float * aixY_d;

	curandGenerator_t gen;
	long long numElems = num * DIM_X * DIM_Y * DIM_Z;
	int * pntsInCir_d,* pntsInCir_h;
	CUDA_CALL (cudaMalloc((void **) &aixX_d,sizeof(float) * numElems));
	CUDA_CALL (cudaMalloc((void **) &aixY_d,sizeof(float) * numElems));
	
	CURAND_CALL (curandCreateGenerator(&gen,CURAND_RNG_PSEUDO_DEFAULT));
	CURAND_CALL (curandSetPseudoRandomGeneratorSeed(gen,12321321ULL));
	CURAND_CALL (curandGenerateUniform(gen, aixX_d, numElems));

	CURAND_CALL (curandSetPseudoRandomGeneratorSeed(gen,21321321ULL));
	CURAND_CALL (curandGenerateUniform(gen, aixY_d, numElems));


	CUDA_CALL  (cudaMalloc((void **) &pntsInCir_d,sizeof(int) * numElems));
	CUDA_CALL  (cudaMemset(pntsInCir_d,0,sizeof(int) * numElems));
	pntsInCir_h = (int *) malloc(sizeof(int) * numElems);
	
	dim3 grid(num);
	dim3 block(DIM_X,DIM_Y,DIM_Z);
	inCircle<<<grid,block>>> (aixX_d,aixY_d,pntsInCir_d);
	CUDA_CALL  (cudaMemcpy(pntsInCir_h,pntsInCir_d,sizeof(int) * numElems,cudaMemcpyDeviceToHost));

	int i = 0;
 	long long totalCnt = 0;
	for (i=0;i<numElems;i++) {
		if (pntsInCir_h[i]) 
			totalCnt++;
	}	

	float pi = (float) totalCnt * 4 / numElems;

	printf("pi is roughly about %f\n",pi);
	free(pntsInCir_h);
	CUDA_CALL  (cudaFree(pntsInCir_d));
	CUDA_CALL  (cudaFree(aixX_d));
	CUDA_CALL  (cudaFree(aixY_d));
	return 0;
}
