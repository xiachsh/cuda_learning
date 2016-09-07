#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <limits.h>
#include <curand.h>



#define CUDA_CALL(x) do { if((x)!=cudaSuccess) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)
#define CURAND_CALL(x) do { if((x)!=CURAND_STATUS_SUCCESS) { \
    printf("Error at %s:%d\n",__FILE__,__LINE__);\
    return EXIT_FAILURE;}} while(0)

__global__ void largerThanOne(float *x,float *y,unsigned int *pnts)
{
	int idx = threadIdx.x;
	if (x[idx] * x[idx] + y[idx] * y[idx] <= 1) {
		pnts[idx] = 1;
	}
}


int main(void)
{

	float * pntsX;
	float * pntsY;

	unsigned int *pnts_h = 0;
	unsigned int *pnts = 0;
	unsigned int totalPnts = 0;

	curandGenerator_t gen;
	
	int elems = 1024;
	int iteration = 1;
	int nBytes = elems * sizeof(float);	

	cudaMalloc((void **) (&pntsX),nBytes);
	cudaMalloc((void **) (&pntsY),nBytes);

	CURAND_CALL(curandCreateGenerator(&gen,CURAND_RNG_PSEUDO_DEFAULT));
	CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen,1234ULL));
	CURAND_CALL(curandGenerateUniform(gen,pntsX,elems));

	
	CURAND_CALL(curandSetPseudoRandomGeneratorSeed(gen,2345ULL));
	CURAND_CALL(curandGenerateUniform(gen,pntsY,elems));
	

	cudaMalloc((void **)&pnts,sizeof(unsigned int)*elems);
	cudaMemset(pnts,0,sizeof(unsigned int)*elems);
	
	largerThanOne<<<1,elems>>>(pntsX,pntsY,pnts);
        pnts_h = (unsigned int *) malloc(sizeof(unsigned int)*elems);	
	cudaMemcpy(pnts_h,pnts,sizeof(unsigned int)*elems,cudaMemcpyDeviceToHost);
	int i = 0;
	int _pnts = 0;
	for (i=0;i<elems;i++)
		_pnts += pnts_h[i];
	printf("pi is roughly about %f\n",(float)_pnts * 4 / elems);


	 CURAND_CALL(curandDestroyGenerator(gen));
	cudaFree(pnts);
	cudaFree(pntsX);
	cudaFree(pntsY);

	free(pnts_h);

	return 0;
}
