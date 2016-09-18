#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <limits.h>

#include <cuda.h>


#include <curand.h>

#define CUDA_CALL(x) \
	if ( cudaSuccess != (x) ) { \
	fprintf(stderr,"cuda call failed at line :%d \n",__LINE__); \
	exit(1); }

#define CURAND_CALL(x) \
	if ((x) != CURAND_STATUS_SUCCESS) { \
		printf("%s %s failed\n",__FILE__,__LINE__); \
		exit(1); }

__global__ void inCircle(float * x ,float * y, int * pnts)
{
	int idx = (gridDim.y * blockIdx.x + blockIdx.y )*blockDim.x*blockDim.y*blockDim.z + blockDim.z*blockDim.y*threadIdx.x + blockDim.z*threadIdx.y + threadIdx.z;

	if (x[idx]*x[idx] + y[idx]*y[idx] <= 1)
		pnts[idx] = 1; 
}

int getMaxThreadsPerBlock(){
	struct cudaDeviceProp prop;
	int devCnt = 0 ;
	CUDA_CALL( cudaGetDeviceCount(&devCnt) );
	if (devCnt == 0) {
		fprintf(stderr,"cannot find cuda device ,exting\n");
		exit(10);
	}
	CUDA_CALL (cudaGetDeviceProperties(&prop,0));
	return prop.maxThreadsPerBlock;
}

void usage(int argc,char ** argv) 
{
	fprintf(stderr,"%s usage:\n",argv[0]);
	fprintf(stderr,"	:%s num\n",argv[0]);
	exit(1);
}
int main(int argc,char **argv)
{
	int threads = getMaxThreadsPerBlock();
	unsigned long long i = 0;
	if(argc != 2) {
		usage(argc,argv);
	}
	else 
		i = atol(argv[1]);
	
	int iter = i / threads;
	int left = i % threads;
	int j = 0;
        curandGenerator_t gen;
	unsigned long long cnt = 0;
	float * aixX_d;
	float * aixY_d;
        dim3 block(threads);
        int * buf_inCircle_h,*buf_inCircle_d ;
        buf_inCircle_h = (int *) malloc(sizeof(int) * threads);
	bzero(buf_inCircle_h,sizeof(int) * threads);

	CUDA_CALL (cudaMalloc((void **) &aixX_d,sizeof(float) * threads));
        CUDA_CALL (cudaMalloc((void **) &aixY_d,sizeof(float) * threads));
	CUDA_CALL (cudaMalloc((void **) &buf_inCircle_d,sizeof(int) * threads));

	for (;j<iter;j++) {
	
        CURAND_CALL (curandSetPseudoRandomGeneratorSeed(gen,12321321ULL));
        CURAND_CALL (curandGenerateUniform(gen, aixX_d, threads));
       

        CURAND_CALL (curandSetPseudoRandomGeneratorSeed(gen,21321321ULL));
        CURAND_CALL (curandGenerateUniform(gen, aixY_d, threads));

	CUDA_CALL (cudaMemset(buf_inCircle_d,0,sizeof(int) * threads));
	
        inCircle <<<1,block>>> (aixX_d,aixY_d,buf_inCircle_d);	
	
	CUDA_CALL (cudaMemcpy(buf_inCircle_h,buf_inCircle_d,sizeof(int) * threads,cudaMemcpyDeviceToHost));

 	int z = 0;
	for (z=0;z<threads;z++) 
		cnt += 	buf_inCircle_h[z];
	}


	
        CURAND_CALL (curandSetPseudoRandomGeneratorSeed(gen,12321321ULL));
        CURAND_CALL (curandGenerateUniform(gen, aixX_d, left));
       

        CURAND_CALL (curandSetPseudoRandomGeneratorSeed(gen,21321321ULL));
        CURAND_CALL (curandGenerateUniform(gen, aixY_d, left));

	CUDA_CALL (cudaMemset(buf_inCircle_d,0,sizeof(int) * threads));
        inCircle<<<1,left>>> (aixX_d,aixY_d,buf_inCircle_d);	
	CUDA_CALL (cudaMemcpy(buf_inCircle_h,buf_inCircle_d,sizeof(int) * threads,cudaMemcpyDeviceToHost));
	int z = 0;
	
	for (z=0;z<left;z++) 
		cnt += 	buf_inCircle_h[z];

	printf("pi is roughly about %1.10f in total cnt :%d\n", 4 * float(cnt) / i,i);

	return 0;
		
}
