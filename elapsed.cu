#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <cuda.h>

#define N  1024 * 1024



#define CHECK_CUDA_ERR(x)   {			\
	err = x;				\
	if (err != cudaSuccess) { 		\
		printf("cuda error with %s in line %d\n",cudaGetErrorString(err),__LINE__);	\
		exit(1);									\
	} }

int main()
{
	
	cudaError_t err;
	cudaEvent_t start_event,stop_event;

	CHECK_CUDA_ERR ( cudaEventCreate(&start_event) );
	CHECK_CUDA_ERR ( cudaEventCreate(&stop_event)  );




	void * buf = malloc(sizeof(float) * N);

	void * dBuf;
	 CHECK_CUDA_ERR ( cudaMalloc(&dBuf,sizeof(float) * N) );

	 CHECK_CUDA_ERR ( cudaEventRecord(start_event,0) ) ;
	 CHECK_CUDA_ERR ( cudaMemcpy(dBuf,buf,sizeof(float) * N, cudaMemcpyHostToDevice));
	 CHECK_CUDA_ERR ( cudaEventRecord(stop_event,0) );
	 CHECK_CUDA_ERR ( cudaDeviceSynchronize() );
	float ms = 100.f;

	 CHECK_CUDA_ERR ( cudaEventElapsedTime(&ms,start_event,stop_event));


	printf("%f m second cost\n",ms);


	CHECK_CUDA_ERR ( cudaEventDestroy(start_event) );
	CHECK_CUDA_ERR ( cudaEventDestroy(stop_event)  );
	free(buf);
	CHECK_CUDA_ERR ( cudaFree(dBuf) );


	return 0;





}
