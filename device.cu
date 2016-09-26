#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>



int main()
{
	int count = 0;
	cudaGetDeviceCount(&count);	
	if (0 == count) {
		fprintf(stderr,"found no GPU device\n");
		exit (1);
	}
	
	fprintf(stdout,"found %d GPU on host\n",count);
	int i = 0;
	
	for (i=0;i<count;i++)
        {
		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp,i);
		// major version 5: Maxwell
		// major versopm 3: kepler
	        // major version 2: Fermi
                // major version 1: Tesla
		fprintf(stdout,"Device :%d has compute capability %d:%d\n",i,deviceProp.major,deviceProp.minor);
	}
	return 0;
}
