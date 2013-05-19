#include <cufft.h>
#include <cuda_runtime.h>

namespace
{
	const int N = 64;
}

void do_fft(float const *h, float *result) {

	cufftHandle plan;
	cufftComplex *data;
	cufftComplex term[N * N];
	for (int i = 0; i < N * N; ++i) {
		term[i].x = h[2 * i];
		term[i].y = h[2 * i + 1];
	}
	cudaMalloc((void**)&data, sizeof(cufftComplex) * N * N);

	if (cudaGetLastError() != cudaSuccess){
		fprintf(stderr, "Cuda error: Failed to allocate\n");
		return;	
	}

	if (cudaMemcpy (data, term, N * N * sizeof(cufftComplex), cudaMemcpyHostToDevice) != cudaSuccess) {
		fprintf(stderr, "Cuda error: Failed to copy data to device\n");
		return;	
	}

	if (cufftPlan2d(&plan, N, N, CUFFT_C2C) != CUFFT_SUCCESS) {
		fprintf(stderr, "CUFFT Error: Unable to create plan\n");
		return;	
	}

	if (cufftSetCompatibilityMode(plan, CUFFT_COMPATIBILITY_NATIVE)!= CUFFT_SUCCESS) {
		fprintf(stderr, "CUFFT Error: Unable to set compatibility mode to native\n");
		return;		
	}

	if (cufftExecC2C(plan, data, data, CUFFT_FORWARD) != CUFFT_SUCCESS) {
		fprintf(stderr, "CUFFT Error: Unable to execute plan\n");
		return;		
	}

	if (cudaThreadSynchronize() != cudaSuccess){
  		fprintf(stderr, "Cuda error: Failed to synchronize\n");
   		return;
	}	

	if (cudaMemcpy (term, data, N * N * sizeof(cufftComplex), cudaMemcpyDeviceToHost) != cudaSuccess) {
		fprintf(stderr, "Cuda error: Failed to copy data to host\n");
		return;	
	}

	//при выводе формулы еще один ч дает * -1
	for (int i = 0; i < N * N; ++i) {
		result[i] = -term[i].x;
	}

	cufftDestroy(plan);
	cudaFree(data);
	return;
}