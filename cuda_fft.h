#include <cstdlib>
#include <cufft.h>
#include <cuda_runtime.h>
#include "fft.cuh"
#define PI 3.141592653589793238f

extern "C" void check();

namespace
{
	const int N = 256;
}

cufftHandle plan;
cufftComplex *data;
cufftComplex term[N * N];

void cuda_init() {
	if (cudaMalloc((void**)&data, sizeof(cufftComplex) * N * N) != cudaSuccess) {
		fprintf(stderr, cudaGetErrorString(cudaGetLastError()));
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
}

void calc (cufftComplex * t) {

	if (cufftExecC2C(plan, t, data, CUFFT_FORWARD) != CUFFT_SUCCESS) {
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
	
}

void do_fft(float *result) {
	
	calc(h_k);

	for (int i = 0; i < N * N; ++i) {
		result[3 * i] = term[i].x;
	}
	
	calc(h_k_normalx);

	for (int i = 0; i < N * N; ++i) {
		result[3 * i + 1] = term[i].x;
	}

	calc(h_k_normaly);

	for (int i = 0; i < N * N; ++i) {
		result[3 * i + 2] = term[i].x;
	}
	return;
}