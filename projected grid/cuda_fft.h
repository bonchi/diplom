#include <cufft.h>
#include <cuda_runtime.h>
#define PI 3.141592653589793238f

namespace
{
	const int N = 128;
}

void do_fft(float const *h, float *result) {

	cufftHandle plan;
	cufftComplex *data;
	cufftComplex term[N * N];
	cudaMalloc((void**)&data, sizeof(cufftComplex) * N * N);

	if (cudaGetLastError() != cudaSuccess){
		fprintf(stderr, "Cuda error: Failed to allocate\n");
		return;	
	}

	if (cudaMemcpy (data, h, N * N * 2 * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess) {
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

	for (int i = 0; i < N * N; ++i) {
		result[i] = term[i].x;
	}

	cufftDestroy(plan);
	cudaFree(data);
	/*float h2[N * N * 2];
	for (int t1 = 0; t1 < N; ++t1) {
		for (int t2 = 0; t2 < N; ++t2) {
			float res = 0;
			for (int i = 0; i < N; ++i) {
				float res_r = 0;
				float res_im = 0;
				for (int j = 0; j < N; ++j) {
					float l = -2 * PI * j * t1 / N;
					res_r += h[i * N * 2 + j * 2] * cos (l) - h[2 * N * i + j * 2 + 1] * sin (l);
					res_im += h[i * N * 2 + j * 2 + 1] * cos (l) + h[2 * N * i + j * 2] * sin (l);
				}
				float l = -2 * PI * i * t2 / N;
				res += res_r* cos (l) - res_im * sin (l);
			}
			result[t1 * N + t2] = res;
		}
	}*/
	return;
}