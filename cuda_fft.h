#include <cufft.h>
#include <cuda_runtime.h>
#include <assert.h>

#define PI 3.141592653589793238f
namespace
{
	const int N = 128;
}

cufftHandle plan;
cufftComplex *data;
cufftComplex term[N * N];

void fft_init() {
	cudaMalloc((void**)&data, sizeof(cufftComplex) * N * N);
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess){
		fprintf(stderr, cudaGetErrorString(err));
		fprintf(stderr, "Cuda error: Failed to allocate \n");
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

int fftshift(int idx) {
	if (idx >= N / 2)
		return idx - N / 2;
	else 
		return idx + N / 2;
}

void calc() {
	if (cudaMemcpy(data, term, N * N * sizeof(cufftComplex), cudaMemcpyHostToDevice) != cudaSuccess) {
		fprintf(stderr, "Cuda error: Failed to copy data to device\n");
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

	if (cudaMemcpy(term, data, N * N * sizeof(cufftComplex), cudaMemcpyDeviceToHost) != cudaSuccess) {
		fprintf(stderr, "Cuda error: Failed to copy data to host\n");
		return;	
	}
	
}

float max(float a, float b) {
	if (a < b) return b;
	return a;
}

float min(float a, float b) {
	if (a < b) return a;
	return b;
}

float max_diff(int i, int j) {
	float max_ = term[N * i + j].x;
	max_ = max(max_, term[N * i + j + 1].x); 
	max_ = max(max_, term[N * (i + 1) + j + 1].x); 
	max_ = max(max_, term[N * (i + 1) + j].x);
	float min_ = term[N * i + j].x;
	min_ = min(min_, term[N * i + j + 1].x); 
	min_ = min(min_, term[N * (i + 1) + j + 1].x); 
	min_ = min(min_, term[N * (i + 1) + j].x);
	return abs(max_ - min_);
}

void do_fft(float const *h, float *result, float *density, float lx, float lz, float koef_density) {

	for (int i = 0; i < N; ++i) {
		for (int j = 0;j < N; ++j) {
			int shifti = fftshift(i);
			int shiftj = fftshift(j);
			term[i * N + j].x = h[shifti * 2 * N + 2 * shiftj];
			term[i * N + j].y = h[shifti * 2 * N + 2 * shiftj + 1];
		}
	}

	calc();
	
	/*float min_ = term[0].x;
	float max_ = term[0].x;
	for (int i = 0; i < N * N; ++i) {
		result[3 * i] = term[i].x;
		if (min_ > term[i].x) {
			min_ = term[i].x;
		}
		if (max_ < term[i].x) {
			max_ = term[i].x;
		}
	}
	float H = max_ - min_;
	for (int i = 0; i < N - 1; ++i) {
		for (int j = 0; j < N - 1; ++j) {
			density[i * (N - 1) + j] = koef_density * max_diff(i, j) / H;
		}
	}*/

	for (int i = 0; i < N * N; ++i) {
		result[3 * i] = term[i].x;
	}

	for (int i = 0; i < N - 1; ++i) {
		for (int j = 0; j < N - 1; ++j) {
			density[i * (N - 1) + j] = koef_density * max_diff(i, j);
		}
	}
	
	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			int shifti = fftshift(i);
			int shiftj = fftshift(j);
			float kx = 2 * PI * (shifti - N / 2) / lx;
			term[i * N + j].x = -kx * h[shifti * 2 * N + 2 * shiftj + 1];
			term[i * N + j].y = kx * h[shifti * 2 * N + 2 * shiftj];
		}
	}

	calc();

	for (int i = 0; i < N * N; ++i) {
		result[3 * i + 1] = term[i].x;
	}

	for (int i = 0; i < N; ++i) {
		for (int j = 0; j < N; ++j) {
			int shifti = fftshift(i);
			int shiftj = fftshift(j);
			float kz = 2 * PI * (shiftj - N / 2) / lz;
			term[i * N + j].x = -kz * h[shifti * 2 * N + 2 * shiftj + 1];
			term[i * N + j].y = kz * h[shifti * 2 * N + 2 * shiftj];
		}
	}

	calc();

	for (int i = 0; i < N * N; ++i) {
		result[3 * i + 2] = term[i].x;
	}
	/*float h2[N * N * 2];
	for (int i = 0; i < 2 * N * N; ++i) {
		h2[i] = 0;
	}
	//h2[2*N] = 1;
	//h2[2*N*(N-1)]=1;
	h2[2 * (N / 2) * N]=1;
	for (int t1 = 0; t1 < N; ++t1) {
		for (int t2 = 0; t2 < N; ++t2) {
			float res = 0;
			for (int i = 0; i < N; ++i) {
				float res_r = 0;
				float res_im = 0;
				for (int j = 0; j < N; ++j) {
					float l = -2 * PI * j * t1 / N;
					res_r += h2[j * N * 2 + i * 2] * cos (l) - h2[2 * N * j + i * 2 + 1] * sin (l);
					res_im += h2[j * N * 2 + j * 2 + 1] * cos (l) + h2[2 * N * j + i * 2] * sin (l);
				}
				float l = -2 * PI * i * t2 / N;
				res += res_r* cos (l) - res_im * sin (l);
			}
			result[t1 * N + t2] = res;
		}
	}*/
	return;
}