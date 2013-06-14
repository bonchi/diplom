#include <cufft.h>
#include <cuda_runtime.h>
#include <assert.h>
#include <math.h>
#include "computing_gpu.cuh"
#include <cuda_runtime_api.h>

#define PI 3.141592653589793238f

cufftHandle plan;
cufftComplex * res;
float * density_mip;

void fft_init() {
	if (cufftPlan2d(&plan, MAX_WAVE_RESOLUTION, MAX_WAVE_RESOLUTION, CUFFT_C2C) != CUFFT_SUCCESS) {
		fprintf(stderr, "CUFFT Error: Unable to create plan\n");
		return;	
	}

	if (cufftSetCompatibilityMode(plan, CUFFT_COMPATIBILITY_NATIVE)!= CUFFT_SUCCESS) {
		fprintf(stderr, "CUFFT Error: Unable to set compatibility mode to native\n");
		return;		
	}

	if (cudaMalloc((void**)&res, MAX_WAVE_RESOLUTION * MAX_WAVE_RESOLUTION * sizeof (cufftComplex)) != CUFFT_SUCCESS) {
		fprintf(stderr, "CUFFT Error: Failed to allocated\n");
		return;
	}

	if (cudaMalloc((void**)&density_mip, MAX_WAVE_RESOLUTION * MAX_WAVE_RESOLUTION * sizeof (float)) != CUFFT_SUCCESS) {
		fprintf(stderr, "CUFFT Error: Failed to allocated\n");
		return;
	}

}

void calc (cufftComplex * t, cudaArray * ca) {

	if (cufftExecC2C(plan, t, res, CUFFT_FORWARD) != CUFFT_SUCCESS) {
		fprintf(stderr, "CUFFT Error: Unable to execute plan\n");
		return;		
	}

	if (cudaThreadSynchronize() != cudaSuccess){
  		fprintf(stderr, "Cuda error: Failed to synchronize\n");
   		return;
	}	

	if( cudaMemcpyToArray(ca, 0, 0, res, MAX_WAVE_RESOLUTION * MAX_WAVE_RESOLUTION * sizeof (cufftComplex), cudaMemcpyDeviceToDevice)  != cudaSuccess) {
		fprintf(stderr, "Cuda error: Failed to copy cudaArray\n");
   		return;
	}

}

void do_fft(cudaArray * hf, cudaArray * hf_nx, cudaArray * hf_ny, cudaGraphicsResource * resource) {
	
	calc(h_k, hf);
	
	cudaArray * mip_level;
	density_mip0(density_mip, res);
	cudaError_t l;
	l = cudaGraphicsSubResourceGetMappedArray (&mip_level,  resource, 0, 0);
	l = cudaMemcpyToArray(mip_level, 0, 0, density_mip, MAX_WAVE_RESOLUTION * MAX_WAVE_RESOLUTION * sizeof (float), cudaMemcpyDeviceToDevice);

	for (int i = 1, size = MAX_WAVE_RESOLUTION >> 1; size >= 1; ++i, size = size >> 1) {
		density_mip_next(density_mip, size);
		l = cudaGraphicsSubResourceGetMappedArray (&mip_level,  resource, 0, i);
		l = cudaMemcpyToArray(mip_level, 0, 0, density_mip, size * size * sizeof (float), cudaMemcpyDeviceToDevice);
	}

	calc(h_k_normalx, hf_nx);

	calc(h_k_normaly, hf_ny);

	return;
}
/*
	for (int i = 0; i < N; ++i) {
		for (int j = 0;j < N; ++j) {
			int shifti = fftshift(i);
			int shiftj = fftshift(j);
			term[i * N + j].x = h[shifti * 2 * N + 2 * shiftj];
			term[i * N + j].y = h[shifti * 2 * N + 2 * shiftj + 1];
		}
	}

	calc();
	
		float min_ = term[0].x;
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
	}

	for (int i = 0; i < N * N; ++i) {
		result[3 * i] = term[i].x;
	}

	for (int i = 0; i < N - 1; ++i) {
		for (int j = 0; j < N - 1; ++j) {
			density[i * 2 * (N - 1) + j] = koef_density * max_diff(i, j);
		}
	}
	for (int i = 2; i <= N; i *= 2) {
		if (i == 2) {
			for (int t= 0; t < int (N * 0.5); ++t) {
				for (int j = 0; j < int(N * 0.5); ++j) {
					float max_ = density[(t * 2) * 2 * (N - 1) + (j * 2)];
					if (t != int(N * 0.5) - 1) {
						max_ = max(max_, density[(t * 2 + 1) * 2 * (N - 1) + (j * 2)]);
					}
					if (j != int(N * 0.5) - 1) {
						max_ = max(max_, density[(t * 2) * 2 * (N - 1) + (j * 2 + 1)]);
					}
					if ((t != int(N * 0.5) - 1) && (j != int(N * 0.5) - 1)) {
						max_ = max(max_, density[(t * 2 + 1) * 2 * (N - 1) + (j * 2 + 1)]);
					}
					density[(t + N - 1) * 2 * (N - 1) + (j + N - 1)] = max_;
				}
			}
		} else {
			int shift = get_shift(i);
			int prev_shift = shift - int(N / (i / 2));
			for (int t= 0; t < int (N / i); ++t) {
				for (int j = 0; j < int(N / i); ++j) {
					float max_ = density[(prev_shift + 2 * t) * 2 * (N - 1) + prev_shift + 2 * j];
					max_ = max(max_, density[(prev_shift + 2 * t) * 2 * (N - 1) + prev_shift + 2 * j + 1]);
					max_ = max(max_, density[(prev_shift + 2 * t + 1) * 2 * (N - 1) + prev_shift + 2 * j]);
					max_ = max(max_, density[(prev_shift + 2 * t + 1) * 2 * (N - 1) + prev_shift + 2 * j + 1]);
					density[(t + shift) * 2 * (N - 1) + (j + shift)] = max_;
				}
			}
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
	return;
}*/