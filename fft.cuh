#include <cuda_runtime.h>
#include <cufft.h>
#include <cuda.h>
#include <curand.h>
//#include <curand_kernel.h>

#define MAX_WAVE_RESOLUTION 256
#define BLOCK_SIZE 64
#define PI 3.141592653589793238f

extern "C" cufftComplex *h0;	
extern "C" cufftComplex *h_k;
extern "C" cufftComplex *h_k_normalx;
extern "C" cufftComplex *h_k_normaly;
extern "C" cufftComplex *h0_minus;

extern "C" void cuda_init_h();
extern "C" void generation_h0(float lx, float lz, float windx, float windy, float anorm);
extern "C" void generation_h(float t);
extern "C" void deshift(cufftComplex *a, cufftComplex *b);