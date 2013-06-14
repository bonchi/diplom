#include "computing_gpu.cuh"

cufftComplex *h0;	
cufftComplex *h_k;
cufftComplex *h_k_normalx;
cufftComplex *h_k_normaly;
cufftComplex *h0_minus;
curandGenerator_t gen;
float *randData;
float *term;
__constant__ cufftComplex _wind;
__constant__ float _Anorm;
__constant__ float _g;
__constant__ float _lx;
__constant__ float _lz;


extern "C" void cuda_init_h() {
	cudaMalloc((void**)&h0, MAX_WAVE_RESOLUTION * MAX_WAVE_RESOLUTION * sizeof (cufftComplex));
	cudaMalloc((void**)&h_k, MAX_WAVE_RESOLUTION * MAX_WAVE_RESOLUTION * sizeof (cufftComplex));
	cudaMalloc((void**)&h_k_normalx, MAX_WAVE_RESOLUTION * MAX_WAVE_RESOLUTION * sizeof (cufftComplex));
	cudaMalloc((void**)&h_k_normaly, MAX_WAVE_RESOLUTION * MAX_WAVE_RESOLUTION * sizeof (cufftComplex));
	cudaMalloc((void**)&h0_minus, MAX_WAVE_RESOLUTION * MAX_WAVE_RESOLUTION * sizeof (cufftComplex));
	float g = 9.81f;
	cudaMemcpyToSymbol(_g, &g, sizeof (float));
	cudaMalloc((void**)&randData, 2 * MAX_WAVE_RESOLUTION * MAX_WAVE_RESOLUTION * sizeof(float));
	cudaMalloc((void**)&term, MAX_WAVE_RESOLUTION * MAX_WAVE_RESOLUTION * sizeof(float));
	curandCreateGenerator(&gen,CURAND_RNG_PSEUDO_DEFAULT);
	curandSetPseudoRandomGeneratorSeed(gen, 1234ULL);
}

__device__ cufftComplex get_h0(cufftComplex a, int i, int j, float *randData) {
	cufftComplex res;
	float norm1 = randData[2 * (i * MAX_WAVE_RESOLUTION + j)];
	float norm2 = randData[2 * (i * MAX_WAVE_RESOLUTION + j) + 1];
	float term;
	float t = sqrt(_wind.x * _wind.x + _wind.y * _wind.y);
	if (((a.x == 0) && (a.y == 0)) || (t == 0)) {
		term = 0;
	} else {
		float l = t * t / _g;
		float kl = sqrt(a.x * a.x + a.y * a.y);
		float n1 = t * kl;
		term = a.x * _wind.x / n1 + a.y * _wind.y / n1;
		if (term < 0) {	
			term = 0;
		} else {
			float t1 = exp(-1 / (kl * kl * l * l));
			term = _Anorm * t1 * term * term/ (kl * kl * kl * kl);
		}
	}
	term =  sqrt(term * 0.5);
	res.x = norm1 * term;
	res.y = norm2 * term;
	return res;
}

__global__ void generationH0(cufftComplex *h0, cufftComplex *h0_minus, float *randData) {
	int bx = threadIdx.x;
	int by = blockIdx.x;
	int num = bx + by * BLOCK_SIZE;
	cufftComplex a;
	cufftComplex am;
	if (num < MAX_WAVE_RESOLUTION * MAX_WAVE_RESOLUTION) {
		int i = num / MAX_WAVE_RESOLUTION;
		int j = num - i * MAX_WAVE_RESOLUTION;
		cufftComplex k;
		cufftComplex km;
		k.x = 2 * PI * (i - MAX_WAVE_RESOLUTION / 2) / _lx;
		k.y = 2 * PI * (j - MAX_WAVE_RESOLUTION / 2) / _lz;
		km.x = -k.x;
		km.y = -k.y;
		a = get_h0(k, i, j, randData);	
		am = get_h0(km, i, j, randData);	
	}
	__syncthreads(); 
	if (num < MAX_WAVE_RESOLUTION * MAX_WAVE_RESOLUTION) {
		h0[num] = a;
		h0_minus[num] = am;
	}
}

//пусть будут передаваться все параметры
extern "C" void generation_h0(float lx, float lz, float windx, float windy, float anorm) {
	cufftComplex wind;
	wind.x = windx;
	wind.y = windy;
	cudaMemcpyToSymbol(_lx, &lx, sizeof (float));
	cudaMemcpyToSymbol(_lz, &lz, sizeof (float));
	cudaMemcpyToSymbol(_wind, &wind, sizeof (cufftComplex));
	cudaMemcpyToSymbol(_Anorm, &anorm, sizeof (float));
	curandGenerateNormal(gen, randData, 2 * MAX_WAVE_RESOLUTION * MAX_WAVE_RESOLUTION, 0.f, 1.f);
	dim3 threads = dim3(BLOCK_SIZE);
    dim3 blocks  = dim3((int) ((MAX_WAVE_RESOLUTION * MAX_WAVE_RESOLUTION - 0.5) / threads.x) + 1); 
	generationH0 <<<blocks, threads>>> (h0, h0_minus, randData);
	cudaDeviceSynchronize();
}

__device__ cufftComplex get_h(int i, int j, float t, cufftComplex *h0, cufftComplex *h0_minus) {
	cufftComplex h_0 = h0[MAX_WAVE_RESOLUTION * i + j];
	cufftComplex h_1 = h0_minus[MAX_WAVE_RESOLUTION * i + j];
	cufftComplex a;
	a.x = 2 * PI * (i - MAX_WAVE_RESOLUTION / 2) / _lx;
	a.y = 2 * PI * (j - MAX_WAVE_RESOLUTION / 2) / _lz;
	cufftComplex p;
	cufftComplex hi;
	cufftComplex pi;
	float x = sqrt(_g * sqrt(a.x * a.x + a.y * a.y)) * t;
	p.x = cos(x);
	p.y = sin(x);
	hi.x = h_1.x;
	hi.y = -h_1.y;
	pi.x = p.x;
	pi.y = -p.y;
	cufftComplex res;
	res.x = h_0.x * p.x - h_0.y * p.y + hi.x * pi.x - hi.y * pi.y;
	res.y = h_0.x * p.y + h_0.y * p.x + hi.x * pi.y + hi.y * pi.x;
	return res;
}

__device__ int fftshift(int idx) {
	if (idx >= MAX_WAVE_RESOLUTION / 2)
		return idx - MAX_WAVE_RESOLUTION / 2;
	else 
		return idx + MAX_WAVE_RESOLUTION / 2;
}

__global__ void generationHeight(float *t, cufftComplex * h_k, cufftComplex * h_k_normalx, cufftComplex * h_k_normaly, cufftComplex *h0, cufftComplex *h0_minus) {
	int bx = threadIdx.x;
	int by = blockIdx.x;
	int num = bx + by * BLOCK_SIZE;
	cufftComplex res;
	int shifti;
	int shiftj;
	int i;
	int j;
	cufftComplex k;
	if (num < MAX_WAVE_RESOLUTION * MAX_WAVE_RESOLUTION) {
		i = num / MAX_WAVE_RESOLUTION;
		j = num - i * MAX_WAVE_RESOLUTION;
		shifti = fftshift(i);
		shiftj = fftshift(j);
		k.x = 2 * PI * (shifti - MAX_WAVE_RESOLUTION / 2) / _lx;
		k.y = 2 * PI * (shiftj - MAX_WAVE_RESOLUTION / 2) / _lz;
		cufftComplex term =  get_h(i , j, *t, h0, h0_minus);
		res = term;
	}
	__syncthreads(); 
	if (num < MAX_WAVE_RESOLUTION * MAX_WAVE_RESOLUTION) {
		h_k[shifti * MAX_WAVE_RESOLUTION + shiftj] = res;
		h_k_normalx[shifti * MAX_WAVE_RESOLUTION + shiftj].x = -k.x * res.y;
		h_k_normalx[shifti * MAX_WAVE_RESOLUTION + shiftj].y = k.x * res.x;
		h_k_normaly[shifti * MAX_WAVE_RESOLUTION + shiftj].x = -k.y * res.y;
		h_k_normaly[shifti * MAX_WAVE_RESOLUTION + shiftj].y = k.y * res.x;
	}
}

extern "C" void generation_h(float t) {
	dim3 threads = dim3(BLOCK_SIZE);
    dim3 blocks  = dim3((int) ((MAX_WAVE_RESOLUTION * MAX_WAVE_RESOLUTION - 0.5) / threads.x) + 1); 
	float* _t = NULL;
	cudaMalloc ((void**)&_t, sizeof (float));
	cudaMemcpy (_t, &t,  sizeof (float), cudaMemcpyHostToDevice);
	generationHeight <<<blocks, threads>>> (_t, h_k, h_k_normalx, h_k_normaly, h0, h0_minus);
	cudaDeviceSynchronize();
}

__global__ void calc_density_mip0(cufftComplex * h, float * res) {
	int bx = threadIdx.x;
	int by = blockIdx.x;
	int num = bx + by * BLOCK_SIZE;
	int i = num / MAX_WAVE_RESOLUTION;
	int j = num - i * MAX_WAVE_RESOLUTION; 
	float ans = 0;
	if ((i < MAX_WAVE_RESOLUTION - 1) && (j  < MAX_WAVE_RESOLUTION - 1)) {
		float max_ = h[MAX_WAVE_RESOLUTION * i + j].x;
		max_ = max(max_, h[MAX_WAVE_RESOLUTION * i + j + 1].x); 
		max_ = max(max_, h[MAX_WAVE_RESOLUTION * (i + 1) + j + 1].x); 
		max_ = max(max_, h[MAX_WAVE_RESOLUTION * (i + 1) + j].x);
		float min_ = h[MAX_WAVE_RESOLUTION * i + j].x;
		min_ = min(min_, h[MAX_WAVE_RESOLUTION * i + j + 1].x); 
		min_ = min(min_, h[MAX_WAVE_RESOLUTION * (i + 1) + j + 1].x); 
		min_ = min(min_, h[MAX_WAVE_RESOLUTION * (i + 1) + j].x);
		ans = abs(max_ - min_);
	}
	__syncthreads(); 
	if (num < MAX_WAVE_RESOLUTION * MAX_WAVE_RESOLUTION) {
		res[i * MAX_WAVE_RESOLUTION + j] = h[MAX_WAVE_RESOLUTION * i + j].x;
	}
}

extern "C" void density_mip0(float * calc, cufftComplex * res) {
	dim3 threads = dim3(BLOCK_SIZE);
    dim3 blocks  = dim3((int) ((MAX_WAVE_RESOLUTION * MAX_WAVE_RESOLUTION - 0.5) / threads.x) + 1); 
	calc_density_mip0 <<<blocks, threads>>> (res, calc);
	cudaDeviceSynchronize();
}

__global__ void calc_density_mip_next(float * den, float * term, int * size) {
	int bx = threadIdx.x;
	int by = blockIdx.x;
	int num = bx + by * BLOCK_SIZE;
	int i = num / (*size);
	int j = num - i * (*size); 
	float ans = 0;
	if (num < (*size) * (*size)) {
		float max_ = den[4 * (*size) * i + 2 * j];
		max_ = max(max_, den[4 * (*size) * i + 2 * (*size) + 2 * j]); 
		max_ = max(max_, den[4 * (*size) * i + 2 * j + 1]); 
		max_ = max(max_, den[4 * (*size) * i + 2 * (*size) + 2 * j + 1]);
		ans = max_;
	}
	__syncthreads(); 
	if (num < (*size) * (*size)) {
		term[i * (*size) + j] = ans;
	}
}


extern "C" void density_mip_next(float * calc, int size) {
	dim3 threads = dim3(BLOCK_SIZE);
    dim3 blocks  = dim3((int) ((size * size - 0.5) / threads.x) + 1); 
	float * md;
	int * size_;
	cudaMalloc ((void**)&size_, sizeof (float));
	cudaMemcpy (size_, &size,  sizeof (float), cudaMemcpyHostToDevice);
	calc_density_mip_next <<<blocks, threads>>> (calc, term, size_);
	cudaError_t l;
	l = cudaDeviceSynchronize();
	l = cudaMemcpy (calc, term,  size * size * sizeof (float), cudaMemcpyDeviceToDevice);
}