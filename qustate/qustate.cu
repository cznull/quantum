#include <assert.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cusolverDn.h"
#include "cufft.h"
#include <gl/glew.h>
#include <cuda_gl_interop.h>
#include "qustate.h"


int m = 0;
int lda = 0;
int lwork = 0;
current2 *d_A = NULL;
current *d_W = NULL;
float *d_ldos = NULL;
current2 *d_work = NULL;
int *devInfo = NULL;
float3 *d_line = NULL;
int *d_ct = NULL;
int *d_img = NULL;
struct cudaGraphicsResource *cu_vbo;
struct cudaGraphicsResource *cu_pbo;
cudaError cudaStatus;

cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;

cusolverDnHandle_t cusolverH = NULL;
cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;
cufftHandle fftPlan;
cufftResult fresu;

__global__ void gldos(current2 *vec, current *val, float *img, int size, float s, int el, float es, float eb,float ce) {
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int i, j;
	float jf;
	for (i = 0; i < size; i++) {
		jf = (val[i] - eb) * s * size*es;
		j = jf;
		if (j < size + el + 1) {
			if (j > -el) {
				float a = vec[i*size + x].x*vec[i*size + x].x + vec[i*size + x].y*vec[i*size + x].y;
				a = a * s;
				for (int k = j - el; k <= j + el; k++) {
					if (k<size && k>-1) {
						img[k*size + x] += exp((k - jf)*(k - jf)*ce)*a;
					}
				}
			}
		}
		else {
			break;
		}
	}
}

__global__ void imgp(float *img, int size, int xl, float cx) {
	int x = threadIdx.x;
	int y = blockIdx.x;
	int i, j;
	float a;
	__shared__ float as[128 + 64 * 3];

	if (x < 64) {
		as[x] = img[y*size + size - 64 + x];
		as[128 + 64 * 2 + x] = img[y*size + x];
		as[128 + 64 + x] = img[y*size + 128 + x];
	}
	__syncthreads();
	as[64 + x] = img[y*size + x];
	__syncthreads();
	a = 0;
	for (j = -xl; j <= xl; j++) {
		a += as[x + 64 + j] * exp(j*j*cx);
	}
	img[y*size + x] = a;

	for (i = 1; i < size / 128 - 1; i++) {
		if (x < 64) {
			as[x] = as[128 + x];
			as[128 + 64 + x] = img[y*size + i * 128 + 128 + x];
		}
		__syncthreads();
		as[64 + x] = img[y*size + i * 128 + x];
		__syncthreads();
		a = 0;
		for (j = -xl; j <= xl; j++) {
			a += as[x + 64 + j] * exp(j*j*cx);
		}
		img[y*size + i * 128 + x] = a;;
	}
	if (x < 64) {
		as[x] = as[128 + x];
		as[128 + 64 + x] = as[128 + 64 * 2 + x];
	}
	__syncthreads();
	as[64 + x] = img[y*size + i * 128 + x];
	__syncthreads();
	a = 0;
	for (j = -xl; j <= xl; j++) {
		a += as[x + 64 + j] * exp(j*j*cx);
	}
	img[y*size + i * 128 + x] = a;
}

__global__ void gline(current2 *vec, current *val, float3 *line, int size, float l, float dx, float m, float es, float eb, float height) {
	int j = threadIdx.x + blockIdx.x * blockDim.x;
	int i = threadIdx.y + blockIdx.y * blockDim.y;
	if (j < size - 1) {
		float y = (val[i] - eb)*es;
		float2 a = { vec[i*size + j].x,vec[i*size + j].y };
		float2 b = { vec[i*size + j + 1].x,vec[i*size + j + 1].y };
		line[i*(size - 1) * 2 + j * 2 + 0] = { l + dx * (j + 0.5f),(a.x*a.x + a.y*a.y)*height *m,-y };
		line[i*(size - 1) * 2 + j * 2 + 1] = { l + dx * (j + 1.5f),(b.x*b.x + b.y*b.y)*height *m,-y };
	}
}

__global__ void gimg(float *ldos, int *ct, int *img, float br) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = log(ldos[i] * br + 1.0f) * 2048.0f + 1024.0f;
	if (j > 4095) {
		j = 4095;
	}
	if (j < 0) {
		j = 0;
	}
	img[i] = ct[j];
}

int cui(int size ,GLuint vbo,GLuint pbo,int *ct) {
	size_t num_bytes;
	m = size;
	lda = size;
	cudaSetDevice(0);
	cudaMalloc((void**)&devInfo, sizeof(int));
	cudaMalloc((void**)&d_A, sizeof(current2) * lda * m);
	cudaMalloc((void**)&d_ldos, sizeof(float) * lda * m);
	cudaMalloc((void**)&d_W, sizeof(current) * m);
	cudaMalloc((void**)&d_ct, sizeof(int) * 4096);
	cudaMemcpy(d_ct, ct, sizeof(int) * 4096, cudaMemcpyHostToDevice);

#ifdef T_d
	cufftPlan1d(&fftPlan, size, CUFFT_Z2Z, size);//d/f
#else
	cufftPlan1d(&fftPlan, size, CUFFT_C2C, size);//d/f
#endif

	cusolverDnCreate(&cusolverH);

#ifdef T_d
	cusolverDnZheevd_bufferSize(cusolverH, jobz, uplo, m, d_A, lda, d_W, &lwork);
#else
	cusolverDnCheevd_bufferSize(cusolverH, jobz, uplo, m, d_A, lda, d_W, &lwork);
#endif

	cudaMalloc((void**)&d_work, sizeof(current2)*lwork);

	cudaGraphicsGLRegisterBuffer(&cu_vbo, vbo, cudaGraphicsMapFlagsWriteDiscard);
	cudaGraphicsMapResources(1, &cu_vbo, 0);
	cudaGraphicsResourceGetMappedPointer((void **)&d_line, &num_bytes, cu_vbo);
	cudaGraphicsUnmapResources(1, &cu_vbo, 0);

	cudaGraphicsGLRegisterBuffer(&cu_pbo, pbo, cudaGraphicsMapFlagsWriteDiscard);
	cudaGraphicsMapResources(1, &cu_pbo, 0);
	cudaGraphicsResourceGetMappedPointer((void **)&d_img, &num_bytes, cu_pbo);
	cudaGraphicsUnmapResources(1, &cu_pbo, 0);
	return 0;
}

int cueigen(void *hm, void *val, void *vec, int size) {
	cudaMemcpy(d_A, hm, sizeof(current2) * lda * m, cudaMemcpyHostToDevice);

#ifdef T_d
	cusolverDnZheevd(cusolverH, jobz, uplo, m, d_A, lda, d_W, d_work, lwork, devInfo);//d/f
	cufftExecZ2Z(fftPlan, d_A, d_A, CUFFT_INVERSE);//d/f
#else
	cusolverDnCheevd(cusolverH, jobz, uplo, m, d_A, lda, d_W, d_work, lwork, devInfo);//d/f
	cufftExecC2C(fftPlan, d_A, d_A, CUFFT_INVERSE);//d/f
#endif

	cudaMemcpy(val, d_W, sizeof(current)*m, cudaMemcpyDeviceToHost);
	cudaMemcpy(vec, d_A, sizeof(current2)*lda*m, cudaMemcpyDeviceToHost);
	return 0;
}

int cuimg(float *img, int size, float s, float el, float xl, float es, float eb, float br) {
	cudaError cudaStatus;
	cudaMemset(d_ldos, 0, sizeof(float)*size*size);
	gldos << <size / 128, 128 >> > (d_A, d_W, d_ldos, size, s, el, es, eb, -4.0 / el / el);
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		return 1;
	}
	imgp << <size, 128 >> > (d_ldos, size, xl, -4.0 / xl / xl);
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		return 1;
	}
	cudaGraphicsMapResources(1, &cu_pbo, 0);
	gimg << <size*size / 128, 128 >> > (d_ldos, d_ct, d_img, br / el / xl);
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		return 1;
	}
	cudaGraphicsUnmapResources(1, &cu_pbo, 0);
	//cudaMemcpy(img, d_ldos, sizeof(float)*size*size, cudaMemcpyDeviceToHost);
	return 0;
}

int culine(int size, float l, float r, float es, float eb, float height) {
	cudaGraphicsMapResources(1, &cu_vbo, 0);
	gline << < dim3(size / 128, size, 1), dim3(128, 1, 1) >> > (d_A, d_W, d_line, size, l, (r - l)*(1.0 / size), 1.0 / (r - l), es, eb, height);
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		return 1;
	}
	cudaGraphicsUnmapResources(1, &cu_vbo, 0);
	return 0;
}