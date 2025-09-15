
#include <cuda_runtime.h>
#include <stdio.h>
#include <math.h>

struct Particle
{
    float3 p;
    float3 v;
    float3 a;
};

__global__ void K_Particle01_NOCOALESCE(Particle *vet)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Update particle properties
    vet[idx].v.x += vet[idx].a.x + vet[idx].p.x + vet[idx].v.y;
    vet[idx].v.y += vet[idx].a.y + vet[idx].p.y + vet[idx].v.x;
    vet[idx].v.z += vet[idx].a.z + vet[idx].p.z + vet[idx].v.z;
}

__global__ void K_Particle01_COALESCE(float *vet_px, float *vet_py, float *vet_pz,
                                      float *vet_vx, float *vet_vy, float *vet_vz,
                                      float *vet_ax, float *vet_ay, float *vet_az)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    // Update particle properties
    vet_vx[idx] += vet_ax[idx] + vet_px[idx] + vet_vy[idx];
    vet_vy[idx] += vet_ay[idx] + vet_py[idx] + vet_vx[idx];
    vet_vz[idx] += vet_az[idx] + vet_pz[idx] + vet_vz[idx];
}

int main()
{

    // create Particle

    int nPart = 1024 * 1024;
    size_t size = nPart * sizeof(Particle);
    Particle *h_vet = (Particle *)malloc(size);

    for (int i = 0; i < nPart; i++)
    {
        h_vet[i].p = make_float3(1.0f, 2.0f, 3.0f);
        h_vet[i].v = make_float3(0.0f, 0.0f, 0.0f);
        h_vet[i].a = make_float3(0.1f, 0.1f, 0.1f);
    }

    Particle *d_vet;
    cudaMalloc((void **)&d_vet, size);
    cudaMemcpy(d_vet, h_vet, size, cudaMemcpyHostToDevice);

    dim3 blockDimension = dim3(256, 1, 1);
    dim3 gridDimension = dim3(ceil(nPart / (float)blockDimension.x), 1, 1);

    printf("Grid: (%d, %d, %d) - Block: (%d, %d, %d)\n",
           gridDimension.x, gridDimension.y, gridDimension.z,
           blockDimension.x, blockDimension.y, blockDimension.z);

    printf("Particle 0: p=(%f, %f, %f), v=(%f, %f, %f), a=(%f, %f, %f)\n",
           h_vet[0].p.x, h_vet[0].p.y, h_vet[0].p.z,
           h_vet[0].v.x, h_vet[0].v.y, h_vet[0].v.z,
           h_vet[0].a.x, h_vet[0].a.y, h_vet[0].a.z);

    // cudaEvents
    cudaEvent_t start, stop;
    float elapsedTime;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    K_Particle01_NOCOALESCE<<<gridDimension, blockDimension>>>(d_vet);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("K_Particle01_NOCOALESCE - elapsedTime: %f ms\n", elapsedTime);

    cudaMemcpy(h_vet, d_vet, size, cudaMemcpyDeviceToHost);

    printf("Particle 0: p=(%f, %f, %f), v=(%f, %f, %f), a=(%f, %f, %f)\n",
           h_vet[0].p.x, h_vet[0].p.y, h_vet[0].p.z,
           h_vet[0].v.x, h_vet[0].v.y, h_vet[0].v.z,
           h_vet[0].a.x, h_vet[0].a.y, h_vet[0].a.z);

    cudaFree(d_vet);
    free(h_vet);

    float *h_vet_px = (float *)malloc(nPart * sizeof(float));
    float *h_vet_py = (float *)malloc(nPart * sizeof(float));
    float *h_vet_pz = (float *)malloc(nPart * sizeof(float));
    float *h_vet_vx = (float *)malloc(nPart * sizeof(float));
    float *h_vet_vy = (float *)malloc(nPart * sizeof(float));
    float *h_vet_vz = (float *)malloc(nPart * sizeof(float));
    float *h_vet_ax = (float *)malloc(nPart * sizeof(float));
    float *h_vet_ay = (float *)malloc(nPart * sizeof(float));
    float *h_vet_az = (float *)malloc(nPart * sizeof(float));

    for (int i = 0; i < nPart; i++)
    {
        h_vet_px[i] = 1.0f;
        h_vet_py[i] = 2.0f;
        h_vet_pz[i] = 3.0f;
        h_vet_vx[i] = 0.0f;
        h_vet_vy[i] = 0.0f;
        h_vet_vz[i] = 0.0f;
        h_vet_ax[i] = 0.1f;
        h_vet_ay[i] = 0.1f;
        h_vet_az[i] = 0.1f;
    };

    float *d_vet_px, *d_vet_py, *d_vet_pz;
    float *d_vet_vx, *d_vet_vy, *d_vet_vz;
    float *d_vet_ax, *d_vet_ay, *d_vet_az;

    cudaMalloc((void **)&d_vet_px, nPart * sizeof(float));
    cudaMalloc((void **)&d_vet_py, nPart * sizeof(float));
    cudaMalloc((void **)&d_vet_pz, nPart * sizeof(float));
    cudaMalloc((void **)&d_vet_vx, nPart * sizeof(float));
    cudaMalloc((void **)&d_vet_vy, nPart * sizeof(float));
    cudaMalloc((void **)&d_vet_vz, nPart * sizeof(float));
    cudaMalloc((void **)&d_vet_ax, nPart * sizeof(float));
    cudaMalloc((void **)&d_vet_ay, nPart * sizeof(float));
    cudaMalloc((void **)&d_vet_az, nPart * sizeof(float));

    cudaMemcpy(d_vet_px, h_vet_px, nPart * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vet_py, h_vet_py, nPart * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vet_pz, h_vet_pz, nPart * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vet_vx, h_vet_vx, nPart * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vet_vy, h_vet_vy, nPart * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vet_vz, h_vet_vz, nPart * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vet_ax, h_vet_ax, nPart * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vet_ay, h_vet_ay, nPart * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_vet_az, h_vet_az, nPart * sizeof(float), cudaMemcpyHostToDevice);
    printf("Particle 0: p=(%f, %f, %f), v=(%f, %f, %f), a=(%f, %f, %f)\n",
           h_vet_px[0], h_vet_py[0], h_vet_pz[0],
           h_vet_vx[0], h_vet_vy[0], h_vet_vz[0],
           h_vet_ax[0], h_vet_ay[0], h_vet_az[0]);

    cudaEventRecord(start, 0);

    K_Particle01_COALESCE<<<gridDimension, blockDimension>>>(
        d_vet_px, d_vet_py, d_vet_pz,
        d_vet_vx, d_vet_vy, d_vet_vz,
        d_vet_ax, d_vet_ay, d_vet_az);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    printf("K_Particle01_COALESCE - elapsedTime: %f ms\n", elapsedTime);

    cudaMemcpy(h_vet_vx, d_vet_vx, nPart * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_vet_vy, d_vet_vy, nPart * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_vet_vz, d_vet_vz, nPart * sizeof(float), cudaMemcpyDeviceToHost);
    printf("Particle 0: p=(%f, %f, %f), v=(%f, %f, %f), a=(%f, %f, %f)\n",
           h_vet_px[0], h_vet_py[0], h_vet_pz[0],
           h_vet_vx[0], h_vet_vy[0], h_vet_vz[0],
           h_vet_ax[0], h_vet_ay[0], h_vet_az[0]);

    // Free device memory
    cudaFree(d_vet_px);
    cudaFree(d_vet_py);
    cudaFree(d_vet_pz);
    cudaFree(d_vet_vx);
    cudaFree(d_vet_vy);
    cudaFree(d_vet_vz);
    cudaFree(d_vet_ax);
    cudaFree(d_vet_ay);
    cudaFree(d_vet_az);
    // Free host memory
    free(h_vet_px);
    free(h_vet_py);
    free(h_vet_pz);
    free(h_vet_vx);
    free(h_vet_vy);
    free(h_vet_vz);
    free(h_vet_ax);
    free(h_vet_ay);
    free(h_vet_az);

    return 0;
}