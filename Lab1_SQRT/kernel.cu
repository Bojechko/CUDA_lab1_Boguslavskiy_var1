
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <ctime>
#include <math.h>
#include <stdio.h>
#include <iostream>
using namespace std;

#define N 3


cudaError_t sqrtWithCuda(float* result, const float* a, unsigned int size);

void get_a(float* a)
{
    for (int i = 0; i < N; i++)
    {
        while (a[i] < 0)
        {
            a[i] = rand() % 100;
        }
    }
}


__global__ void sqrtKernel( float* result, float* a)
{
    int x = threadIdx.x;
    result[x] = sqrt(a[x]);
}

void sqrt_for_CPU(float* result, float* a)
{
    for (int i = 0; i < N; i++)
        result[i] = sqrt(a[i]);
}


void post_result(float *result)
{
    for (int i = 0; i < N; i++)
    {
        cout << result[i] << ";  ";
    }
}



int main()
{

    float* a = new float[N];
    float* result = new float[N];

    clock_t begin = clock();

    get_a(a);

    sqrt_for_CPU(result, a);

    post_result(result);

    clock_t end = clock();
    double cpu = (double)(end - begin) / CLOCKS_PER_SEC * 1000;

    cout << "CPU time = " << cpu << "\n";


//gpu

    float* a1 = new float[N];
    float* result1 = new float[N];

    cudaEvent_t start, stop;
    float gpu = 0.0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);
    get_a(a1);
 
    cudaError_t cudaStatus = sqrtWithCuda(result1, a1, N);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "sqrtKernel failed!");
        return 0;
    }

    

    cout << "Result " << "\n";
    post_result(result1);

    cudaEventRecord(stop, 0);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&gpu, start, stop);

    cout << "GPU time = " << gpu << "\n";


    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }
    cudaDeviceReset();
    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t sqrtWithCuda(float* res, const float* a, unsigned int size)
{
    float* dev_a = 0;
    float* dev_result = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_result, size * sizeof(float));
    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(float));

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(float), cudaMemcpyHostToDevice);

    sqrtKernel << <1, size >> > (dev_result, dev_a);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(res, dev_result, size * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dev_result);
    cudaFree(dev_a);

    return cudaStatus;
}
