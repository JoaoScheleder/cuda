#include "cuda_runtime.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <stdio.h>

#include <math.h>

__global__ void grayscaleKernel(unsigned char *inputImage, unsigned char *outputImage, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int pixelIndex = (y * width + x) * 3;

    unsigned char r = inputImage[pixelIndex];
    unsigned char g = inputImage[pixelIndex + 1];
    unsigned char b = inputImage[pixelIndex + 2];

    // Using luminosity method for better grayscale representation
    unsigned char gray = static_cast<unsigned char>(0.21f * r + 0.72f * g + 0.07f * b);

    outputImage[pixelIndex] = gray;
    outputImage[pixelIndex + 1] = gray;
    outputImage[pixelIndex + 2] = gray;
}

int main()
{
    int width, height, channels;

    unsigned char *h_inputImage = stbi_load("input.jpg", &width, &height, &channels, 3);

    if (h_inputImage == nullptr)
    {
        printf("Error loading image\n");
        return -1;
    }

    size_t imageSize = width * height * 3 * sizeof(unsigned char);

    unsigned char *h_outputImage = (unsigned char *)malloc(imageSize);

    unsigned char *d_inputImage, *d_outputImage;
    cudaMalloc((void **)&d_inputImage, imageSize);
    cudaMalloc((void **)&d_outputImage, imageSize);

    cudaMemcpy(d_inputImage, h_inputImage, imageSize, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16, 1);
    dim3 gridSize(ceil((float)width / blockSize.x), ceil((float)height / blockSize.y), 1);

    printf("Image dimensions: %d x %d\n", width, height);
    printf("Launching kernel with grid size (%d, %d, %d) and block size (%d, %d, %d)\n", gridSize.x, gridSize.y, gridSize.z, blockSize.x, blockSize.y, blockSize.z);

    grayscaleKernel<<<gridSize, blockSize>>>(d_inputImage, d_outputImage, width, height);

    cudaMemcpy(h_outputImage, d_outputImage, imageSize, cudaMemcpyDeviceToHost);

    // Save the output image
    stbi_write_jpg("output.jpg", width, height, 3, h_outputImage, 100);

    // Clean up
    stbi_image_free(h_inputImage);
    free(h_outputImage);
    cudaFree(d_inputImage);
    cudaFree(d_outputImage);

    return 0;
}