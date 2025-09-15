#include "cuda_runtime.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <stdio.h>

__global__ void gaussianBlurKernel(unsigned char *inputImage, unsigned char *outputImage,
                                   int width, int height, int radius, float sigma)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height)
        return;

    int pixelIndex = (y * width + x) * 3;

    float r = 0.0f, g = 0.0f, b = 0.0f;
    float weightSum = 0.0f;

    // Precompute Gaussian weights
    for (int ky = -radius; ky <= radius; ky++)
    {
        for (int kx = -radius; kx <= radius; kx++)
        {
            int nx = x + kx;
            int ny = y + ky;

            if (nx >= 0 && nx < width && ny >= 0 && ny < height)
            {
                // Compute Gaussian weight
                float distance = kx * kx + ky * ky;
                // using expf for better performance on GPU
                float weight = expf(-distance / (2.0f * sigma * sigma));

                int nIndex = (ny * width + nx) * 3;
                r += inputImage[nIndex] * weight;
                g += inputImage[nIndex + 1] * weight;
                b += inputImage[nIndex + 2] * weight;
                weightSum += weight;
            }
        }
    }

    // Normalize
    outputImage[pixelIndex] = (unsigned char)(r / weightSum);
    outputImage[pixelIndex + 1] = (unsigned char)(g / weightSum);
    outputImage[pixelIndex + 2] = (unsigned char)(b / weightSum);
}

int main()
{
    int width, height, channels;
    int blurRadius = 16;
    unsigned char *h_inputImage = stbi_load("input.jpg", &width, &height, &channels, 3);
    // force 3 channels (RGB)

    if (!h_inputImage)
    {
        printf("Failed to load image!\n");
        return -1;
    }

    size_t imageSize = width * height * 3;

    unsigned char *h_outputImage = (unsigned char *)malloc(imageSize);

    unsigned char *d_inputImage, *d_outputImage;
    cudaMalloc(&d_inputImage, imageSize);
    cudaMalloc(&d_outputImage, imageSize);

    cudaMemcpy(d_inputImage, h_inputImage, imageSize, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    gaussianBlurKernel<<<grid, block>>>(d_inputImage, d_outputImage, width, height, blurRadius, 5.0f);

    cudaMemcpy(h_outputImage, d_outputImage, imageSize, cudaMemcpyDeviceToHost);

    stbi_write_jpg("output.jpg", width, height, 3, h_outputImage, 100);

    // Cleanup
    cudaFree(d_inputImage);
    cudaFree(d_outputImage);
    stbi_image_free(h_inputImage);
    free(h_outputImage);

    return 0;
}