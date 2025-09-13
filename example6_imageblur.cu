#include "cuda_runtime.h"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <stdio.h>


/// @brief 
/// @param inputImage 
/// @param outputImage 
/// @param width 
/// @param height 
/// @return 
__global__ void blurImageKernel(unsigned char* inputImage, unsigned char* outputImage, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) return;

    int pixelIndex = (y * width + x) * 3; // Assuming 3 channels (RGB)

    // Simple box blur kernel
    int r = 0, g = 0, b = 0;
    int count = 0;

    for (int ky = -1; ky <= 1; ky++) {
        for (int kx = -1; kx <= 1; kx++) {
            int nx = x + kx;
            int ny = y + ky;

            if (nx >= 0 && nx < width && ny >= 0 && ny < height) {
                int nIndex = (ny * width + nx) * 3;
                r += inputImage[nIndex];
                g += inputImage[nIndex + 1];
                b += inputImage[nIndex + 2];
                count++;
            }
        }
    }

    outputImage[pixelIndex] = r / count;
    outputImage[pixelIndex + 1] = g / count;
    outputImage[pixelIndex + 2] = b / count;
}


int main() {
    int width, height, channels;
    unsigned char* h_inputImage = stbi_load("input.jpg", &width, &height, &channels, 3); 
    // force 3 channels (RGB)

    if (!h_inputImage) {
        printf("Failed to load image!\n");
        return -1;
    }

    size_t imageSize = width * height * 3;

    unsigned char *h_outputImage = (unsigned char*)malloc(imageSize);

    unsigned char *d_inputImage, *d_outputImage;
    cudaMalloc(&d_inputImage, imageSize);
    cudaMalloc(&d_outputImage, imageSize);

    cudaMemcpy(d_inputImage, h_inputImage, imageSize, cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((width + block.x - 1) / block.x, (height + block.y - 1) / block.y);

    blurImageKernel<<<grid, block>>>(d_inputImage, d_outputImage, width, height);

    cudaMemcpy(h_outputImage, d_outputImage, imageSize, cudaMemcpyDeviceToHost);

    stbi_write_jpg("output.jpg", width, height, 3, h_outputImage, 100);

    // Cleanup
    cudaFree(d_inputImage);
    cudaFree(d_outputImage);
    stbi_image_free(h_inputImage);
    free(h_outputImage);

    return 0;
}
