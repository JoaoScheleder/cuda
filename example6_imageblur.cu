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
/// @param radius
/// @return
__global__ void blurImageKernel(unsigned char *inputImage, unsigned char *outputImage, int width, int height, int radius)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    // in case the image dimensions are not multiples of block size
    // avoids processing out-of-bounds threads/pixels
    if (x >= width || y >= height)
        return;

    // each image is stored in a single array in RGB format, thats why we dont need the height here
    int pixelIndex = (y * width + x) * 3;

    // Accumulators for RGB values and count of pixels
    int r = 0, g = 0, b = 0;

    // keeps track of number of pixels considered for averaging
    int count = 0;

    // Iterate over the kernel window
    for (int ky = -radius; ky <= radius; ky++)
    {
        // Iterate over the kernel window in x direction
        for (int kx = -radius; kx <= radius; kx++)
        {
            // Neighbor pixel coordinates
            int nx = x + kx;
            // Neighbor pixel coordinates
            int ny = y + ky;

            // Check if neighbor pixel is within image bounds
            if (nx >= 0 && nx < width && ny >= 0 && ny < height)
            {
                // Get the index of the neighbor pixel
                int nIndex = (ny * width + nx) * 3;

                // Accumulate RGB values
                r += inputImage[nIndex];
                g += inputImage[nIndex + 1];
                b += inputImage[nIndex + 2];
                // Increment count of valid pixels
                count++;
            }
        }
    }

    // Compute average and assign to output image
    outputImage[pixelIndex] = r / count;
    outputImage[pixelIndex + 1] = g / count;
    outputImage[pixelIndex + 2] = b / count;
}

int main()
{
    int width, height, channels;
    int blurRadius = 32;
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

    blurImageKernel<<<grid, block>>>(d_inputImage, d_outputImage, width, height, blurRadius);

    cudaMemcpy(h_outputImage, d_outputImage, imageSize, cudaMemcpyDeviceToHost);

    stbi_write_jpg("output.jpg", width, height, 3, h_outputImage, 100);

    // Cleanup
    cudaFree(d_inputImage);
    cudaFree(d_outputImage);
    stbi_image_free(h_inputImage);
    free(h_outputImage);

    return 0;
}
