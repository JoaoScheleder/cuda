#include  "cuda_runtime.h"
#include <stdio.h>


__global__ void myKernel(void) {
    printf("Hello From GPU \n");
}   

int main () {
    // blocks, threads (max threads per block = 1024)
    myKernel<<<1,10>>>();
    printf("Hello from CPU \n");
    return 0;
}