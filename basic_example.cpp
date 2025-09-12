#include  "cuda_runtime.h"
#include <stdio.h>


__global__ void myKernel(void) {
    printf("Hello From GPU");
}   

int main () {
    printf("Hello from CPU");
    // blocks, threads
    myKernel<<<1,10>>>();
    
    return 0;
}