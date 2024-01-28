#include <stdio.h>

#include <stdlib.h>

#include <time.h>



// CUDA kernel to perform one iteration of the Ising model simulation with one thread per moment

__global__ void updateIsingModelV1(int size, char *isingModel) {

    // Calculate the global thread index

    int idx = blockIdx.x * blockDim.x + threadIdx.x;



    // Calculate the 2D coordinates from the 1D index

    int i = idx / size;

    int j = idx % size;



    // Ensure the index is within bounds

    if (i < size && j < size) {

        // Update the Ising model based on the majority of spins among neighbors

        int sum = (isingModel[((i - 1 + size) % size) * size + j] == '*') +

                  (isingModel[i * size + (j - 1 + size) % size] == '*') +

                  (isingModel[i * size + j] == '*') +

                  (isingModel[((i + 1) % size) * size + j] == '*') +

                  (isingModel[i * size + (j + 1) % size] == '*');



        isingModel[idx] = (sum > 2) ? '*' : '`';

    }

}



// Function to print the Ising model

void printIsingModel(int size, char *isingModel) {

    for (int i = 0; i < size; ++i) {

        for (int j = 0; j < size; ++j) {

            printf("%c ", isingModel[i * size + j]);

        }

        printf("\n");

    }

    printf("\n");

}



int main() {

    srand(42);  // Seed for random number generation



    int size = 25;       // Size of the 2D lattice

    int iterations = 35; // Number of iterations



    char *isingModel = (char *)malloc(size * size * sizeof(char));



    // Initialize the Ising model with a random initial state

    for (int i = 0; i < size * size; ++i) {

        isingModel[i] = (rand() % 2) ? '*' : '`';  // Randomly assign '*' or ' '

    }



    // Print the initial state

    printf("Initial Ising Model:\n");

    printIsingModel(size, isingModel);



    // Allocate device memory and copy data to device

    char *d_isigModel;

    cudaMalloc((void **)&d_isigModel, size * size * sizeof(char));

    cudaMemcpy(d_isigModel, isingModel, size * size * sizeof(char), cudaMemcpyHostToDevice);



    // Explore different block sizes to find the fastest configuration

    int blockSize = 256;  // You can experiment with different block sizes

    int gridSize = (size * size + blockSize - 1) / blockSize;



    // Perform iterations of the Ising model simulation using the V1 implementation

    for (int k = 0; k < iterations; ++k) {

        updateIsingModelV1<<<gridSize, blockSize>>>(size, d_isigModel);

        cudaDeviceSynchronize();  // Wait for the kernel to finish



        printf("After Iteration %d:\n", k + 1);



        // Copy the updated state back to the host

        cudaMemcpy(isingModel, d_isigModel, size * size * sizeof(char), cudaMemcpyDeviceToHost);



        printIsingModel(size, isingModel);

    }



    // Free device memory

    cudaFree(d_isigModel);

    free(isingModel);



    return 0;

}


