#include <stdio.h>

#include <stdlib.h>

#include <time.h>



// Function to initialize the Ising model with a random initial state

void initializeIsingModel(int size, char *isingModel) {

    for (int i = 0; i < size * size; ++i) {

        isingModel[i] = (rand() % 2) ? '*' : ' ';  // Randomly assign '*' or ' '

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



// Kernel function for GPU parallelization

__global__ void updateIsingModelGPU(int size, char *isingModel) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    int j = blockIdx.y * blockDim.y + threadIdx.y;



    if (i < size && j < size) {

        int sum = (isingModel[((i - 1 + size) % size) * size + j] == '*') +

                  (isingModel[i * size + (j - 1 + size) % size] == '*') +

                  (isingModel[i * size + j] == '*') +

                  (isingModel[((i + 1) % size) * size + j] == '*') +

                  (isingModel[i * size + (j + 1) % size] == '*');



        isingModel[i * size + j] = (sum > 2) ? '*' : ' ';

    }

}



// Function to copy Ising model from device to host

void copyModelFromDevice(int size, char *hostModel, char *deviceModel) {

    cudaMemcpy(hostModel, deviceModel, size * size * sizeof(char), cudaMemcpyDeviceToHost);

}



int main() {

    srand(time(NULL));  // Seed for random number generation



    int size = 5;       // Size of the 2D lattice

    int iterations = 3; // Number of iterations



    char *hostModel = (char *)malloc(size * size * sizeof(char));

    char *deviceModel;



    // Initialize the Ising model with a random initial state

    initializeIsingModel(size, hostModel);



    // Print the initial state

    printf("Initial Ising Model:\n");

    printIsingModel(size, hostModel);



    // Allocate memory on the GPU for Ising model

    cudaMalloc((void **)&deviceModel, size * size * sizeof(char));



    // Copy the initial Ising model from host to device

    cudaMemcpy(deviceModel, hostModel, size * size * sizeof(char), cudaMemcpyHostToDevice);



    // Define GPU grid and block dimensions

    dim3 blockDim(16, 16);  // Adjust as needed

    dim3 gridDim((size + blockDim.x - 1) / blockDim.x, (size + blockDim.y - 1) / blockDim.y);



    // Perform iterations of the Ising model simulation on GPU

    for (int k = 0; k < iterations; ++k) {

        updateIsingModelGPU<<<gridDim, blockDim>>>(size, deviceModel);

        cudaDeviceSynchronize();  // Wait for GPU to finish



        // Copy the updated Ising model from device to host

        copyModelFromDevice(size, hostModel, deviceModel);



        printf("After Iteration %d:\n", k + 1);

        printIsingModel(size, hostModel);

    }



    // Free allocated memory on the GPU

    cudaFree(deviceModel);



    free(hostModel);



    return 0;

}


