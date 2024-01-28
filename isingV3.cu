#include <stdio.h>

#include <stdlib.h>

#include <time.h>



#define BLOCK_SIZE 3



// Function to initialize the Ising model with a random initial state

void initializeIsingModel(int size, char *isingModel) {

    for (int i = 0; i < size * size; ++i) {

        isingModel[i] = (rand() % 2) ? '*' : '`';  // Randomly assign '*' or '`'

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



// Kernel function for GPU parallelization with shared memory

__global__ void updateIsingModelGPUV3(int size, char *isingModel) {

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    int j = blockIdx.y * blockDim.y + threadIdx.y;



    __shared__ char sharedModel[BLOCK_SIZE * 2][BLOCK_SIZE * 2];



    // Load data into shared memory

    for (int i_local = threadIdx.x; i_local < blockDim.x * 2; i_local += blockDim.x) {

        for (int j_local = threadIdx.y; j_local < blockDim.y * 2; j_local += blockDim.y) {

            int i_global = i - blockDim.x + i_local;

            int j_global = j - blockDim.y + j_local;



            sharedModel[i_local][j_local] = (i_global >= 0 && i_global < size && j_global >= 0 && j_global < size)

                                                ? isingModel[i_global * size + j_global]

                                                : '`';

        }

    }



    __syncthreads();  // Ensure all threads have loaded data into shared memory



    // Compute a block of moments using shared memory

    for (int i_local = threadIdx.x; i_local < blockDim.x * 2; i_local += blockDim.x) {

        for (int j_local = threadIdx.y; j_local < blockDim.y * 2; j_local += blockDim.y) {

            if (i + i_local < size && j + j_local < size) {

                int sum = (sharedModel[i_local - 1][j_local] == '*') +

                          (sharedModel[i_local][j_local - 1] == '*') +

                          (sharedModel[i_local][j_local] == '*') +

                          (sharedModel[i_local + 1][j_local] == '*') +

                          (sharedModel[i_local][j_local + 1] == '*');



                isingModel[(i + i_local) * size + (j + j_local)] = (sum > 2) ? '*' : '`';

            }

        }

    }

}



// Function to copy Ising model from device to host

void copyModelFromDevice(int size, char *hostModel, char *deviceModel) {

    cudaMemcpy(hostModel, deviceModel, size * size * sizeof(char), cudaMemcpyDeviceToHost);

}



int main() {

    srand(time(NULL));  // Seed for random number generation



    int size = 15;       // Size of the 2D lattice

    int iterations = 12; // Number of iterations



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

    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);

    dim3 gridDim((size + blockDim.x * 2 - 1) / (blockDim.x * 2), (size + blockDim.y * 2 - 1) / (blockDim.y * 2));



    // Perform iterations of the Ising model simulation on GPU

    for (int k = 0; k < iterations; ++k) {

        updateIsingModelGPUV3<<<gridDim, blockDim>>>(size, deviceModel);

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


