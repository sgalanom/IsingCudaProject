#include <stdio.h>

#include <stdlib.h>

#include <time.h>



// Function to initialize the Ising model with a random initial state

void initializeIsingModel(int size, char *isingModel) {

    for (int i = 0; i < size * size; ++i) {

        isingModel[i] = (rand() % 2) ? '*' : '`';  // Randomly assign '*' or ' '

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



// Function to perform one iteration of the Ising model simulation

void updateIsingModel(int size, char *isingModel) {

    char *tempModel = malloc(size * size * sizeof(char));



    // Update the Ising model based on the majority of spins among neighbors

    for (int i = 0; i < size; ++i) {

        for (int j = 0; j < size; ++j) {

            int sum = (isingModel[((i - 1 + size) % size) * size + j] == '*') +

                      (isingModel[i * size + (j - 1 + size) % size] == '*') +

                      (isingModel[i * size + j] == '*') +

                      (isingModel[((i + 1) % size) * size + j] == '*') +

                      (isingModel[i * size + (j + 1) % size] == '*');



            tempModel[i * size + j] = (sum > 2) ? '*' : '`';

        }

    }



    // Copy the updated state back to the original array

    for (int i = 0; i < size * size; ++i) {

        isingModel[i] = tempModel[i];

    }



    free(tempModel);

}



int main() {

    srand(42);  // Seed for random number generation



    int size = 25;       // Size of the 2D lattice

    int iterations = 35; // Number of iterations



    char *isingModel = malloc(size * size * sizeof(char));



    // Initialize the Ising model with a random initial state

    initializeIsingModel(size, isingModel);



    // Print the initial state

    printf("Initial Ising Model:\n");

    printIsingModel(size, isingModel);



    // Perform iterations of the Ising model simulation

    for (int k = 0; k < iterations; ++k) {

        updateIsingModel(size, isingModel);

        printf("After Iteration %d:\n", k + 1);

        printIsingModel(size, isingModel);

    }



    free(isingModel);



    return 0;

}






