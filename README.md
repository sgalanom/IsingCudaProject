# Ising Model Simulation using CUDA
This repository contains the implementation of the Ising model simulation in two dimensions using CUDA, a parallel computing platform. The Ising model is a mathematical model of ferromagnetism in statistical mechanics, which consists of discrete magnetic dipole moments of atomic spins arranged in a 2D lattice.

# Versions Implemented
V0. Sequential implementation in C to simulate the Ising model.

V1. CUDA implementation with one thread per moment.

V2. CUDA implementation with optimized thread assignments.

V3. CUDA implementation utilizing shared memory for optimization.

# Example Usage in aristotelis Remote Shell
#Load modules

module load gcc/9.4.0-eewq4j6 cuda/11.2.2-kkrwdua

#Clone the repository:

git clone https://github.com/sgalanom/IsingCudaProject

#Navigate to the repository directory:

cd IsingCudaProject

#Compile the code, for example:

nvcc isingV0.cu -o output

#Run the executable:

./output
