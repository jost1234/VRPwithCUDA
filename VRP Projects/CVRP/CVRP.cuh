#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cooperative_groups.h>
#include <iostream>

#include "CVRP_Defines.cuh"

// VRP : Vehicle Routing Problem
// Input : n nodes (0th node: warehouse, other n-1: terminals), k trucks
// Goal : Going through every node, while sum of distances is minimal

/// DIAGNOSTIC FUNCTIONS

// Diagnostic printing of a matrix
__host__ __device__ void print(float* A, size_t size) {
    for (int ii = 0; ii < size; ii++) {
        for (int jj = 0; jj < size; jj++)
            printf("%.2f ", A[ii * size + jj]);
        printf("\n");
    }
    printf("\n");
}

__host__ __device__ void print(float* A, size_t col, size_t row) {
    for (int ii = 0; ii < row; ii++) {
        for (int jj = 0; jj < col; jj++)
            printf("%.2f ", A[ii * col + jj]);
        printf("\n");
    }
    printf("\n");
}

// ceil function used to calculate block count
__device__ __host__ int my_ceil(int osztando, int oszto) {
    if (!(osztando % oszto)) return osztando / oszto;
    else return	osztando / oszto + 1;
}

namespace CVRP {
    /// CUDA LAUNCH AND KERNEL FUNCTIONS

    // Host function for File Handling and Memory allocation
    int Host_main(FILE* pfile, int size, char* weightType);

    // Main CUDA function
    cudaError_t CUDA_main(CUDA_Main_ParamTypedef h_params);

    // Testing input for main CUDA function
    // Returns true if input data syntax is good
    // Disclaimer: Only tests NULL property of pointers, does not 100% guarantee perfect data
    __host__ inline bool inputGood(CUDA_Main_ParamTypedef* ph_params);

    // Inicializes a random seed for each different threads
    __global__ void setup_kernel(curandState* state, unsigned long seed);

    // Frees device memory
    void Free_device_memory(Kernel_ParamTypedef params);

    __device__ inline bool inputGood(Kernel_ParamTypedef* params);

    // Diagnostic function for printing given sequence
    __host__ float sequencePrint(CUDA_Main_ParamTypedef* params);

    // Reduction to TSP
    __host__ __device__ inline int RouteSize(int size, int maxVehicles)
    {
        return size + maxVehicles - 1;
    };

    // 1 block sized kernel
    __global__ void Kernel_1Block(
        Kernel_ParamTypedef params,
        Kernel_ConfigParamTypedef configParams
    );

    // Multiblock sized kernel
    __global__ void Kernel_multiBlock(
        Kernel_ParamTypedef params,
        Kernel_ConfigParamTypedef configParams);



    // Gets initial value of Route arrays
    __device__ void initAntRoute(
        Kernel_ParamTypedef* pkernelParams,
        int antIndex
    );

    // Generates a random sequence of numbers between 0 and (size - 1) starting with 0
    // secondVertex: Variable used for giving an arbitrary second vertex
    //      0 < secondvertex < size : valid input (condition = 1)
    //      else: invalid input, no mandatory second vertex (condition = 0)
    __device__ void generateRandomSolution(
        Kernel_ParamTypedef* pkernelParams,
        int antIndex
    );

    // Returns bool value of whether newParam is already listed in the route
    // Special care for node 0, which can be in the route [maxVehicles] times.
    __device__ bool alreadyListed(
        Kernel_ParamTypedef* pkernelParams,
        int antIndex,
        int idx,    // serial number of node in route
        int newParam
    );

    // Returns the sum length of the given route of trucks
    // Returns -1 if route not possible (for example has dead end) or if cap. condition not met
    // FUNCTION USAGE: CapacityCondition
    __device__ float antRouteLength(Kernel_ParamTypedef* pkernelParams, int antIndex);

    // Represents az ant who follows other ants' pheromones
    // Generates a route with Roulette wheel method given the values of the Pheromone matrix
    __device__ void followPheromones(
        Kernel_ParamTypedef* pkernelParams,
        int antIndex,
        int maxTryNumber
    );

    // If the last node was 0 in route, we have to calculate the row index
    // we need
    __device__ inline int correctRow(int size, int vehicleIdx, int sourceNode = 0);

    /// Scans that the given solution is suitable for the capacity condition
    // Returns a bool value of the condition evaluation
    // FUNCTION USED BY: evaluateSolution
    __device__ bool CapacityCondition(Kernel_ParamTypedef* pkernelParams, int antIndex);

    // FUNCTION USED BY: sequencePrint
    __host__ bool CapacityCondition(CUDA_Main_ParamTypedef* params);

    // Manipulating the pheromone values according to the given solution
    // The longer the route is, the smaller amount we are adding
    // Sets the route vector if we found a best yet solution
    // FUNCTION USAGE: CapacityCondition
    __device__ void evaluateSolution(
        Kernel_ParamTypedef* pkernelParams,
        int antIndex,
        float multiplConstant,
        float rewardMultiplier,
        int repNumber
    );

    // Auxilary function for greedy sequence
    // Returns the highest vertex index not yet chosen
    /// row : row of previous route element (decides, which row to watch in the function)
    __device__ int maxInIdxRow(Kernel_ParamTypedef* pkernelParams, int row, int idx);

    // Generates a sequnce using greedy algorithm
    // Always chooses the highest possible value for the next vertex
    __device__ void greedySequence(Kernel_ParamTypedef* pkernelParams);

    // Copies a route into the answer vector
    __device__ void copyAntRoute(Kernel_ParamTypedef* pkernelParams, int antIndex);

    // Validates the output vector
    __device__ bool validRoute(Kernel_ParamTypedef* pkernelParams);

    // How many times does the given node appear in the sequence 
    __device__ int nodeCount(Kernel_ParamTypedef* pkernelParams, int node);

    // Finds a value in the route vector
    __device__ bool routeContain(Kernel_ParamTypedef* pkernelParams, int value);
}