#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand_kernel.h>
#include <iostream>


// Thread block size
#define BLOCK_SIZE 1024

///
/// CONTROL PANEL
///

// Number of threads = number of ants
// Default value: 1024
int ants = 1024;

// Repetition constants
#define REPETITIONS 10
#define RANDOM_GENERATIONS 20
#define FOLLOWER_GENERATIONS 500

// Pheromone matrix constants
#define RHO 0.75  // Reduction ratio of previous pheromon value
#define REWARD_MULTIPLIER 10   // Reward multiplier after finding a shortest path until then
#define INITIAL_PHEROMONE_VALUE 1000    // Initial value of elements in the Pheromone matrix

#define SERIALMAXTRIES 10 // Number of serial processes (for debug purposes)

namespace TSP {

	/// Struct definitions
	/// 
	/// Naming convention:	firstSecond skalar and vector variables
	///						FirstSecond Matrices

	// Struct for Main CUDA function call
	typedef struct {
		int antNum;
		float* Dist;
		bool* foundRoute;
		float* Pheromone;
		int* route;
		int size;
	} CUDA_Main_ParamTypedef;


	// Struct for kernel call
	typedef struct {
		int antNum;         // Number of ants
		int* antRoute;      // Temp array
		float* Dist;     // Cost function input
		bool* foundRoute;   // Existence output
		float* Pheromone;
		int* route;         // Sequence output
		int size;        // Number of graph vertices
		curandState* state; // CURAND random state
	} Kernel_ParamTypedef;

	typedef struct {
		// Repetition constants
		int Repetitions;
		int Random_Generations;
		int Follower_Generations;
		int maxTryNumber;   // Follower ants use this to stop weighted roulette
		// Pheromone matrix constants
		float Initial_Pheromone_Value;
		float Rho;
		float Reward_Multiplier;
	} Kernel_ConfigParamTypedef;

	// Variables allocated in global memory for communication between different thread blocks
	// Either as extra function parameter or as a global variable (hence the name)
	typedef struct {
		bool invalidInput;   // Variable used for detecting invalid input
		bool isolatedVertex;  // Variable used for detecting isolated vertex (for optimization purposes)
		float averageDist;
		float minRes;    // Minimal found Route distance
	} Kernel_GlobalParamTypedef;

}