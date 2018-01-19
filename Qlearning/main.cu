#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <time.h>
#include "win-gettimeofday.h"


#define NUMBER_OF_STATES 11
#define NUMBER_OF_ACTIONS 4
#define NUMBER_OF_TRIALS 50
#define NUMBER_OF_EPISODES 200

void seedRandom()
{
	srand(time(NULL));
}
int transferFunction(int state, int action)
{
	int newState = 0;
	newState = state;

	switch (action)
	{
	case 0: 
		if (state == 0 || state == 1 || state == 2 || state == 3)
		{
			newState = state + 3;
			break;
		}
		else if (state == 4)
		{
			newState = state + 4;
			break;
		}
		else if (state == 5)
		{
			newState = state + 5; 
			break;
		}
	case 1:
		if (state == 6 || state == 7 || state == 8 || state == 9)
		{
			newState = state + 1;
			break;
		}
	case 2:
		if (state == 3 || state == 4 || state == 5 || state == 6)
		{
			newState = state - 3;
			break;
		}
		else if (state == 8)
		{
			newState = 4;
			break;
		}
		else if (state == 10)
		{
			newState = 5;
			break;
		}
	case 3:
		if (state == 10 || state == 7 || state == 8 || state == 9)
		{
			newState = state - 1;
			break;
		}
	}
	return newState;
}
void generateQtable(float **qTable)
{
	for (int i = 0; i < NUMBER_OF_STATES; i++)
	{
		for (int j = 0; j < NUMBER_OF_ACTIONS; j++)
		{
			qTable[i][j] = (double)rand() / (double)((unsigned)RAND_MAX + 1)* (0.1);
		}
	}
}
int getAction(float **qTable, int state)
{
	int action;
	double maxValue = -1;

	for (int i = 0; i < NUMBER_OF_ACTIONS; i++)
	{
		if (qTable[state][i] >maxValue)
		{
			maxValue = qTable[state][i];
				action = i;
		}
	}

	return action;
}
int getRndAction(float **qTable, int state)
{
	int action = 0;
	double maxValue = -1;
	int rndValue;
	rndValue = rand() % (10 + 1 - 1) + 1;
	if (rndValue <= 9)
	{
		for (int i = 0; i < NUMBER_OF_ACTIONS; i++)
		{
			if (qTable[state][i] > maxValue)
			{
				action = i;
				maxValue = qTable[state][i];
			}
		}
	}
	else
	{
		rndValue = rand() % (3 + 1 - 0) + 0;
		action = rndValue;
	}
	return action;
}
int getReward(int state, int action)
{
	int reward = 0;
	if (state == 4 && action == 2)
	{
		reward = 10;
	}
	return reward;
}
double updateQTable(int state, int action, int nextState,int reward, float** qTable )
{
	double updateValue = 0;
	int nextAction;
	double discount = 0.9;
	double learningRate =0.2;
	double diff;

	nextAction = getAction(qTable, nextState);
	diff = reward + discount * qTable[nextState][nextAction] - qTable[state][action];
	updateValue = qTable[state][action] + learningRate * diff;

	return updateValue;
}
int rndState()
{
	int state = 0;
	int Max = 10;
	state = rand() % Max;
	while (state == 1)
	{
		state = rand() % Max;
	}
	
	return state;
}
__device__ float calculateSTD(int episodeCounter[],int stepTotal,int size)
{
	float std;
	float mean = stepTotal / size;
	for (int i = 0; i < size; i++)
	{
		std += pow(episodeCounter[i] - mean, 2);
	}
	std = sqrt(std / size);
	return std;
}
__global__ void calculateAllSteps(int *StepsArray)
{
	__shared__ int episodeStorage[NUMBER_OF_EPISODES];
	__shared__ int trialStorage[NUMBER_OF_TRIALS];
 	__shared__ int total;
	__shared__ int *allSteps;
	allSteps = StepsArray;
	__shared__ int episodeTotal;
	__shared__ float std[NUMBER_OF_TRIALS];
	__shared__ int worstSteps;
	__shared__ int bestSteps;
	__shared__ int totalSteps;
	 bestSteps = 100000;
	 worstSteps = 0;
	total = 0; 
	episodeTotal = 0;
	totalSteps = 0;
	__syncthreads();
	for (int i = 0; i < NUMBER_OF_TRIALS;i++)
	{
		for (int j = 0; j < NUMBER_OF_EPISODES; j+=20)
		{
			/* gets the element from the array using the stride values for example if i is 2 and j is 1 then the index
			 would be 201 meaning it is the second episode of the second trial;*/
			total += allSteps[i*NUMBER_OF_EPISODES + j];
			total += allSteps[i*NUMBER_OF_EPISODES + j+1];
			total += allSteps[i*NUMBER_OF_EPISODES + j+2];
			total += allSteps[i*NUMBER_OF_EPISODES + j+3];
			total += allSteps[i*NUMBER_OF_EPISODES + j+4];
			total += allSteps[i*NUMBER_OF_EPISODES + j+5];
			total += allSteps[i*NUMBER_OF_EPISODES + j+6];
			total += allSteps[i*NUMBER_OF_EPISODES + j+7];
			total += allSteps[i*NUMBER_OF_EPISODES + j+8];
			total += allSteps[i*NUMBER_OF_EPISODES + j+9];
			total += allSteps[i*NUMBER_OF_EPISODES + j+10];
			total += allSteps[i*NUMBER_OF_EPISODES + j+11];
			total += allSteps[i*NUMBER_OF_EPISODES + j+12];
			total += allSteps[i*NUMBER_OF_EPISODES + j+13];
			total += allSteps[i*NUMBER_OF_EPISODES + j+14];
			total += allSteps[i*NUMBER_OF_EPISODES + j+15];
			total += allSteps[i*NUMBER_OF_EPISODES + j+16];
			total += allSteps[i*NUMBER_OF_EPISODES + j+17];
			total += allSteps[i*NUMBER_OF_EPISODES + j+18];
			total += allSteps[i*NUMBER_OF_EPISODES + j+19];
			episodeStorage[j] = allSteps[i*NUMBER_OF_EPISODES + j];
			episodeStorage[j] = allSteps[i*NUMBER_OF_EPISODES + j+1];
			episodeStorage[j] = allSteps[i*NUMBER_OF_EPISODES + j+2];
			episodeStorage[j] = allSteps[i*NUMBER_OF_EPISODES + j+3];
			episodeStorage[j] = allSteps[i*NUMBER_OF_EPISODES + j+4];
			episodeStorage[j] = allSteps[i*NUMBER_OF_EPISODES + j+5];
			episodeStorage[j] = allSteps[i*NUMBER_OF_EPISODES + j+6];
			episodeStorage[j] = allSteps[i*NUMBER_OF_EPISODES + j+7];
			episodeStorage[j] = allSteps[i*NUMBER_OF_EPISODES + j+8];
			episodeStorage[j] = allSteps[i*NUMBER_OF_EPISODES + j+9];
			episodeStorage[j] = allSteps[i*NUMBER_OF_EPISODES + j+10];
			episodeStorage[j] = allSteps[i*NUMBER_OF_EPISODES + j+11];
			episodeStorage[j] = allSteps[i*NUMBER_OF_EPISODES + j+12];
			episodeStorage[j] = allSteps[i*NUMBER_OF_EPISODES + j+13];
			episodeStorage[j] = allSteps[i*NUMBER_OF_EPISODES + j+14];
			episodeStorage[j] = allSteps[i*NUMBER_OF_EPISODES + j+15];
			episodeStorage[j] = allSteps[i*NUMBER_OF_EPISODES + j+16];
			episodeStorage[j] = allSteps[i*NUMBER_OF_EPISODES + j+17];
			episodeStorage[j] = allSteps[i*NUMBER_OF_EPISODES + j+18];
			episodeStorage[j] = allSteps[i*NUMBER_OF_EPISODES + j+19];
		}
		std[i] = calculateSTD(episodeStorage,total,NUMBER_OF_EPISODES);
		trialStorage[i] = total;
		totalSteps += total;
		episodeTotal += (total / 200);
		total = 0;
	}

	float aveStd = 0;
	float mean = totalSteps / NUMBER_OF_TRIALS;
	for (int i = 0; i < NUMBER_OF_TRIALS; i++)
	{
		aveStd += pow(std[i] - mean, 2);
	}
	aveStd = sqrt(aveStd / NUMBER_OF_TRIALS);

	for (int i = 0; i < NUMBER_OF_EPISODES*NUMBER_OF_TRIALS; i++)
	{
		if (bestSteps > allSteps[i])
		{
			bestSteps = allSteps[i];
		}
		if (worstSteps < allSteps[i])
		{
			worstSteps = allSteps[i];
		}
		__syncthreads();
	}

	__syncthreads();
	int block_id = blockIdx.x + gridDim.x * blockIdx.y;
	unsigned int id = blockDim.x * block_id + threadIdx.x;
	//printf("id : %d\n", id);
	if (id == 1)
	{
		printf("Data from trials \n\n");
		printf("avarge Standard deviation : %f\n", aveStd);
		printf("Best value %d\n", bestSteps);
		printf("worst value %d\n", worstSteps);
		printf("Avarage steps per tiral : %d\n",episodeTotal);
		printf("Avarage steps per episode : %d\n", episodeTotal/50);
		printf("\n\n");
	}
}
int main()
{
	long long totalProgrammeTimmer = start_timer();
	seedRandom();
	int size = sizeof(int) * NUMBER_OF_EPISODES * NUMBER_OF_TRIALS;
	int returnSize = sizeof(int) * 4;
	float **qTable;
	qTable = (float**)  malloc(NUMBER_OF_STATES * sizeof(float*));
	for (int i = 0; i < NUMBER_OF_STATES; i++)
	{
		qTable[i] = (float*)malloc(sizeof(float)*NUMBER_OF_ACTIONS);
	}
	generateQtable(qTable);
		
	int *allSteps;
	allSteps = (int*)malloc((NUMBER_OF_TRIALS * NUMBER_OF_EPISODES)* sizeof(int));
	/*allSteps[0] = (int*)malloc(NUMBER_OF_EPISODES * sizeof(int));
	for (size_t i = 1; i < NUMBER_OF_TRIALS; i++)
	{
		allSteps[i] = allSteps[i - 1] + NUMBER_OF_EPISODES;
	}*/
	/*for (int i = 0; i < NUMBER_OF_TRIALS; i++)
	{
		allSteps[i] = (int*)malloc(sizeof(int)* NUMBER_OF_EPISODES);
	}*/
	

	int state = 0;
	int action = 0;
	int newState = 0;
	int reward = 0;
	int steps = 0;
	float updatedValue = 0;

	for (int trial = 0; trial < NUMBER_OF_TRIALS; trial++)
	{
		for (int i = 0; i < NUMBER_OF_EPISODES; i++)
		{
			state = rndState();
			while (state != 1)
			{
				action = getRndAction(qTable, state);
				reward = getReward(state, action);
				newState = transferFunction(state, action);
				updatedValue = updateQTable(state, action, newState, reward, qTable);
				qTable[state][action] = updatedValue;
				
				state = newState;
				steps++;
			}
			
			allSteps[trial*NUMBER_OF_EPISODES + i] = steps;
			
			steps = 0;
			
		}
		generateQtable(qTable);
	}


	/*int *returnData;
	returnData =(int*) malloc(4 * sizeof(int));*/
	int *d_allsteps;
	//int *d_returnData;
	long long transferTimmer = start_timer();

	cudaStream_t stream1;
	cudaError_t err;
	err = cudaStreamCreate(&stream1);

	printf("CUDA Stream: %s\n", cudaGetErrorString(err));

	err = cudaMalloc((void**)&d_allsteps, size);
	
	printf("CUDA malloc 1D array: %s\n", cudaGetErrorString(err));

	//err = cudaMalloc((void**)&d_returnData, returnSize);

	//printf("CUDA malloc 1D array: %s\n", cudaGetErrorString(err));

	err = cudaMemcpyAsync(d_allsteps, allSteps,size, cudaMemcpyHostToDevice,stream1);

	printf("CUDA memcpy 1D array: %s\n", cudaGetErrorString(err));

	//err = cudaMemset(d_returnData, 0, returnSize); //(d_returnData,returnData,returnSize,cudaMemcpyHostToDevice);

	//printf("CUDA memcpy 1D array of nothing: %s\n", cudaGetErrorString(err));

	stop_timer(transferTimmer,"Transfer timer");

	long long GPUComput = start_timer();
	int gridSize = (int)ceil(NUMBER_OF_EPISODES*NUMBER_OF_TRIALS / 50);
	dim3 dimGrid(gridSize,1,1);
	dim3 dimBlock(50,1,1);

	calculateAllSteps << <gridSize, 50,0 ,stream1 >> > (d_allsteps);

	//err = cudaMemcpy(returnData, d_returnData, returnSize, cudaMemcpyDeviceToHost);

	//printf("CUDA memcpy data back: %s\n", cudaGetErrorString(err));
	stop_timer(GPUComput, "Computaion time");
	/*printf("Best value %d\n", returnData[2]);
	printf("worst value %d\n", returnData[3]);
	printf("Avarage steps per tiral : %d\n", returnData[0]);
	printf("Avarage steps per episode : %d\n", returnData[1]);*/
	stop_timer(totalProgrammeTimmer, "Total programme time");

	cudaFree(d_allsteps);

	//cudaFree(d_returnData);

	free(allSteps);

	free(qTable);
}