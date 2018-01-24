#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <time.h>
#include "win-gettimeofday.h"

// number of states
#define NUMBER_OF_STATES 11
// number of actions
#define NUMBER_OF_ACTIONS 4
// number of trials the agent will proform
#define NUMBER_OF_TRIALS 50
// number of epiosdes the agent will proform in each trial
#define NUMBER_OF_EPISODES 200

/*				 ___________________
GRID WORLD USED | 6 | 7 | 8 | 9 | 10|
				| 3 |   | 4 |   | 5 |
				| 0 |   | 1 |   | 2 |
*/

///////////////////////////////////////////////
//				seedRandom  				 //
// seedRanom is only used to seed the random //
// generator at the start of the programe	 //
///////////////////////////////////////////////

void seedRandom()
{
	//seeds the number generator to the system clock
	srand(time(NULL));
}

///////////////////////////////////////////////
//				transfer Function			 //
// transfer function is used to get the		 //
// next state after the agent proformes its	 //
// action in its current state				 //
// state : current state the agent is in	 //
// action : the move that the agent is going //
// to make in this grid world				 //
///////////////////////////////////////////////

int transferFunction(int state, int action)
{
	// creates value to store new state
	int newState = 0;
	// sets new state to state meaning if the agent does not
	// move its new state = its old state
	newState = state;
	// switch based on the action
	switch (action)
	{
	case 0: 
		// checks if this action can be proformed in one of the states, if it 
		// can be proformed move the agent to its new state if not then dont
		// move the agent
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
	// returns new state
	return newState;
}

///////////////////////////////////////////////
//				generateQtable				 //
// generate Qtable is used to create an 11 x //
// 4 table of values between 0.1-0.01		 //
// qTable : 2-D array that is filled with the//
// Q-values which corresponds to the state	 //
// and action								 //
///////////////////////////////////////////////

void generateQtable(float **qTable)
{
	// loop over the number of states
	for (int i = 0; i < NUMBER_OF_STATES; i++)
	{ // loops over the number of actions
		for (int j = 0; j < NUMBER_OF_ACTIONS; j++)
		{
			// geneates a random double between 0.1 -0.01
			qTable[i][j] = (double)rand() / (double)((unsigned)RAND_MAX + 0.1)* (0.01);
		}
	}
}

///////////////////////////////////////////////
//				getAction					 //
// get Action will return an action that the //
// agent will proform in its state. This is  //
// tipically the highest Q-value for that	 //
// state									 //
// state : the state the agent is in		 //
// qTable : 2-D array that is filled with	 //
// the Q-Values of states and actions		 //
///////////////////////////////////////////////

int getAction(float **qTable, int state)
{
	int action;
	double maxValue = -1;
	// loops over the number of actions
	for (int i = 0; i < NUMBER_OF_ACTIONS; i++)
	{
		// finds the max value in the qTable
		if (qTable[state][i] >maxValue)
		{
			// sets the max value to this Q- value
			maxValue = qTable[state][i];
			// sets the action to the best action
				action = i;
		}
	}
	// returns the action
	return action;
}

///////////////////////////////////////////////
//				getAction					 //
// get Action will return an action that the //
// agent will proform in its state. This is  //
// tipically the highest Q-value for that	 //
// state how every there is a 1/10 chance	 //
// that a random action will be chosen for	 //
// the agent to proform						 //
// state : the state the agent is in		 //
// qTable : 2-D array that is filled with	 //
// the Q-Values of states and actions		 //
///////////////////////////////////////////////

int getRndAction(float **qTable, int state)
{
	int action = 0;
	double maxValue = -1;
	int rndValue;
	// gets a random value for the exploration rate
	rndValue = rand() % (10 + 1 - 1) + 1;
	// if its not exploring
	if (rndValue <= 9)
	{
		// loops over the action
		for (int i = 0; i < NUMBER_OF_ACTIONS; i++)
		{
			// checks the best value against the current Q-value
			if (qTable[state][i] > maxValue)
			{
				// sets the action to the best action
				action = i;
				// updates the max value
				maxValue = qTable[state][i];
			}
		}
	}
	// else explore
	else
	{
		// randomly generates an action
		rndValue = rand() % (3 + 1 - 0) + 0;
		// sets the action
		action = rndValue;
	}
	return action;
}

///////////////////////////////////////////////
//				getReward					 //
// get reward is used to give the agent its  //
// reward based on its state and action it	 //
// would take								 //
///////////////////////////////////////////////

int getReward(int state, int action)
{
	int reward = 0;
	// checks if the action and state will
	// lead to the goal state being reach
	if (state == 4 && action == 2)
	{
		// set reward to 10
		reward = 10;
	}

	// return the reward
	return reward;
}

///////////////////////////////////////////////
//				update Qtable   			 //
// update Q-Table is used to update the last //
// move the agent profomed in the grid world //
// using its reward, next state and currrent //
// state and action	and updates the Q value  //
// for that move							 //
// qTable : is an array that stores all of   //
// actions q - values for a given state		 //
// state : is the current state the agent is //
// in										 //
// action : the action the agent profomed    //
// in its given state						 //
// newState : is the state the agent ended	 //
// up in based on its state and action		 //
// that the agent proformed					 //
// reward : reward that the agent earns		 //
// based on its state and action			 //
///////////////////////////////////////////////

double updateQTable(int state, int action, int nextState,int reward, float** qTable )
{
	double updateValue = 0;
	int nextAction;
	// discounts the learning by 0.9
	double discount = 0.9;
	double learningRate =0.2;
	// used to store the diffrence in the Q-table values
	double diff;
	// gets the best action that is mostlikely going to be proformed
	// in the agents next state
	nextAction = getAction(qTable, nextState);
	// calculates how good the move is
	diff = reward + discount * qTable[nextState][nextAction] - qTable[state][action];
	// work out the new Q values using the learning rate and the diffrence
	updateValue = qTable[state][action] + learningRate * diff;
	return updateValue;
}

///////////////////////////////////////////////
//				rndState		   			 //
// rnd state is used to randomlly generate a //
// starting state that is not the goal state //
///////////////////////////////////////////////

int rndState()
{
	int state = 0;
	int Max = 10;
	// calculates a random number betwen 0 - 10
	state = rand() % Max;
	// checks the state is not the goal state
	while (state == 1)
	{
		// generates a new state if it is
		state = rand() % Max;
	}
	
	return state;
}

///////////////////////////////////////////////
//				calculateSTD	  			 //
// calculate STD is used to calculates the	 //
// standard deviation on each trial the		 //
// agent has proformed, this method is only	 //
// run on the device						 //
// episodeCount : stores all of the episodes //
// that the agent proformed in this trial	 //
// total steps is the total number of steps	 //
// for that trial							 //
// size  : is the number of episodes the	 //
// agent completed							 //
///////////////////////////////////////////////

__device__ float calculateSTD(int episodeCounter[],int stepTotal,int size)
{
	float std;
	// gets the mean of the data
	float mean = stepTotal / size;
	// loops over the data
	for (int i = 0; i < size; i++)
	{
		// calculates the total of the data set squared
		std += pow(episodeCounter[i] - mean, 2);
	}
	// square roots the data devided by the size
	std = sqrt(std / size);
	return std;
}

///////////////////////////////////////////////
//				calculateAllSteps  			 //
// calculate all steps: is used to sum all	 //
// steps and there standard deviation per	 //
// trial									 //
///////////////////////////////////////////////

__global__ void calculateAllSteps(int *StepsArray)
{
	// creates arrrys to store the total number of steps in
	// the each trial and the steps in all episodes
	 int episodeStorage[NUMBER_OF_EPISODES];
	 int trialStorage[NUMBER_OF_TRIALS];
 	__shared__ int total;
	__shared__ int *allSteps;
	// sets the array the kernal is passed to one in shared memory
	allSteps = StepsArray;
	__shared__ int episodeAvrage;
	// creates array to store the standard deviation
	float std[NUMBER_OF_TRIALS];
	__shared__ int worstSteps;
	__shared__ int bestSteps;
	__shared__ int totalSteps;
	 bestSteps = 100000;
	 worstSteps = 0;
	total = 0; 
	episodeAvrage = 0;
	totalSteps = 0;
	// syncthreads to make sure all data has been created
	__syncthreads();
	// loops over the number of trials
	for (int i = 0; i < NUMBER_OF_TRIALS;i++)
	{
		// loops over the number of episodes
		for (int j = 0; j < NUMBER_OF_EPISODES; j+=20)
		{
			/* gets the element from the array using the stride values for example if i is 2 and j is 1 then the index
			 would be 201 meaning it is the second episode of the second trial;*/
			/*
			 a number of diffrent unravling amounts where used here and 20 seemed to yeld the best
			 results in terms of proformance and computation time.
			*/
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
		// calculates the STD of the episode
		std[i] = calculateSTD(episodeStorage,total,NUMBER_OF_EPISODES);
		// stores the total number of steps accross this episode
		trialStorage[i] = total;
		// adds to the total steps across all trials
		totalSteps += total;
		// works out the avrage number of steps for that episode
		episodeAvrage += (total / NUMBER_OF_EPISODES);
		total = 0;
	}
	// calculates the avrage standard deviation
	float aveStd = 0;
	float mean = totalSteps / NUMBER_OF_TRIALS;
	for (int i = 0; i < NUMBER_OF_TRIALS; i++)
	{
		aveStd += pow(std[i] - mean, 2);
	}
	aveStd = sqrt(aveStd / NUMBER_OF_TRIALS);
	// loops over the data set
	for (int i = 0; i < NUMBER_OF_EPISODES*NUMBER_OF_TRIALS; i++)
	{
		// finds the best and worst number of steps from all of the
		// episodes the agent proformed
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
	// makes sure all of the threads have finished
	__syncthreads();
	// works out the threads block id
	int block_id = blockIdx.x + gridDim.x * blockIdx.y;
	// works out the threads id
	unsigned int id = blockDim.x * block_id + threadIdx.x;
	if (id == 1)
	{
		// thread one prints all of the data to the user so that no time is spent 
		// tranfering data back to the host when it is not going to be used
		printf("Data from trials \n\n");
		printf("avarge Standard deviation : %f\n", aveStd);
		printf("Best value %d\n", bestSteps);
		printf("worst value %d\n", worstSteps);
		printf("Avarage steps per tiral : %d\n",episodeAvrage);
		printf("Avarage steps per episode : %d\n", episodeAvrage/50);
		printf("\n\n");
	}
}
int main()
{
	// starts the programme timmer
	long long totalProgrammeTimmer = start_timer();
	// seeds the random number generator
	seedRandom();
	// sets the size of the arrys
	int size = sizeof(int) * NUMBER_OF_EPISODES * NUMBER_OF_TRIALS;
	// creates the qTable
	float **qTable;
	// allocates space to the q-table 
	qTable = (float**)  malloc(NUMBER_OF_STATES * sizeof(float*));
	// loops over the q-table and allocates space to each row equal to the number
	// of actions
	for (int i = 0; i < NUMBER_OF_STATES; i++)
	{
		qTable[i] = (float*)malloc(sizeof(float)*NUMBER_OF_ACTIONS);
	}
	// generates the q-table
	generateQtable(qTable);
		// creates array to store all steps across trials
	int *allSteps;
	// allocates space to the array 
	allSteps = (int*)malloc(size);
	// creates the data to store the values for the
	// q-learning algorithum
	int state = 0;
	int action = 0;
	int newState = 0;
	int reward = 0;
	int steps = 0;
	float updatedValue = 0;
	// loops over the number of trials
	for (int trial = 0; trial < NUMBER_OF_TRIALS; trial++)
	{
		// loops over the number of episodes
		for (int i = 0; i < NUMBER_OF_EPISODES; i++)
		{
			// randomly generates a starting state
			state = rndState();
			// loops until goal state is reached
			while (state != 1)
			{
				// gets the action for the state
				action = getRndAction(qTable, state);
				// gets the agents reward for the move
				reward = getReward(state, action);
				// gets the resulting state
				newState = transferFunction(state, action);
				// gets the new Q-value for that state and action
				updatedValue = updateQTable(state, action, newState, reward, qTable);
				// sets q-table at the state and action to be the new value
				qTable[state][action] = updatedValue;
				// updates the state
				state = newState;
				// updates the step count
				steps++;
			}
			// updates the array to have the total steps for that episode
			allSteps[trial*NUMBER_OF_EPISODES + i] = steps;
			// resest the steps
			steps = 0;
			
		}
		// generates a new qTable after each trial
		generateQtable(qTable);
	}
	// creates an aray to store all steps on the device
	int *d_allsteps;
	// startes the transfer timmer
	long long transferTimmer = start_timer();

	// creates error variable
	cudaError_t err;
	// mallocs space for the data on the device
	err = cudaMalloc((void**)&d_allsteps, size);
	
	printf("CUDA malloc 1D array: %s\n", cudaGetErrorString(err));
	// copys the data over to the device
	err = cudaMemcpy(d_allsteps, allSteps, size, cudaMemcpyHostToDevice);
	printf("CUDA memcpy 1D array: %s\n", cudaGetErrorString(err));
	// stops timmer
	stop_timer(transferTimmer,"Transfer timer");
	// starts computation timer
	long long GPUComput = start_timer();
	// calculates the grid size
	int gridSize = (int)ceil(NUMBER_OF_EPISODES*NUMBER_OF_TRIALS / NUMBER_OF_TRIALS);
	dim3 dimGrid(gridSize,1,1);
	// runs the kernal with the calculated block size and 50 threads
	calculateAllSteps << <gridSize, 50>> > (d_allsteps);
	// stops the timer
	stop_timer(GPUComput, "Computaion time");
	// stopes the total programme timmer
	stop_timer(totalProgrammeTimmer, "Total programme time");
	// frees memory
	cudaFree(d_allsteps);

	free(allSteps);

	free(qTable);
}