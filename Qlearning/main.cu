#include <stdio.h>
#include <stdio.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <time.h>

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
void generateQtable(double **qTable)
{
	for (int i = 0; i < NUMBER_OF_STATES; i++)
	{
		for (int j = 0; j < NUMBER_OF_ACTIONS; j++)
		{
			qTable[i][j] = (double)rand() / (double)((unsigned)RAND_MAX + 1)* (0.1);
		}
	}
}
int getAction(double **qTable, int state)
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
int getRndAction(double **qTable, int state)
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
double updateQTable(int state, int action, int nextState,int reward,double** qTable )
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
__device__ double calculateSTD(int episodeCounter[],int stepTotal)
{
	double std;
	double mean = stepTotal / NUMBER_OF_EPISODES;
	for (int i = 0; i < NUMBER_OF_EPISODES; i++)
	{
		std += pow(episodeCounter[i] - mean, 2);
	}
	std = sqrt(std / NUMBER_OF_EPISODES);
	return std;
}
__global__ void calculateAllSteps(int **StepsArray)
{
	__shared__ int epsiodeCounter[NUMBER_OF_EPISODES];
	__shared__ int trialCounter[NUMBER_OF_TRIALS];
	__shared__ int total;
	double std[NUMBER_OF_TRIALS];
	total = 0; 
	for (int i = 0; i < NUMBER_OF_TRIALS;i++)
	{
		for (int j = 0; j < NUMBER_OF_EPISODES; j++)
		{
			total += StepsArray[i][j];
			epsiodeCounter[j] = StepsArray[i][j];
		}
		std[i] = calculateSTD(epsiodeCounter,total);
		trialCounter[i] = total;
		total = 0;
	}
}
int main()
{
	seedRandom();

	double **qTable;
	qTable = (double**)  malloc(NUMBER_OF_STATES * sizeof(double*));
	for (int i = 0; i < NUMBER_OF_STATES; i++)
	{
		qTable[i] = (double*)malloc(sizeof(double)*NUMBER_OF_ACTIONS);
	}
	generateQtable(qTable);
		/*for (int i = 0; i < NUMBER_OF_STATES; i++)
		{
			for (int j = 0; j < NUMBER_OF_ACTIONS; j++)
			{
				printf("vlaue at [%d][%d] : %f\n", i, j, qTable[i][j]);
			}
		}*/
	int **allSteps;
	allSteps = (int**)malloc(NUMBER_OF_TRIALS * sizeof(int*));
	for (int i = 0; i < NUMBER_OF_TRIALS; i++)
	{
		allSteps[i] = (int*)malloc(sizeof(int)* NUMBER_OF_EPISODES);
	}

	int state = 0;
	int action = 0;
	int newState = 0;
	int reward = 0;
	int steps = 0;
	double updatedValue = 0;
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
				//printf("state : %d , action : %d newState : %d\n", state, action, newState);
				state = newState;
				steps++;
			}
			allSteps[trial][i] = steps;
			steps = 0;
			
		}
		generateQtable(qTable);
	}

	for (int i = 0; i < NUMBER_OF_TRIALS; i++)
	{
		for (int j = 0; j < NUMBER_OF_EPISODES; j++)
		{
			printf("[%d][%d] : %d\n",i,j,allSteps[i][j]);
		}
	}

	/*for (int trial = 0; trial < NUMBER_OF_TRIALS; trial++)
	{
		for (int i = 0; i < NUMBER_OF_EPISODES; i++)
		{
			printf("number of steps at [%d][%d] : %d\n", trial, i, allSteps[trial][i]);
		}
	}*/
	free(allSteps);
	free(qTable);
}