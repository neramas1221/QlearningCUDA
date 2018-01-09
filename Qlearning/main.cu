#include <stdio.h>
#include <stdio.h>
#include <math.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <time.h>

#define NUMBER_OF_STATES 11
#define NUMBER_OF_ACTIONS 4


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
		if (state == 6 || state == 3 || state == 5 || state == 4)
		{
			newState = state - 3;
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
			qTable[i][j] = rand();
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
	int maxValue = -1;
	// rnd number betwene 1-10
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
int getReward(int state, int action)
{
	int reward = 0;
	if (state == 4 && action == 1)
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
	double learningRate;
	double diff;

	nextAction = getAction(qTable, nextState);
	diff = reward + discount * qTable[nextState][nextAction] - qTable[state][action];
	updateValue = qTable[state][action] + learningRate * diff;

	return updateValue;
}
int main()
{

}