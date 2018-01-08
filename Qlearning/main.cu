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
int getAction()
{
	int action;
	return action;
}
int main()
{

}