#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "assignment.h"

//Reads from the .txt dataset file and assigns values in it into the corresponding attribute arrays
void read_txt(const char *filename, double trainingInput[TRAINSIZE][9], double trainingOutput[TRAINSIZE], double testingInput[TESTSIZE][9], double testingOutput[TESTSIZE])
{
    FILE *ptr;
    ptr=fopen(filename,"r");
    int i=0, k=0;
    char *token;
    if(ptr==NULL)
    {
        printf("fertility_Diagnosis_Data_Group1_4.txt file not found\n");
    }
    while(fscanf(ptr,"%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf",&trainingInput[i][0],&trainingInput[i][1],&trainingInput[i][2],&trainingInput[i][3],&trainingInput[i][4],&trainingInput[i][5],&trainingInput[i][6],&trainingInput[i][7],&trainingInput[i][8],&trainingOutput[i])!=EOF)
    {
        if(i==89)
        break;
        else
        i++;
    }
    while(fscanf(ptr,"%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf",&testingInput[k][0],&testingInput[k][1],&testingInput[k][2],&testingInput[k][3],&testingInput[k][4],&testingInput[k][5],&testingInput[k][6],&testingInput[k][7],&testingInput[k][8],&testingOutput[k])!=EOF)
    {
        k++;
    }
}

double sigmoid(double x)
{
    double result;
    result = 1 / (1 + exp(-x));
    return result;
}

void randWeight(double x[], int n)
{
    int i;
    for (i = 0; i < n; i++)
    {
        x[i] = randFrom(-1.0, 1.0);
    }
}

double randFrom(double min, double max)
{
    double range = (max - min);
    double div = RAND_MAX / range;
    return min + (rand() / div);
}