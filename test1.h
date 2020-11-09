#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#define TXT_LINE_SIZE 45 //maximum number of chars in per line in .txt file
#define SIZE 100         //size of dataset
#define TRAINSIZE 90
#define TESTSIZE 10

void read_txt(const char *filename, char c[TXT_LINE_SIZE], double trainingInput[TRAINSIZE][9], double trainingOutput[TRAINSIZE], double testingInput[TESTSIZE][9], double testingOutput[TESTSIZE]);
void train(double input[90][9], double outputarray[90][1]);
void randWeight(double x[], int n);
double randFrom(double min, double max);
void read_floatArray(float array[]);
void read_intArray(float array[]);
double sigmoid(double x);

//Reads from the .txt dataset file and assigns values in it into the corresponding attribute arrays
void read_txt(const char *filename, char c[TXT_LINE_SIZE], double trainingInput[TRAINSIZE][9], double trainingOutput[TRAINSIZE], double testingInput[TESTSIZE][9], double testingOutput[TESTSIZE])
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

void read_floatArray(float array[])
{
    int i = 0;
    for (i = 0; i < SIZE; i++)
    {
        printf("Attribute in [%d]: %f\n", i, array[i]);
    }
}
void read_intArray(float array[])
{
    int i = 0;
    for (i = 0; i < SIZE; i++)
    {
        printf("Attribute in [%d]: %f\n", i, array[i]);
    }
}