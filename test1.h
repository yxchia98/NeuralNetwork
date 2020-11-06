#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#define TXT_LINE_SIZE 41 //maximum number of chars in per line in .txt file
#define SIZE 100         //size of dataset
#define TRAINSIZE 90
#define TESTSIZE 10

void read_txt(char *filename, char c[TXT_LINE_SIZE], double trainingInput[TRAINSIZE][9], double trainingOutput[TRAINSIZE], double testingInput[TESTSIZE][9], double testingOutput[TESTSIZE]);
void train(double input[90][9], double outputarray[90][1]);
void randWeight(double x[], int n);
double randFrom(double min, double max);
void read_floatArray(float array[]);
void read_intArray(float array[]);
double sigmoid(double x);

//Reads from the .txt dataset file and assigns values in it into the corresponding attribute arrays
void read_txt(char *filename, char c[TXT_LINE_SIZE], double trainingInput[TRAINSIZE][9], double trainingOutput[TRAINSIZE], double testingInput[TESTSIZE][9], double testingOutput[TESTSIZE])
{
    int i = 0, k = 0;
    FILE *fp; //file pointer
    char *ptr;
    char delim[] = ",";
    fp = fopen(filename, "r"); //open file, read only
    if (fp == NULL)
    {
        printf("Could not open filename %s", filename);
        exit;
    }
    while (fgets(c, TXT_LINE_SIZE, fp) != NULL)
    {
        if (i < TRAINSIZE)
        {
            ptr = strtok(c, delim);
            while (ptr != NULL)
            {
                for (int j = 0; j < 9; j++)
                {
                    trainingInput[i][j] = atof(ptr);
                }
                trainingOutput[i] = atof(ptr);
                ptr = strtok(NULL, delim);
            }
            i++;
        }
        else
        {
            ptr = strtok(c, delim);
            while (ptr != NULL)
            {
                for (int j = 0; j < 9; j++)
                {
                    trainingInput[k][j] = atof(ptr);
                }
                trainingOutput[k] = atof(ptr);
                ptr = strtok(NULL, delim);
            }
            k++;
        }
    }
    fclose(fp);
}

void train(double inputarray[SIZE][9], double outputarray[SIZE][1])
{
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
    for(i=0;i<n;i++)
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