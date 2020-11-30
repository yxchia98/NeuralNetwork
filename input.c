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
    int i=0, k=0;   //initiliaze both counters, i counter for training set, k counter for testing set.
    if(ptr==NULL)   //error checking for file not found
    {
        printf("fertility_Diagnosis_Data_Group1_4.txt file not found\n");
    }
    //use fscanf to scan inputs from the file into respective elements and arrays of training sets, delimited by a ',' separator
    while(fscanf(ptr,"%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf",&trainingInput[i][0],&trainingInput[i][1],&trainingInput[i][2],&trainingInput[i][3],&trainingInput[i][4],&trainingInput[i][5],&trainingInput[i][6],&trainingInput[i][7],&trainingInput[i][8],&trainingOutput[i])!=EOF)
    {
        if(i==89)   //once it hits the 90th row, break out of this loop and into the next loop
        break;
        else
        i++;
    }
    //utilizes fscanf to populate testing set, same methodology as the training set.
    while(fscanf(ptr,"%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf,%lf",&testingInput[k][0],&testingInput[k][1],&testingInput[k][2],&testingInput[k][3],&testingInput[k][4],&testingInput[k][5],&testingInput[k][6],&testingInput[k][7],&testingInput[k][8],&testingOutput[k])!=EOF)
    {
        k++;    //increment k, which will be used as index to populate the training set arrays
    }
}

//randWeight() takes in an array and randomizes every element in the array
void randWeight(double x[], int n)
{
    int i;
    for (i = 0; i < n; i++)
    {
        x[i] = randFrom(-1.0, 1.0);
    }
}

//randFrom() randomizes a double within -1 to 1
double randFrom(double min, double max)
{
    double range = (max - min);
    double div = RAND_MAX / range;
    return min + (rand() / div);
}