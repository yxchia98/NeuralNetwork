#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#define TXT_LINE_SIZE 41                    //maximum number of chars in per line in .txt file
#define SIZE 100                            //size of dataset
#define TRAINSIZE 90
#define TESTSIZE  10

void read_txt(char* filename, char c[TXT_LINE_SIZE], double trainingInput[TRAINSIZE][9], double trainingOutput[TRAINSIZE][1], double testingInput[TESTSIZE][9], double testingOutput[TESTSIZE][1]);
void train(double input[90][9], double outputarray[90][1]);
void randWeight(double x[], int n);
double randFrom(double min, double max);
void read_floatArray(float array[]);
void read_intArray(float array[]);
double sigmoid(double x);

//Reads from the .txt dataset file and assigns values in it into the corresponding attribute arrays
void read_txt(char* filename, char c[TXT_LINE_SIZE], double trainingInput[TRAINSIZE][9], double trainingOutput[TRAINSIZE][1], double testingInput[TESTSIZE][9], double testingOutput[TESTSIZE][1])
{
    int i=0, j=0, k;
    FILE *fp;                               //file pointer
    char *ptr;
    char delim[]=",";
    fp=fopen(filename,"r");                 //open file, read only
    if(fp==NULL)
    {
        printf("Could not open filename %s", filename);
        exit;
    }
    while(fgets(c,TXT_LINE_SIZE,fp)!=NULL)
    {
        //strcpy(txt_array[i],c);             //Copy line from .txt file into string array
        if(i<TRAINSIZE)
        {
            k=0;
            ptr=strtok(c,delim);
            while(ptr!=NULL)
            {
                switch (k)
                {
                    case 0:
                    trainingInput[i][0]=atof(ptr);
                    break;
                    case 1:
                    trainingInput[i][1]=atof(ptr);
                    break;
                    case 2:
                    trainingInput[i][2]=atoi(ptr);
                    break;
                    case 3:
                    trainingInput[i][3]=atoi(ptr);
                    break;
                    case 4:
                    trainingInput[i][4]=atoi(ptr);
                    break;
                    case 5:
                    trainingInput[i][5]=atoi(ptr);
                    break;
                    case 6:
                    trainingInput[i][6]=atof(ptr);
                    break;
                    case 7:
                    trainingInput[i][7]=atoi(ptr);
                    break;
                    case 8:
                    trainingInput[i][8]=atof(ptr);
                    break;
                    case 9:
                    trainingOutput[i][0]=atoi(ptr);
                    break;
                }
                k++;
                ptr=strtok(NULL,delim);
            }
            i++;
        }
        else
        {
            k=0;
            ptr=strtok(c,delim);
            while(ptr!=NULL)
            {
                switch (k)
                {
                    case 0:
                    testingInput[j][0]=atof(ptr);
                    break;
                    case 1:
                    testingInput[j][1]=atof(ptr);
                    break;
                    case 2:
                    testingInput[j][2]=atoi(ptr);
                    break;
                    case 3:
                    testingInput[j][3]=atoi(ptr);
                    break;
                    case 4:
                    testingInput[j][4]=atoi(ptr);
                    break;
                    case 5:
                    testingInput[j][5]=atoi(ptr);
                    break;
                    case 6:
                    testingInput[j][6]=atof(ptr);
                    break;
                    case 7:
                    testingInput[j][7]=atoi(ptr);
                    break;
                    case 8:
                    testingInput[j][8]=atof(ptr);
                    break;
                    case 9:
                    testingOutput[j][0]=atoi(ptr);
                    break;
                }
                k++;
                ptr=strtok(NULL,delim);
            }
            j++;
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
    result=1/(1+exp(-x));
    return result;
}

void randWeight(double x[],int n)
{
    int i;
    for(i=0;i<n;i++)
    {
        x[i]=randFrom(-1.0,1.0);
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
    int i=0;
    for(i=0;i<SIZE;i++)
    {
        printf("Attribute in [%d]: %f\n", i, array[i]);
    }
}
void read_intArray(float array[])
{
    int i=0;
    for(i=0;i<SIZE;i++)
    {
        printf("Attribute in [%d]: %f\n", i, array[i]);
    }
}