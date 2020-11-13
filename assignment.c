#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "assignment.h"
#define SIZE 100         //size of dataset
#define TRAINSIZE 90
#define TESTSIZE 10
#define LEARNING_RATE 0.05
#define NUM_INPUT 9

/**********************************************************************************************
Basic elements inside Input arrays(trainingInput/testingInput)          Datatype(range)
[0]=Season of analysis                                                  Double(-1,-0.33,0.33,1)
[1]=Age of analysis                                                     Double(0-1)
[2]=Childish disease                                                    Int(0,1)
[3]=Accident or serious trauma                                          Int(0,1)
[4]=Surgical intervention                                               Int(0,1)
[5]=High fevers last year                                               Int(-1,0,1)
[6]=Frequency of alcohol consumption                                    Double(0.2,0.4,0.6,0.8,1)
[7]=Smoking habit                                                       Int(-1,0,1)
[8]=Number of hours spent sitting per day                               Double(0-1)
--------------------------------------------------------------------------------------------------
Basic elements inside Output arrays(trainingOutput/testingOutput)       Datatype(range)
[0]=Semen Diagnosis                                                     Int(0,1)
***********************************************************************************************/

int main()
{
    static double weight[NUM_INPUT], trainingInput[TRAINSIZE][NUM_INPUT], trainingOutput[TRAINSIZE], testingInput[TESTSIZE][NUM_INPUT], testingOutput[TESTSIZE], sumWeightChange[NUM_INPUT];
    char *filename = "fertility_Diagnosis_Data_Group1_4.txt";
    FILE *plotptr;
    double bias, error, mae_summation, untrained_mae, untrained_mmse, mmse_summation, mae, mmse, sumBiasChange, linear_regression_val, current_error, delta;
    int i, j, k, l;
    clock_t start, elapsed;
    k = 1;
    read_txt(filename, trainingInput, trainingOutput, testingInput, testingOutput); // reads txt file and assigns it into txt_array
    randWeight(weight, 9);
    bias = randFrom(-1, 1);
    start = clock();
    if((plotptr = fopen("MAEGraph.txt","w"))==NULL)
    {
        printf("\nMAEGraph.txt does not exist.");
        exit(1);
    }
    do
    {
        linear_regression_val=0;
        for (i = 0; i < TRAINSIZE; i++)
        {
            linear_regression_val = linear_regression(trainingInput[i], weight, bias);
            current_error = sigmoid(linear_regression_val) - trainingOutput[i];
            mmse_summation += pow(current_error, 2);
            mae_summation += fabs(current_error);
            sumBiasChange += backward_propogation(current_error, linear_regression_val, 1);
            for (j = 0; j < NUM_INPUT; j++)
            {
                sumWeightChange[j] += backward_propogation(current_error, linear_regression_val, trainingInput[i][j]);
            }
        }
        mmse = mmse_summation / TRAINSIZE;
        mae = mae_summation / TRAINSIZE;
        if(k==1)
        {
            untrained_mmse = mmse;
            untrained_mae = mae;
        }
        printf("\nAt Iteration %d, MAE is: %lf, MMSE is: %lf ", k, mae, mmse);
        fprintf(plotptr,"%lf\n", mae);
        //update weight
        for (int l = 0; l < NUM_INPUT; l++)
        {
            delta = sumWeightChange[l] / TRAINSIZE;
            weight[l] -= LEARNING_RATE * delta;
            //printf("\nWEIGHT %d value is: %f", l, weight[l]);
            sumWeightChange[l] = 0;
        }
        //update bias
        delta = sumBiasChange / 90;
        bias -= LEARNING_RATE * delta;
        sumBiasChange = 0;
        mmse_summation = 0;
        mae_summation = 0;
        k++;
    } while (mae > 0.25);
    elapsed = (clock() - start)*1000/CLOCKS_PER_SEC;
    printf("\nUntrained MAE is: %lf, untrained MMSE is: %lf", untrained_mae, untrained_mmse);
    printf("\nTime taken: %dms", elapsed);
    fclose(plotptr);

    return 0;
}
