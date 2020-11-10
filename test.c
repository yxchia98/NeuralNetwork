#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "test1.h"
#include "yongting.h"
#include "MAE_MMSE.h"
#define TXT_LINE_SIZE 41 //maximum number of chars in per line in .txt file
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
    double bias, error, sumAbsError, sumErrorSq, mae, mmse, sumBiasChange, linear_regression_val, current_mae, current_error, delta;
    int i, j, k, l;
    k = 1;
    read_txt(filename, trainingInput, trainingOutput, testingInput, testingOutput); // reads txt file and assigns it into txt_array
    randWeight(weight, 9);
    bias = randFrom(-1, 1);
    do
    {
        for(int m =0; m<NUM_INPUT; m++){
            sumWeightChange[m] = 0;
        }
        for (i = 0; i < TRAINSIZE; i++)
        {
            linear_regression_val = linear_regression(trainingInput[i], weight, bias);
            sumErrorSq += m_m_s_e(sigmoid(linear_regression_val), testingOutput[i]);
            current_mae = m_a_e(sigmoid(linear_regression_val), testingOutput[i]);
            sumAbsError += current_mae;
            current_error = sigmoid(linear_regression_val) - testingOutput[i];
            for (j = 0; j < NUM_INPUT; j++)
            {
                sumWeightChange[j] += backward_propogation(current_error, trainingInput[i][j], linear_regression_val);
            }
            sumBiasChange += backward_propogation(current_error, 1, linear_regression_val);
        }
        mae = sumAbsError / TRAINSIZE;
        mmse = sumErrorSq / TRAINSIZE;
        printf("\nMAE of iteration %d is: %f", k, mae);
        //update weight
        for (int l = 0; l < NUM_INPUT; l++)
        {
            delta = sumWeightChange[l] / 90;
            weight[l] -= LEARNING_RATE * delta;
            //printf("\nWEIGHT %d value is: %f", l, weight[l]);
            sumWeightChange[l] = 0;
        }
        //update bias
        delta = sumBiasChange / 90;
        bias -= LEARNING_RATE * delta;
        sumBiasChange = 0;
        sumAbsError = 0;
        sumErrorSq = 0;
        k++;
    } while (mae > 0.0);

    return 0;
}
