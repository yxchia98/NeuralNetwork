#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "assignment.h"
#define TXT_LINE_SIZE 45 //maximum number of chars in per line in .txt file
#define SIZE 100         //size of dataset
#define TRAINSIZE 90
#define TESTSIZE 10
#define LEARNING_RATE 0.05
#define NUM_INPUT 9
#define NUM_LAYER1 7

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
    static double trainingInput[TRAINSIZE][NUM_INPUT], trainingOutput[TRAINSIZE], testingInput[TESTSIZE][NUM_INPUT], testingOutput[TESTSIZE];
    static double input_weight[NUM_LAYER1][NUM_INPUT], layer1_weight[NUM_LAYER1];
    static double layer1_bias[NUM_LAYER1], output_bias;
    static double layer1_output[TRAINSIZE][NUM_LAYER1], layer1_summation[TRAINSIZE][NUM_LAYER1], output_error[TRAINSIZE], output_summation[TRAINSIZE];
    static double output_bias_update, layer1_weight_update[NUM_LAYER1], layer1_bias_update[NUM_LAYER1], input_weight_update[NUM_LAYER1][NUM_INPUT];
    char *filename = "fertility_Diagnosis_Data_Group1_4.txt";
    double sumAbsError, sumErrorSq, mae, mmse, sumBiasChange, layer1Sum, outputSum, current_error, error_output, error_layer1[NUM_LAYER1];
    int i, j, k, l, m;
    m = 1;
    read_txt(filename, trainingInput, trainingOutput, testingInput, testingOutput); // reads txt file and assigns it into txt_array
    for(i = 0; i < NUM_LAYER1; i++)
    {
        randWeight(input_weight[i], NUM_INPUT);
    }
    randWeight(layer1_weight, NUM_LAYER1);
    randWeight(layer1_bias, NUM_LAYER1);
    output_bias = randFrom(-1, 1);
    do
    {
        for(i = 0; i < TRAINSIZE; i++)
        {
            for(j = 0; j < NUM_LAYER1; j++)
            {
                for(k = 0; k < NUM_INPUT; k++)
                {
                    layer1Sum += trainingInput[i][k] * input_weight[j][k];
                }
                layer1Sum += layer1_bias[j];
                layer1_summation[i][j] = layer1Sum;
                layer1_output[i][j] = sigmoid(layer1Sum);
                layer1Sum = 0;

                outputSum += layer1_output[i][j] * layer1_weight[j];
            }
            outputSum += output_bias;
            output_summation[i] = outputSum;
            current_error = sigmoid(outputSum) - trainingOutput[i];
            output_error[i] = current_error;
            sumAbsError += fabs(current_error);
            sumErrorSq += pow(current_error,2);
            outputSum = 0;
        }
        mae = sumAbsError/90;
        mmse = sumErrorSq/90;
        printf("MAE for iteration %d is: %lf, MMSE is: %lf\n", m, mae, mmse);
        sumErrorSq = 0;
        sumAbsError = 0;
        m++;

        for(i = 0; i < TRAINSIZE; i++)
        {
            error_output = (output_error[i] * deSigmoid(output_summation[i])) / 90;
            output_bias_update += error_output;
            for(j = 0; j < NUM_LAYER1; j++)
            {
                error_layer1[j] = error_output * layer1_weight[j] * deSigmoid(layer1_summation[i][j]);
                layer1_bias_update[j] += error_layer1[j];
                layer1_weight_update[j] += error_output * layer1_output[i][j];
                for(k = 0; k < NUM_INPUT; k++)
                {
                    input_weight_update[j][k] += error_layer1[j] * trainingInput[i][k];
                }
            }
        }

        for(i = 0; i < NUM_LAYER1; i++)
        {
            layer1_weight[i] -= LEARNING_RATE * layer1_weight_update[i];
            layer1_bias[i] -= LEARNING_RATE * layer1_bias_update [i];
            layer1_weight_update[i] = 0;
            layer1_bias_update[i] = 0;
            for(j = 0; j < NUM_INPUT; j++)
            {
                input_weight[i][j] -= LEARNING_RATE * input_weight_update[i][j];
                input_weight_update[i][j] = 0;
            }
        }
        
    } while (mae);

    return 0;
}
