#include <stdio.h>
#include "assignment.h"
void testWeights(int data_size, int confusionCount[4], double *mmse_ptr, double *mae_ptr, double input_arr[][NUM_INPUT], double *output_arr, double input_weight[NUM_LAYER1][NUM_INPUT], double layer1_weight[NUM_LAYER2][NUM_LAYER1], double layer2_weight[NUM_LAYER2], double layer1_bias[NUM_LAYER1], double layer2_bias[NUM_LAYER2], double *output_bias)
{
    static double layer1_output[TRAINSIZE][NUM_LAYER1], layer1_summation[TRAINSIZE][NUM_LAYER1], layer2_output[TRAINSIZE][NUM_LAYER2], layer2_summation[TRAINSIZE][NUM_LAYER2], output[TRAINSIZE];
    double sumAbsError, sumErrorSq, current_error, predictedY, mae, mmse, layer1Sum, layer2Sum, outputSum;
    sumErrorSq = 0;
    sumAbsError = 0;
    //confusion matrix of testing set, untrained weights
    //TESTING SET
    for (int i = 0; i < data_size; i++)
    {
        for (int j = 0; j < NUM_LAYER2; j++) //layer 2
        {
            for (int k = 0; k < NUM_LAYER1; k++) //layer 1
            {
                for (int l = 0; l < NUM_INPUT; l++) //input layer
                {
                    layer1Sum += input_arr[i][l] * input_weight[k][l]; //regression for input layer
                }
                layer1Sum += layer1_bias[k];              //add bias after weight*input portion
                layer1_summation[i][k] = layer1Sum;       //store summation to be used for backpropagation later
                layer1_output[i][k] = sigmoid(layer1Sum); //store output
                layer1Sum = 0;                            //reset sum for next row of data

                layer2Sum += layer1_output[i][k] * layer1_weight[j][k]; //regression for second layer
            }
            layer2Sum += layer2_bias[j];              //adding bais to finalize regression
            layer2_summation[i][j] = layer2Sum;       //store layer2 summation
            layer2_output[i][j] = sigmoid(layer2Sum); //store layer2 output
            layer2Sum = 0;                            //reset for next row of data

            outputSum += layer2_output[i][j] * layer2_weight[j]; //regression for output neuron
        }
        outputSum += *output_bias; //adding bias to finalize regression
        predictedY = sigmoid(outputSum);
        confusionMatrix(confusionCount, predictedY, output_arr[i]); //pass in predicted output for confusion matrix
        current_error = predictedY - output_arr[i];
        sumErrorSq += pow(current_error, 2);
        sumAbsError += fabs(current_error);
        /**************************************************************************************/
        outputSum = 0;
    }
    *mmse_ptr = sumErrorSq / data_size;
    *mae_ptr = sumAbsError / data_size;
}