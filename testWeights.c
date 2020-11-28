#include <stdio.h>
#include "assignment.h"
void testWeights(int size, int confusionCount[4], double *mmse_ptr, double input_arr[][NUM_INPUT], double *output_arr, double input_weight[NUM_LAYER1][NUM_INPUT], double layer1_weight[NUM_LAYER2][NUM_LAYER1], double layer2_weight[NUM_LAYER2], double layer1_bias[NUM_LAYER1], double layer2_bias[NUM_LAYER2], double *output_bias)
{
    static double layer1_output[TRAINSIZE][NUM_LAYER1], layer1_summation[TRAINSIZE][NUM_LAYER1], layer2_output[TRAINSIZE][NUM_LAYER2], layer2_summation[TRAINSIZE][NUM_LAYER2], output[TRAINSIZE];
    double sumErrorSq, mae, mmse, layer1Sum, layer2Sum, outputSum;
    //confusion matrix of testing set, untrained weights
    //TESTING SET
    sumErrorSq = 0;
    for (int i = 0; i < size; i++)
    {
        for (int j = 0; j < NUM_LAYER2; j++) //layer 2
        {
            for (int k = 0; k < NUM_LAYER1; k++) //layer 1
            {
                for (int l = 0; l < NUM_INPUT; l++) //input layer
                {
                    layer1Sum += input_arr[i][l] * input_weight[k][l];
                }
                layer1Sum += layer1_bias[k];
                layer1_summation[i][k] = layer1Sum;
                layer1_output[i][k] = sigmoid(layer1Sum);
                layer1Sum = 0;

                layer2Sum += layer1_output[i][k] * layer1_weight[j][k];
            }
            layer2Sum += layer2_bias[j];
            layer2_summation[i][j] = layer2Sum;
            layer2_output[i][j] = sigmoid(layer2Sum);
            layer2Sum = 0;

            outputSum += layer2_output[i][j] * layer2_weight[j];
        }
        outputSum += *output_bias;
        double predictedY = sigmoid(outputSum);
        confusionMatrix(confusionCount, predictedY, output_arr[i]);
        sumErrorSq += pow(predictedY - output_arr[i], 2);
        /**************************************************************************************/
        outputSum = 0;
    }
    *mmse_ptr = sumErrorSq / size;
}