#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "assignment.h"

void feedforward(double trainingInput[][NUM_INPUT], double trainingOutput[TRAINSIZE], double input_weight[][NUM_INPUT], double layer1_weight[][NUM_LAYER1], double layer2_weight[NUM_LAYER2], double layer1_bias[NUM_LAYER1], double layer2_bias[NUM_LAYER2], double *output_bias, double layer1_output[][NUM_LAYER1], double layer1_summation[][NUM_LAYER1], double layer2_output[][NUM_LAYER2], double layer2_summation[][NUM_LAYER2], double output_error[TRAINSIZE], double output_summation[TRAINSIZE], double output[TRAINSIZE], double *sumAbsError, double *sumErrorSq, double *layer1Sum, double *layer2Sum, double *outputSum, double *current_error)
{
    int i,j,k,l;
    //FEEDFORAWRD PORTION
    for (i = 0; i < TRAINSIZE; i++)
    {
        for (j = 0; j < NUM_LAYER2; j++) //layer 2
        {
            for (k = 0; k < NUM_LAYER1; k++) //layer 1
            {
                for (l = 0; l < NUM_INPUT; l++) //input layer
                {
                    *layer1Sum += trainingInput[i][l] * input_weight[k][l];
                }
                *layer1Sum += layer1_bias[k];
                layer1_summation[i][k] = *layer1Sum;
                layer1_output[i][k] = sigmoid(*layer1Sum);
                *layer1Sum = 0;

                *layer2Sum += layer1_output[i][k] * layer1_weight[j][k];
            }
            *layer2Sum += layer2_bias[j];
            layer2_summation[i][j] = *layer2Sum;
            layer2_output[i][j] = sigmoid(*layer2Sum);
            *layer2Sum = 0;

            *outputSum += layer2_output[i][j] * layer2_weight[j];
        }
        *outputSum += *output_bias;
        output_summation[i] = *outputSum;
        output[i] = sigmoid(*outputSum);
        *current_error = output[i] - trainingOutput[i];
        output_error[i] = *current_error;
        *sumAbsError += fabs(*current_error);
        *sumErrorSq += pow(*current_error, 2);
        *outputSum = 0;
    }
}


void backpropagate_summation(double trainingInput[][NUM_INPUT], double layer1_weight[][NUM_LAYER1], double layer2_weight[NUM_LAYER2], double layer1_output[][NUM_LAYER1], double layer1_summation[][NUM_LAYER1], double layer2_output[][NUM_LAYER2], double layer2_summation[][NUM_LAYER2], double output_error[TRAINSIZE], double output_summation[TRAINSIZE], double *output_bias_update, double layer2_weight_update[NUM_LAYER2], double layer2_bias_update[NUM_LAYER2], double layer1_weight_update[][NUM_LAYER1], double layer1_bias_update[NUM_LAYER1], double input_weight_update[][NUM_INPUT])
{
    int i,j,k,l,n;
    double error_output, error_layer2[NUM_LAYER2], error_layer1[NUM_LAYER1];
    for (i = 0; i < TRAINSIZE; i++)
    {
        error_output = (output_error[i] * deSigmoid(output_summation[i])) / 90;     //calculate error at output neuron
        *output_bias_update += error_output;
        for (j = 0; j < NUM_LAYER2; j++)
        {
            error_layer2[j] = error_output * layer2_weight[j] * deSigmoid(layer2_summation[i][j]);  //calculate error at layer2 neurons
            layer2_weight_update[j] += error_output * layer2_output[i][j];
        }
        for (j = 0; j < NUM_LAYER2; j++)
        {
            for (k = 0; k < NUM_LAYER1; k++)
            {
                error_layer1[k] += error_layer2[j] * layer1_weight[j][k] * deSigmoid(layer1_summation[i][k]);   //sum up error linking to neurons in layer1
            }
        }
        for (j = 0; j < NUM_LAYER2; j++)
        {
            for (k = 0; k < NUM_LAYER1; k++)
            {
                layer1_weight_update[j][k] += error_layer2[j] * layer1_output[i][k];
                for (l = 0; l < NUM_INPUT; l++)
                {
                    input_weight_update[k][l] += error_layer1[k] * trainingInput[i][l];
                }
                
            }
            
        }
        for(j = 0; j < NUM_LAYER2; j++)
        {
            layer2_bias_update[j] += error_layer2[j];     
            error_layer2[j] = 0;       
        }
        for(j = 0; j < NUM_LAYER1; j++)
        {
            layer1_bias_update[j] += error_layer1[j];   
            error_layer1[j] = 0;
        }
    } 
}
void backpropagate_update(double input_weight[][NUM_INPUT], double layer1_weight[][NUM_LAYER1], double layer2_weight[NUM_LAYER2], double layer1_bias[NUM_LAYER1], double layer2_bias[NUM_LAYER2], double *output_bias, double *output_bias_update, double layer2_weight_update[NUM_LAYER2], double layer2_bias_update[NUM_LAYER2], double layer1_weight_update[][NUM_LAYER1], double layer1_bias_update[NUM_LAYER1], double input_weight_update[][NUM_INPUT])
{
    int i, j, k;
    for (i = 0; i < NUM_LAYER2; i++)
    {
        layer2_bias[i] -= LEARNING_RATE * layer2_bias_update[i];
        layer2_weight[i] -= LEARNING_RATE * layer2_weight_update[i];
        layer2_bias_update[i] = 0;
        layer2_weight_update[i] = 0;
        for (j = 0; j < NUM_LAYER1; j++)
        {
            layer1_bias[j] -= LEARNING_RATE * layer1_bias_update[j];
            layer1_bias_update[j] = 0;
            layer1_weight_update[i][j] -= LEARNING_RATE * layer1_weight_update[i][j];
            layer1_weight_update[i][j] = 0;
            for (k = 0; k < NUM_INPUT; k++)
            {
                input_weight[j][k] -= LEARNING_RATE * input_weight_update[j][k];
                input_weight_update[j][k] = 0;
            }
        }
    }
    *output_bias -= LEARNING_RATE * *output_bias_update;
    *output_bias_update = 0;
}

double deSigmoid(double x)
{
    double result;
    result = exp(x) / pow(1 + exp(x), 2);
    return result;
}

double sigmoid(double x)
{
    double result;
    result = 1 / (1 + exp(-x));
    return result;
}