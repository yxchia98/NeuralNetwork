#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "assignment.h"

//FEED FORWARD PORTION
void feedforward(double trainingInput[][NUM_INPUT], double trainingOutput[TRAINSIZE], double input_weight[][NUM_INPUT], double layer1_weight[][NUM_LAYER1], double layer2_weight[NUM_LAYER2], double layer1_bias[NUM_LAYER1], double layer2_bias[NUM_LAYER2], double *output_bias, double layer1_output[][NUM_LAYER1], double layer1_summation[][NUM_LAYER1], double layer2_output[][NUM_LAYER2], double layer2_summation[][NUM_LAYER2], double output_error[TRAINSIZE], double output_summation[TRAINSIZE], double *sumAbsError, double *sumErrorSq, double *layer1Sum, double *layer2Sum, double *outputSum, double *current_error)
{
    int i, j, k, l;         //declare counters

    for (i = 0; i < TRAINSIZE; i++)     //trainsize = training set size of 90
    {
        for (j = 0; j < NUM_LAYER2; j++) //layer 2 
        {
            for (k = 0; k < NUM_LAYER1; k++) //layer 1
            {
                for (l = 0; l < NUM_INPUT; l++) //input layer
                {
                    *layer1Sum += trainingInput[i][l] * input_weight[k][l];     //sum of inputs * weights of input layer to layer1
                }
                *layer1Sum += layer1_bias[k];               //adding bias at the end of the summation of layer1 neuron
                layer1_summation[i][k] = *layer1Sum;        //store the summation of the neuron into a array, to be used layer on at backpropagation
                layer1_output[i][k] = sigmoid(*layer1Sum);  //pass through sigmoid activation function, and store the output into a array, to be used for backpropagation.
                *layer1Sum = 0;                             //reset the sum of the neuron, to service the next neuron in layer 1

                *layer2Sum += layer1_output[i][k] * layer1_weight[j][k];  //sum of layer1 output * weights, to be feeded into 2nd layer neuron
            }
            *layer2Sum += layer2_bias[j];                   // adding bias at the end of the summation of layer2 neuron
            layer2_summation[i][j] = *layer2Sum;            //storing summation for backpropagation use
            layer2_output[i][j] = sigmoid(*layer2Sum);      // storing output for backpropagation use
            *layer2Sum = 0;                                 //reset the sum, to service the next neuron in layer 2

            *outputSum += layer2_output[i][j] * layer2_weight[j];   //sum of layer2 output * weights, to be feeded into output neuron
        }
        *outputSum += *output_bias;                         //add output bias to finalize the summation
        output_summation[i] = *outputSum;                   //store the summation for backpropagation use later on
        *current_error = sigmoid(*outputSum) - trainingOutput[i];     //calculate error for the current row
        output_error[i] = *current_error;                   //store the error into an array, to be used for backpropagation
        *sumAbsError += fabs(*current_error);               //get the absolute value and feed into the summation, to be used to calculate MAE
        *sumErrorSq += pow(*current_error, 2);              //square the error and feed it into another summation, to be used to calculate MMSE
        *outputSum = 0;                                     //reset outputsum, to service the next row 
    }
}

//similar to feedforward(), only to be called at the first iteration to calculate untrained confusion matrix
//calls confusion matrix
void feedforward_first_iteration(int confusionCount[4], double trainingInput[][NUM_INPUT], double trainingOutput[TRAINSIZE], double input_weight[][NUM_INPUT], double layer1_weight[][NUM_LAYER1], double layer2_weight[NUM_LAYER2], double layer1_bias[NUM_LAYER1], double layer2_bias[NUM_LAYER2], double *output_bias, double layer1_output[][NUM_LAYER1], double layer1_summation[][NUM_LAYER1], double layer2_output[][NUM_LAYER2], double layer2_summation[][NUM_LAYER2], double output_error[TRAINSIZE], double output_summation[TRAINSIZE], double *sumAbsError, double *sumErrorSq, double *layer1Sum, double *layer2Sum, double *outputSum, double *current_error)
{
    int i, j, k, l;
    float predictedY;

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
        predictedY = sigmoid(*outputSum);           //get output of output neuron
        //confusion matrix for first iteration of training set
        confusionMatrix(confusionCount, predictedY, trainingOutput[i]);     //calls function to calculate FP FN TP TN
        *current_error = predictedY - trainingOutput[i];
        output_error[i] = *current_error;
        *sumAbsError += fabs(*current_error);
        *sumErrorSq += pow(*current_error, 2);
        *outputSum = 0;
    }
}

//BACK PROPAGATION PORTION (BATCH GRADIENT DESCENT)
//SUMMATION OF ERRORS FOR BACK PROPAGATION
//Errors for the batch of 90 training data will be summed up in backpropagate_summation() , and respective weights and biases will be updated accordingly in backpropagate_update()
void backpropagate_summation(double trainingInput[][NUM_INPUT], double layer1_weight[][NUM_LAYER1], double layer2_weight[NUM_LAYER2], double layer1_output[][NUM_LAYER1], double layer1_summation[][NUM_LAYER1], double layer2_output[][NUM_LAYER2], double layer2_summation[][NUM_LAYER2], double output_error[TRAINSIZE], double output_summation[TRAINSIZE], double *output_bias_update, double layer2_weight_update[NUM_LAYER2], double layer2_bias_update[NUM_LAYER2], double layer1_weight_update[][NUM_LAYER1], double layer1_bias_update[NUM_LAYER1], double input_weight_update[][NUM_INPUT])
{
    int i, j, k, l, n;
    double error_output, error_layer2[NUM_LAYER2], error_layer1[NUM_LAYER1];    //error arrays for respective neurons
    for (i = 0; i < TRAINSIZE; i++)
    {
        error_output = (output_error[i] * deSigmoid(output_summation[i])) / 90; //calculate error at output neuron
        *output_bias_update += error_output;                                    //feed current row's error into output bias update sum
        for (j = 0; j < NUM_LAYER2; j++)
        {
            error_layer2[j] = error_output * layer2_weight[j] * deSigmoid(layer2_summation[i][j]); //calculate error at layer2 neurons
            layer2_weight_update[j] += error_output * layer2_output[i][j];                         //feed error of weghts into respective error sums
            layer2_bias_update[j] += error_layer2[j];       //summation of bias update for layer 2 neurons

        }
        for (j = 0; j < NUM_LAYER2; j++)
        {
            for (k = 0; k < NUM_LAYER1; k++)
            {
                error_layer1[k] += error_layer2[j] * layer1_weight[j][k] * deSigmoid(layer1_summation[i][k]); //sum up error linking to neurons in layer1
            }
        }
        for (j = 0; j < NUM_LAYER2; j++)
        {
            for (k = 0; k < NUM_LAYER1; k++)
            {
                layer1_weight_update[j][k] += error_layer2[j] * layer1_output[i][k];    //summation of layer 1 update, by differenciating layer 2 neuron and layer1 neuron output
                error_layer2[j] = 0;            //reset for next batch
                for (l = 0; l < NUM_INPUT; l++)
                {
                    input_weight_update[k][l] += error_layer1[k] * trainingInput[i][l]; //summation of input weight update, using  the sum of error till layer 1, differenciate with input parameter to get weight update for input
                }
            }
        }
        for (j = 0; j < NUM_LAYER1; j++)
        {
            layer1_bias_update[j] += error_layer1[j];       //summation of bias update for layer 1 neurons
            error_layer1[j] = 0;                            //reset for next batch
        }
    }
}

//UPDATE OF WEIGHTS AND BIASES FOR BACK PROPAGATION, USING SUMMATION OF RESPECTIVE ERRORS
void backpropagate_update(double input_weight[][NUM_INPUT], double layer1_weight[][NUM_LAYER1], double layer2_weight[NUM_LAYER2], double layer1_bias[NUM_LAYER1], double layer2_bias[NUM_LAYER2], double *output_bias, double *output_bias_update, double layer2_weight_update[NUM_LAYER2], double layer2_bias_update[NUM_LAYER2], double layer1_weight_update[][NUM_LAYER1], double layer1_bias_update[NUM_LAYER1], double input_weight_update[][NUM_INPUT])
{
    int i, j;
    for (i = 0; i < NUM_LAYER2; i++) // layer 2
    {
        layer2_bias[i] -= LEARNING_RATE * layer2_bias_update[i];        //apply the gradient descent formula, to update biases of layer 2 neurons
        layer2_weight[i] -= LEARNING_RATE * layer2_weight_update[i];    //update weights of layer 2 neruons
        layer2_bias_update[i] = 0;              //reset update summations for next batch
        layer2_weight_update[i] = 0;
        //the number of layer1 weights to update will be: number of layer1 neurons * number of layer2 neurons, hence nested for loop
        for (j = 0; j < NUM_LAYER1; j++)            
        {
            layer1_bias[j] -= LEARNING_RATE * layer1_bias_update[j];    //update respective weights and biases, and reset update summations for next batch
            layer1_bias_update[j] = 0;
            layer1_weight_update[i][j] -= LEARNING_RATE * layer1_weight_update[i][j];
            layer1_weight_update[i][j] = 0;
        }
    }
    //number of input weights to be updated will be: number of input weights * number of layer1 neurons
    for (i = 0; i < NUM_LAYER1; i++)            
    {
        for (j = 0; j < NUM_INPUT; j++)
        {
            input_weight[i][j] -= LEARNING_RATE * input_weight_update[i][j];
            input_weight_update[i][j] = 0;
        }
    }
    *output_bias -= LEARNING_RATE * *output_bias_update;
    *output_bias_update = 0;
}

double deSigmoid(double x)      //function to differenciate summation
{
    double result;
    result = exp(x) / pow(1 + exp(x), 2);
    return result;
}

double sigmoid(double x)        //sigmoid activation function
{
    double result;
    result = 1 / (1 + exp(-x));
    return result;
}