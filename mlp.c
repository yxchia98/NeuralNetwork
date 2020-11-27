#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "assignment.h"
#include "input.c"
#include "regression.c"
#include "confusionMatrix.c"

#define TXT_LINE_SIZE 45 //maximum number of chars in per line in .txt file
#define SIZE 100         //size of dataset
#define TRAINSIZE 90
#define TESTSIZE 10
#define LEARNING_RATE 0.10
#define TARGETED_MAE 0.22
#define NUM_INPUT 9
#define NUM_LAYER1 7
#define NUM_LAYER2 2

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
    static double input_weight[NUM_LAYER1][NUM_INPUT], layer1_weight[NUM_LAYER2][NUM_LAYER1], layer2_weight[NUM_LAYER2];
    static double layer1_bias[NUM_LAYER1], layer2_bias[NUM_LAYER2], output_bias;
    static double layer1_output[TRAINSIZE][NUM_LAYER1], layer1_summation[TRAINSIZE][NUM_LAYER1], layer2_output[TRAINSIZE][NUM_LAYER2], layer2_summation[TRAINSIZE][NUM_LAYER2], output_error[TRAINSIZE], output_summation[TRAINSIZE];
    static double output_bias_update, layer2_weight_update[NUM_LAYER2], layer2_bias_update[NUM_LAYER2], layer1_weight_update[NUM_LAYER2][NUM_LAYER1], layer1_bias_update[NUM_LAYER1], input_weight_update[NUM_LAYER1][NUM_INPUT];
    int confusionCount[4][4];
    confusionCount[0][0] = 0;
    confusionCount[0][1] = 0;
    confusionCount[0][2] = 0;
    confusionCount[0][3] = 0;
    confusionCount[1][0] = 0;
    confusionCount[1][1] = 0;
    confusionCount[1][2] = 0;
    confusionCount[1][3] = 0;
    confusionCount[2][0] = 0;
    confusionCount[2][1] = 0;
    confusionCount[2][2] = 0;
    confusionCount[2][3] = 0;
    confusionCount[3][0] = 0;
    confusionCount[3][1] = 0;
    confusionCount[3][2] = 0;
    confusionCount[3][3] = 0;
    double mmse_arr[4];
    int done = 0;
    //1 = training set before training, 2 = testing set before training, 3 = training set after training, 4 = testing set after training
    //1 = TP, 2 = FP, 3 = TN, 4 = FN
    char *filename = "fertility_Diagnosis_Data_Group1_4.txt";
    FILE *plotptr;
    double sumAbsError, sumErrorSq, mae, mmse, untrained_mae, untrained_mmse, sumBiasChange, layer1Sum, layer2Sum, outputSum, current_error, error_output, error_layer1[NUM_LAYER1], error_layer2[NUM_LAYER2];
    int i, j, k, l, m, n, tp, fp, fn, tn;
    clock_t start, elapsed;
    m = 1;
    read_txt(filename, trainingInput, trainingOutput, testingInput, testingOutput); // reads txt file and assigns it into txt_array
    for (i = 0; i < NUM_LAYER1; i++)
    {
        randWeight(input_weight[i], NUM_INPUT);
    }
    for (i = 0; i < NUM_LAYER2; i++)
    {
        randWeight(layer1_weight[i], NUM_LAYER1);
    }
    randWeight(layer1_bias, NUM_LAYER1);
    randWeight(layer2_weight, NUM_LAYER2);
    randWeight(layer2_bias, NUM_LAYER2);
    output_bias = randFrom(-1, 1);
    start = clock();
    if ((plotptr = fopen("MAEGraph.txt", "w")) == NULL)
    {
        printf("\nMAEGraph.txt does not exist.");
        exit(1);
    }
    //TESTING SET BEFORE TRAINING
    for (i = 0; i < TESTSIZE; i++)
    {
        for (j = 0; j < NUM_LAYER2; j++) //layer 2
        {
            for (k = 0; k < NUM_LAYER1; k++) //layer 1
            {
                for (l = 0; l < NUM_INPUT; l++) //input layer
                {
                    layer1Sum += testingInput[i][l] * input_weight[k][l];
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
        outputSum += output_bias;

        //output_summation[i] = outputSum;
        current_error = sigmoid(outputSum) - testingOutput[i];

        output_error[i] = current_error;
        //sumAbsError += fabs(current_error);
        sumErrorSq += pow(current_error, 2);

        double predictedY = sigmoid(outputSum);
        int output = testingOutput[i];
        /* Negative predicted result will be 0, Postive predicted result will be 1 for confusion matrix*******/
        confusionMatrix(confusionCount[1], predictedY, output);
        outputSum = 0;
    }
    mmse = sumErrorSq / TESTSIZE;
    mmse_arr[1] = mmse;
    do
    {
        //FEEDFORAWRD PORTION
        for (i = 0; i < TRAINSIZE; i++)
        {
            for (j = 0; j < NUM_LAYER2; j++) //layer 2
            {
                for (k = 0; k < NUM_LAYER1; k++) //layer 1
                {
                    for (l = 0; l < NUM_INPUT; l++) //input layer
                    {
                        layer1Sum += trainingInput[i][l] * input_weight[k][l];
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
            outputSum += output_bias;
            output_summation[i] = outputSum;
            double predictedY = sigmoid(outputSum);
            int output = trainingOutput[i];
            if (done == 0)
            {
                //TRAINING SET BEFORE TRAINING
                confusionMatrix(confusionCount[0], predictedY, output);
            }
            current_error = sigmoid(outputSum) - trainingOutput[i];
            output_error[i] = current_error;
            sumAbsError += fabs(current_error);
            sumErrorSq += pow(current_error, 2);
            outputSum = 0;
        }
        done = 1;
        mae = sumAbsError / 90;
        mmse = sumErrorSq / 90;
        mmse_arr[0] = mmse;
        if (m == 1)
        {
            untrained_mmse = mmse;
            untrained_mae = mae;
        }
        printf("MAE for iteration %d is: %lf, MMSE is: %lf\n", m, mae, mmse);
        fprintf(plotptr, "%lf\n", mae);
        sumErrorSq = 0;
        sumAbsError = 0;
        m++;

        //BACKPROPAGATE PORTION
        for (i = 0; i < TRAINSIZE; i++)
        {
            error_output = (output_error[i] * deSigmoid(output_summation[i])) / 90;
            output_bias_update += error_output;
            for (j = 0; j < NUM_LAYER2; j++)
            {
                error_layer2[j] = error_output * layer2_weight[j] * deSigmoid(layer2_summation[i][j]);
                layer2_bias_update[j] += error_layer2[j];
                layer2_weight_update[j] += error_output * layer2_output[i][j];
            }
            for (j = 0; j < NUM_LAYER2; j++)
            {
                for (k = 0; k < NUM_LAYER1; k++)
                {
                    error_layer1[k] += error_layer2[j] * layer1_weight[j][k] * deSigmoid(layer1_summation[i][k]);
                }
            }
            for (j = 0; j < NUM_LAYER2; j++)
            {
                for (k = 0; k < NUM_LAYER1; k++)
                {
                    layer1_bias_update[k] += error_layer1[k];
                    layer1_weight_update[j][k] += error_layer2[j] * layer1_output[i][k];
                    for (l = 0; l < NUM_INPUT; l++)
                    {
                        input_weight_update[k][l] += error_layer1[k] * trainingInput[i][l];
                    }
                    error_layer1[k] = 0;
                }
            }
        }
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
        output_bias -= LEARNING_RATE * output_bias_update;
        output_bias_update = 0;
    } while (mae > TARGETED_MAE);

    elapsed = (clock() - start) * 1000 / CLOCKS_PER_SEC;
    printf("\nUntrained MAE is: %lf, untrained MMSE is: %lf", untrained_mae, untrained_mmse);
    printf("\nTime taken: %dms", elapsed);
    sumErrorSq = 0;

    //TESTING SET
    for (i = 0; i < TRAINSIZE; i++)
    {
        for (j = 0; j < NUM_LAYER2; j++) //layer 2
        {
            for (k = 0; k < NUM_LAYER1; k++) //layer 1
            {
                for (l = 0; l < NUM_INPUT; l++) //input layer
                {
                    layer1Sum += trainingInput[i][l] * input_weight[k][l];
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
        outputSum += output_bias;

        //output_summation[i] = outputSum;
        current_error = sigmoid(outputSum) - trainingOutput[i];

        output_error[i] = current_error;
        //sumAbsError += fabs(current_error);
        sumErrorSq += pow(current_error, 2);

        double predictedY = sigmoid(outputSum);
        int output = trainingOutput[i];
        /* Negative predicted result will be 0, Postive predicted result will be 1 for confusion matrix*******/
        confusionMatrix(confusionCount[2], predictedY, output);
        outputSum = 0;
    }
    mmse = sumErrorSq / TRAINSIZE;
    mmse_arr[2] = mmse;
    sumErrorSq = 0;

    //TESTING SET
    for (i = 0; i < TESTSIZE; i++)
    {
        for (j = 0; j < NUM_LAYER2; j++) //layer 2
        {
            for (k = 0; k < NUM_LAYER1; k++) //layer 1
            {
                for (l = 0; l < NUM_INPUT; l++) //input layer
                {
                    layer1Sum += testingInput[i][l] * input_weight[k][l];
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
        outputSum += output_bias;

        //output_summation[i] = outputSum;
        current_error = sigmoid(outputSum) - testingOutput[i];

        output_error[i] = current_error;
        //sumAbsError += fabs(current_error);
        sumErrorSq += pow(current_error, 2);

        double predictedY = sigmoid(outputSum);
        int output = testingOutput[i];
        /* Negative predicted result will be 0, Postive predicted result will be 1 for confusion matrix*******/
        confusionMatrix(confusionCount[3], predictedY, output);
        outputSum = 0;
    }
    mmse = sumErrorSq / TESTSIZE;
    mmse_arr[3] = mmse;
    sumErrorSq = 0;
    printConfusionMatrix(confusionCount, mmse_arr);
    fclose(plotptr);
    system("gnuplot -p plotcmd.txt");
    return 0;
}
