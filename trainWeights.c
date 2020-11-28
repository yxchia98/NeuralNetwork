#include <stdio.h>
#include "assignment.h"

void trainWeights(double *mmse_arr, int confusionCount[4], FILE *plotptr, double trainingInput[TRAINSIZE][NUM_INPUT], double trainingOutput[TRAINSIZE], double input_weight[NUM_LAYER1][NUM_INPUT], double layer1_weight[NUM_LAYER2][NUM_LAYER1], double layer2_weight[NUM_LAYER2], double layer1_bias[NUM_LAYER1], double layer2_bias[NUM_LAYER2], double *output_bias)
{
    clock_t start, elapsed;
    static double layer1_output[TRAINSIZE][NUM_LAYER1], layer1_summation[TRAINSIZE][NUM_LAYER1], layer2_output[TRAINSIZE][NUM_LAYER2], layer2_summation[TRAINSIZE][NUM_LAYER2], output_error[TRAINSIZE], output_summation[TRAINSIZE], output[TRAINSIZE];
    static double output_bias_update, layer2_weight_update[NUM_LAYER2], layer2_bias_update[NUM_LAYER2], layer1_weight_update[NUM_LAYER2][NUM_LAYER1], layer1_bias_update[NUM_LAYER1], input_weight_update[NUM_LAYER1][NUM_INPUT];
    double sumAbsError, sumErrorSq, mae, mmse, untrained_mae, untrained_mmse, layer1Sum, layer2Sum, outputSum, current_error;
    int m;
    m = 1;
    start = clock();
    do
    {
        if (m == 1)
        {
            //feed forward for first iteration
            //to get confusion matrix for training set, untrained weights
            feedforward_first_iteration(confusionCount, trainingInput, trainingOutput, input_weight, layer1_weight, layer2_weight, layer1_bias, layer2_bias, output_bias, layer1_output, layer1_summation, layer2_output, layer2_summation, output_error, output_summation, output, &sumAbsError, &sumErrorSq, &layer1Sum, &layer2Sum, &outputSum, &current_error);
            mae = sumAbsError / 90;
            mmse = sumErrorSq / 90;
            untrained_mmse = mmse;
            untrained_mae = mae;
            *mmse_arr = untrained_mmse;
        }
        else
        {
            //call feedforward function
            feedforward(trainingInput, trainingOutput, input_weight, layer1_weight, layer2_weight, layer1_bias, layer2_bias, output_bias, layer1_output, layer1_summation, layer2_output, layer2_summation, output_error, output_summation, output, &sumAbsError, &sumErrorSq, &layer1Sum, &layer2Sum, &outputSum, &current_error);
            mae = sumAbsError / 90;
            mmse = sumErrorSq / 90;
        }
        printf("MAE for iteration %d is: %lf, MMSE is: %lf\n", m, mae, mmse);
        //Yi Xuan, help me check the plotting
        fprintf(plotptr, "%lf\n", mae);
        sumErrorSq = 0;
        sumAbsError = 0;
        m++;
        if (mae > TARGETED_MAE)
        {
            //call function to to sum up update parameters
            backpropagate_summation(trainingInput, layer1_weight, layer2_weight, layer1_output, layer1_summation, layer2_output, layer2_summation, output_error, output_summation, &output_bias_update, layer2_weight_update, layer2_bias_update, layer1_weight_update, layer1_bias_update, input_weight_update);

            //call function to update weight and biases
            backpropagate_update(input_weight, layer1_weight, layer2_weight, layer1_bias, layer2_bias, output_bias, &output_bias_update, layer2_weight_update, layer2_bias_update, layer1_weight_update, layer1_bias_update, input_weight_update);
        }
    } while (mae > TARGETED_MAE);
    elapsed = (clock() - start) * 1000 / CLOCKS_PER_SEC;
    printf("\nUntrained MAE is: %lf, untrained MMSE is: %lf", untrained_mae, untrained_mmse);
    printf("\nTime taken: %dms", elapsed);
}