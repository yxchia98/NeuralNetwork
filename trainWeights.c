#include <stdio.h>
#include "assignment.h"

void trainWeights(double *mmse_arr, double *mae_arr, int confusionCount[4], FILE *plotptr, double trainingInput[TRAINSIZE][NUM_INPUT], double trainingOutput[TRAINSIZE], double input_weight[NUM_LAYER1][NUM_INPUT], double layer1_weight[NUM_LAYER2][NUM_LAYER1], double layer2_weight[NUM_LAYER2], double layer1_bias[NUM_LAYER1], double layer2_bias[NUM_LAYER2], double *output_bias)
{
    static double layer1_output[TRAINSIZE][NUM_LAYER1], layer1_summation[TRAINSIZE][NUM_LAYER1], layer2_output[TRAINSIZE][NUM_LAYER2], layer2_summation[TRAINSIZE][NUM_LAYER2], output_error[TRAINSIZE], output_summation[TRAINSIZE], output[TRAINSIZE];
    static double output_bias_update, layer2_weight_update[NUM_LAYER2], layer2_bias_update[NUM_LAYER2], layer1_weight_update[NUM_LAYER2][NUM_LAYER1], layer1_bias_update[NUM_LAYER1], input_weight_update[NUM_LAYER1][NUM_INPUT];
    double sumAbsError, sumErrorSq, mae, mmse, untrained_mae, untrained_mmse, layer1Sum, layer2Sum, outputSum, current_error;
    int num_iteration;
    printf("training...");
    num_iteration = 1;
    do
    {
        if (num_iteration == 1)
        {
            //feed forward for first iteration
            //to get confusion matrix for training set, untrained weights
            feedforward_first_iteration(confusionCount, trainingInput, trainingOutput, input_weight, layer1_weight, layer2_weight, layer1_bias, layer2_bias, output_bias, layer1_output, layer1_summation, layer2_output, layer2_summation, output_error, output_summation, &sumAbsError, &sumErrorSq, &layer1Sum, &layer2Sum, &outputSum, &current_error);
            mae = sumAbsError / 90; //get mean absolute error from sum of absolute errors
            mmse = sumErrorSq / 90; //get mean square error from sum of square errors
            untrained_mmse = mmse;  //capture untrained mmse and mae
            untrained_mae = mae;
            *mmse_arr = untrained_mmse; //store untrained mmse and mae for printing
            *mae_arr = untrained_mae;
        }
        else
        {
            //call feedforward function
            feedforward(trainingInput, trainingOutput, input_weight, layer1_weight, layer2_weight, layer1_bias, layer2_bias, output_bias, layer1_output, layer1_summation, layer2_output, layer2_summation, output_error, output_summation, &sumAbsError, &sumErrorSq, &layer1Sum, &layer2Sum, &outputSum, &current_error);
            mae = sumAbsError / 90; //get mean absolute error from sum of absolute errors
            mmse = sumErrorSq / 90; //get mean square error from sum of square errors
        }
        // printf("\nIteration %d, MAE is: %lf, MMSE is: %lf", num_iteration, mae, mmse);
        fprintf(plotptr, "%lf\n", mae); //write MAE of every iteration into a txt file
        sumErrorSq = 0;
        sumAbsError = 0;
        num_iteration++;
        if (mae > TARGETED_MAE)
        {
            //call function to to sum up update parameters
            backpropagate_summation(trainingInput, layer1_weight, layer2_weight, layer1_output, layer1_summation, layer2_output, layer2_summation, output_error, output_summation, &output_bias_update, layer2_weight_update, layer2_bias_update, layer1_weight_update, layer1_bias_update, input_weight_update);

            //call function to update weight and biases
            backpropagate_update(input_weight, layer1_weight, layer2_weight, layer1_bias, layer2_bias, output_bias, &output_bias_update, layer2_weight_update, layer2_bias_update, layer1_weight_update, layer1_bias_update, input_weight_update);
        }
    } while (mae > TARGETED_MAE);
    printf("\n\nTotal Number of iterations: %d", num_iteration);

    printf(BOLDRED "\n\nUntrained" RESET);
    printf("\n MAE is: %lf", untrained_mae);
    printf("\n MMSE is: %lf", untrained_mmse);

    printf(BOLDGREEN "\n\nTrained" RESET);
    printf("\n MAE is: %lf", mae);
    printf("\n MMSE is: %lf", mmse);
}