#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define TXT_LINE_SIZE 45 //maximum number of chars in per line in .txt file
#define SIZE 100         //size of dataset
#define TRAINSIZE 90
#define TESTSIZE 10
#define LEARNING_RATE 0.40
#define TARGETED_MAE 0.15
#define NUM_INPUT 9
#define NUM_LAYER1 10
#define NUM_LAYER2 5
// definition of NUM_LAYER1 and NUM_LAYER2 can be changed to adjust the neurons in respective layers
void read_txt(const char *filename, double trainingInput[TRAINSIZE][9], double trainingOutput[TRAINSIZE], double testingInput[TESTSIZE][9], double testingOutput[TESTSIZE]);
void randWeight(double x[], int n);
double randFrom(double min, double max);
void feedforward(double trainingInput[][NUM_INPUT], double trainingOutput[TRAINSIZE], double input_weight[][NUM_INPUT], double layer1_weight[][NUM_LAYER1], double layer2_weight[NUM_LAYER2], double layer1_bias[NUM_LAYER1], double layer2_bias[NUM_LAYER2], double *output_bias, double layer1_output[][NUM_LAYER1], double layer1_summation[][NUM_LAYER1], double layer2_output[][NUM_LAYER2], double layer2_summation[][NUM_LAYER2], double output_error[TRAINSIZE], double output_summation[TRAINSIZE], double *sumAbsError, double *sumErrorSq, double *layer1Sum, double *layer2Sum, double *outputSum, double *current_error);
void feedforward_first_iteration(int confusionCount[4], double trainingInput[][NUM_INPUT], double trainingOutput[TRAINSIZE], double input_weight[][NUM_INPUT], double layer1_weight[][NUM_LAYER1], double layer2_weight[NUM_LAYER2], double layer1_bias[NUM_LAYER1], double layer2_bias[NUM_LAYER2], double *output_bias, double layer1_output[][NUM_LAYER1], double layer1_summation[][NUM_LAYER1], double layer2_output[][NUM_LAYER2], double layer2_summation[][NUM_LAYER2], double output_error[TRAINSIZE], double output_summation[TRAINSIZE], double *sumAbsError, double *sumErrorSq, double *layer1Sum, double *layer2Sum, double *outputSum, double *current_error);
void backpropagate_summation(double trainingInput[][NUM_INPUT], double layer1_weight[][NUM_LAYER1], double layer2_weight[NUM_LAYER2], double layer1_output[][NUM_LAYER1], double layer1_summation[][NUM_LAYER1], double layer2_output[][NUM_LAYER2], double layer2_summation[][NUM_LAYER2], double output_error[TRAINSIZE], double output_summation[TRAINSIZE], double *output_bias_update, double layer2_weight_update[NUM_LAYER2], double layer2_bias_update[NUM_LAYER2], double layer1_weight_update[][NUM_LAYER1], double layer1_bias_update[NUM_LAYER1], double input_weight_update[][NUM_INPUT]);
void backpropagate_update(double input_weight[][NUM_INPUT], double layer1_weight[][NUM_LAYER1], double layer2_weight[NUM_LAYER2], double layer1_bias[NUM_LAYER1], double layer2_bias[NUM_LAYER2], double *output_bias, double *output_bias_update, double layer2_weight_update[NUM_LAYER2], double layer2_bias_update[NUM_LAYER2], double layer1_weight_update[][NUM_LAYER1], double layer1_bias_update[NUM_LAYER1], double input_weight_update[][NUM_INPUT]);
double sigmoid(double x);
double deSigmoid(double x);
void confusionMatrix(int confusionCount[4], float predictedY, int output);
void printConfusionMatrix(int confusionCount[4][4], double MMSE[4]);
void trainWeights(double *mmse_arr, int confusionCount[4], FILE *plotptr, double trainingInput[TRAINSIZE][NUM_INPUT], double trainingOutput[TRAINSIZE], double input_weight[NUM_LAYER1][NUM_INPUT], double layer1_weight[NUM_LAYER2][NUM_LAYER1], double layer2_weight[NUM_LAYER2], double layer1_bias[NUM_LAYER1], double layer2_bias[NUM_LAYER2], double *output_bias);
void testWeights(int size, int confusionCount[4], double *mmse_ptr, double input_arr[][NUM_INPUT], double *output_arr, double input_weight[NUM_LAYER1][NUM_INPUT], double layer1_weight[NUM_LAYER2][NUM_LAYER1], double layer2_weight[NUM_LAYER2], double layer1_bias[NUM_LAYER1], double layer2_bias[NUM_LAYER2], double *output_bias);