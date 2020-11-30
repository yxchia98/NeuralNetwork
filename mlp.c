#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "assignment.h"
// #include "input.c"
// #include "process.c"
// #include "confusionMatrix.c"
// #include "trainWeights.c"
// #include "testWeights.c"

/**********************************************************************************************
Basic elements inside Input arrays(trainingInput/testingInput)          
[0]=Season of analysis                                                  
[1]=Age of analysis                                                     
[2]=Childish disease                                                    
[3]=Accident or serious trauma                                          
[4]=Surgical intervention                                               
[5]=High fevers last year                                               
[6]=Frequency of alcohol consumption                                    
[7]=Smoking habit                                                       
[8]=Number of hours spent sitting per day                               
--------------------------------------------------------------------------------------------------
Basic elements inside Output arrays(trainingOutput/testingOutput)       Datatype(range)
[0]=Semen Diagnosis                                                     Int(0,1)
***********************************************************************************************/

/************************************MULTI-LAYER PERCEPTRON(2 HIDDEN LAYERS, 10 IN LAYER1, 5 IN LAYER 2)***************************************/

int main()
{
    static double trainingInput[TRAINSIZE][NUM_INPUT], trainingOutput[TRAINSIZE], testingInput[TESTSIZE][NUM_INPUT], testingOutput[TESTSIZE];
    static double input_weight[NUM_LAYER1][NUM_INPUT], layer1_weight[NUM_LAYER2][NUM_LAYER1], layer2_weight[NUM_LAYER2];
    static double layer1_bias[NUM_LAYER1], layer2_bias[NUM_LAYER2], output_bias;
    char *filename = "fertility_Diagnosis_Data_Group1_4.txt";
    FILE *plotptr; //file pointer for plotting of graph
    int i, j, k, l, n, tp, fp, fn, tn;
    int confusionCount[4][4] = {};
    //1 = training set before training, 2 = testing set before training, 3 = training set after training, 4 = testing set after training
    double mmse_arr[4];
    //1 = TP, 2 = FP, 3 = TN, 4 = FN
    double predictedY;
    //used in confusion matrix
    double output_untrained_arr[90];

    read_txt(filename, trainingInput, trainingOutput, testingInput, testingOutput); // reads txt file and assigns it into txt_array
    //randomize weights below
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

    //Open MAEGraph.txt, the file used to store the MAE for every iteration
    if ((plotptr = fopen("MAEGraph.txt", "w")) == NULL) 
    {
        printf("\nMAEGraph.txt does not exist.");
        exit(1);
    }
    //confusion matrix of testing set, untrained weights
    //TESTING SET
    testWeights(TESTSIZE, confusionCount[1], &mmse_arr[1], testingInput, testingOutput, input_weight, layer1_weight, layer2_weight, layer1_bias, layer2_bias, &output_bias);

    //train the weights
    trainWeights(&mmse_arr[0], confusionCount[0], plotptr, trainingInput, trainingOutput, input_weight, layer1_weight, layer2_weight, layer1_bias, layer2_bias, &output_bias);

    //confusion matrix of testing set, trained weights
    testWeights(TESTSIZE, confusionCount[3], &mmse_arr[3], testingInput, testingOutput, input_weight, layer1_weight, layer2_weight, layer1_bias, layer2_bias, &output_bias);

    //confusion matrix of training set, trained weights
    testWeights(TRAINSIZE, confusionCount[2], &mmse_arr[2], trainingInput, trainingOutput, input_weight, layer1_weight, layer2_weight, layer1_bias, layer2_bias, &output_bias);

    printConfusionMatrix(confusionCount, mmse_arr);
    fclose(plotptr);
    system("gnuplot -p plotcmd.txt");
    return 0;
}
