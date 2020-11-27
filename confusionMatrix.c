#include <stdio.h>
#include "assignment.h"
//1 = TP, 2 = FP, 3 = TN, 4 = FN
void confusionMatrix(int confusionCount[4], float predictedY, int output)
{
    if (predictedY < 0.5)
    {
        if (output == 0)
        {
            confusionCount[2] += 1; //true negative
        }
        else if (output == 1)
        {
            confusionCount[3] += 1; //false negative
        }
    }
    else if (predictedY >= 0.5)
    {
        if (output == 1)
        {
            confusionCount[0] += 1; //true postive
        }
        else if (output == 0)
        {
            confusionCount[1] += 1; //false postive
        }
    }
}
void printConfusionMatrix(int confusionCount[4][4], double MMSE[4])
{
    printf("\n\naaaConfusion Matrix for 90 training dataset before training\nTrue Positive  : %d \nFalse Positive : %d \nTrue Negative  : %d \nFalse Negative : %d ", confusionCount[0][0], confusionCount[0][1], confusionCount[0][2], confusionCount[0][3]);
    printf("\nMMSE of testing dataset: %lf", MMSE[0]);
    printf("\n\naaaConfusion Matrix for 10 testing dataset before training\nTrue Positive  : %d \nFalse Positive : %d \nTrue Negative  : %d \nFalse Negative : %d ", confusionCount[1][0], confusionCount[1][1], confusionCount[1][2], confusionCount[1][3]);
    printf("\nMMSE of testing dataset: %lf", MMSE[1]);
    printf("\n\naaaConfusion Matrix for 90 training dataset after training\nTrue Positive  : %d \nFalse Positive : %d \nTrue Negative  : %d \nFalse Negative : %d ", confusionCount[2][0], confusionCount[2][1], confusionCount[2][2], confusionCount[2][3]);
    printf("\nMMSE of testing dataset: %lf", MMSE[2]);
    printf("\n\naaaConfusion Matrix for 10 testing dataset after training\nTrue Positive  : %d \nFalse Positive : %d \nTrue Negative  : %d \nFalse Negative : %d ", confusionCount[3][0], confusionCount[3][1], confusionCount[3][2], confusionCount[3][3]);
    printf("\nMMSE of testing dataset: %lf", MMSE[3]);
}