#include <stdio.h>
#include "assignment.h"
//calculate confusion matrix
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
void printConfusionMatrix(int confusionCount[4][4], double mmse[4], double mae[4])
{
    float accuracy = 0;
    printf(BOLDRED "\n\n========= Confusion Matrix before training =========" RESET);

    accuracy = (((float)confusionCount[0][0] + confusionCount[0][2]) / TRAINSIZE) * 100;
    printf(BOLDBLACK "\n\nTraining dataset" RESET);
    printf("\nTrue Positive  : %d \nFalse Positive : %d \nTrue Negative  : %d \nFalse Negative : %d " RESET, confusionCount[0][0], confusionCount[0][1], confusionCount[0][2], confusionCount[0][3]);
    printf("\n-------------------");
    printf("\nMMSE: %lf\nMAE: %lf", mmse[0], mae[0]);
    printf("\nAccuracy: %.2f%c", accuracy, '%');

    accuracy = (((float)confusionCount[1][0] + confusionCount[1][2]) / TESTSIZE) * 100;
    printf(BOLDBLACK "\n\nTesting dataset" RESET);
    printf("\nTrue Positive  : %d \nFalse Positive : %d \nTrue Negative  : %d \nFalse Negative : %d ", confusionCount[1][0], confusionCount[1][1], confusionCount[1][2], confusionCount[1][3]);
    printf("\n-------------------");
    printf("\nMMSE: %lf\nMAE: %lf", mmse[1], mae[1]);
    printf("\nAccuracy: %.2f%c", accuracy, '%');

    printf(BOLDGREEN "\n\n========= Confusion Matrix after training =========" RESET);

    accuracy = (((float)confusionCount[2][0] + confusionCount[2][2]) / TRAINSIZE) * 100;
    printf(BOLDBLACK "\n\nTraining dataset" RESET);
    printf("\nTrue Positive  : %d \nFalse Positive : %d \nTrue Negative  : %d \nFalse Negative : %d ", confusionCount[2][0], confusionCount[2][1], confusionCount[2][2], confusionCount[2][3]);
    printf("\n-------------------");
    printf("\nMMSE: %lf\nMAE: %lf", mmse[2], mae[2]);
    printf("\nAccuracy: %.2f%c", accuracy, '%');

    accuracy = (((float)confusionCount[3][0] + confusionCount[3][2]) / TESTSIZE) * 100;
    printf(BOLDBLACK "\n\nTesting dataset" RESET);
    printf("\nTrue Positive  : %d \nFalse Positive : %d \nTrue Negative  : %d \nFalse Negative : %d ", confusionCount[3][0], confusionCount[3][1], confusionCount[3][2], confusionCount[3][3]);
    printf("\n-------------------");
    printf("\nMMSE: %lf\nMAE: %lf", mmse[3], mae[3]);
    printf("\nAccuracy: %.2f%c\n\n", accuracy, '%');
}
