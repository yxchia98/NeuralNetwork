#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#define TXT_LINE_SIZE 45 //maximum number of chars in per line in .txt file
#define SIZE 100         //size of dataset
#define TRAINSIZE 90
#define TESTSIZE 10

void read_txt(const char *filename, double trainingInput[TRAINSIZE][9], double trainingOutput[TRAINSIZE], double testingInput[TESTSIZE][9], double testingOutput[TESTSIZE]);
void train(double input[90][9], double outputarray[90][1]);
void randWeight(double x[], int n);
double randFrom(double min, double max);
double deSigmoid(double x);
double sigmoid(double x);
double linear_regression(double arrayX[9], double arrayW[9], double bias);
double backward_propogation(double current_error, double z, float x);
void confusionMatrix(int confusionCount[4], float predictedY, int output);
void printConfusionMatrix(int confusionCount[4][4], double MMSE[4]);
