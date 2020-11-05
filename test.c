#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "test1.h"
#define TXT_LINE_SIZE 41                    //maximum number of chars in per line in .txt file
#define SIZE 100                            //size of dataset
//training set range 0-89
//test set range 90-99

int main()
{
    static double season[SIZE], age[SIZE], alcFreq[SIZE], sitHour[SIZE], weight[9];    // arrays for attributes with float data types
    static int disease[SIZE], acci[SIZE], surgInt[SIZE], fever[SIZE], smoke[SIZE], semenDiag[SIZE]; // arrays for attributes with int data types
    char c[TXT_LINE_SIZE];
    char txt_array[SIZE][TXT_LINE_SIZE]={};
    char* filename="fertility_Diagnosis_Data_Group1_4.txt";
    double error, mae;
    printf("Size of array in bytes:%d",sizeof(weight));
    read_txt(filename, c, txt_array, season, age, alcFreq, sitHour, disease, acci, surgInt, fever, smoke, semenDiag);               // reads txt file and assigns it into txt_array
    randWeight(weight,9);
    return 0;
}

