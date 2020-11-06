#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "test1.h"
#include "yongting.h"
#define TXT_LINE_SIZE 41                    //maximum number of chars in per line in .txt file
#define SIZE 100                            //size of dataset
#define TRAINSIZE 90
#define TESTSIZE  10

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
    static double weight[9], trainingInput[TRAINSIZE][9], trainingOutput[TRAINSIZE][1], testingInput[TESTSIZE][9], testingOutput[TESTSIZE][1];
    char c[TXT_LINE_SIZE];
    char txt_array[SIZE][TXT_LINE_SIZE]={};
    char* filename="fertility_Diagnosis_Data_Group1_4.txt";
    double bias, error, mae;
    read_txt(filename, c, txt_array, trainingInput, trainingOutput, testingInput, testingOutput);               // reads txt file and assigns it into txt_array
    randWeight(weight,9);
    bias=randFrom(-1,1);
    linear_regression(trainingInput[1], weight, 1);
    return 0;
}

