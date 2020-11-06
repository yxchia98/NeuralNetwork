#include<stdio.h>


double linear_regression(double arrayX[9], double arrayW[9], double bias);
double linear_regression(double arrayX[9], double arrayW[9], double bias){
    double sum;
    for(int i =0; i<9; i++){
        printf("\n arrayX%d %f", i, arrayX[i]);
        printf("\n arrayW%d %f", i, arrayW[i]);
        sum += arrayX[i] * arrayW[i];
    }
    printf("\nhit");
    printf("this is the value %f", sum);
    return (sum + bias);
}