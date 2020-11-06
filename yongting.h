#include<stdio.h>


double linear_regression(double arrayX[9], double arrayW[9], double bias);
double linear_regression(double arrayX[9], double arrayW[9], double bias){
    double sum;
    for(int i =0; i<9; i++){
        sum += arrayX[i] * arrayW[i];
    }
    return (sum + bias);
}