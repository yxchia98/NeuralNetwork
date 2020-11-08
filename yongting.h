#include <stdio.h>
#include <math.h>

double linear_regression(double arrayX[9], double arrayW[9], double bias);
double backward_propogation(double MAE, float x, double z);
double linear_regression(double arrayX[9], double arrayW[9], double bias)
{
    double sum;
    for (int i = 0; i < 9; i++)
    {
        sum += arrayX[i] * arrayW[i];
    }
    return (sum + bias);
}
double backward_propogation(double MAE, float x, double linear_regression_val)
{
    return (MAE * (exp(linear_regression_val) / pow(1 + exp(linear_regression_val), 2)) * x);
}
