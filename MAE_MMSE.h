#include <stdio.h>
#include <math.h>

double m_a_e(double predictedOutput, double testingOutput);
double m_m_s_e(double predictedOutput, double testingOutput);

double m_a_e(double predictedOutput, double testingOutput)
{
return abs(predictedOutput - testingOutput); // absolute value, no negative
}

double m_m_s_e(double predictedOutput, double testingOutput)
{
return pow((predictedOutput-testingOutput),2);
}