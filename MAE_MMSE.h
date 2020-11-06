#include <stdio.h>
#include <math.h>

double m_a_e(double predictedOutput, double testingOutput);
double m_m_s_e(double predictedOutput, double testingOutput);

double m_a_e(double predictedOutput, double testingOutput)
{
return predictedOutput - testingOutput;
}

double m_m_s_e(double predictedOutput, double testingOutput)
{
return (predictedOutput-testingOutput)*(predictedOutput-testingOutput);
}