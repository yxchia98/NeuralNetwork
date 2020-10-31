#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#define TXT_LINE_SIZE 41                    //maximum number of chars in per line in .txt file
#define SIZE 100                            //size of dataset
void read_txt(char* filename, char c[TXT_LINE_SIZE], char txt_array[SIZE][TXT_LINE_SIZE], double season[], double age[], double alcFreq[], double sitHour[], int disease[], int acci[], int surgInt[], int fever[], int smoke[], int semenDiag[]);
void randWeight(double x[], int n);
double randomize(double min, double max);
void read_floatArray(float array[]);
void read_intArray(float array[]);

//Reads from the .txt dataset file and assigns values in it into the corresponding attribute arrays
void read_txt(char* filename, char c[TXT_LINE_SIZE], char txt_array[SIZE][TXT_LINE_SIZE], double season[], double age[], double alcFreq[], double sitHour[], int disease[], int acci[], int surgInt[], int fever[], int smoke[], int semenDiag[])
{
    int i=0,k;
    FILE *fp;                               //file pointer
    char *ptr;
    char delim[]=",";
    fp=fopen(filename,"r");                 //open file, read only
    if(fp==NULL)
    {
        printf("Could not open filename %s", filename);
        exit;
    }
    while(fgets(c,TXT_LINE_SIZE,fp)!=NULL)
    {
        strcpy(txt_array[i],c);             //Copy line from .txt file into string array
        k=0;
        ptr=strtok(c,delim);
        while(ptr!=NULL)
        {
            switch (k)
            {
                case 0:
                season[i]=atof(ptr);
                break;
                case 1:
                age[i]=atof(ptr);
                break;
                case 2:
                disease[i]=atoi(ptr);
                break;
                case 3:
                acci[i]=atoi(ptr);
                break;
                case 4:
                surgInt[i]=atoi(ptr);
                break;
                case 5:
                fever[i]=atoi(ptr);
                break;
                case 6:
                alcFreq[i]=atof(ptr);
                break;
                case 7:
                smoke[i]=atoi(ptr);
                break;
                case 8:
                sitHour[i]=atof(ptr);
                break;
                case 9:
                semenDiag[i]=atoi(ptr);
                break;
            }
            k++;
            ptr=strtok(NULL,delim);
        }


        i++;
    }
    fclose(fp);

}

void randWeight(double x[],int n)
{
    int i;
    printf("Size of array in bytes:%d",sizeof(x));
    for(i=0;i<n;i++)
    {
        x[i]=randomize(-1.0,1.0);
    }
}

double randomize(double min, double max) 
{
    double range = (max - min); 
    double div = RAND_MAX / range;
    return min + (rand() / div);
}

void read_floatArray(float array[])
{
    int i=0;
    for(i=0;i<SIZE;i++)
    {
        printf("Attribute in [%d]: %f\n", i, array[i]);
    }
}
void read_intArray(float array[])
{
    int i=0;
    for(i=0;i<SIZE;i++)
    {
        printf("Attribute in [%d]: %f\n", i, array[i]);
    }
}