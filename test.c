#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#define TXT_LINE_SIZE 41                    //maximum number of chars in per line in .txt file
#define SIZE 100                            //size of dataset
void read_txt(char* filename, char c[TXT_LINE_SIZE], char txt_array[SIZE][TXT_LINE_SIZE], float season[], float age[], float alcFreq[], float sitHour[], int disease[], int acci[], int surgInt[], int fever[], int smoke[], int semenDiag[]);
void read_floatArray(float array[]);
void read_intArray(float array[]);
int main()
{
    static float season[SIZE], age[SIZE], alcFreq[SIZE], sitHour[SIZE];    // arrays for attributes with float data types
    static int disease[SIZE], acci[SIZE], surgInt[SIZE], fever[SIZE], smoke[SIZE], semenDiag[SIZE]; // arrays for attributes with int data types
    char c[TXT_LINE_SIZE];
    char txt_array[SIZE][TXT_LINE_SIZE]={};
    char* filename="fertility_Diagnosis_Data_Group1_4.txt";
    read_txt(filename, c, txt_array, season, age, alcFreq, sitHour, disease, acci, surgInt, fever, smoke, semenDiag);               // reads txt file and assigns it into txt_array
    return 0;
}

void read_txt(char* filename, char c[TXT_LINE_SIZE], char txt_array[SIZE][TXT_LINE_SIZE], float season[], float age[], float alcFreq[], float sitHour[], int disease[], int acci[], int surgInt[], int fever[], int smoke[], int semenDiag[])
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