//
// Created by Miss Guo on 2018/11/28.
//

#define _USE_MATH_DEFINES
#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <math.h>
#define NR_END 1
#define I2D(Nx, i, j) ((i) + (Nx)*(j))
#define I3D(x_width, i, j, t) ((i) + (x_width)*(j) + (x_width)*(t))
float *dvector (int N)
{
    float *v;
    v = (float *)malloc(sizeof(float)*N);
    return v;
}

void zero_matrix(float *a, int Nx, int Ny)
{
    int i, j;

    for (j = 0; j < Ny; j++) {
        for (i = 0; i < Nx; i++) {
            a[I2D(Nx, i, j)] = 0.0;
        }
    }
}
void zero_matrix(float *a, int Nx, int Ny, int Nz)
{
    int i, j, t;

    for (j = 0; j < Ny; j++) {
        for (i = 0; i < Nx; i++) {
            for (t = 0; t < Nz; t++)
            {
                a[I3D(Nx, i, j, t)] = 0.0;
            }
        }
    }
}

void print_mat(FILE *fptr,
               float *a, int Nx, int Ny)
{
    int i, j;

    for (j = 0; j < Ny; j++) {
        for (i = 0; i < Nx; i++) {
            fprintf(fptr, "%.16f ", a[I2D(Nx, i, j)]);
        }
        fprintf(fptr, "\n");
    }
    fclose(fptr);
}
void print_mat(FILE *fptr,
               float *a, int Nx, int Ny, int Nz)
{
    int i, j, t;

    for (j = 0; j < Ny; j++) {
        for (i = 0; i < Nx; i++) {
            for (t = 0; t < Nz; t++)
            {
                fprintf(fptr, "%.16f ", a[I3D(Nx, i, j, t)]);
            }
        }
        fprintf(fptr, "\n");
    }
    fclose(fptr);
}

void initialize(float *a, float *x, float *y, int Nx, int Ny)
{
    int i, j;

    for (j = 1; j < Ny-1; j++) {
        for (i = 1; i < Nx-1; i++) {
            a[I2D(Nx, i, j)] = sin(M_PI*y[i])*sin(M_PI*x[j]);
        }
    }
}
void initialize(float *a, float *x, float *y, float *z, int Nx, int Ny, int Nz)
{
    int i, j, t;

    for (j = 1; j < Ny-1; j++) {
        for (i = 1; i < Nx-1; i++) {
            for (t = 1; t < Nz-1; t++)
            {
                a[I3D(Nx, i, j, t)] = sin(M_PI*y[i])*sin(M_PI*x[j])*sin(M_PI*z[t]);
            }
        }
    }
}

int iDivUp(int a, int b){
    return (((a) + (b) - 1)/(b));
}