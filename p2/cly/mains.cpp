//
// Created by Peng Guo on 2018/11/26.
//

#define _USE_MATH_DEFINES
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <fstream>
#include <cassert>
#include <string>
#include "utils.h"
#include "dev_matrix.h"

//I2D, I3D
//#define I2D(x_width, i, j) ((i) + (x_width)*(j))
//#define I3D(x_width, i, j, t) ((i) + (x_width)*(j) + (x_width)*(t))

#define BLOCK_SIZE_X 16
#define BLOCK_SIZE_Y 16
#define BLOCK_SIZE_Z 16

using namespace std;

struct BLOCK
{
    int location_x;
    int location_y;
    int location_z;
    int bound_x;
    int bound_y;
    int bound_z;
    float fix_tmp;
};

struct CONF
{
    int dimension;
    float k;
    int num_step;
    int x_width;
    int y_height;
    int z_depth;
    float start_temp; //default starting temperature for nodes
    vector<BLOCK> heat_block;
};

// kernel to update temperatures - GPU version (not using shared mem)
__global__ void heat2d_gpu(int Nx, int Ny, float k, float *in, float *out)
{
    int i, j, P, W, E, S, N;
    float d2tdx2, d2tdy2;
    // find i and j indices of this thread
    i = blockIdx.x*(BLOCK_SIZE_X) + threadIdx.x;
    j = blockIdx.y*(BLOCK_SIZE_Y) + threadIdx.y;

    // find indices into linear memory
    P = I2D(Nx, i, j);
    W = I2D(Nx, i-1, j); E = I2D(Nx, i+1, j);
    S = I2D(Nx, i, j-1); N = I2D(Nx, i, j+1);

    // check that thread is within domain (not on boundary or outside domain)
    if (i > 0 && i < Nx-1 && j > 0 && j < Ny-1) {
        d2tdx2 = in[W] - 2.0*in[P] + in[E];
        d2tdy2 = in[N] - 2.0*in[P] + in[S];
        if (srcBlock(i, j, t, ))
        out[P] = in[P] + k * (d2tdx2 + d2tdy2);
    }
}

__global__ void heat3d_gpu(int Nx, int Ny, int Nz, float k, float *in, float *out)
{
    int i, j, t, P, W, E, S, N, T, D;
    float d3tdx3, d3tdy3, d3tdz3;
    // find i and j indices of this thread
    i = blockIdx.x*(BLOCK_SIZE_X) + threadIdx.x;
    j = blockIdx.y*(BLOCK_SIZE_Y) + threadIdx.y;
    t = blockIdx.z*(BLOCK_SIZE_Z) + threadIdx.z;

    // find indices into linear memory
    P = I3D(Nx, i, j, t);
    W = I3D(Nx, i-1, j, t);
    E = I3D(Nx, i+1, j, t);
    S = I3D(Nx, i, j-1, t);
    N = I3D(Nx, i, j+1, t);
    T = I3D(Nx, i, j, t+1);
    D = I3D(Nx, i, j, t-1);


    // check that thread is within domain (not on boundary or outside domain)
    if (i > 0 && i < Nx-1 && j > 0 && j < Ny-1 && t > 0 && t < Nz-1) {
        d3tdx3 = in[W] + in[E] - 2.0*in[P];
        d3tdy3 = in[N] + in[S] - 2.0*in[P];
        d3tdz3 = in[T] + in[D] - 2.0*in[P];

        out[P] = in[P] + k * (d3tdx3 + d3tdy3 + d3tdz3);
    }
}

// read configuration file
void read(string & path, CONF & data){
    fsream fin(path, ios_base::in);
    string s;
    int s_num;

    ifstream theconf(argv[1]);
    if (!theconf.is_open()) {
        cerr << "file cannot be opened.\n";
        return -1;
    }
    while (getline(theconf, s)) {
        if (s[0] == '#' || s[0] == ' ') {
            continue;
        }
        if (s_num == 0) {
            if (s[0] == '2') {
                data.dimension = 2;
            } else {
                data.dimension = 3;
            }
        } else if (s_num == 1) {
            data.k = stof(s);
            cout << "k = " << k << endl;
        } else if (s_num == 2) {
            data.num_step = stoi(s);
            cout << "timestep = " << num_step << endl;
        } else if (s_num == 3) {
            if (data.dimension == 2) {
                data.x_width = stoi(s[0]);
                data.y_height = stoi(s[2]);
                data.z_depth = 1;
                cout << "width = " << data.x_width << endl;
                cout << "height = " << data.y_height << endl;
            } else {
                data.x_width = stoi(s[0]);
                data.y_height = stoi(s[2]);
                data.z_depth = stoi(s[4]);
                cout << "width = " << data.x_width << endl;
                cout << "height = " << data.y_height << endl;
                cout << "depth = " << data.z_depth << endl;
            }
        }
        else if (s_num == 4) {
            data.start_tmp = stof(s);
            cout << "default starting temperature = " << data.start_tmp << endl;
        }
        else if (s_num == 5) {
            stringstream ss(s);
            ss.str(s);
            char c;
            BLOCK b;
            if (data.dimension == 2){
                ss >> b.location_x >> c >> b.location_y >> c >> b.bound_x >> c >> b.bound_y >> c >> b.fix_tmp;
                b.location_z = 0;
                b.bound_z = 1;
            }
            else if (data.dimension == 3){
                ss >> b.location_x >> c >> b.location_y >> c >> b.location_z >> c >> b.bound_x >> c >> b.bound_y >> c >> b.bound_z >> c >> b.fix_tmp;
            }
            data.heat_block.push_back(b);
        }
        s_num++;
    }
    fin.close();
}

//Decide the node to be change(0) or not(1)
__device__ bool srcBlock(int x0, int y0, int z0, BLOCK* &heat_block, int count) {
    for (int i = 0; i < count; i++) {
        if (x0 >= heat_block[i].location_x && x0 < heat_block[i].location_x + heat_block[i].bound_x &&
            y0 >= heat_block[i].location_y && y0 < heat_block[i].location_y + heat_block[i].bound_y &&
            z0 >= heat_block[i].location_z && z0 < heat_block[i].location_z + heat_block[i].bound_z)
            return true;
    }
    return false;
}

int main(int argc, char* argv[]) {
    float *x, *y, *z, *u_h, *oldu_h, *tmp_h,;
    float *u_d;
    int i, j, t, iter;
    dim3 numBlocks, threadsPerBlock;
    FILE *fp;

    int dimension, num_step;
    int bound_x, bound_y, bound_z = 1;
    int location_x, location_y, location_z = 0;
    int x_width, y_height, z_depth = 0;
    float k, start_tmp, fix_tmp;

    
    

    if(dimension == 3)
    {
        x = dvector(x_width);
        y = dvector(y_height);
        z = dvector(z_depth);
        u_h = dvector(x_width*y_height*z_depth);
        oldu_h = dvector(x_width*y_height*z_depth);
        u_d = dvector(x_width*y_height*z_depth);

        for (j = 0; j < x_width - 1; j++) x[j] = start_tmp;
        for (i = 0; i < y_height - 1; i++) y[i] = start_tmp;
        for (t = 0; t < z_depth - 1; t++) z[t] = start_tmp;

        for (i = location_x1; i < location_x1+bound_x1; i++) x[i] = fix_tmp1;
        for (i = location_x2; i < location_x2+bound_x2; i++) x[i] = fix_tmp2;
        for (j = location_y1; j < location_y1+bound_y1; j++) y[j] = fix_tmp1;
        for (j = location_y2; j < location_y2+bound_y2; j++) y[j] = fix_tmp2;
        for (t = location_z1; t < location_z1+bound_z1; t++) z[t] = fix_tmp1;
        for (t = location_z2; t < location_z2+bound_z2; t++) z[t] = fix_tmp2;

        zero_matrix(u_h, x_width, y_height, z_depth);
        zero_matrix(oldu_h, x_width, y_height, z_depth);

        //initial
        initialize(u_h, x, y, z, x_width, y_height, z_depth);
        initialize(oldu_h, x, y, z, x_width, y_height, z_depth);

        //allocate temperature arrays on device
        dev_matrix<float> ud(x_width, y_height, z_depth);
        ud.set(u_h, x_width, y_height, z_depth);
        dev_matrix<float> oldud(x_width, y_height, z_depth);
        oldud.set(u_h, x_width, y_height, z_depth);
        dev_matrix<float> tmp_d(x_width, y_height, z_depth);

        //set threads and blocks
        numBlocks = dim3(iDivUp(x_width, BLOCK_SIZE_X), iDivUp(y_height, BLOCK_SIZE_Y), iDivUp(z_depth, BLOCK_SIZE_Z));
        threadsPerBlock = dim3(BlOCK_SIZE_X, BLOCK_SIZE_Y, BLOCK_SIZE_Z);

        //gpu loop
        printf("GPU start.\n");
        for (iter = 0; iter < num_step; iter++) {
            heat3d_gpu << < numBlocks, threadsPerBlock >> > (x_width, y_height, z_depth, k, oldud.getData(), ud.getData());
            tmp_d = ud;
            ud = oldud;
            oldud = tmp_d;
        }
        cudaThreadSynchronize();
        printf("GPU end.\n");

        //copy temperature array from device to host
        oldud.get(&u_d[0], x_width, y_height, z_depth);

        fp = fopen("dev_out.dat", "w");
        print_mat(fp, u_d, x_width, y_height, z_depth);
    } else {
        x = dvector(x_width);
        y = dvector(y_height);
        u_h = dvector(x_width * y_height);
        oldu_h = dvector(x_width * y_height);
        u_d = dvector(x_width * y_height);

        for (j = 0; j < x_width - 1; j++) x[j] = 0.0 + (j * h);
        for (i = 0; i < y_height - 1; i++) y[i] = 0.0 + (i * h);

        zero_matrix(u_h, x_width, y_height);
        zero_matrix(oldu_h, x_width, y_height);

        //initial
        initialize(u_h, x, y, x_width, y_height);
        initialize(oldu_h, x, y, x_width, y_height);

        //allocate temperature arrays on device
        dev_matrix<float> ud(x_width, y_height);
        ud.set(u_h, x_width, y_height);
        dev_matrix<float> oldud(x_width, y_height);
        oldud.set(u_h, x_width, y_height);
        dev_matrix<float> tmp_d(x_width, y_height);

        //set threads and blocks
        numBlocks = dim3(iDivUp(x_width, BLOCK_SIZE_X), iDivUp(y_height, BLOCK_SIZE_Y));
        threadsPerBlock = dim3(BlOCK_SIZE_X, BLOCK_SIZE_Y);

        //gpu loop
        printf("GPU start.\n");
        for (iter = 0; iter < num_step; iter++) {
            heat2d_gpu << < numBlocks, threadsPerBlock >> > (x_width, y_height, k, oldud.getData(), ud.getData());
            tmp_d = ud;
            ud = oldud;
            oldud = tmp_d;
        }
        cudaThreadSynchronize();
        printf("GPU end.\n");

        //copy temperature array from device to host
        oldud.get(&u_d[0], x_width, y_height);

        fp = fopen("dev_out.dat", "w");
        print_mat(fp, u_d, x_width, y_height);
    }

        oldud.~dev_matrix();
        ud.~dev_matrix();
        free(x);
        free(y);
        free(z);
        free(u_h);
        free(oldu_h);
        free(u_d);

        return 0;

}
