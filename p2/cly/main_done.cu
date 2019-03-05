#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <stdlib.h>
#include <cuda.h>

using namespace std;
#define THREAD_SIZE 8
#define DIVIDE(a,b) ((a) + (b) - 1)/(b)

#define HD2(x0,y0) ((x0) + (y0) * data->x)
#define HD2_1(x0,y0) ((x0) + (y0) * data.x)
#define HD3(x0,y0,z0) ((x0) + (y0) * data->x + (z0) * data->x * data->y)
#define HD3_1(x0,y0,z0) ((x0) + (y0) * data.x + (z0) * data.x * data.y)

struct BLOCK
{
    int x;
    int y;
    int z;
    int width;
    int height;
    int depth;
    float temp;
};

struct CONF
{
    int dimension;
    float n;
    int num_step;
    int x;
    int y;
    int z;
    float def_temp; //default starting temperature for nodes
    vector<BLOCK> heat_block;
};

void read(string & path, CONF & data) {
    fstream theconf(path, ios_base::in);
    string s;
    int s_num;
    if (!theconf.is_open()) {
        cerr << "file cannot be opened.\n";
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
            data.n = stof(s);
            cout << "n = " << data.n << endl;
        } else if (s_num == 2) {
            data.num_step = stoi(s);
            cout << "timestep = " << data.num_step << endl;
        } else if (s_num == 3) {
            stringstream ss(s);
            ss.str(s);
            char c;
            if (data.dimension == 2) {
                ss >> data.x >> c >> data.y;
                data.z = 1;
                cout << "width = " << data.x << endl;
                cout << "height = " << data.y << endl;
                cout << "depth = " << data.z << endl;
            }
            else if (data.dimension == 3) {
                ss >> data.x >> c >> data.y >> c >> data.z;
                cout << "width = " << data.x << endl;
                cout << "height = " << data.y << endl;
                cout << "depth = " << data.z << endl;
            }
        }
        else if (s_num == 4) {
            data.def_temp = stof(s);
        }
        else{
            stringstream ss(s);
            ss.str(s);
            char c;
            BLOCK b;
            if (data.dimension == 2) {
                ss >> b.x >> c >> b.y >> c >> b.width >> c >> b.height >> c >> b.temp;
                b.z = 0; b.depth = 1;
                cout << "fixed blocks: " << endl;
                cout << "(" << b.x << ',' << b.y << ',' << b.z << "): width = " << b.width << " height = " << b.height << " depth = " << b.depth << " temp = " << b.temp << endl;
            }
            else if (data.dimension == 3) {
                ss >> b.x >> c >> b.y >> c >> b.z >> c >> b.width >> c >> b.height >> c >> b.depth >> c >> b.temp;
                cout << "fixed blocks: " << endl;
                cout << "(" << b.x << ',' << b.y << ',' << b.z << "): width = " << b.width << " height = " << b.height << " depth = " << b.depth << " temp = " << b.temp << endl;
            }
            data.heat_block.push_back(b);
        }
        s_num++;
    }
    theconf.close();
}

//default temperature in heat blocks
void initialize(CONF &data, vector<float> &u) {
    u.assign(data.x * data.y * data.z, data.def_temp);
    for (int i = 0; i < data.heat_block.size(); i++) {
        BLOCK b = data.heat_block[i];
        for (int m = b.z; m < b.z + b.depth; m++) {
            for (int j = b.y; j < b.y + b.height; j++) {
                for (int n = b.x; n < b.x + b.width; n++) {
                    u[HD3_1(n, j, m)] = b.temp;
                }
            }
        }
    }
}

//output csv file
void print(const CONF &data, const vector<float> &u) {
    char filename[] = "heat2D3Doutput.csv";
    ofstream fout(filename);
    for (int m = 0; m < data.z; m++) {
        for (int j = 0; j < data.y; j++) {
            for (int n = 0; n < data.x - 1; n++) {
                fout << u[HD3_1(n, j, m)] << ',';
            }
            fout << u[HD3_1(data.x - 1, j, m)] << endl;
        }
        fout << endl;
    }
    fout.close();
}

//Decide the node to be change(0) or not(1)
__device__ bool changeBLO(int x0, int y0, BLOCK* &heat_block, int count) {
    for (int i = 0; i < count; i++) {
        if (x0 >= heat_block[i].x && x0 < heat_block[i].x + heat_block[i].width &&
            y0 >= heat_block[i].y && y0 < heat_block[i].y + heat_block[i].height)
            //z0 >= heat_block[i].z && z0 < heat_block[i].z + heat_block[i].depth
            return true;
    }
    return false;
}
__device__ bool changeBLO(int x0, int y0, int z0, BLOCK* &heat_block, int count) {
    for (int i = 0; i < count; i++) {
        if (x0 >= heat_block[i].x && x0 < heat_block[i].x + heat_block[i].width &&
            y0 >= heat_block[i].y && y0 < heat_block[i].y + heat_block[i].height &&
            z0 >= heat_block[i].z && z0 < heat_block[i].z + heat_block[i].depth)
            return true;
    }
    return false;
}

__global__ void heat2D(float *u, float *u_new, CONF *data, BLOCK *pHeater, int count){
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    //int z = blockDim.z * blockIdx.z;
    if (x < data->x && y < data->y) {
        if (changeBLO(x, y, pHeater, count)) {
            // fixed blocks, u_new = u
            u_new[HD2(x, y)] = u[HD2(x, y)];
        }
        else
        {
            if (x == 0) {
                if (y == 0) {
                    u_new[HD2(x, y)] = u[HD2(x, y)] + data->n * (u[HD2(x + 1, y)] + u[HD2(x, y + 1)] - 2 * u[HD2(x, y)]);
                }
                else if (y == data->y - 1) {
                    u_new[HD2(x, y)] = u[HD2(x, y)] + data->n * (u[HD2(x + 1, y)] + u[HD2(x, y - 1)] - 2 * u[HD2(x, y)]);
                }
                else {
                    u_new[HD2(x, y)] = u[HD2(x, y)] + data->n * (u[HD2(x + 1, y)] + u[HD2(x, y - 1)] + u[HD2(x, y + 1)] - 3 * u[HD2(x, y)]);
                }
            }
            else if (x == data->x - 1) {
                if (y == 0) {
                    u_new[HD2(x, y)] = u[HD2(x, y)] + data->n * (u[HD2(x - 1, y)] + u[HD2(x, y + 1)] - 2 * u[HD2(x, y)]);
                }
                else if (y == data->y - 1) {
                    u_new[HD2(x, y)] = u[HD2(x, y)] + data->n * (u[HD2(x - 1, y)] + u[HD2(x, y - 1)] - 2 * u[HD2(x, y)]);
                }
                else {
                    u_new[HD2(x, y)] = u[HD2(x, y)] + data->n * (u[HD2(x - 1, y)] + u[HD2(x, y - 1)] + u[HD2(x, y + 1)] - 3 * u[HD2(x, y)]);
                }
            }
            else {
                if (y == 0) {
                    u_new[HD2(x, y)] = u[HD2(x, y)] + data->n * (u[HD2(x - 1, y)] + u[HD2(x + 1, y)] + u[HD2(x, y + 1)] - 3 * u[HD2(x, y)]);
                }
                else if (y == data->y - 1) {
                    u_new[HD2(x, y)] = u[HD2(x, y)] + data->n * (u[HD2(x - 1, y)] + u[HD2(x + 1, y)] + u[HD2(x, y - 1)] - 3 * u[HD2(x, y)]);
                }
                else {
                    u_new[HD2(x, y)] = u[HD2(x, y)] + data->n * (u[HD2(x - 1, y)] + u[HD2(x + 1, y)] + u[HD2(x, y - 1)] + u[HD2(x, y + 1)] - 4 * u[HD2(x, y)]);
                }
            }
        }
    }
    float *temp = u_new;
    u_new = u;
    u = temp;
}

__global__ void heat3D(float *u, float *u_new, CONF *data, BLOCK *pHeater, int count) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int z = blockDim.z * blockIdx.z;

    if (x < data->x && y < data->y && z < data->z) {
        if (changeBLO(x, y, z, pHeater, count)) {
            u_new[HD3(x, y, z)] = u[HD3(x, y, z)];
        }
        else {
            if (x == 0) {
                if (y == 0) {
                    if (z == 0) {
                        u_new[HD3(x, y, z)] = u[HD3(x, y, z)] + data->n * (u[HD3(x + 1, y, z)] + u[HD3(x, y + 1, z)] + u[HD3(x, y, z + 1)] - 3 * u[HD3(x, y, z)]);
                    }
                    else if (z == data->z - 1) {
                        u_new[HD3(x, y, z)] = u[HD3(x, y, z)] + data->n * (u[HD3(x + 1, y, z)] + u[HD3(x, y + 1, z)] + u[HD3(x, y, z - 1)] - 3 * u[HD3(x, y, z)]);
                    }
                    else {
                        u_new[HD3(x, y, z)] = u[HD3(x, y, z)] + data->n * (u[HD3(x + 1, y, z)] + u[HD3(x, y + 1, z)] + u[HD3(x, y, z - 1)] + u[HD3(x, y, z + 1)] - 4 * u[HD3(x, y, z)]);
                    }
                }
                else if (y == data->y - 1) {
                    if (z == 0) {
                        u_new[HD3(x, y, z)] = u[HD3(x, y, z)] + data->n * (u[HD3(x + 1, y, z)] + u[HD3(x, y - 1, z)] + u[HD3(x, y, z + 1)] - 3 * u[HD3(x, y, z)]);
                    }
                    else if (z == data->z - 1) {
                        u_new[HD3(x, y, z)] = u[HD3(x, y, z)] + data->n * (u[HD3(x + 1, y, z)] + u[HD3(x, y - 1, z)] + u[HD3(x, y, z - 1)] - 3 * u[HD3(x, y, z)]);
                    }
                    else {
                        u_new[HD3(x, y, z)] = u[HD3(x, y, z)] + data->n * (u[HD3(x + 1, y, z)] + u[HD3(x, y - 1, z)] + u[HD3(x, y, z - 1)] + u[HD3(x, y, z + 1)] - 4 * u[HD3(x, y, z)]);
                    }
                }
                else {
                    if (z == 0) {
                        u_new[HD3(x, y, z)] = u[HD3(x, y, z)] + data->n * (u[HD3(x + 1, y, z)] + u[HD3(x, y - 1, z)] + u[HD3(x, y + 1, z)] + u[HD3(x, y, z + 1)] - 4 * u[HD3(x, y, z)]);
                    }
                    else if (z == data->z - 1) {
                        u_new[HD3(x, y, z)] = u[HD3(x, y, z)] + data->n * (u[HD3(x + 1, y, z)] + u[HD3(x, y - 1, z)] + u[HD3(x, y + 1, z)] + u[HD3(x, y, z - 1)] - 4 * u[HD3(x, y, z)]);
                    }
                    else {
                        u_new[HD3(x, y, z)] = u[HD3(x, y, z)] + data->n * (u[HD3(x + 1, y, z)] + u[HD3(x, y - 1, z)] + u[HD3(x, y + 1, z)] + u[HD3(x, y, z - 1)] + u[HD3(x, y, z + 1)] - 5 * u[HD3(x, y, z)]);
                    }
                }
            }
            else if (x == data->x - 1) {
                if (y == 0) {
                    if (z == 0) {
                        u_new[HD3(x, y, z)] = u[HD3(x, y, z)] + data->n * (u[HD3(x - 1, y, z)] + u[HD3(x, y + 1, z)] + u[HD3(x, y, z + 1)] - 3 * u[HD3(x, y, z)]);
                    }
                    else if (z == data->z - 1) {
                        u_new[HD3(x, y, z)] = u[HD3(x, y, z)] + data->n * (u[HD3(x - 1, y, z)] + u[HD3(x, y + 1, z)] + u[HD3(x, y, z - 1)] - 3 * u[HD3(x, y, z)]);
                    }
                    else {
                        u_new[HD3(x, y, z)] = u[HD3(x, y, z)] + data->n * (u[HD3(x - 1, y, z)] + u[HD3(x, y + 1, z)] + u[HD3(x, y, z - 1)] + u[HD3(x, y, z + 1)] - 4 * u[HD3(x, y, z)]);
                    }
                }
                else if (y == data->y - 1) {
                    if (z == 0) {
                        u_new[HD3(x, y, z)] = u[HD3(x, y, z)] + data->n * (u[HD3(x - 1, y, z)] + u[HD3(x, y - 1, z)] + u[HD3(x, y, z + 1)] - 3 * u[HD3(x, y, z)]);
                    }
                    else if (z == data->z - 1) {
                        u_new[HD3(x, y, z)] = u[HD3(x, y, z)] + data->n * (u[HD3(x - 1, y, z)] + u[HD3(x, y - 1, z)] + u[HD3(x, y, z - 1)] - 3 * u[HD3(x, y, z)]);
                    }
                    else {
                        u_new[HD3(x, y, z)] = u[HD3(x, y, z)] + data->n * (u[HD3(x - 1, y, z)] + u[HD3(x, y - 1, z)] + u[HD3(x, y, z - 1)] + u[HD3(x, y, z + 1)] - 4 * u[HD3(x, y, z)]);
                    }
                }
                else {
                    if (z == 0) {
                        u_new[HD3(x, y, z)] = u[HD3(x, y, z)] + data->n * (u[HD3(x - 1, y, z)] + u[HD3(x, y - 1, z)] + u[HD3(x, y + 1, z)] + u[HD3(x, y, z + 1)] - 4 * u[HD3(x, y, z)]);
                    }
                    else if (z == data->z - 1) {
                        u_new[HD3(x, y, z)] = u[HD3(x, y, z)] + data->n * (u[HD3(x - 1, y, z)] + u[HD3(x, y - 1, z)] + u[HD3(x, y + 1, z)] + u[HD3(x, y, z - 1)] - 4 * u[HD3(x, y, z)]);
                    }
                    else {
                        u_new[HD3(x, y, z)] = u[HD3(x, y, z)] + data->n * (u[HD3(x - 1, y, z)] + u[HD3(x, y - 1, z)] + u[HD3(x, y + 1, z)] + u[HD3(x, y, z - 1)] + u[HD3(x, y, z + 1)] - 5 * u[HD3(x, y, z)]);
                    }
                }
            }
            else {
                if (y == 0) {
                    if (z == 0) {
                        u_new[HD3(x, y, z)] = u[HD3(x, y, z)] + data->n * (u[HD3(x - 1, y, z)] + u[HD3(x + 1, y, z)] + u[HD3(x, y + 1, z)] + u[HD3(x, y, z + 1)] - 4 * u[HD3(x, y, z)]);
                    }
                    else if (z == data->z - 1) {
                        u_new[HD3(x, y, z)] = u[HD3(x, y, z)] + data->n * (u[HD3(x - 1, y, z)] + u[HD3(x + 1, y, z)] + u[HD3(x, y + 1, z)] + u[HD3(x, y, z - 1)] - 4 * u[HD3(x, y, z)]);
                    }
                    else {
                        u_new[HD3(x, y, z)] = u[HD3(x, y, z)] + data->n * (u[HD3(x - 1, y, z)] + u[HD3(x + 1, y, z)] + u[HD3(x, y + 1, z)] + u[HD3(x, y, z - 1)] + u[HD3(x, y, z + 1)] - 5 * u[HD3(x, y, z)]);
                    }

                }
                else if (y == data->y - 1) {
                    if (z == 0) {
                        u_new[HD3(x, y, z)] = u[HD3(x, y, z)] + data->n * (u[HD3(x - 1, y, z)] + u[HD3(x + 1, y, z)] + u[HD3(x, y - 1, z)] + u[HD3(x, y, z + 1)] - 4 * u[HD3(x, y, z)]);
                    }
                    else if (z == data->z - 1) {
                        u_new[HD3(x, y, z)] = u[HD3(x, y, z)] + data->n * (u[HD3(x - 1, y, z)] + u[HD3(x + 1, y, z)] + u[HD3(x, y - 1, z)] + u[HD3(x, y, z - 1)] - 4 * u[HD3(x, y, z)]);
                    }
                    else {
                        u_new[HD3(x, y, z)] = u[HD3(x, y, z)] + data->n * (u[HD3(x - 1, y, z)] + u[HD3(x + 1, y, z)] + u[HD3(x, y - 1, z)] + u[HD3(x, y, z - 1)] + u[HD3(x, y, z + 1)] - 5 * u[HD3(x, y, z)]);
                    }
                }
                else {
                    if (z == 0) {
                        u_new[HD3(x, y, z)] = u[HD3(x, y, z)] + data->n * (u[HD3(x - 1, y, z)] + u[HD3(x + 1, y, z)] + u[HD3(x, y - 1, z)] + u[HD3(x, y + 1, z)] + u[HD3(x, y, z + 1)] - 5 * u[HD3(x, y, z)]);
                    }
                    else if (z == data->z - 1) {
                        u_new[HD3(x, y, z)] = u[HD3(x, y, z)] + data->n * (u[HD3(x - 1, y, z)] + u[HD3(x + 1, y, z)] + u[HD3(x, y - 1, z)] + u[HD3(x, y + 1, z)] + u[HD3(x, y, z - 1)] - 5 * u[HD3(x, y, z)]);
                    }
                    else {
                        u_new[HD3(x, y, z)] = u[HD3(x, y, z)] + data->n * (u[HD3(x - 1, y, z)] + u[HD3(x + 1, y, z)] + u[HD3(x, y - 1, z)] + u[HD3(x, y + 1, z)] + u[HD3(x, y, z - 1)] + u[HD3(x, y, z + 1)] - 6 * u[HD3(x, y, z)]);
                    }
                }
            }
        }
    }
    float *temp = u_new;
    u_new = u;
    u = temp;
}

//global variable
CONF data;

int main(int argc, char* argv[]) {

    if (argc != 2) {
        cout << "Input error." << endl;
        return -1;
    }

    dim3 numBlocks, threadsPerBlock;

    //read file
    string path = argv[1];
    read(path, data);

    //initialize
    vector<float> u;
    vector<float> u_new;
    initialize(data, u);
    u_new = u;

    //set threads and blocks
    threadsPerBlock = dim3(THREAD_SIZE, THREAD_SIZE);
    numBlocks = dim3(DIVIDE(data.x, THREAD_SIZE), DIVIDE(data.y, THREAD_SIZE), data.z);

    //allocate temperature array space on device
    float *d_u, *d_u_new, *temp;
    CONF* d_data;
    int size = data.x * data.y * data.z;
    BLOCK* d_heater;
    int count = data.heat_block.size();

    cudaMalloc((void **)&d_u, size * sizeof(float));
    cudaMalloc((void **)&d_u_new, size * sizeof(float));
    cudaMalloc((void **)&d_data, sizeof(CONF));
    cudaMalloc((void **)&d_heater, count * sizeof(BLOCK));

    //copy inputs to device
    cudaMemcpy(d_u_new, &u_new[0], size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_data, &data, sizeof(CONF), cudaMemcpyHostToDevice);
    cudaMemcpy(d_heater, data.heat_block.data(), count * sizeof(BLOCK), cudaMemcpyHostToDevice);
    cudaMemcpy(d_u, &u[0], size * sizeof(float), cudaMemcpyHostToDevice);

    //gpu loop
    for (int i = 0; i < data.num_step; i++) {
        if (data.dimension == 2){
            heat2D <<< numBlocks, threadsPerBlock>>>(d_u, d_u_new, d_data, d_heater, count);
        }
        else{
            heat3D <<< numBlocks, threadsPerBlock>>>(d_u, d_u_new, d_data, d_heater, count);
        }
        temp = d_u_new;
        d_u_new = d_u;
        d_u = temp;
    }
    //copy temperature array from device to host
    cudaMemcpy(&u[0], d_u, size * sizeof(float), cudaMemcpyDeviceToHost);

    print(data, u);

    //free memory
    cudaFree(d_u);
    cudaFree(d_u_new);
    cudaFree(d_data);
    cudaFree(d_heater);

    return 0;
}