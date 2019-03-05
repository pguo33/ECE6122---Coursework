#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <stdlib.h>
#include <cuda.h>

using namespace std;

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
    int type;
    float k;
    int timestep;
    int x;
    int y;
    int z;
    float def_temp; //default starting temperature for nodes
    vector<BLOCK> Heater;
};

#define THREAD_SIZE 8
#define DIVIDE(a,b) ((a) + (b) - 1)/(b)

//the position of (x0,y0,z0) in 1D array
#define POS(x0,y0,z0) ((x0) + (y0) * data->x + (z0) * data->x * data->y)
#define POS1(x0,y0,z0) ((x0) + (y0) * data.x + (z0) * data.x * data.y)

void ReadFile(string & file_path, CONF & data) {
    fstream fin(file_path, ios_base::in);
    string tmp;

    while (getline(fin, tmp)) {
        if (tmp.find('#') != string::npos) {
            //this line should be ignored
            continue;
        }
        else break;
    }
    int flag = 0;
    //read type
    if (tmp.find("2D") != string::npos) {
        data.type = 2;
    }
    else if (tmp.find("3D") != string::npos) {
        data.type = 3;
    }
    flag = 1;

    while (getline(fin, tmp)) {
        if (tmp.find('#') != string::npos || tmp[0] == '\r') {
            //this line should be ignored
            continue;
        }
        else if (flag == 1) {
            //read k
            data.k = stof(tmp);
            flag = 2;
        }
        else if (flag == 2) {
            //read timesteps
            data.timestep = stoi(tmp);
            flag = 3;
        }
        else if (flag == 3) {
            //read x,y,z
            stringstream ss(tmp);
            ss.str(tmp);
            char c;
            if (data.type == 2) {
                ss >> data.x >> c >> data.y;
                data.z = 1;
            }
            else if (data.type == 3) {
                ss >> data.x >> c >> data.y >> c >> data.z;
            }
            flag = 4;
        }
        else if (flag == 4) {
            //read default starting temperature
            data.def_temp = stof(tmp);
            flag = 5;
        }
        else if (flag == 5) {
            //read temperature blocks
            stringstream ss(tmp);
            ss.str(tmp);
            char c;
            BLOCK b;
            if (data.type == 2) {
                ss >> b.x >> c >> b.y >> c >> b.width >> c >> b.height >> c >> b.temp;
                b.z = 0; b.depth = 1;
            }
            else if (data.type == 3) {
                ss >> b.x >> c >> b.y >> c >> b.z >> c >> b.width >> c >> b.height >> c >> b.depth >> c >> b.temp;
            }
            data.Heater.push_back(b);
        }
    }
    fin.close();
}

void PrintConf(CONF & data) {
    cout.setf(ios::fixed);
    cout << "Dimension: " << data.type << "D" << endl;
    cout << "k = " << data.k << endl;
    cout << "timestep = " << data.timestep << endl;
    cout << "grid axis: \n" << "x = " << data.x << ", y = " << data.y << ", z = " << data.z << endl;
    cout << "default temperature =  " << data.def_temp << endl;
    cout << "fixed blocks: " << endl;
    for (auto &i : data.Heater) {
        cout << "(" << i.x << ',' << i.y << ',' << i.z << "): width = " << i.width << " height = " << i.height << " depth = " << i.depth << " temp = " << i.temp << endl;
    }
    cout << endl;
}

void initialize(CONF &data, vector<float> &u) {
    // set all nodes to default temperature
    u.assign(data.x * data.y * data.z, data.def_temp);

    // set the heater blocks' temperature
    for (int i = 0; i < data.Heater.size(); i++) {
        BLOCK b = data.Heater[i];
        for (int m = b.z; m < b.z + b.depth; m++) {
            for (int j = b.y; j < b.y + b.height; j++) {
                for (int k = b.x; k < b.x + b.width; k++) {
                    u[POS1(k, j, m)] = b.temp;
                }
            }
        }
    }
}

void print(const CONF &data, const vector<float> &u) {
    ofstream ofp;
    ofp.open("output.csv", ios_base::out);
    cout.setf(ios::fixed);
    for (int m = 0; m < data.z; m++) {
        for (int j = 0; j < data.y; j++) {
            for (int k = 0; k < data.x - 1; k++) {
                //cout << u[POS1(k, j, m)] << "  ";
                ofp << u[POS1(k, j, m)] << ',';
            }
            //cout << u[POS1(data.x - 1, j, m)] << endl;
            ofp << u[POS1(data.x - 1, j, m)] << endl;
        }
        //cout << endl;
        ofp << endl;
    }
    //cout << endl;
    ofp.close();
}

// if node belongs to the fixed blocks, return true; otherwise return false.
__device__ bool srcBlock(int x0, int y0, int z0, BLOCK* &Heater, int count) {
    for (int i = 0; i < count; i++) {
        if (x0 >= Heater[i].x && x0 < Heater[i].x + Heater[i].width &&
                y0 >= Heater[i].y && y0 < Heater[i].y + Heater[i].height &&
                z0 >= Heater[i].z && z0 < Heater[i].z + Heater[i].depth)
            return true;
    }
    return false;
}
__global__ void heat2D3D(float *u, float *u_new, CONF *data, BLOCK *pHeater, int count) {
    int x = blockDim.x * blockIdx.x + threadIdx.x;
    int y = blockDim.y * blockIdx.y + threadIdx.y;
    int z = blockDim.z * blockIdx.z;
    //2D: Tnew = Told + k ∗ (Ttop + Tbottom + Tleft + Tright − 4 ∗ Told)
    //3D: Tnew = Told + k ∗ (Tfront + Tback + Ttop + Tbottom + Tleft + Tright − 6 ∗ Told)

    if (x < data->x && y < data->y && z < data->z) {
        if (srcBlock(x, y, z, pHeater, count)) {
            // fixed blocks, u_new = u
            u_new[POS(x, y, z)] = u[POS(x, y, z)];
        }
        else if (data->type == 2) {
            if (x == 0) {
                if (y == 0) {
                    u_new[POS(x, y, z)] = u[POS(x, y, z)] + data->k * (u[POS(x + 1, y, z)] + u[POS(x, y + 1, z)] - 2 * u[POS(x, y, z)]);
                }
                else if (y == data->y - 1) {
                    u_new[POS(x, y, z)] = u[POS(x, y, z)] + data->k * (u[POS(x + 1, y, z)] + u[POS(x, y - 1, z)] - 2 * u[POS(x, y, z)]);
                }
                else {
                    //x = 0, 0 < y < Y-1
                    u_new[POS(x, y, z)] = u[POS(x, y, z)] + data->k * (u[POS(x + 1, y, z)] + u[POS(x, y - 1, z)] + u[POS(x, y + 1, z)] - 3 * u[POS(x, y, z)]);
                }
            }
            else if (x == data->x - 1) {
                if (y == 0) {
                    u_new[POS(x, y, z)] = u[POS(x, y, z)] + data->k * (u[POS(x - 1, y, z)] + u[POS(x, y + 1, z)] - 2 * u[POS(x, y, z)]);
                }
                else if (y == data->y - 1) {
                    u_new[POS(x, y, z)] = u[POS(x, y, z)] + data->k * (u[POS(x - 1, y, z)] + u[POS(x, y - 1, z)] - 2 * u[POS(x, y, z)]);
                }
                else {
                    //x = X-1, 0 < y < Y-1
                    u_new[POS(x, y, z)] = u[POS(x, y, z)] + data->k * (u[POS(x - 1, y, z)] + u[POS(x, y - 1, z)] + u[POS(x, y + 1, z)] - 3 * u[POS(x, y, z)]);
                }
            }
            else {
                //0 < x < X-1
                if (y == 0) {
                    u_new[POS(x, y, z)] = u[POS(x, y, z)] + data->k * (u[POS(x - 1, y, z)] + u[POS(x + 1, y, z)] + u[POS(x, y + 1, z)] - 3 * u[POS(x, y, z)]);
                }
                else if (y == data->y - 1) {
                    u_new[POS(x, y, z)] = u[POS(x, y, z)] + data->k * (u[POS(x - 1, y, z)] + u[POS(x + 1, y, z)] + u[POS(x, y - 1, z)] - 3 * u[POS(x, y, z)]);
                }
                else {
                    u_new[POS(x, y, z)] = u[POS(x, y, z)] + data->k * (u[POS(x - 1, y, z)] + u[POS(x + 1, y, z)] + u[POS(x, y - 1, z)] + u[POS(x, y + 1, z)] - 4 * u[POS(x, y, z)]);
                }
            }
        }
        else if (data->type == 3) {
            if (x == 0) {
                if (y == 0) {
                    if (z == 0) {
                        u_new[POS(x, y, z)] = u[POS(x, y, z)] + data->k * (u[POS(x + 1, y, z)] + u[POS(x, y + 1, z)] + u[POS(x, y, z + 1)] - 3 * u[POS(x, y, z)]);
                    }
                    else if (z == data->z - 1) {
                        u_new[POS(x, y, z)] = u[POS(x, y, z)] + data->k * (u[POS(x + 1, y, z)] + u[POS(x, y + 1, z)] + u[POS(x, y, z - 1)] - 3 * u[POS(x, y, z)]);
                    }
                    else {
                        u_new[POS(x, y, z)] = u[POS(x, y, z)] + data->k * (u[POS(x + 1, y, z)] + u[POS(x, y + 1, z)] + u[POS(x, y, z - 1)] + u[POS(x, y, z + 1)] - 4 * u[POS(x, y, z)]);
                    }
                }
                else if (y == data->y - 1) {
                    if (z == 0) {
                        u_new[POS(x, y, z)] = u[POS(x, y, z)] + data->k * (u[POS(x + 1, y, z)] + u[POS(x, y - 1, z)] + u[POS(x, y, z + 1)] - 3 * u[POS(x, y, z)]);
                    }
                    else if (z == data->z - 1) {
                        u_new[POS(x, y, z)] = u[POS(x, y, z)] + data->k * (u[POS(x + 1, y, z)] + u[POS(x, y - 1, z)] + u[POS(x, y, z - 1)] - 3 * u[POS(x, y, z)]);
                    }
                    else {
                        u_new[POS(x, y, z)] = u[POS(x, y, z)] + data->k * (u[POS(x + 1, y, z)] + u[POS(x, y - 1, z)] + u[POS(x, y, z - 1)] + u[POS(x, y, z + 1)] - 4 * u[POS(x, y, z)]);
                    }
                }
                else {
                    //x = 0, 0 < y < Y-1
                    if (z == 0) {
                        u_new[POS(x, y, z)] = u[POS(x, y, z)] + data->k * (u[POS(x + 1, y, z)] + u[POS(x, y - 1, z)] + u[POS(x, y + 1, z)] + u[POS(x, y, z + 1)] - 4 * u[POS(x, y, z)]);
                    }
                    else if (z == data->z - 1) {
                        u_new[POS(x, y, z)] = u[POS(x, y, z)] + data->k * (u[POS(x + 1, y, z)] + u[POS(x, y - 1, z)] + u[POS(x, y + 1, z)] + u[POS(x, y, z - 1)] - 4 * u[POS(x, y, z)]);
                    }
                    else {
                        u_new[POS(x, y, z)] = u[POS(x, y, z)] + data->k * (u[POS(x + 1, y, z)] + u[POS(x, y - 1, z)] + u[POS(x, y + 1, z)] + u[POS(x, y, z - 1)] + u[POS(x, y, z + 1)] - 5 * u[POS(x, y, z)]);
                    }
                }
            }
            else if (x == data->x - 1) {
                if (y == 0) {
                    if (z == 0) {
                        u_new[POS(x, y, z)] = u[POS(x, y, z)] + data->k * (u[POS(x - 1, y, z)] + u[POS(x, y + 1, z)] + u[POS(x, y, z + 1)] - 3 * u[POS(x, y, z)]);
                    }
                    else if (z == data->z - 1) {
                        u_new[POS(x, y, z)] = u[POS(x, y, z)] + data->k * (u[POS(x - 1, y, z)] + u[POS(x, y + 1, z)] + u[POS(x, y, z - 1)] - 3 * u[POS(x, y, z)]);
                    }
                    else {
                        u_new[POS(x, y, z)] = u[POS(x, y, z)] + data->k * (u[POS(x - 1, y, z)] + u[POS(x, y + 1, z)] + u[POS(x, y, z - 1)] + u[POS(x, y, z + 1)] - 4 * u[POS(x, y, z)]);
                    }
                }
                else if (y == data->y - 1) {
                    if (z == 0) {
                        u_new[POS(x, y, z)] = u[POS(x, y, z)] + data->k * (u[POS(x - 1, y, z)] + u[POS(x, y - 1, z)] + u[POS(x, y, z + 1)] - 3 * u[POS(x, y, z)]);
                    }
                    else if (z == data->z - 1) {
                        u_new[POS(x, y, z)] = u[POS(x, y, z)] + data->k * (u[POS(x - 1, y, z)] + u[POS(x, y - 1, z)] + u[POS(x, y, z - 1)] - 3 * u[POS(x, y, z)]);
                    }
                    else {
                        u_new[POS(x, y, z)] = u[POS(x, y, z)] + data->k * (u[POS(x - 1, y, z)] + u[POS(x, y - 1, z)] + u[POS(x, y, z - 1)] + u[POS(x, y, z + 1)] - 4 * u[POS(x, y, z)]);
                    }
                }
                else {
                    //x = X-1, 0 < y < Y-1
                    if (z == 0) {
                        u_new[POS(x, y, z)] = u[POS(x, y, z)] + data->k * (u[POS(x - 1, y, z)] + u[POS(x, y - 1, z)] + u[POS(x, y + 1, z)] + u[POS(x, y, z + 1)] - 4 * u[POS(x, y, z)]);
                    }
                    else if (z == data->z - 1) {
                        u_new[POS(x, y, z)] = u[POS(x, y, z)] + data->k * (u[POS(x - 1, y, z)] + u[POS(x, y - 1, z)] + u[POS(x, y + 1, z)] + u[POS(x, y, z - 1)] - 4 * u[POS(x, y, z)]);
                    }
                    else {
                        u_new[POS(x, y, z)] = u[POS(x, y, z)] + data->k * (u[POS(x - 1, y, z)] + u[POS(x, y - 1, z)] + u[POS(x, y + 1, z)] + u[POS(x, y, z - 1)] + u[POS(x, y, z + 1)] - 5 * u[POS(x, y, z)]);
                    }
                }
            }
            else {
                //0 < x < X-1
                if (y == 0) {
                    if (z == 0) {
                        u_new[POS(x, y, z)] = u[POS(x, y, z)] + data->k * (u[POS(x - 1, y, z)] + u[POS(x + 1, y, z)] + u[POS(x, y + 1, z)] + u[POS(x, y, z + 1)] - 4 * u[POS(x, y, z)]);
                    }
                    else if (z == data->z - 1) {
                        u_new[POS(x, y, z)] = u[POS(x, y, z)] + data->k * (u[POS(x - 1, y, z)] + u[POS(x + 1, y, z)] + u[POS(x, y + 1, z)] + u[POS(x, y, z - 1)] - 4 * u[POS(x, y, z)]);
                    }
                    else {
                        u_new[POS(x, y, z)] = u[POS(x, y, z)] + data->k * (u[POS(x - 1, y, z)] + u[POS(x + 1, y, z)] + u[POS(x, y + 1, z)] + u[POS(x, y, z - 1)] + u[POS(x, y, z + 1)] - 5 * u[POS(x, y, z)]);
                    }

                }
                else if (y == data->y - 1) {
                    if (z == 0) {
                        u_new[POS(x, y, z)] = u[POS(x, y, z)] + data->k * (u[POS(x - 1, y, z)] + u[POS(x + 1, y, z)] + u[POS(x, y - 1, z)] + u[POS(x, y, z + 1)] - 4 * u[POS(x, y, z)]);
                    }
                    else if (z == data->z - 1) {
                        u_new[POS(x, y, z)] = u[POS(x, y, z)] + data->k * (u[POS(x - 1, y, z)] + u[POS(x + 1, y, z)] + u[POS(x, y - 1, z)] + u[POS(x, y, z - 1)] - 4 * u[POS(x, y, z)]);
                    }
                    else {
                        u_new[POS(x, y, z)] = u[POS(x, y, z)] + data->k * (u[POS(x - 1, y, z)] + u[POS(x + 1, y, z)] + u[POS(x, y - 1, z)] + u[POS(x, y, z - 1)] + u[POS(x, y, z + 1)] - 5 * u[POS(x, y, z)]);
                    }
                }
                else {
                    if (z == 0) {
                        u_new[POS(x, y, z)] = u[POS(x, y, z)] + data->k * (u[POS(x - 1, y, z)] + u[POS(x + 1, y, z)] + u[POS(x, y - 1, z)] + u[POS(x, y + 1, z)] + u[POS(x, y, z + 1)] - 5 * u[POS(x, y, z)]);
                    }
                    else if (z == data->z - 1) {
                        u_new[POS(x, y, z)] = u[POS(x, y, z)] + data->k * (u[POS(x - 1, y, z)] + u[POS(x + 1, y, z)] + u[POS(x, y - 1, z)] + u[POS(x, y + 1, z)] + u[POS(x, y, z - 1)] - 5 * u[POS(x, y, z)]);
                    }
                    else {
                        u_new[POS(x, y, z)] = u[POS(x, y, z)] + data->k * (u[POS(x - 1, y, z)] + u[POS(x + 1, y, z)] + u[POS(x, y - 1, z)] + u[POS(x, y + 1, z)] + u[POS(x, y, z - 1)] + u[POS(x, y, z + 1)] - 6 * u[POS(x, y, z)]);
                    }
                }
            }
        }
    }

    float *temp = u_new;
    u_new = u;
    u = temp;
}

//glocal variables
CONF data;

int main(int argc, char* argv[]) {

    if (argc != 2) {
        cout << "Input parameters error!" << endl;
        return 1;
    }

    /*****Read Configuration File*****/
    string file_path = argv[1];
    //string file_path = "C:\\Users\\leyic\\Documents\\Sublime\\6122-p2\\3D.conf";
    ReadFile(file_path, data);
    PrintConf(data);

    /*****Initialize Data*****/
    vector<float> u;
    vector<float> u_new;
    initialize(data, u);
    u_new = u;
    //print(data, u);
    // for (auto &i : u) {
    //     cout << i << " ";
    // } cout << endl;

    /******Cuda Initialize*****/
    dim3 threads_num = dim3(THREAD_SIZE, THREAD_SIZE);
    dim3 blocks_num = dim3(DIVIDE(data.x, THREAD_SIZE), DIVIDE(data.y, THREAD_SIZE), data.z);

    //allocate space for device copies
    float *d_u, *d_u_new, *temp;
    CONF* d_data;
    int size = data.x * data.y * data.z;
    BLOCK* d_heater;
    int count = data.Heater.size();

    cudaMalloc((void **)&d_u, size * sizeof(float));
    cudaMalloc((void **)&d_u_new, size * sizeof(float));
    cudaMalloc((void **)&d_data, sizeof(CONF));
    cudaMalloc((void **)&d_heater, count * sizeof(BLOCK));


    //copy inputs to device
    cudaMemcpy(d_u_new, &u_new[0], size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_data, &data, sizeof(CONF), cudaMemcpyHostToDevice);
    cudaMemcpy(d_heater, data.Heater.data(), count * sizeof(BLOCK), cudaMemcpyHostToDevice);
    cudaMemcpy(d_u, &u[0], size * sizeof(float), cudaMemcpyHostToDevice);

    /*****Compute Heat Diffusion*****/
    for (int i = 0; i < data.timestep; i++) {
        heat2D3D <<< blocks_num, threads_num>>>(d_u, d_u_new, d_data, d_heater, count);
        temp = d_u_new;
        d_u_new = d_u;
        d_u = temp;
    }
    cudaMemcpy(&u[0], d_u, size * sizeof(float), cudaMemcpyDeviceToHost);

    print(data, u);

    /*****Free*****/
    cudaFree(d_u);
    cudaFree(d_u_new);
    cudaFree(d_data);
    cudaFree(d_heater);

    return 0;
}