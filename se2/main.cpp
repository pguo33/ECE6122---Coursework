#include <iostream>
#include <fstream>
#include <stdio.h>
#include <string>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>

using namespace std;

int main(int argc, char *argv[]){
    const double t1temp = atoi(argv[1]);
    const double t2temp = atoi(argv[2]);
    const double timeStep = atoi(argv[4]);
    const int gridPoint = atoi(argv[3]);
    int num_task;
    int rank;

    MPI_Status stat[4];
    MPI_Request req[4];

    int rc = MPI_Init(&argc, &argv);
    if (rc != MPI_SUCCESS)
    {
        printf("Error starting MPI program. Terminating.\n");
        MPI_Abort(MPI_COMM_WORLD, rc);
    }

    MPI_Comm_size(MPI_COMM_WORLD, &num_task);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int ntmp;
    /*for (int i = 0; i < num_task; i++)
    {*/
    if (rank < gridPoint % num_task)
    {
        ntmp = gridPoint / num_task + 3;
    } else {
        ntmp = gridPoint / num_task + 2;
    }
    //}
    double Tp[ntmp];
    for (int i = 0; i < ntmp; i++)
    {
        Tp[i] = 0;
    }

    //boundary conditions
    if(rank == 0)
    {
        Tp[0] = t1temp;
    }
    if (rank == (num_task - 1))
    {
        Tp[gridPoint/num_task+1] = t2temp;
    }

    //initialize Tp1
    double Tp1[ntmp];
    for (int i = 0; i < ntmp; i++)
    {
        Tp1[i] = Tp[i];
    }

    int t = 0;
    double Tt[gridPoint];
    for (int i = 0; i < gridPoint; i++)
    {
        Tt[i] = 0;
    }
    while(t < timeStep)
    {
        int row_start = 0;
        if (rank == 0)
        {
            row_start = 1;
        }
        int row_end = ntmp;
        if(rank == num_task - 1)
        {
            row_end = ntmp - 1;
        }

        for (int i = row_start; i < row_end; i++)
        {
            Tp1[i] = 0.5 * Tp[i] + 0.25 * Tp[i-1] + 0.25 * Tp[i+1];
        }
        /*if(rank == 0)
        {
            Tp1[0] = t1temp;
        }
        if (rank == (num_task - 1))
        {
            Tp1[gridPoint/num_task+1] = t2temp;
        }*/
        for(int i = 1; i < ntmp-1; i++)
        {
            Tp[i] = Tp1[i];
        }
        int fwd =(rank+1) % num_task;
        int back =(rank-1) % num_task;
        while(back < 0){
            back = back + num_task;
        }

        rc = MPI_Isend(&Tp[ntmp-2], 1, MPI_DOUBLE, fwd , 1, MPI_COMM_WORLD, &req[0]);
        if (rc != MPI_SUCCESS) {
            printf("Error sending from %d to %d \n", rank, fwd);
            MPI_Abort(MPI_COMM_WORLD, rc);
        }

        rc = MPI_Isend(&Tp[1], 1, MPI_DOUBLE, back, 2, MPI_COMM_WORLD, &req[1]);
        if (rc != MPI_SUCCESS) {
            printf("Error sending from %d to %d \n", rank, back);
            MPI_Abort(MPI_COMM_WORLD, rc);
        }

        rc = MPI_Irecv(&Tp[ntmp-1], 1, MPI_DOUBLE, fwd, 2, MPI_COMM_WORLD, &req[2]);
        if (rc != MPI_SUCCESS) {
            printf("Error receiving from %d at %d \n", rank, fwd);
            MPI_Abort(MPI_COMM_WORLD, rc);
        }

        rc = MPI_Irecv(&Tp[0], 1, MPI_DOUBLE, back, 1, MPI_COMM_WORLD, &req[3]);
        if (rc != MPI_SUCCESS) {
            printf("Error receiving from %d at %d \n", rank, back);
            MPI_Abort(MPI_COMM_WORLD, rc);
        }
        MPI_Waitall(4, req, stat);
        if(rank == 0)
        {
            Tp[0] = t1temp;
        }
        if (rank == (num_task - 1))
        {
            Tp[gridPoint/num_task+1] = t2temp;
        }

        for(int i = 0; i < ntmp; i++)
        {
            cout << Tp[i] << ", ";
        }

        MPI_Request reqs2[2*num_task];
        MPI_Status stat2[num_task];
        if (rank >= 0 && rank <= num_task)
        {
            MPI_Isend(&Tp[1], ntmp-2, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &reqs2[rank]);
        }

        if (rank == 0) {
            int size = 0;
            for (int i = 0; i < num_task; ++i) {
                int tmp_size;
                if (i < gridPoint % num_task) {
                    tmp_size = gridPoint / num_task + 1;
                } else {
                    tmp_size = gridPoint / num_task;
                }
                MPI_Irecv(&Tt[size], tmp_size, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, &reqs2[num_task + i]);
                size += tmp_size;

            }
            MPI_Waitall(num_task, &reqs2[num_task], stat2);
        }
        t += 1;
    }
    if (rank == 0){
        char filename[] = "heat1Doutput.csv";
        ofstream fout(filename);
        for(int i = 0; i < gridPoint-1; i++)
        {
            fout << Tt[i] << ", ";
        }
        fout << Tt[gridPoint-1] << endl;
        fout.close();
    }
    MPI_Finalize();
}