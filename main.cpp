#include <omp.h>
#include <iostream>
#include <vector>
#include <iomanip>
#include <fstream>


using namespace std;

double ** arr;
int n;
double timer;

void read_array(string s){
    ifstream fin(s);
    fin>>n;
    arr = new double*[n];
    for(int i = 0; i < n; i++){
        arr[i] = new double[n];
        int k = 0;
        fin>>k;
        for(int j = 0; j < k; j++){
            int ind;
            double val;
            fin>>ind>>val;
            arr[i][ind] = val;
        }
    }
    fin.close();
}

double LU(double **A, double** L,
        double** U, int n)
{
    U = new double*[n];
    for(int i = 0; i < n; i++) {
        U[i] = new double[n];
        for (int j = 0; j < n; j++)
            U[i][j] = A[i][j];
    }
    for(int i = 0; i < n; i++)
        for(int j = i; j < n; j++)
            L[j][i]=U[j][i]/U[i][i];

    for(int k = 1; k < n; k++)
    {
        for(int i = k-1; i < n; i++)
            for(int j = i; j < n; j++)
                L[j][i]=U[j][i]/U[i][i];

        for(int i = k; i < n; i++)
            for(int j = k-1; j < n; j++)
                U[i][j]=U[i][j]-L[i][k-1]*U[k-1][j];
    }
    double def = 1.0;
    for(int i =0 ; i < n; i++)
        def*= U[i][i];
    return def;
}

double Parallel_LU(double** A, double** L,
        double** U, int n)
{
    //cout<<"1"<<endl;
    U = new double*[n];
#pragma omp parallel for
    for(int i = 0; i < n; i++) {
        U[i] = new double[n];
        for (int j = 0; j < n; j++)
            U[i][j] = A[i][j];
    }
//    cout<<"2"<<endl;
#pragma omp parallel for
    for(int i = 0; i < n; i++)
        for(int j = i; j < n; j++)
            L[j][i]=U[j][i]/U[i][i];
//    cout<<"3"<<endl;

    for(int k = 1; k < n; k++)
    {
#pragma omp parallel for
        for(int i = k-1; i < n; i++)
            for(int j = i; j < n; j++)
                L[j][i]=U[j][i]/U[i][i];
#pragma omp parallel for
        for(int i = k; i < n; i++)
            for(int j = k-1; j < n; j++)
                U[i][j]=U[i][j]-L[i][k-1]*U[k-1][j];
    }
//    cout<<"4"<<endl;
    double def = 1.0;
#pragma omp parallel for reduction (* : def)
    for(int i =0 ; i < n; i++)
        def*= U[i][i];
    return def;
}

/*void proisv(double** A, double** B,
            double** R, int n)
{
    for(int i = 0; i < n; i++)
        for(int j = 0; j < n; j++)
            for(int k = 0; k < n; k++)
                R[i][j] += A[i][k] * B[k][j];
}*/

void show(double** A, int n)
{
    for(int i = 0; i < n; i++)
    {
        for(int j = 0; j < n; j++)
        {
            cout<<setw(10) <<"\t"<< A[i][j] << "\t";
        }
        cout << endl;
    }
}

int main()
{
    const int n = 1000;
    double** L = new double*[n];
    double** U = new double*[n];
    double** R = new double*[n];
    for(int i = 0; i < n; i++)
    {
        L[i] = new double[n];
        U[i] = new double[n];
        R[i] = new double[n];
        for(int j = 0; j < n; j++)
        {
            L[i][j] = 0;
            U[i][j] = 0;
            R[i][j] = 0;
        }
    }

    read_array("matrix4.txt");
    cout<<n<<endl;
    timer = omp_get_wtime();
    cout<<"def(non_parallel) = "<<LU(arr,L,U,n);
    double time = omp_get_wtime() - timer;
    cout<<" time = "<<time<<endl;
    timer = omp_get_wtime();
    cout<<"def(parallel) = "<<Parallel_LU(arr,L,U,n);
    time = omp_get_wtime() - timer;
    cout<<" time = "<<time<<endl;
    cout<<"finish"<<endl;
    return 0;
}
