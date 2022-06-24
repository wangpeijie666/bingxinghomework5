#include "mpi.h"
#include <iostream>
#include<algorithm>
using namespace std;
const int mpisize = 8,n=128;
int main(int argc, char* argv[])
{
	int rank;
	double st, ed;
	MPI_Status status;
	MPI_Init(&argc, &argv);
	float m1[n][n];
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	int r1 = rank * (n / mpisize), r2 = (rank == mpisize - 1) ? n - 1 : (rank + 1)*(n / mpisize) - 1;
	if (rank == 0) {
		for (int i = 0; i < n; i++) {
			for (int j = 0; j < n; j++)
				m1[i][j] = 0;
			m1[i][i] = 1.0;
			for (int j = i + 1; j < n; j++)
				m1[i][j] = rand() % 1000 + 1;
		}
		for (int k = 0; k < n; k++)
			for (int i = k + 1; i < n; i++)
				for (int j = 0; j < n; j++)
					m1[i][j] = int((m1[i][j] + m1[k][j])) % 1000 + 1.0;
		for (int j = 1; j < mpisize; j++) {
			int t1 = j * (n / mpisize), t2 = (j == mpisize - 1) ? n - 1 : (j + 1)*(n / mpisize) - 1;
			MPI_Send(&m1[t1][0], n*(t2-t1+1), MPI_FLOAT, j, n + 1, MPI_COMM_WORLD);
		}
	}
	else
		MPI_Recv(&m1[r1][0], n*(r2-r1+1), MPI_FLOAT,0, n+1, MPI_COMM_WORLD, &status);
	MPI_Barrier(MPI_COMM_WORLD);
	st = MPI_Wtime();
	for (int k = 0; k < n; k++) {
		if (rank == 0) {
			for (int j = k + 1; j < n; j++)
				m1[k][j] /= m1[k][k];
			m1[k][k] = 1.0;
			for (int j = 1; j < mpisize; j++)
				MPI_Send(&m1[k][0], n, MPI_FLOAT, j, k + 1, MPI_COMM_WORLD);
		}
		else
			MPI_Recv(&m1[k][0], n, MPI_FLOAT, 0, k + 1, MPI_COMM_WORLD, &status);
		if (r2 >= k + 1) {
			for (int i = max(k + 1, r1); i <= r2; i++) {
				for (int j = k + 1; j < n; j++)
					m1[i][j] -= m1[i][k] * m1[k][j];
				m1[i][k] = 0;
				if (i == k + 1 && rank != 0)
					MPI_Send(&m1[i][0], n, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
			}
		}
		if (rank == 0 && k + 1 > r2&&k+1<n)
			MPI_Recv(&m1[k + 1][0], n, MPI_FLOAT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
	}
	MPI_Barrier(MPI_COMM_WORLD);
	ed = MPI_Wtime();
	MPI_Finalize();
	if (rank == 0) {
		cout<<"N="<<n<<" mpsize="<<mpisize<<" time:="<<(ed - st)*1000<<"ms"<<endl;
	}
	return 0;
}
