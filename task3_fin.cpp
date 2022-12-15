#include <iostream>
#include <mpi.h>
#include <omp.h>
#include <ctime>
#include <stdio.h>
#include <cmath>

#define N 500
#define M 500	
int SIZE_X = M + 1;
int SIZE_Y = N + 1;
double H1 = 5.0 / M;
double H2 = 5.0 / N;
double X[M + 1];
double Y[N + 1];

double u(int i, int j) {
	return 2 / (1 + pow(X[i], 2) + pow(Y[j], 2));
}

double k(int i, int j, double sx = 0, double sy = 0) {
	return 1 + pow(X[i] + Y[j] + sy * H2 + sx * H1, 2);
}

double F(int i, int j) {
	return u(i, j) + 8 / pow(1 + X[i] * X[i] + Y[j] * Y[j], 2) + 32 * X[i] * Y[j] / pow(1 + X[i] * X[i] + Y[j] * Y[j], 3);
}

double phi(int i, int j) {
	return u(i, j);
}

double psi(int i, int j, int sign) {
	return sign * 4 * Y[j] * k(i, j) / pow(1 + X[i] * X[i] + Y[j] * Y[j], 2) + u(i, j);
}

void array_copy(double *to, double *from, int starti = 0, int endi = M) {
	#pragma omp parallel for
	for (int i = starti + 1; i < endi; ++i) {
		for (int j = 0; j < N + 1; ++j) {
			to[i * SIZE_Y + j] = from[i * SIZE_Y + j];
		}
	}
}

void array_minus(double *result, double *left, double *right, double coef = 1, int starti = 0, int endi = M) {
	#pragma omp parallel for
	for (int i = starti + 1; i < endi; ++i) {
		for (int j = 0; j < N + 1; ++j) {
			result[i * SIZE_Y + j] = left[i * SIZE_Y + j] - coef * right[i * SIZE_Y + j];
		}
	}
}

double rho(int i, int j) {
	double rho_1 = 1;
	double rho_2 = 1;
	if (i == M || i == 0) {
		rho_1 = 0.5;
	}
	if (j == N || j == 0) {
		rho_2 = 0.5;
	}
	return rho_1 * rho_2;
}

double array_dot(double *left, double *right, int starti = 0, int endi = M) {
	double sum = 0.0;
	#pragma omp parallel for reduction (+:sum)
	for (int i = starti + 1; i < endi; ++i) {
		for (int j = 0; j < N + 1; ++j) {
			sum += H1 * H2 * rho(i, j) * left[i * SIZE_Y + j] * right[i * SIZE_Y + j];
		}
	}
	return sum;
}

double array_norm(double *object, int starti = 0, int endi = M) {
	return pow(array_dot(object, object, starti, endi), 0.5);
}

void array_B_calc(double *B, int starti = 0, int endi = M) {
	#pragma omp parallel for
	for (int i = starti + 1; i < endi; ++i) {
		for (int j = 1; j < N; ++j) {
			B[i * SIZE_Y + j] = F(i, j);
		}
	}
	for (int i = starti + 1; i < endi; ++i){
		B[i * SIZE_Y + N] = F(i, N) + 2 * psi(i, N, -1) / H2;
		B[i * SIZE_Y + 0] = F(i, 0) + 2 * psi(i, 0, +1) / H2;
	}
}

void array_A_calc(double *r, double *w, int starti = 0, int endi = M) {
	double a = 0, b = 0, c = 0, d = 0;
	#pragma omp parallel for
	for (int i = starti + 1; i < endi; ++i) {
		for (int j = 1; j < N; ++j) {
			a = (k(i, j, 0.5, 0) * (w[(i + 1) * SIZE_Y + j] - w[i * SIZE_Y + j]) / H1 - k(i, j, -0.5, 0) * (w[i * SIZE_Y + j] - w[(i - 1) * SIZE_Y + j]) / H1) / H1;
			b = (k(i, j, 0, 0.5) * (w[i * SIZE_Y + j + 1] - w[i * SIZE_Y + j]) / H2 - k(i, j, 0, -0.5) * (w[i * SIZE_Y + j] - w[i * SIZE_Y + j - 1]) / H2) / H2;
			r[i * SIZE_Y + j] = -(a + b) + w[i * SIZE_Y + j];
		}
	}
	for (int i = starti + 1; i < endi; ++i) {
		a = 2 * k(i, N, 0, -0.5) * (w[i * SIZE_Y + N] - w[i * SIZE_Y + N - 1]) / H2 / H2;
		b = (1 + 2 / H2) * w[i * SIZE_Y + N];
		c = (k(i, N, 0.5, 0) * (w[(i + 1) * SIZE_Y + N] - w[i * SIZE_Y + N]) / H1 - k(i, N, -0.5, 0) * (w[i * SIZE_Y +N] - w[(i - 1) * SIZE_Y + N]) / H1) / H1;
		r[i * SIZE_Y + N] = a + b - c;
	}
	for (int i = starti + 1; i < endi; ++i) {
		a = 2 * k(i, 1, 0, -0.5) * (w[i * SIZE_Y + 1] - w[i * SIZE_Y + 0]) / H2 / H2;
		b = (1 + 2 / H2) * w[i * SIZE_Y + 0];
		c = (k(i, 0, 0.5, 0) * (w[(i + 1) * SIZE_Y + 0] - w[i * SIZE_Y + 0]) / H1 - k(i, 0, -0.5, 0) * (w[i * SIZE_Y + 0] - w[(i - 1) * SIZE_Y + 0]) / H1) / H1;
		r[i * SIZE_Y + 0] = -a + b - c;
	}
}

int main(int argc, char *argv[]) {

	int num_proc, id_proc, block_size;
	int starti, endi, i = 0;
	int endStatus = 1;
	double tau, diff, total_diff;
	double start_time, finish_time, local_time, max_time;
	
	double w[SIZE_X * SIZE_Y];
	double pre_w[SIZE_X * SIZE_Y];
	double w_w_pre[SIZE_X * SIZE_Y];
	double r[SIZE_X * SIZE_Y];
	double Ar[SIZE_X * SIZE_Y];
	double Aw[SIZE_X * SIZE_Y];
	double b[SIZE_X * SIZE_Y];

	X[0] = -2.0;
	X[M] = 3.0;
	for (int i = 1; i < M; ++i) {
		X[i] = -2.0 + H1 * i; 
	}

	Y[0] = -1.0;
	Y[N] = 4.0;
	for (int j = 1; j < N; ++j) {
		Y[j] = -1.0 + H2 * j;
	}

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &num_proc);
	MPI_Comm_rank(MPI_COMM_WORLD, &id_proc);
	
	MPI_Request requestTo[2], requestFrom[2];
	MPI_Status statusTo[2], statusFrom[2];

	start_time = MPI_Wtime();
	block_size = (M + 2 * num_proc - 2) / num_proc + ((M + 2 * num_proc - 2) % num_proc != 0);

	starti = id_proc * block_size - 2 * (id_proc - 1) - 2;
	if (id_proc == num_proc - 1)
		endi = M;
	else
		endi = starti + block_size - 1;
	block_size = endi - starti + 1;
	array_B_calc(b, starti, endi);
	
	#pragma omp parallel for
	for (int i = starti; i < endi; ++i) {
		for (int j = 0; j < N + 1; ++j) {
			if (i == 0 || i == M) {
				w[i * SIZE_Y + j] = phi(i, j);
				pre_w[i * SIZE_Y + j] = phi(i, j);
			}
			else {
				w[i * SIZE_Y + j] = 0.0;
				pre_w[i * SIZE_Y + j] = 0.0;
			}
		}
	}
	for (i = 0; i < 100000; i++) {
		array_A_calc(Aw, w, starti, endi);		
		array_minus(r, Aw, b, 1, starti, endi);
		array_A_calc(Ar, r, starti, endi);
		tau = array_dot(Ar, r, starti, endi) / array_dot(Ar, Ar, starti, endi);
		array_minus(w, w, r, tau, starti, endi);
		array_minus(w_w_pre, w, pre_w, 1,  starti, endi);		
		diff = array_dot(w_w_pre, w_w_pre,  starti, endi);
	
		if (num_proc == 1 && pow(diff, 0.5) < 0.000001) {
			break;
		}
		else {
			MPI_Reduce(&diff, &total_diff, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
			if (id_proc == 0) {
				total_diff = pow(total_diff, 0.5);
				//std::cout << "Diff: " << total_diff << std::endl;
				if (total_diff < 0.000001)
					endStatus = 0;
				MPI_Bcast(&endStatus, 1, MPI_INT, 0, MPI_COMM_WORLD);
				if (endStatus == 0)
					break;
			}
			if (id_proc != 0) {
				MPI_Bcast(&endStatus, 1, MPI_INT, 0, MPI_COMM_WORLD);
				if (endStatus == 0)
					break;
			}
			if (id_proc != 0) {
				MPI_Isend(w + SIZE_Y, SIZE_Y, MPI_DOUBLE, id_proc - 1, 0, MPI_COMM_WORLD, &requestFrom[0]);
				MPI_Irecv(w        , SIZE_Y, MPI_DOUBLE, id_proc - 1, 0, MPI_COMM_WORLD, &requestFrom[1]);
				MPI_Waitall(2, requestFrom, statusFrom);
			}
			if (id_proc != num_proc - 1) {
				MPI_Isend(w + (endi - 1) * SIZE_Y, SIZE_Y, MPI_DOUBLE, id_proc + 1, 0, MPI_COMM_WORLD, &requestTo[0]);
				MPI_Irecv(w + endi * SIZE_Y      , SIZE_Y, MPI_DOUBLE, id_proc + 1, 0, MPI_COMM_WORLD, &requestTo[1]);
				MPI_Waitall(2, requestTo, statusTo);
			}
		}
		array_copy(pre_w, w, starti, endi);
	}
	
	finish_time = MPI_Wtime();
	local_time = finish_time - start_time;

	//printf("%d\n", i);
	//printf("%f\n", total_diff);
	printf("%4f\n", local_time);	

	MPI_Finalize();
	return 0;
}
