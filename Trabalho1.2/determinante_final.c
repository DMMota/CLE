#include <stdio.h>
#include <stddef.h>
#include <stdlib.h>
#include <stdbool.h>
#include <unistd.h>
#include <libgen.h>
#include <math.h>
#include <stdbool.h>
#include <time.h>
#include <errno.h>
#include <string.h>

#include <mpi.h>

int main (int argc, char *argv[]){
	char *fName = "coefData.bin";                                                         
	
	
	int rank, cluster_size;
	
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &cluster_size);									
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);										

	MPI_Status status;
	double *matrix;
	double *matrixCalc;
	double *buffer;
	int matrix_size = 0;
	//master node
	if(rank == 0){
		int matrix_number, current_matrix = 0, matrixreceived = 0;
		FILE *matrix_file = fopen(fName, "rb");
		int *NodeMatrixNumber = (int*)malloc(sizeof(int) * cluster_size-1);
		
        if(matrix_file == NULL) {
            printf("Cannot open the %s file\n"
                   "Please check if is a valid matrix file and the user read permissions!\n", argv[1]);
            return 1;
        }
		
		fread(&matrix_number, sizeof(int), 1, matrix_file);
		fread(&matrix_size, sizeof(int), 1, matrix_file);

		printf("Matrix size: %d | Matrix count: %d\n", matrix_size, matrix_number);
		
		clock_t begin_timer = clock();
		
		for(int node = 1; (node < cluster_size) && (current_matrix < matrix_number); node++){
			matrix = (double*)malloc(sizeof(double) * (matrix_size * matrix_size));
			for(int i = 0; i < matrix_size * matrix_size; i++)
				fread(&matrix[i], sizeof(double), 1, matrix_file);
			
			MPI_Send(0, 0, MPI_INT, node, 0, MPI_COMM_WORLD);
			MPI_Send(&matrix_size, 1, MPI_INT, node, 0, MPI_COMM_WORLD);
			MPI_Send(matrix, matrix_size*matrix_size, MPI_DOUBLE, node, 0, MPI_COMM_WORLD);
			NodeMatrixNumber[node-1] = current_matrix;
			current_matrix++;
		}
		double result = 0;
		while(current_matrix < matrix_number){
			matrix = (double*)malloc(sizeof(double) * (matrix_size * matrix_size));
			for(int i = 0; i < matrix_size * matrix_size; i++)
				fread(&matrix[i], sizeof(double), 1, matrix_file);
			
			MPI_Recv(&result, 1, MPI_DOUBLE, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
			printf("result from matrix[%d] received from %d: %f\n",NodeMatrixNumber[status.MPI_SOURCE-1]+1,status.MPI_SOURCE, result);
			
			matrixreceived++;
			MPI_Send(0, 0, MPI_INT, status.MPI_SOURCE, 0, MPI_COMM_WORLD);
			MPI_Send(&matrix_size, 1, MPI_INT, status.MPI_SOURCE, 0, MPI_COMM_WORLD);
			MPI_Send(matrix, matrix_size*matrix_size, MPI_DOUBLE, status.MPI_SOURCE, 0, MPI_COMM_WORLD);
			NodeMatrixNumber[status.MPI_SOURCE-1] = current_matrix;
			current_matrix++;
		}
		while(matrixreceived < matrix_number){
			MPI_Recv(&result, 1, MPI_DOUBLE, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
			printf("result from matrix[%d] received from %d: %f\n",NodeMatrixNumber[status.MPI_SOURCE-1]+1,status.MPI_SOURCE, result);
			matrixreceived++;
		}
		int end_flag = 1;
		for(int node = 1; node < cluster_size;node++)
			MPI_Send(&end_flag, 1, MPI_INT, node, 0, MPI_COMM_WORLD);

		printf("Execution time: %f\n", (double)(clock() - begin_timer) / CLOCKS_PER_SEC);
	}
	//slaves
	else{
		while(1){
			int end_flag = 0;
			MPI_Recv(&end_flag, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			if(end_flag == 1)
				break;
			MPI_Recv(&matrix_size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			matrix = (double*)malloc(sizeof(double) * (matrix_size * matrix_size));
			MPI_Recv(matrix, matrix_size*matrix_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			double result = 1;
			// Determinant Calculation
	        matrixCalc = (double **) malloc((n) * sizeof(double[n]));
			buffer = matrix;
	        for (k = 0; k < n; ++k)
	            matrixCalc[k] = (double *) malloc((n) * sizeof(double));

	        for (i = 0; i < n; i++)
	            for (j = 0; j < n; j++) {
	                matrixCalc[i][j] = buffer[i * n + j];
	            }


	        result = Partition(matrix, start, end, n);
	        int h = 0;
	        for (h = 0; h < n; ++h)
	            free(matrixCalc[h]);
	        free(matrixCalc);
	        free(buffer);
			MPI_Send(&result, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
		}
	}
	
	printf("Node %d finalized\n", rank);
    MPI_Finalize();
    return 0;
}

double MatrixDeterminant(int nDim, double *pfMatr) {
    double fDet = 1.;
    double fMaxElem;
    double fAcc;
    int i, j, k, m;

    for (k = 0; k < (nDim - 1); k++) // base row of matrix
    {
        // search of line with max element
        fMaxElem = fabs(pfMatr[k * nDim + k]);
        m = k;
        for (i = k + 1; i < nDim; i++) {
            if (fMaxElem < fabs(pfMatr[i * nDim + k])) {
                fMaxElem = pfMatr[i * nDim + k];
                m = i;
            }
        }

        // permutation of base line (index k) and max element line(index m)
        if (m != k) {
            for (i = k; i < nDim; i++) {
                fAcc = pfMatr[k * nDim + i];
                pfMatr[k * nDim + i] = pfMatr[m * nDim + i];
                pfMatr[m * nDim + i] = fAcc;
            }
            fDet *= (-1.);
        }

        if (pfMatr[k * nDim + k] == 0.) return 0.0;

        // trianglulation of matrix
        for (j = (k + 1); j < nDim; j++) // current row of matrix
        {
            fAcc = -pfMatr[j * nDim + k] / pfMatr[k * nDim + k];
            for (i = k; i < nDim; i++) {
                pfMatr[j * nDim + i] = pfMatr[j * nDim + i] + fAcc * pfMatr[k * nDim + i];
            }
        }
    }

    for (i = 0; i < nDim; i++) {
        fDet *= pfMatr[i * nDim + i]; // diagonal elements multiplication
    }

    return fDet;
}

double Partition(double **a, int s, int end, int n) {
    int i, j, j1, j2;
    double det = 0;
    double **m = NULL;

    det = 0;                      // initialize determinant of sub-matrix

    // for each column in sub-matrix
    for (j1 = s; j1 < end; j1++) {
        // get space for the pointer list
        m = (double **) malloc((n - 1) * sizeof(double *));

        for (i = 0; i < n - 1; i++)
            m[i] = (double *) malloc((n - 1) * sizeof(double));

        for (i = 1; i < n; i++) {
            j2 = 0;

            for (j = 0; j < n; j++) {
                if (j == j1) continue;

                m[i - 1][j2] = a[i][j];


                j2++;
            }
        }
        int dim = n - 1;
        double fMatr[dim * dim];
        for (i = 0; i < dim; i++) {
            for (j = 0; j < dim; j++) {
                fMatr[i * dim + j] = m[i][j];
                // printf("%3.2lf    ",fMatr[i*nDim+j]);
            }
            //printf("\n");
        }

        det += pow(-1.0, 1.0 + j1 + 1.0) * a[0][j1] * MatrixDeterminant(dim, fMatr);

        //free(fMatr);
        for (i = 0; i < n - 1; i++) free(m[i]);

        free(m);

    }

    return (det);
}