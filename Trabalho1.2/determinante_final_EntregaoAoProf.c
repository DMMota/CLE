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

/** \brief variables to be used to calc determinant */
double mult, deter;

int main (int argc, char *argv[]){
	char *fName;
	int rank, cluster_size;
	
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &cluster_size);									
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);										

	MPI_Status status;
	double *matrix;
	int matrix_size = 0;
	//Granting that the executable will open with a filename to compute the matrices.
	if(rank == 0){
		if (argc != 2){
			printf("Usage: ./<ExecutableFilename> <MatrixFilename>\n");
			printf("%6s MatrixFilename: Name the file that contains the matrices\n", "->");
			exit(1);
		}
		
		
		fName = argv[1];
		
		//Master node here the matrix will be read and the Master will send to slaves for calculation
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
		
		//Calculate the time to process the files
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
			printf("result from matrix %d received from %d: %.3e\n",NodeMatrixNumber[status.MPI_SOURCE-1]+1,status.MPI_SOURCE, result);
			
			matrixreceived++;
			MPI_Send(0, 0, MPI_INT, status.MPI_SOURCE, 0, MPI_COMM_WORLD);
			MPI_Send(&matrix_size, 1, MPI_INT, status.MPI_SOURCE, 0, MPI_COMM_WORLD);
			MPI_Send(matrix, matrix_size*matrix_size, MPI_DOUBLE, status.MPI_SOURCE, 0, MPI_COMM_WORLD);
			NodeMatrixNumber[status.MPI_SOURCE-1] = current_matrix;
			current_matrix++;
		}
		while(matrixreceived < matrix_number){
			MPI_Recv(&result, 1, MPI_DOUBLE, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
			printf("result from matrix %d received from %d: %.3e\n",NodeMatrixNumber[status.MPI_SOURCE-1]+1,status.MPI_SOURCE, result);
			matrixreceived++;
		}
		int end_flag = 1;
		for(int node = 1; node < cluster_size;node++)
			MPI_Send(&end_flag, 1, MPI_INT, node, 0, MPI_COMM_WORLD);

		printf("Execution time: %f\n", (double)(clock() - begin_timer) / CLOCKS_PER_SEC);
	}
	// Slaves Node - here they will receive the matrices for calculation. The matrices are in one dimension
	else{
		while(1){
			int end_flag = 0;
			MPI_Recv(&end_flag, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			if(end_flag == 1)
				break;
			MPI_Recv(&matrix_size, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			matrix = (double*)malloc(sizeof(double) * (matrix_size * matrix_size));
			MPI_Recv(matrix, matrix_size*matrix_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			//Transforming the matrix from one dimension to two dimension
			double **matrix_calc = (double **) malloc(sizeof(double *) * matrix_size);
			for (int i = 0; i < matrix_size; i++) {
				matrix_calc[i] = (double *) malloc(sizeof(double) * matrix_size);
				for (int j = 0; j < matrix_size; j++) {
					matrix_calc[i][j] = matrix[matrix_size*i + j];
				} 
			}
			
			/* Determinant Calculation */
			// Gauss Elimination
			for(int k = 0; k < matrix_size-1; k++) {
				for(int i = k+1; i < matrix_size; i++) {
					mult = matrix_calc[i][k]/matrix_calc[k][k];
					matrix_calc[i][k] = 0;
					for(int j = k+1; j <= matrix_size; j++)
						matrix_calc[i][j] -= mult * matrix_calc[k][j];
				}
			}
			//printf ("Gauss\n");

			// determinant calculation
			deter = 1;
			for(int i = 0; i < matrix_size; i++)
				deter *= matrix_calc[i][i];
			
			MPI_Send(&deter, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
		}
	}
	
	printf("Node %d finalized\n", rank);
    MPI_Finalize();
    return 0;
}