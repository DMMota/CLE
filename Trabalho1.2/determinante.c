/**
 *
 *  \author Diogo Martins Mota - May 2019
 *  \author Clony - May 2019
 */

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
#include "determinante.h"

/** \brief number of square matrices whose determinant is to be computed */
int nMat;

/** \brief order of the square matrices whose determinant is to be computed */
int order;

/** \brief pointer to the storage area of matrices coefficients */
double *mat;

/** \brief pointer to the storage area of matrices determinants */
double *det;

/** \brief pointer to the binary stream associated with the file in processing */
static FILE *f;

/** \brief amount of matrix calculation per process */
int amountPerProcess;

/** \brief matrix to be filled */
double **matrix;

/** \brief buffer of matrix to be filled */
double *buffer;

/** \brief buffer of matrix to be filled */
double *bufPerProc;

/** \brief variables to be used to calc determinant */
double mult, deter;

/** \brief variables to be used to calc time */
double StartTime, EndTime;

/**
 *  \param argc number arguments in the command line
 *  \param argv array of pointers to the arguments
 */
int main (int argc, char *argv[]){
	char *fName = "coefData.bin";                                                         /* file name, set to default */

	int totalProcesses, process_id;														/* number of processes and ids */
	int count;

	/* Initializing text processing threads application defined thread id arrays */
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &totalProcesses);									/* MPI size com numero total de processos */
	MPI_Comm_rank(MPI_COMM_WORLD, &process_id);										/* MPI rank com id do processo */

	MPI_Status status;

	/* Start time */
	StartTime = MPI_Wtime();

	/* master work */
	if(process_id == MASTER) {
		printf ("Entrei processo master.\n");

		/* open the file for reading */
		openFile (fName);
		amountPerProcess = nMat / totalProcesses;
		buffer = (double *) malloc(sizeof(double)  * nMat * order * order);
		bufPerProc = (double *) malloc(sizeof(double) * order * order * nMat);

		/* fill the buffer with matrix values */
		fread(buffer, sizeof(double), nMat * order * order, f);

		for(int proc = 0; proc < totalProcesses; proc++){

			/* divide into smaller buffers */
			count = 0;
			for(int w = 0; w < nMat; w++){
				for (int i = 0; i < order; i++){
					for (int j = 0; j < order; j++)
						bufPerProc[count] = buffer[((order*order)) + (i * order + j)];
					count++;
					if(count == amountPerProcess)
						count = 0;
				}
			}

			/* do master work */
			if(proc == MASTER){

				/* aloca memoria para matriz */
				matrix = (double **) malloc((order) * sizeof(double[order]));

				/* faz calculo do determinante das matrizes */
				for(int x = 0; x < amountPerProcess; x++){

					matrix[x] = (double *) malloc((order) * sizeof(double));

					/* preenche matriz atraves do buffer */
					for (int i = 0; i < order; i++){
						matrix[i] = (double *) malloc((order) * sizeof(double));
						for (int j = 0; j < order; j++)
							matrix[i][j] = bufPerProc[/* (((MASTER+1)*x) */ ((order*order)) + (i * order + j)];
					}
					printf ("Preenchimento\n");

					/* eliminacao de gauss */
					for(int k = 0; k < order-1; k++) {
						for(int i = k+1; i < order; i++) {
							mult = matrix[i][k]/matrix[k][k];
							matrix[i][k] = 0;
							for(int j = k+1; j <= order; j++)
								matrix[i][j] -= mult * matrix[k][j];
						}
					}
					printf ("Gauss\n");

					/* determinante */
					deter = 1;
					for(int i = 0; i < order; i++)
						deter *= matrix[i][i];

					printf ("Det %.3e\n", deter);
					det[(MASTER+1)*x] = deter;
				}

				/* sincronizacao */
				//MPI_Barrier(MPI_COMM_WORLD);

				//for(int j = 1; j < totalProcesses; j++){
				/* recebe dos slaves */
				//MPI_Recv(&det, amountPerProcess, MPI_DOUBLE, j, FROM_SLAVE, MPI_COMM_WORLD, &status);
				//}

				//closeFileAndPrintDetValues();

				/* Final time */
				//EndTime = MPI_Wtime();

				/* Print total time */
				//printf ("\nElapsed time = %.6f s\n", EndTime - StartTime);

			}else if(proc > MASTER){
				/* send to slaves */
				MPI_Send(&order, 1, MPI_INT, proc, FROM_MASTER, MPI_COMM_WORLD);
				MPI_Send(&amountPerProcess, 1, MPI_INT, proc, FROM_MASTER, MPI_COMM_WORLD);
				MPI_Send(&bufPerProc, amountPerProcess, MPI_DOUBLE, proc, FROM_MASTER, MPI_COMM_WORLD);
				MPI_Send(&det, amountPerProcess, MPI_DOUBLE, proc, FROM_MASTER, MPI_COMM_WORLD);
				printf ("Enviou para o slave %d.\n", proc);
			}
		}
	} else if(process_id > MASTER) { /* slave work */
		printf ("Entrei processo worker %d.\n", process_id);

		/* recebe do master */
		MPI_Recv(&order, 1, MPI_INT, MASTER, FROM_MASTER, MPI_COMM_WORLD, &status);
		MPI_Recv(&amountPerProcess, 1, MPI_INT, MASTER, FROM_MASTER, MPI_COMM_WORLD, &status);
		bufPerProc = (double *) malloc(sizeof(double) * amountPerProcess * order * order);
		MPI_Recv(bufPerProc, amountPerProcess, MPI_DOUBLE, MASTER, FROM_MASTER, MPI_COMM_WORLD, &status);
		MPI_Recv(&det, amountPerProcess, MPI_DOUBLE, MASTER, FROM_MASTER, MPI_COMM_WORLD, &status);
		printf ("Slave %d recebeu.\n", process_id);

		/* aloca memoria para matriz */
		matrix = (double **) malloc((order) * sizeof(double[order]));

		/* faz calculo do determinante das matrizes */
		for(int x = 0; x < amountPerProcess; x++){

			matrix[x] = (double *) malloc((order) * sizeof(double));

			/* preenche matriz atraves do buffer */
			for (int i = 0; i < order; i++){
				matrix[i] = (double *) malloc((order) * sizeof(double));
				for (int j = 0; j < order; j++)
					matrix[i][j] = bufPerProc[/*(((process_id+1)*x) */ ((order*order)) + (i * order + j)];
			}
			printf ("Preenchimento.\n");

			/* eliminacao de gauss */
			for(int k = 0; k < order-1; k++) {
				for(int i = k+1; i < order; i++) {
					mult = matrix[i][k]/matrix[k][k];
					matrix[i][k] = 0;
					for(int j = k+1; j <= order; j++)
						matrix[i][j] -= mult * matrix[k][j];
				}
			}
			printf ("Gauss.\n");

			/* determinante */
			deter = 1;
			for(int i = 0; i < order; i++)
				deter *= matrix[i][i];

			printf ("Det %.3e\n", deter);
			det[(process_id+1)*x] = deter;
		}

		/* sincronizacao */
		MPI_Barrier(MPI_COMM_WORLD);

		/* send para o master */
		MPI_Send(&det, amountPerProcess, MPI_DOUBLE, MASTER, FROM_SLAVE, MPI_COMM_WORLD);
	}

	MPI_Finalize();
	return 0;
}

/**
 *  \brief Open file and initialize internal data structure.
 *
 *  Operation carried out by the master.
 *
 *  \param fName file name
 */
void openFile (char fName[]){
	int i;                                                                                        /* counting variable */

	if (strlen (fName) > M)
		fprintf (stderr, "file name too long");

	if ((f = fopen (fName, "r")) == NULL)
		perror ("error on file opening for reading");

	if (fread (&nMat, sizeof (nMat), 1, f) != 1)
		fprintf (stderr, "%s\n", "error on reading header - number of stored matrices\n");

	if (fread (&order, sizeof (order), 1, f) != 1)
		fprintf (stderr, "%s\n", "error on reading header - order of stored matrices\n");

	if ((mat = malloc (N * sizeof (double) * order * order)) == NULL)
		fprintf (stderr, "%s\n", "error on allocating storage area for matrices coefficients\n");

	if ((det = malloc (nMat * sizeof (double))) == NULL)
		fprintf (stderr, "%s\n", "error on allocating storage area for determinant values\n");
}

/**
 *  \brief Close file and print the values of the determinants.
 *
 *  Operation carried out by the master
 */
void closeFileAndPrintDetValues (void){
	printf ("Closing and Printing Values...\n");

	int i, n;                                                                                     /* counting variable */

	if (fclose (f) == EOF)
		perror ("error on closing file");
	printf ("\n");

	for (n = 0; n < nMat; n++)
		printf ("The determinant of matrix %d is %.3e\n", n, det[n]);
	printf ("\n");
}
