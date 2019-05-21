/**
 *  \file computeDet.c (implementation file)
 *
 *  \brief Computation of the determinant of a square matrix through the application of the Gaussian elimination method.
 *
 *  It reads the number of matrices whose determinant is to be computed and their order from a binary file. The
 *  coefficients of each matrix are stored line wise.
 *  The file name may be supplied by the user.
 *  Multithreaded implementation.
 *
 *  Generator thread of the intervening entities and definition of the intervening entities.
 *
 *  SYNOPSIS:
 *  <P><PRE>                computeDet [OPTIONS]
 *
 *                OPTIONS:
 *                 -f name --- set the file name (default: "coefData.bin")
 *                 -h      --- print this help.</PRE>
 *
 *  \author António Rui Borges - February 2019
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
static int nMat;

/** \brief order of the square matrices whose determinant is to be computed */
static unsigned int order;

/** \brief pointer to the storage area of matrices coefficients */
static double *mat;

/** \brief pointer to the storage area of matrices determinants */
static double *det;

/** \brief pointer to the binary stream associated with the file in processing */
static FILE *f;

/** \brief amount of matrix calculation per process */
int amountPerProcess;

/** \brief matrix to be filled */
double **matrix;

double *buffer;

double mult;

double deter;

/**
 *  \brief Main thread.
 *
 *  Its role is starting the simulation by generating the intervening entities (dispatcher and determinant computing
 *  threads) and waiting for their termination.
 *
 *  \param argc number arguments in the command line
 *  \param argv array of pointers to the arguments
 */
int main (int argc, char *argv[]){
	char *fName = "coefData.bin";                                                         /* file name, set to default */

	int totalProcesses, process_id;														/* number of processes and ids */

	/* Initializing text processing threads application defined thread id arrays */
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &totalProcesses);									/* MPI size com numero total de processos */
	MPI_Comm_rank(MPI_COMM_WORLD, &process_id);										/* MPI rank com id do processo */

	double StartTime, EndTime;                                                                                     /* time limits */
	/* Start time */
	StartTime = MPI_Wtime();

	/* master work */
	if(process_id == MASTER) {
		printf ("Entrei processo master.\n");

		/* open the file for reading */
		openFile (fName);
		amountPerProcess = nMat / totalProcesses;
		buffer = (double *) malloc(sizeof(double) * order * order);

		for(int i = 0; i < totalProcesses; i++){

			/* fill the buffer with matrix values */
			fread(buffer, sizeof(double), nMat * order * order, f);

			/* do master work */
			if(i == MASTER){

				/* aloca memoria para matriz */
				matrix = (double **) malloc((order) * sizeof(double[order]));

				/* faz calculo do determinante das matrizes */
				for(int x = 0; x < amountPerProcess; x++){

					matrix[x] = (double *) malloc((order) * sizeof(double));
					printf ("oioioi\n");

					/* preenche matriz atraves do buffer */
					for (int i1 = 0; i1 < order; i1++)
						for (int j = 0; j < order; j++)
							matrix[i1][j] = buffer[((order*order)) + (i1 * order + j)];
					printf ("preenchimento\n");

					/* eliminacao de gauss */
					for(int k = 0; k < order-1; k++) {
						for(int i = k+1; i < order; i++) {
							mult = matrix[i][k]/matrix[k][k];
							matrix[i][k] = 0;
							for(int j = k+1; j <= order; j++)
								matrix[i][j] -= mult * matrix[k][j];
						}
					}
					printf ("gauss\n");

					/* determinante */
					deter = 1;
					for(int i = 0; i < order; i++)
						deter *= matrix[i][i];

					printf ("Det %d.\n", deter);
				}

				/* sincronizacao */
				//MPI_Barrier(MPI_COMM_WORLD);

				/* close file and print the values of the determinants */
				closeFileAndPrintDetValues();

				for(int j = 1; j < totalProcesses; j++){
					/* recebe dos slaves */
					MPI_Recv(&matrix, amountPerProcess, MPI_DOUBLE, j, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					MPI_Recv(&det, amountPerProcess, MPI_DOUBLE, j, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

					/* close file and print the values of the determinants */
					closeFileAndPrintDetValues ();
				}

				/* Final time */
				EndTime = MPI_Wtime();

				/* Print total time */
				printf ("\nElapsed time = %.6f s\n", EndTime - StartTime);

			}
			else{
				/* send to slaves */
				MPI_Send(&amountPerProcess, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
				MPI_Send(&buffer, amountPerProcess, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
				MPI_Send(&det, amountPerProcess, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
			}
		}
	} else if(process_id > MASTER) { /* slave work */
		printf ("Entrei processo worker %d.\n", process_id);

		/* recebe do master */
		MPI_Recv(&amountPerProcess, 1, MPI_INT, MASTER, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		printf ("recebi as cenas 0\n");
		buffer = (double *) malloc(sizeof(double) * order * order);
		MPI_Recv(&buffer, amountPerProcess, MPI_DOUBLE, MASTER, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Recv(&det, amountPerProcess, MPI_DOUBLE, MASTER, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		printf ("recebi as cenas\n");

		/* aloca memoria para matriz */
		matrix = (double **) malloc((order) * sizeof(double[order]));

		/* faz calculo do determinante das matrizes */
		for(int x = 0; x < amountPerProcess; x++){

			matrix[x] = (double *) malloc((order) * sizeof(double));

			/* preenche matriz atraves do buffer */
			for (int i = 0; i < order; i++)
				for (int j = 0; j < order; j++)
					matrix[i][j] = buffer[((order*order)) + (i * order + j)];
			printf ("preenchimento\n");

			/* eliminacao de gauss */
			for(int k = 0; k < order-1; k++) {
				for(int i = k+1; i < order; i++) {
					mult = matrix[i][k]/matrix[k][k];
					matrix[i][k] = 0;
					for(int j = k+1; j <= order; j++)
						matrix[i][j] -= mult * matrix[k][j];
				}
			}
			printf ("gauss\n");

			/* determinante */
			deter = 1;
			for(int i = 0; i < order; i++)
				deter *= matrix[i][i];

			printf ("Det %d.\n", deter);
		}

		/* sincronizacao */
		//MPI_Barrier(MPI_COMM_WORLD);

		/* send para o master */
		MPI_Send(&matrix, amountPerProcess, MPI_DOUBLE, MASTER, 0, MPI_COMM_WORLD);
		MPI_Send(&det, amountPerProcess, MPI_DOUBLE, MASTER, 0, MPI_COMM_WORLD);
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
