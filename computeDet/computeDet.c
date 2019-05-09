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
#include <stdlib.h>
#include <stdbool.h>
#include <unistd.h>
#include <libgen.h>
#include <pthread.h>
#include <math.h>
#include <stdbool.h>
#include <time.h>

#include <mpi.h>
#include "probConst.h"
#include "dataStruct.h"
#include "procFile.h"

/** Master default value set to 0 */
#define MASTER 0

/** \brief print command usage */
static void printUsage (char *cmdName);

/** \brief determinant computing threads life cycle routine */
static void *worker (void *par);

/** \brief determinant processing threads return status array */
int statusT[K+2];

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

	/* process command line options */

	int opt;                                                                                        /* selected option */

	opterr = 0;
	do{
		switch ((opt = getopt (argc, argv, "f:h"))){
		case 'f': /* file name */
			if ((optarg[0] == '-') || ((optarg[0] == '\0'))){
				fprintf (stderr, "%s: file name is missing\n", basename (argv[0]));
				printUsage (basename (argv[0]));
				return EXIT_FAILURE;
			}
			fName = optarg;
			break;
		case 'h': /* help mode */
			printUsage (basename (argv[0]));
			return EXIT_SUCCESS;
		case '?': /* invalid option */
			fprintf (stderr, "%s: invalid option\n", basename (argv[0]));
			printUsage (basename (argv[0]));
			return EXIT_FAILURE;
		case -1:  break;
		}
	} while (opt != -1);
	if (optind < argc){
		fprintf (stderr, "%s: invalid format\n", basename (argv[0]));
		printUsage (basename (argv[0]));
		return EXIT_FAILURE;
	}

	/* Initializing text processing threads application defined thread id arrays */

	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &totalProcesses);									/* MPI size com numero total de processos */
	MPI_Comm_rank(MPI_COMM_WORLD, &process_id);										/* MPI rank com id do processo */

	/* Create Data type for structure MATRIXINFO to send and receive with MPI */
	int nitems = 4;
	int blocklengths[4] = {1,1,1,1};

	MPI_Datatype types[4] = {MPI_UNSIGNED, MPI_UNSIGNED, MPI_DOUBLE, MPI_DOUBLE};
	MPI_Aint offsets[4];

	offsets[0] = offsetof(MATRIXINFO, n);														/* setof identification */
	offsets[1] = offsetof(MATRIXINFO, order);												    /* setof order */
	offsets[2] = offsetof(MATRIXINFO, *mat);												    /* setof pointer to the storage area of matrix coefficients */
	offsets[3] = offsetof(MATRIXINFO, detValue);											    /* setof value of the determinant */

	MPI_Datatype MPI_MATRIXINFO;
	MPI_Type_create_struct(nitems, blocklengths, offsets, types, &MPI_MATRIXINFO);
	MPI_Type_commit(&MPI_MATRIXINFO);

	double t0, t1;                                                                                     /* time limits */
	t0 = ((double) clock ()) / CLOCKS_PER_SEC;

	if(process_id == MASTER) {
		/* open the file for reading */
		openFile (fName);
		readMatrixCoef ();

		int amountPerProcess = nMat / numWorkers;

		for(int i = 0; i < numWorkers; i++){
			if(i == MASTER){
				worker(*mat);
				MPI_BARRIER(MPI_COMM_WORLD);

				//MPI_Recv(&, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
				//MPI_Recv(&, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
			}
			else{
				//MPI_Send(&, amountPerProcess, MPI_INT, i, 0, MPI_COMM_WORLD);
				//MPI_Send(&mat, amountPerProcess, MPI_MATRIXINFO, i, 0, MPI_COMM_WORLD);
			}
		}
	} else if(process_id > MASTER) {
		MPI_Status status;
		//MPI_Recv(&, 1, MPI_INT, MASTER, 0, MPI_COMM_WORLD, &status);
		//MPI_Recv(&, 1, MPI_INT, MASTER, 0, MPI_COMM_WORLD, &status);
		worker(*mat);
		MPI_BARRIER(MPI_COMM_WORLD)
		//MPI_Send(&mat, amountPerProcess, MPI_MATRIXINFO, i, 0, MPI_COMM_WORLD);
	}

	/* waiting for the termination of the dispatcher and determinant computing thread */
	//MPI_Barrier(MPI_COMM_WORLD);

	t1 = ((double) clock ()) / CLOCKS_PER_SEC;

	/* close file and print the values of the determinants */
	printf ("\nFinal report\n");
	closeFileAndPrintDetValues ();
	printf ("\nElapsed time = %.6f s\n", t1 - t0);

	MPI_Type_free(&MPI_MATRIXINFO);
	MPI_Finalize();
	return 0;
}

/**
 *  \brief Function determinant computing thread.
 *
 *  It fetches a data buffer with the coefficients of a square matrix and computes its determinant using the method of
 *  Gaussian elimination.
 *  Its role is to simulate the life cycle of a worker.
 *
 *  \param par pointer to application defined worker identification
 */
static void *worker (void *par){
	unsigned int id = *((unsigned int *) par);                                                            /* worker id */
	MATRIXINFO *buf;                                                                       /* pointer to a data buffer */
	int k, m, r;                                                                                 /* counting variables */
	double tmp;                                                                                     /* temporary value */
	bool found;                                                                             /* non-null value is found */

	/* fetch a data buffer */

	while (getMatrixCoef (id, &buf)){
		found = false;

		/* apply Gaussian elimination procedure */

		buf->detValue = 1.0;
		for (k = 0; k < buf->order - 1; k++){ /* check if pivot element is null */
			if (fabs (*(buf->mat + k * buf->order + k)) < 1.0e-20){ /* try and get a non-null element */
				found = false;
				for (m = k + 1; m < buf->order; m++)
					if (fabs (*(buf->mat + k * buf->order + m)) >= 1.0e-20){ /* swap columns */
						for (r = 0; r < buf->order; r++){
							tmp = *(buf->mat + r * buf->order + k);
							*(buf->mat + r * buf->order + k) = *(buf->mat + r * buf->order + m);
							*(buf->mat + r * buf->order + m) = tmp;
						}
						buf->detValue = -buf->detValue;                           /* reverse the signal of the determinant */
						found = true;
						break;
					}
			}
			else found = true;
			if (!found){ /* null row */
				break;
			}
			/* apply row transformation */
			for (m = k + 1; m < buf->order; m++){
				tmp = *(buf->mat + m * buf->order + k) / *(buf->mat + k * buf->order + k);
				for (r = k; r < buf->order; r++)
					*(buf->mat + m * buf->order + r) -= tmp * *(buf->mat + k * buf->order + r);
			}
		}

		/* compute the determinant */

		if (found)
			for (k = 0; k < buf->order; k++)
				buf->detValue *= *(buf->mat + k * buf->order + k);
		else buf->detValue = 0.0;

		/* return computed value */

		returnDetValue (id, buf);
	}
}

/**
 *  \brief Print command usage.
 *
 *  A message specifying how the program should be called is printed.
 *
 *  \param cmdName string with the name of the command
 */
static void printUsage (char *cmdName){
	fprintf (stderr, "\nSynopsis: %s [OPTIONS]\n"
			"  OPTIONS:\n"
			"  -f name --- set the file name (default: \"coefData.bin\")\n"
			"  -h      --- print this help\n", cmdName);
}
