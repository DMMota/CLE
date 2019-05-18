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

/** \brief array of buffers to store matrix coefficients */
static MATRIXINFO info[N];

/** \brief storage area for the FIFO of pointers to buffers with data */
MATRIXINFO *dataBuff[N];

/** \brief insertion pointer to FIFO of pointers to buffers with data */
unsigned int iiDataBuff;

/** \brief retrieval pointer to FIFO of pointers to buffers with data */
unsigned int riDataBuff;

/** \brief flag signaling FIFO of pointers to buffers with data is empty */
bool emptyDataBuff;

/** \brief storage area for the FIFO of pointers to buffers with no data */
MATRIXINFO *noDataBuff[N];

/** \brief insertion pointer to FIFO of pointers to buffers with no data */
unsigned int iiNoDataBuff;

/** \brief retrieval pointer to FIFO of pointers to buffers with no data */
unsigned int riNoDataBuff;

/** \brief flag signaling FIFO of pointers to buffers with no data is empty */
bool emptyNoDataBuff;

/** \brief number of determinant computing threads waiting for a data buffer with data */
static unsigned int nBlocks;

/** \brief flag signaling end of processing */
static bool end;

/** \brief pointer to the binary stream associated with the file in processing */
static FILE *f;

/** \brief number of entries in the monitor per thread */
static unsigned int nEntries[K+2];

/** \brief dispatcher synchronization point when all data buffers have data */
int noDataBuffEmpty;

/** \brief determinant computing threads synchronization point when all data buffers are empty */
int dataBuffEmpty;

/** \brief amount of matrix calculation per process */
int amountPerProcess;

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
	offsets[2] = offsetof(MATRIXINFO, mat);												    /* setof pointer to the storage area of matrix coefficients */
	offsets[3] = offsetof(MATRIXINFO, detValue);											    /* setof value of the determinant */
	MPI_Datatype MPI_MATRIXINFO;
	MPI_Type_create_struct(nitems, blocklengths, offsets, types, &MPI_MATRIXINFO);
	MPI_Type_commit(&MPI_MATRIXINFO);

	double StartTime, EndTime;                                                                                     /* time limits */
	StartTime = MPI_Wtime();

	if(process_id == MASTER) {
		printf ("Entrei processo master.\n");
		/* open the file for reading */
		openFile (fName);
		readMatrixCoef();

		amountPerProcess = nMat / totalProcesses;
		MATRIXINFO *infoMat = (MATRIXINFO*) malloc(amountPerProcess*sizeof(MATRIXINFO));
		int count;

		for(int i = 0; i < totalProcesses; i++){
			count = 0;
			for(int j = 0; j < nMat; j++){
				infoMat[count] = info[j];
				if(count == amountPerProcess)
					break;
			}
			if(i == MASTER){
				// do calc
				for(int x = 0; x < amountPerProcess; x++)
					worker(i, &infoMat[x]);
				MPI_Barrier(MPI_COMM_WORLD);

				// print work
				closeFileAndPrintDetValues ();

				for(int j = 1; j < totalProcesses; j++){
					MPI_Recv(&infoMat, amountPerProcess, MPI_DOUBLE, j, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
					MPI_Recv(&det, amountPerProcess, MPI_DOUBLE, j, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

					/* close file and print the values of the determinants */
					closeFileAndPrintDetValues ();
				}
				EndTime = MPI_Wtime();

				printf ("\nElapsed time = %.6f s\n", EndTime - StartTime);
			}
			else{
				MPI_Send(&amountPerProcess, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
				MPI_Send(&infoMat, amountPerProcess, MPI_MATRIXINFO, i, 0, MPI_COMM_WORLD);
				MPI_Send(&det, amountPerProcess, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);
			}
		}
	} else if(process_id > MASTER) {
		printf ("Entrei processo worker %d.\n", process_id);
		MPI_Recv(&amountPerProcess, 1, MPI_INT, MASTER, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MATRIXINFO *infoMat = (MATRIXINFO*) malloc(amountPerProcess*sizeof(MATRIXINFO));
		MPI_Recv(&infoMat, amountPerProcess, MPI_DOUBLE, MASTER, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		MPI_Recv(&det, amountPerProcess, MPI_DOUBLE, MASTER, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

		for(int x = 0; x < amountPerProcess; x++)
			worker(process_id, &infoMat[x]);
		MPI_Barrier(MPI_COMM_WORLD);

		MPI_Send(&infoMat, amountPerProcess, MPI_MATRIXINFO, MASTER, 0, MPI_COMM_WORLD);
		MPI_Send(&det, amountPerProcess, MPI_DOUBLE, MASTER, 0, MPI_COMM_WORLD);
	}

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
static void *worker (int process_id, MATRIXINFO *infoMat){
	int id;																				/* worker id */
	MATRIXINFO *buf;                                                                       /* pointer to a data buffer */
	int k, m, r;                                                                                 /* counting variables */
	double tmp;                                                                                     /* temporary value */
	bool found;                                                                             /* non-null value is found */

	id = process_id;
	*buf = *infoMat;

	/* fetch a data buffer */
	while (getMatrixCoef(id, &buf)){
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
 *  \brief Initialization of the data transfer region.
 *
 *  Internal monitor operation.
 */
static void initialization (void){
	int i;                                                                                        /* counting variable */

	iiDataBuff = riDataBuff = 0;                /* insertion and retrieval pointers of the FIFO of pointers to buffers
                                                                                     with data set to the same value */
	for (i = 0; i < K + 2; i++)
		nEntries[i] = 0;

	emptyDataBuff = true;                                            /* FIFO of pointers to buffers with data is empty */

	for (i = 0; i < N; i++)
		noDataBuff[i] = info + i;                  /* fill all positions of the FIFO of pointers to buffers with no data */

	iiNoDataBuff = riNoDataBuff = 0;            /* insertion and retrieval pointers of the FIFO of pointers to buffers
                                                                                  with no data set to the same value */
	emptyNoDataBuff = false;                                  /* FIFO of pointers to buffers with no data is not empty */
	nBlocks = 0;                                       /* no determinant computing threads are presently blocked */
	end = false;                                                                  /* processing has not terminated yet */
}

/**
 *  \brief Open file and initialize internal data structure.
 *
 *  Operation carried out by the master.
 *
 *  \param fName file name
 */
void openFile (char fName[]){
	printf ("Opening File...\n");

	int i;                                                                                        /* counting variable */
	initialization();
	nEntries[K+1] += 1;

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

	for (i = 0; i < N; i++){
		info[i].order = order;
		info[i].mat = mat + i * order * order;
	}
}

/**
 *  \brief Read matrix coefficients from the file.
 *
 *  Operation carried out by the master.
 */
void readMatrixCoef (void){
	printf ("Reading Matrix Coef...\n");

	int n;                                                                                        /* counting variable */
	MATRIXINFO *buf;                                                                       /* pointer to a data buffer */
	initialization();
	nEntries[K] += 1;

	for (n = 0; n < nMat; n++){
		/* retrieve a pointer to an empty data buffer from the FIFO of pointers to buffers with no data */
		buf = noDataBuff[riNoDataBuff];
		riNoDataBuff = (riNoDataBuff + 1) % N;
		emptyNoDataBuff = (iiNoDataBuff == riNoDataBuff);

		/* initialize and read matrix coefficients into the data buffer */
		buf->n = n;
		if (fread (buf->mat, sizeof (double), order * order, f) != (order * order))
			fprintf (stderr, "error on reading matrix coefficients in iteration %d\n", n);

		/* insert the pointer to the buffer into the FIFO of pointers to buffers with data */
		dataBuff[iiDataBuff] = buf;
		iiDataBuff = (iiDataBuff + 1) % N;
		emptyDataBuff = false;
	}

	/* signal end of processing */
	end = true;

	MPI_Send(&end, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
}

/**
 *  \brief Close file and print the values of the determinants.
 *
 *  Operation carried out by the master
 */
void closeFileAndPrintDetValues (void){
	printf ("Closing and Printing Values...\n");

	int i, n;                                                                                     /* counting variable */
	initialization();
	nEntries[K+1] += 1;

	if (fclose (f) == EOF)
		perror ("error on closing file");
	printf ("\n");

	for (n = 0; n < nMat; n++)
		printf ("The determinant of matrix %d is %.3e\n", n, det[n]);
	printf ("\n");
}

/**
 *  \brief Get a buffer with matrix coefficients to compute its determinant.
 *
 *  Operation carried out by the determinant computing threads.
 *  The determinant computing thread is blocked if there are no data buffers with data.
 *
 *  \param id determinant computing thread id
 *  \param bufPnt pointer to the pointer to the data buffer
 *
 *  \return true, if it could get a buffer
 *          false, if there are no more matrices whose determinant has to be computed
 */
bool getMatrixCoef (unsigned int id, MATRIXINFO **bufPnt){
	printf ("Getting Matrix Coef...\n");

	MATRIXINFO *buf;                                                                       /* pointer to a data buffer */
	initialization();
	nEntries[id] += 1;

	MPI_Recv(&end, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

	/* check for end of processing */
	if (emptyDataBuff && end){											 /* let possible determinant computing threads know the processing is terminated */
		while (nBlocks > 0)
			nBlocks -= 1;
		return false;
	}

	/* wait for a data buffer with data to become available */
	while (emptyDataBuff){
		if (emptyDataBuff && end)
			return false;
		else nBlocks -= 1;
	}

	/* retrieve a pointer to an empty data buffer from the FIFO of pointers to buffers with data */
	buf = dataBuff[riDataBuff];
	riDataBuff = (riDataBuff + 1) % N;
	emptyDataBuff = (iiDataBuff == riDataBuff);
	*bufPnt = buf;

	return true;
}

/**
 *  \brief Return a buffer with the matrix determinant already computed.
 *
 *  Operation carried out by the determinant computing threads.
 *
 *  \param id determinant computing process id
 *  \param buf pointer to matrix buffer
 */
void returnDetValue (unsigned int id, MATRIXINFO *buf){
	printf ("Returning Det Value...\n");

	initialization();
	nEntries[id] += 1;

	/* store the value of the determinant */
	det[buf->n] = buf->detValue;

	/* insert the pointer to the buffer into the FIFO of pointers to buffers with no data */
	noDataBuff[iiNoDataBuff] = buf;
	iiNoDataBuff = (iiNoDataBuff + 1) % N;
	emptyNoDataBuff = false;
}

double detMatrix(double **a, int s, int end, int n) {
	int i, j, j1, j2;
	double det;
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
		for (i = 0; i < dim; i++)
			for (j = 0; j < dim; j++)
				fMatr[i * dim + j] = m[i][j];


		det += pow(-1.0, 1.0 + j1 + 1.0) * a[0][j1] * detMatrixHelper(dim, fMatr);

		for (i = 0; i < n - 1; i++) free(m[i]);

		free(m);

	}

	return (det);
}

double detMatrixHelper(int nDim, double *pfMatr) {
	double fDet = 1.;
	double fMaxElem;
	double fAcc;
	int i, j, k, m;

	for (k = 0; k < (nDim - 1); k++){ 										// base row of matrix

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
		for (j = (k + 1); j < nDim; j++){ 						// current row of matrix
			fAcc = -pfMatr[j * nDim + k] / pfMatr[k * nDim + k];
			for (i = k; i < nDim; i++)
				pfMatr[j * nDim + i] = pfMatr[j * nDim + i] + fAcc * pfMatr[k * nDim + i];
		}
	}

	for (i = 0; i < nDim; i++)
		fDet *= pfMatr[i * nDim + i]; // diagonal elements multiplication

	return fDet;
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
