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

/** \brief matrix struct */
static MATRIXINFO matInfo;

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
	offsets[0] = offsetof(matInfo, matInfo.n);														/* setof identification */
	offsets[1] = offsetof(matInfo, matInfo.order);												    /* setof order */
	offsets[2] = offsetof(matInfo, matInfo.mat);												    /* setof pointer to the storage area of matrix coefficients */
	offsets[3] = offsetof(matInfo, matInfo.detValue);											    /* setof value of the determinant */
	MPI_Datatype MPI_MATRIXINFO;
	MPI_Type_create_struct(nitems, blocklengths, offsets, types, &MPI_MATRIXINFO);
	MPI_Type_commit(&MPI_MATRIXINFO);

	double t0, t1;                                                                                     /* time limits */
	t0 = ((double) clock ()) / CLOCKS_PER_SEC;
	int amountPerProcess;

	if(process_id == MASTER) {

		/* open the file for reading */
		openFile (fName);
		readMatrixCoef();

		amountPerProcess = nMat / totalProcesses;

		for(int i = 0; i < totalProcesses; i++){
			if(i == MASTER){
				worker(process_id);
				MPI_Barrier(MPI_COMM_WORLD);

				//MPI_Recv(&, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
				MPI_Recv(&info, nMat, MPI_INT, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
			}
			else{
				//MPI_Send(&, amountPerProcess, MPI_INT, i, 0, MPI_COMM_WORLD);
				MPI_Send(&info, amountPerProcess, MPI_MATRIXINFO, i, 0, MPI_COMM_WORLD);
			}
		}
	} else if(process_id > MASTER) {
		MPI_Status status;
		//MPI_Recv(&, 1, MPI_INT, MASTER, 0, MPI_COMM_WORLD, &status);
		MPI_Recv(&info, nMat, MPI_INT, MASTER, 0, MPI_COMM_WORLD, &status);

		worker(process_id);
		MPI_Barrier(MPI_COMM_WORLD);
		MPI_Send(&info, amountPerProcess, MPI_MATRIXINFO, process_id, 0, MPI_COMM_WORLD);
	}

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
static void *worker (int process_id){
	int id;																				/* worker id */
	MATRIXINFO *buf;                                                                       /* pointer to a data buffer */
	int k, m, r;                                                                                 /* counting variables */
	double tmp;                                                                                     /* temporary value */
	bool found;                                                                             /* non-null value is found */

	id = process_id;

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

	MPI_T_init_thread(MPI_THREAD_SERIALIZED, MPI_THREAD_SERIALIZED);
}

/**
 *  \brief Open file and initialize internal data structure.
 *
 *  Operation carried out by the main thread.
 *
 *  \param fName file name
 */
void openFile (char fName[]){
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

	MPI_T_finalize();
}

/**
 *  \brief Read matrix coefficients from the file.
 *
 *  Operation carried out by the dispatcher.
 *  The dispatcher is blocked if there are no empty data buffers.
 */
void readMatrixCoef (void){
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

	MPI_T_finalize();
}

/**
 *  \brief Close file and print the values of the determinants.
 *
 *  Operation carried out by the main thread.
 */
void closeFileAndPrintDetValues (void){
	int i, n;                                                                                     /* counting variable */
	initialization();
	nEntries[K+1] += 1;

	if (fclose (f) == EOF)
		perror ("error on closing file");
	printf ("\n");

	for (n = 0; n < nMat; n++)
		printf ("The determinant of matrix %d is %.3e\n", n, det[n]);
	printf ("\n");

	for (i = 0; i < K; i++)
		printf ("N. ent. thread %u ", i);
	printf ("  Dispatcher       Main thread\n");

	for (i = 0; i < K + 2; i++)
		printf ("%8u         ", nEntries[i]);
	printf ("\n");

	MPI_T_finalize();
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
	MATRIXINFO *buf;                                                                       /* pointer to a data buffer */
	initialization();
	nEntries[id] += 1;

	/* check for end of processing */

	if (emptyDataBuff && end){											 /* let possible determinant computing threads know the processing is terminated */
		while (nBlocks > 0){
			nBlocks -= 1;
		}
		return false;
	}

	/* wait for a data buffer with data to become available */
	while (emptyDataBuff){
		nBlocks += 1;
		if (emptyDataBuff && end)
			return false;
		else nBlocks -= 1;
	}

	/* retrieve a pointer to an empty data buffer from the FIFO of pointers to buffers with data */
	buf = dataBuff[riDataBuff];
	riDataBuff = (riDataBuff + 1) % N;
	emptyDataBuff = (iiDataBuff == riDataBuff);
	*bufPnt = buf;

	MPI_T_finalize();
	return true;
}

/**
 *  \brief Return a buffer with the matrix determinant already computed.
 *
 *  Operation carried out by the determinant computing threads.
 *
 *  \param id determinant computing thread id
 *  \param buf pointer to matrix buffer
 */
void returnDetValue (unsigned int id, MATRIXINFO *buf){
	initialization();
	nEntries[id] += 1;

	/* store the value of the determinant */
	det[buf->n] = buf->detValue;

	/* insert the pointer to the buffer into the FIFO of pointers to buffers with no data */
	noDataBuff[iiNoDataBuff] = buf;
	iiNoDataBuff = (iiNoDataBuff + 1) % N;
	emptyNoDataBuff = false;

	MPI_T_finalize();
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
