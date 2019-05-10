/**
 *  \file procFile.c (implementation file)
 *
 *  \brief Problem name: Computation of the determinant of a square matrix through the application of the Gaussian
 *                       elimination method.
 *
 *  Synchronization based on monitors.
 *  Both threads and the monitor are implemented using the pthread library which enables the creation of a
 *  monitor of the Lampson / Redell type.
 *
 *  File processing utilities.
 *
 *  It reads the number of matrices whose determinant is to be computed and their order from a binary file. It reads
 *  next the coefficients of each matrix which are stored line wise into an array of buffers. The buffers are then
 *  supplied upon request for processing and the determinant value is returned.
 *
 *  Definition of the operations carried out by the main thread, the dispatcher and the determinant computing threads:
 *     \li openFile
 *     \li readMatrixCoef
 *     \li closeFileAndPrintDetValues
 *     \li getMatrixCoef
 *     \li returnDetValue.
 *
 *  \author António Rui Borges - March 2019
 */

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <unistd.h>
#include <pthread.h>
#include <errno.h>
#include <string.h>

#include <mpi.h>
#include "probConst.h"
#include "dataStruct.h"

/** \brief main, dispatcher and determinant computing threads return status array */
extern int statusT[K+2];

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

/** \brief locking flag which warrants mutual exclusion inside the monitor */
//static pthread_mutex_t accessCR = PTHREAD_MUTEX_INITIALIZER;

/** \brief flag which warrants that the file processing region is initialized exactly once */
//static pthread_once_t init = PTHREAD_ONCE_INIT;


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


	//pthread_cond_init (&noDataBuffEmpty, NULL);                         /* initialize dispatcher synchronization point */
	//pthread_cond_init (&dataBuffEmpty, NULL); 							/* initialize determinant computing threads synchronization point */
	MPI_T_init_thread(MPI_THREAD_SERIALIZED, MPI_THREAD_SERIALIZED);
}

/**
 *  \brief Open file and initialize internal data structure.
 *
 *  Operation carried out by the main thread.
 *
 *  \param fName file name
 */
MATRIXINFO*** openFile (char fName[]){
	int i;                                                                                        /* counting variable */

	//if ((statusT[K+1] = pthread_mutex_lock (&accessCR)) != 0){                                        /* enter monitor */
	//	errno = statusT[K+1];                                                                  /* save error in errno */
	//	perror ("error on entering monitor(CF)");
	//	statusT[K+1] = EXIT_FAILURE;
	//	pthread_exit (&statusT[K+1]);
	//}
	//pthread_once (&init, initialization);                                              /* internal data initialization */
	initialization();

	nEntries[K+1] += 1;

	if (strlen (fName) > M){
		fprintf (stderr, "file name too long");
		//statusT[K+1] = EXIT_FAILURE;
		//pthread_exit (&statusT[K]);
	}

	if ((f = fopen (fName, "r")) == NULL){
		perror ("error on file opening for reading");
		//statusT[K+1] = EXIT_FAILURE;
		//pthread_exit (&statusT[K+1]);
	}

	if (fread (&nMat, sizeof (nMat), 1, f) != 1){
		fprintf (stderr, "%s\n", "error on reading header - number of stored matrices\n");
		//statusT[K+1] = EXIT_FAILURE;
		//pthread_exit (&statusT[K+1]);
	}

	if (fread (&order, sizeof (order), 1, f) != 1){
		fprintf (stderr, "%s\n", "error on reading header - order of stored matrices\n");
		//statusT[K+1] = EXIT_FAILURE;
		//pthread_exit (&statusT[K+1]);
	}

	if ((mat = malloc (N * sizeof (double) * order * order)) == NULL){
		fprintf (stderr, "%s\n", "error on allocating storage area for matrices coefficients\n");
		//statusT[K+1] = EXIT_FAILURE;
		//pthread_exit (&statusT[K+1]);
	}

	if ((det = malloc (nMat * sizeof (double))) == NULL){
		fprintf (stderr, "%s\n", "error on allocating storage area for determinant values\n");
		//statusT[K+1] = EXIT_FAILURE;
		//pthread_exit (&statusT[K+1]);
	}

	for (i = 0; i < N; i++){
		info[i].order = order;
		info[i].mat = mat + i * order * order;
	}

	//if ((statusT[K+1] = pthread_mutex_unlock (&accessCR)) != 0){                                       /* exit monitor */
	//	errno = statusT[K+1];                                                                  /* save error in errno */
	//	perror ("error on exiting monitor(CF)");
	//	statusT[K+1] = EXIT_FAILURE;
	//	pthread_exit (&statusT[K+1]);
	//}
	MPI_T_finalize();
	return info;
}

/**
 *  \brief Read matrix coefficients from the file.
 *
 *  Operation carried out by the dispatcher.
 *  The dispatcher is blocked if there are no empty data buffers.
 */
int readMatrixCoef (void){
	int n;                                                                                        /* counting variable */
	MATRIXINFO *buf;                                                                       /* pointer to a data buffer */
	initialization();
	nEntries[K] += 1;

	for (n = 0; n < nMat; n++){
		/* wait for an empty data buffer to become available */

		//while (emptyNoDataBuff){
		//	if ((statusT[K] = pthread_cond_wait (&noDataBuffEmpty, &accessCR)) != 0){
		//		errno = statusT[K];                                                                /* save error in errno */
		//		perror ("error on waiting for an empty data buffer");
		//		statusT[K] = EXIT_FAILURE;
		//		pthread_exit (&statusT[K]);
		//	}
		//}

		/* retrieve a pointer to an empty data buffer from the FIFO of pointers to buffers with no data */

		buf = noDataBuff[riNoDataBuff];
		riNoDataBuff = (riNoDataBuff + 1) % N;
		emptyNoDataBuff = (iiNoDataBuff == riNoDataBuff);

		/* initialize and read matrix coefficients into the data buffer */
		buf->n = n;
		if (fread (buf->mat, sizeof (double), order * order, f) != (order * order)){
			fprintf (stderr, "error on reading matrix coefficients in iteration %d\n", n);
			//statusT[K] = EXIT_FAILURE;
			//pthread_exit (&statusT[K]);
		}

		/* insert the pointer to the buffer into the FIFO of pointers to buffers with data */
		dataBuff[iiDataBuff] = buf;
		iiDataBuff = (iiDataBuff + 1) % N;
		emptyDataBuff = false;

		/* let a determinant computing thread know a buffer with data is available */
		if ((statusT[K] = pthread_cond_signal (&dataBuffEmpty)) != 0){
			//errno = statusT[K];                                                                  /* save error in errno */
			perror ("error on signaling for a buffer with data");
			//statusT[K] = EXIT_FAILURE;
			//pthread_exit (&statusT[K]);
		}
	}

	/* signal end of processing */
	end = true;

	//if ((statusT[K] = pthread_mutex_unlock (&accessCR)) != 0){ 									/* exit monitor */
	//	errno = statusT[K];                                                                    /* save error in errno */
	//	perror ("error on exiting monitor(CF)");
	//	statusT[K] = EXIT_FAILURE;
	//	pthread_exit (&statusT[K]);
	//}
	MPI_T_finalize();
	return nMat;
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
			//if ((statusT[id] = pthread_cond_signal (&dataBuffEmpty)) != 0){
			//	errno = statusT[id];                                                            /* save error in errno */
			//	perror ("error on signaling for a buffer with data");
			//	statusT[id] = EXIT_FAILURE;
			//	pthread_exit (&statusT[id]);
			//}
			nBlocks -= 1;
		}
		//if ((statusT[id] = pthread_mutex_unlock (&accessCR)) != 0){                                    /* exit monitor */
		//	errno = statusT[id];                                                              /* save error in errno */
		//	perror ("error on exiting monitor(CF)");
		//	statusT[id] = EXIT_FAILURE;
		//	pthread_exit (&statusT[id]);
		//}
		return false;
	}

	/* wait for a data buffer with data to become available */

	while (emptyDataBuff){
		nBlocks += 1;
		//if ((statusT[id] = pthread_cond_wait (&dataBuffEmpty, &accessCR)) != 0){
		//	errno = statusT[id];                                                                 /* save error in errno */
		//	perror ("error on waiting for a data buffer with data");
		//	statusT[id] = EXIT_FAILURE;
		//	pthread_exit (&statusT[id]);
		//}
		if (emptyDataBuff && end){
			//if ((statusT[id] = pthread_mutex_unlock (&accessCR)) != 0){                                  /* exit monitor */
			//	errno = statusT[id];                                                            /* save error in errno */
			//	perror ("error on exiting monitor(CF)");
			//	statusT[id] = EXIT_FAILURE;
			//	pthread_exit (&statusT[id]);
			//}
			return false;
		}
		else nBlocks -= 1;
	}

	/* retrieve a pointer to an empty data buffer from the FIFO of pointers to buffers with data */

	buf = dataBuff[riDataBuff];
	riDataBuff = (riDataBuff + 1) % N;
	emptyDataBuff = (iiDataBuff == riDataBuff);
	*bufPnt = buf;

	//if ((statusT[id] = pthread_mutex_unlock (&accessCR)) != 0){                                         /* exit monitor */
	//	errno = statusT[id];                                                                   /* save error in errno */
	//	perror ("error on exiting monitor(CF)");
	//	statusT[id] = EXIT_FAILURE;
	//	pthread_exit (&statusT[id]);
	//}

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

	/* let the dispatcher know a buffer with no data is available */

	//if ((statusT[id] = pthread_cond_signal (&noDataBuffEmpty)) != 0){
	//	errno = statusT[id];                                                                   /* save error in errno */
	//	perror ("error on signaling for a buffer with data");
	//	statusT[id] = EXIT_FAILURE;
	//	pthread_exit (&statusT[id]);
	//}

	//if ((statusT[id] = pthread_mutex_unlock (&accessCR)) != 0){                                         /* exit monitor */
	//	errno = statusT[id];                                                                   /* save error in errno */
	//	perror ("error on exiting monitor(CF)");
	//	statusT[id] = EXIT_FAILURE;
	//	pthread_exit (&statusT[id]);
	//}

	MPI_T_finalize();
}
