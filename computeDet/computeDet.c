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

#include "probConst.h"
#include "dataStruct.h"
#include "procFile.h"

/** \brief print command usage */
static void printUsage (char *cmdName);

/** \brief dispatcher life cycle routine */
static void *dispatcher (void *par);

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

int main (int argc, char *argv[])
{
  char *fName = "matCoef_128_64.bin";                                                         /* file name, set to default */

  /* process command line options */

  int opt;                                                                                        /* selected option */

  opterr = 0;
  do
  { switch ((opt = getopt (argc, argv, "f:h")))
    { case 'f': /* file name */
    	        if ((optarg[0] == '-') || ((optarg[0] == '\0')))
                { fprintf (stderr, "%s: file name is missing\n", basename (argv[0]));
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
  if (optind < argc)
     { fprintf (stderr, "%s: invalid format\n", basename (argv[0]));
       printUsage (basename (argv[0]));
       return EXIT_FAILURE;
     }

  pthread_t tIdDispatcher,                                                                   /* dispatcher thread id */
            tIdWorker[K];                                  /* determinant computing threads internal thread id array */
  unsigned int dispatcherId,                                             /* dispatcher application defined thread id */
               workId[K];                       /* determinant computing threads application defined thread id array */
  int t;                                                                                        /* counting variable */
  int *status_p;                                                                      /* pointer to execution status */

  /* initializing text processing threads application defined thread id arrays */

  dispatcherId = K;
  for (t = 0; t < K; t++)
    workId[t] = t;

  /* open the file for reading */

  double t0, t1;                                                                                      /* time limits */

  t0 = ((double) clock ()) / CLOCKS_PER_SEC;
  openFile (fName);

  /* generation of the dispatcher and determinant computing threads */

  if (pthread_create (&tIdDispatcher, NULL, dispatcher, &dispatcherId) != 0)                    /* dispatcher thread */
     { perror ("error on creating determinant computing thread");
       exit (EXIT_FAILURE);
     }
  for (t = 0; t < K; t++)
    if (pthread_create (&tIdWorker[t], NULL, worker, &workId[t]) != 0)               /* determinant computing thread */
       { perror ("error on creating determinant computing thread");
         exit (EXIT_FAILURE);
       }

  /* waiting for the termination of the dispatcher and determinant computing thread */

  printf ("\nFinal report\n");
  if (pthread_join (tIdDispatcher, (void *) &status_p) != 0)                                    /* dispatcher thread */
     { perror ("error on waiting for dispatcher thread to terminate");
       exit (EXIT_FAILURE);
     }
  printf ("dispatcher thread has terminated: ");
  printf ("its status was %d\n", *status_p);
  for (t = 0; t < K; t++)
  { if (pthread_join (tIdWorker[t], (void *) &status_p) != 0)                        /* determinant computing thread */
       { perror ("error on waiting for determinant computing thread to terminate");
         exit (EXIT_FAILURE);
       }
    printf ("determinant computing thread, with id %u, has terminated: ", t);
    printf ("its status was %d\n", *status_p);
  }
  t1 = ((double) clock ()) / CLOCKS_PER_SEC;

  /* close file and print the values of the determinants */

  closeFileAndPrintDetValues ();
  printf ("\nElapsed time = %.6f s\n", t1 - t0);

  exit (EXIT_SUCCESS);
}

/**
 *  \brief Function dispatcher thread.
 *
 *  It reads in succession the coefficients of square matrices into data buffers for subsequent processing.
 *  Its role is to simulate the life cycle of the dispatcher.
 *
 *  \param par dummy parameter
 */

static void *dispatcher (void *par)
{
  readMatrixCoef ();

  statusT[K] = EXIT_SUCCESS;
  pthread_exit (&statusT[K]);
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

static void *worker (void *par)
{
  unsigned int id = *((unsigned int *) par);                                                            /* worker id */
  MATRIXINFO *buf;                                                                       /* pointer to a data buffer */
  int k, m, r;                                                                                 /* counting variables */
  double tmp;                                                                                     /* temporary value */
  bool found;                                                                             /* non-null value is found */

  /* fetch a data buffer */

  while (getMatrixCoef (id, &buf))
  { found = false;

    /* apply Gaussian elimination procedure */

    buf->detValue = 1.0;
    for (k = 0; k < buf->order - 1; k++)
    { /* check if pivot element is null */
      if (fabs (*(buf->mat + k * buf->order + k)) < 1.0e-20)
         { /* try and get a non-null element */
           found = false;
    	   for (m = k + 1; m < buf->order; m++)
             if (fabs (*(buf->mat + k * buf->order + m)) >= 1.0e-20)
    	        { /* swap columns */
                  for (r = 0; r < buf->order; r++)
                  { tmp = *(buf->mat + r * buf->order + k);
                    *(buf->mat + r * buf->order + k) = *(buf->mat + r * buf->order + m);
                    *(buf->mat + r * buf->order + m) = tmp;
                  }
                  buf->detValue = -buf->detValue;                           /* reverse the signal of the determinant */
                  found = true;
                  break;
    	        }
         }
         else found = true;
      if (!found)
         { /* null row */
           break;
         }
      /* apply row transformation */
      for (m = k + 1; m < buf->order; m++)
      { tmp = *(buf->mat + m * buf->order + k) / *(buf->mat + k * buf->order + k);
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

  statusT[id] = EXIT_SUCCESS;
  pthread_exit (&statusT[id]);
}

/**
 *  \brief Print command usage.
 *
 *  A message specifying how the program should be called is printed.
 *
 *  \param cmdName string with the name of the command
 */

static void printUsage (char *cmdName)
{
  fprintf (stderr, "\nSynopsis: %s [OPTIONS]\n"
           "  OPTIONS:\n"
           "  -f name --- set the file name (default: \"coefData.bin\")\n"
           "  -h      --- print this help\n", cmdName);
}
