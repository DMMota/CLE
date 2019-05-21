/**
 *  \file computeDet.h (interface file)
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

/** \brief master default value set to 0 */
#define MASTER 0

/** \brief number of buffers to store matrix coefficients */
#define N 8

/** \brief number of determinant computing processes */
#define K 4

/** \brief maximum number of characters in file name */
#define M 48

/** \brief print command usage */
static void printUsage (char *cmdName);

/** \brief Open file and initialize internal data structure. */
extern void openFile (char fName[]);

/** \brief Read matrix coefficients from the file. */
extern void readMatrixCoef (void);

/** \brief Close file and print the values of the determinants. */
extern void closeFileAndPrintDetValues (void);

extern double detMatrix(double **a, int s, int end, int n);

extern double detMatrixHelper(int nDim, double *pfMatr);




