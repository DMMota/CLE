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

/** \brief matrix struct */
typedef struct{
	unsigned int n;                                                            /* identification of the matrix */
	unsigned int order;                                                                 /* order of the matrix */
	double *mat;                                         /* pointer to the storage area of matrix coefficients */
	double detValue;                                                               /* value of the determinant */
}MATRIXINFO;

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

/** \brief determinant computing threads life cycle routine */
static void *worker (int process_id, MATRIXINFO *infoMat);

/** \brief Open file and initialize internal data structure. */
extern void openFile (char fName[]);

/** \brief Read matrix coefficients from the file. */
extern void readMatrixCoef (void);

/** \brief Close file and print the values of the determinants. */
extern void closeFileAndPrintDetValues (void);

/** \brief Get a buffer with matrix coefficients to compute its determinant. */
extern bool getMatrixCoef (unsigned int id, MATRIXINFO **bufPnt);

/** \brief Return a buffer with the matrix determinant already computed. */
extern void returnDetValue (unsigned int id, MATRIXINFO *buf);

extern double detMatrix(double **a, int s, int end, int n);

extern double detMatrixHelper(int nDim, double *pfMatr);




