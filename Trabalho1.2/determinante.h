/**
 *
 *  \author Diogo Martins Mota - May 2019
 *  \author Clony - May 2019
 */

/** \brief master default value set to 0 */
#define MASTER 0

/** \brief master default value set to 0 */
#define FROM_MASTER 0

/** \brief master default value set to 0 */
#define FROM_SLAVE 1

/** \brief number of buffers to store matrix coefficients */
#define N 8

/** \brief number of determinant computing processes */
#define K 4

/** \brief maximum number of characters in file name */
#define M 48

/** \brief Open file and initialize internal data structure. */
extern void openFile (char fName[]);

/** \brief Close file and print the values of the determinants. */
extern void closeFileAndPrintDetValues (void);
