/**
 *  \file procFile.h (interface file)
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

#ifndef PROCFILE_H_
#define PROCFILE_H_

#include  <stdbool.h>

#include  "dataStruct.h"

/**
 *  \brief Open file and initialize internal data structure.
 *
 *  Operation carried out by the main thread.
 *
 *  \param fName file name
 */

extern MATRIXINFO openFile (char fName[]);

/**
 *  \brief Read matrix coefficients from the file.
 *
 *  Operation carried out by the dispatcher.
 *  The dispatcher is blocked if there are no empty data buffers.
 */

extern int readMatrixCoef (void);

/**
 *  \brief Close file and print the values of the determinants.
 *
 *  Operation carried out by the main thread.
 */

extern void closeFileAndPrintDetValues (void);

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

extern bool getMatrixCoef (unsigned int id, MATRIXINFO **bufPnt);

/**
 *  \brief Return a buffer with the matrix determinant already computed.
 *
 *  Operation carried out by the determinant computing threads.
 *
 *  \param id determinant computing thread id
 *  \param bufPnt pointer to the pointer to the data buffer
 */

extern void returnDetValue (unsigned int id, MATRIXINFO *buf);

#endif /* PROCFILE_H_ */
