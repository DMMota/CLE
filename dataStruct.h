/**
 *  \file dataStruct.h (interface file)
 *
 *  \brief Problem name: Computation of the determinant of a square matrix through the application of the Gaussian
 *                       elimination method.
 *
 *  Definition of common data structures.
 *
 *  \author António Rui Borges - March 2019
 */

#ifndef DATASTRUCT_H_
#define DATASTRUCT_H_

#include <stdbool.h>

/** \brief matrix coefficients data structure */

typedef struct
        { unsigned int n;                                                            /* identification of the matrix */
          unsigned int order;                                                                 /* order of the matrix */
          double *mat;                                         /* pointer to the storage area of matrix coefficients */
          double detValue;                                                               /* value of the determinant */
        } MATRIXINFO;

#endif /* DATASTRUCT_H_ */
