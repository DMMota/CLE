#include "common.h"
#include <math.h>
#include <cuda_runtime.h>
#include <stdio.h>

int matrix_size;
int cluster_size;
float *matrix, *gpuRef, *hostRef;
float *dev_matrix;
float dev_det;

void DeviceFunc(float *temp_h , int numvar , float *temp1_h){
    float *a_d , *b_d;
    //Memory allocation on the device
    cudaMalloc(&a_d,sizeof(float)*(numvar)*(numvar+1));
    cudaMalloc(&b_d,sizeof(float)*(numvar)*(numvar+1));
    //Copying data to device from host
    cudaMemcpy(a_d, temp_h, sizeof(float)*numvar*(numvar+1),cudaMemcpyHostToDevice);
    //Defining size of Thread Block
    dim3 dimBlock(numvar+1,numvar,1);
    dim3 dimGrid(1,1,1);
    //Kernel call 
    //Kernel<<<dimGrid , dimBlock>>>(a_d , b_d , numvar);
    //Coping data to host from device
    cudaMemcpy(temp1_h,b_d,sizeof(float)*numvar*(numvar+1),cudaMemcpyDeviceToHost);
    //Deallocating memory on the device
    cudaFree(a_d);
    cudaFree(b_d);
}

__global__ void detMatrixOnGPUMix(float *matrix, int nx, int ny){
    int matrix_size, matrix_number, current_matrix;
    float deter, pivot, *line;

    matrix_size = nx;
    matrix_number = ny;
    current_matrix = blockIdx.y;

    unsigned int idxCollumn = threadIdx.x;
    unsigned int idxLine = threadIdx.y;
    unsigned int idxPosition = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int idxPivot = current_matrix + idxCollumn * matrix_size + idxCollumn;

    pivot = matrix[0];
    printf("%d %d %d %d %f\n", current_matrix, idxCollumn, idxLine, idxPivot, pivot);
    //printf("idxCurrentMatrix: %d current_matrix: %d\n", idxCurrentMatrix, current_matrix);

    //for (int i = idxCurrentMatrix; i < (idxCurrentMatrix+1) * matrix_size * matrix_size; i += matrix_size) {
        //if (current_matrix == 0)
            //printf("%d %d %d %d\n", current_matrix, i, idxCollumn, i * matrix_size + idxCollumn);
    //}

    if(idxCollumn < matrix_size && idxLine < matrix_size){
	    /* Pivot Verification */
	    if(pivot == 0.000000){
			// Procurar novo pivot, diferente de 0
			int i = idxPivot;
			bool newpivot_found = false;
		        while(!newpivot_found){
				printf("A procurar novo pivot.\n");
				i = i + matrix_size;
				if(matrix[i] != 0){
					printf("Encontrou novo pivot.\n");
					newpivot_found = true;
					double aux = (i-1) / matrix_size;
					i = floor(aux) * matrix_size + 1;
					// Guardar valores da linha num novo array. E trocar valores entre linhas
					for(int k = idxPosition; k < idxPosition + matrix_size; k++) {
						line[k-1] = matrix[k];
						matrix[k] = matrix[i];
						matrix[i] = line[k-1];
						i++;
					}
				}
			}
	    }

	    // thread N-1 calcula determinante no final
	    if((threadIdx.x == nx-1) && (threadIdx.y == nx-1)) {
		    /* Determinant Calculation */
		    deter = 1;
		    //printf("A calcular determinante.\n");
		    for(int i = 0; i < matrix_size; i++)
			    deter *= matrix[i*i];

		    printf("Matrix number - %d; Determinante - %d.\n", current_matrix, deter);
		    
	    }// restantes threads preenchem colunas a 0
	    else {
		    //printf("A aplicar eliminacao de Gauss.\n");
		    /* Gauss Elimination */
		//printf("%d %d\n", matrix_size - idxLine+1, matrix_size - idxCollumn+1);
		for(int i = idxCollumn; i < matrix_size; i++) {
		    for(int k = idxLine; k < matrix_size; k++) {
			   	matrix[(k*matrix_size)+(i+1)] = matrix[i+1] * matrix[k*matrix_size] - matrix[(k*matrix_size)+(i+1)] * matrix[0];
		   	 }
		}
		    //__syncthreads();
	    }
	}	    
}

int main(int argc, char **argv){
    char *fName;
	
    if (argc != 2){
        printf("Usage: ./<ExecutableFilename> <MatrixFilename>\n");
	printf("%6s MatrixFilename: Name the file that contains the matrices\n", "->");
	exit(1);
    }
	
    fName = argv[1];
    
    printf("%s \nStarting...\n", argv[0]);

    int matrix_number, current_matrix, matrixreceived = 0;
    FILE *matrix_file = fopen(fName, "rb");
	
    if(matrix_file == NULL) {
        printf("Cannot open the %s file\n"
               "Please check if is a valid matrix file and the user read permissions!\n", argv[1]);
        return 1;
    }
	
    fread(&matrix_number, sizeof(int), 1, matrix_file);
    fread(&matrix_size, sizeof(int), 1, matrix_file);
    int dimension  = matrix_number * matrix_size;

    printf("Matrix size: %d | Matrix count: %d\n", matrix_size, matrix_number);
    
    //Read the matrices on file 
    matrix = (float*) malloc(sizeof(float) * (matrix_size * matrix_size * matrix_number));
    for(current_matrix = 0; current_matrix < matrix_number; current_matrix++){
        for(int i = current_matrix * matrix_size * matrix_size; i < (current_matrix + 1) * matrix_size * matrix_size; i++) {
            fread(&matrix[i], sizeof(float), 1, matrix_file);
            //printf("%d %f\n", i, matrix[i]);
        }
    }
    
    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("Using Device %d: %s\n", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    // set up data size of matrix
    int nx = matrix_size;
    int ny = matrix_number;

    int nxy = nx * nx * ny;
    int nBytes = nxy * sizeof(float);
    printf("Matrix size: nx %d ny %d\n", nx, ny);

    // malloc host memory
    //hostRef = (float *)malloc(nBytes);
    gpuRef = (float *)malloc(nBytes);

    // initialize data at host side
    double iStart = seconds();

    double iElaps = seconds() - iStart;
    printf("Matrix initialization elapsed %f sec\n", iElaps);

    //memset(hostRef, 0, nBytes);
    memset(gpuRef, 0, nBytes);

    // add matrix at host side for result checks
    iStart = seconds();

    //Multiply Matrix OnHost(h_A, h_B, hostRef, nx, ny);
    //multMatrixOnHost(h_A, h_B, hostRef, nx, ny);
    iElaps = seconds() - iStart;
    
    //printf("multMatrixOnHost elapsed %f sec\n", iElaps);

    CHECK(cudaMalloc((void **)&dev_matrix, nBytes));

    // transfer data from host to device
    CHECK(cudaMemcpy(dev_matrix, matrix, nBytes, cudaMemcpyHostToDevice));
    
    dim3 grid(1,ny);
    dim3 block(nx,nx);

    iStart = seconds();
    detMatrixOnGPUMix<<<grid, block>>>(dev_matrix, nx, ny);
    
    //CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    printf("detMatrixOnGPUMix <<<(%d,%d), (%d,%d)>>> elapsed %f sec\n", grid.x, grid.y, block.x, block.y, iElaps);

    // check kernel error
    CHECK(cudaGetLastError());

    // copy kernel result back to host side
    //CHECK(cudaMemcpy(gpuRef, dev_matrix, nBytes, cudaMemcpyDeviceToHost));

    // check device results
    //checkResult(hostRef, gpuRef, nxy);

    // free device global memory
    CHECK(cudaFree(dev_matrix));

    // free host memory
    free(matrix);
    //free(hostRef);
    free(gpuRef);

    // reset device
    CHECK(cudaDeviceReset());

    return (0);
}
