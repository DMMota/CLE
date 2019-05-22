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


int main (int argc, char *argv[]){

	int totalProcesses, process_id;														/* number of processes and ids */

	/* Initializing text processing threads application defined thread id arrays */
	MPI_Init(&argc, &argv);
	MPI_Comm_size(MPI_COMM_WORLD, &totalProcesses);									/* MPI size com numero total de processos */
	MPI_Comm_rank(MPI_COMM_WORLD, &process_id);										/* MPI rank com id do processo */

	int value;

	if(process_id == 0) {
		value = 37;
		MPI_Send(&value, 1, MPI_INT, 1, 0, MPI_COMM_WORLD);
		MPI_Send(&value, 1, MPI_INT, 2, 0, MPI_COMM_WORLD);
		MPI_Send(&value, 1, MPI_INT, 3, 0, MPI_COMM_WORLD);


	} else if (process_id > 0) {
		MPI_Recv(&value, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
		printf ("%d...\n", value);

		MPI_Finalize();
		return 0;
	}
}

