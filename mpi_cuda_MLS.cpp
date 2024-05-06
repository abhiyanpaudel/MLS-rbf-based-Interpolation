
#include <mpi.h>
#include <Kokkos_Core.hpp>
#include "points.hpp"
#include "MLSCoefficients.hpp"
#include "MLSInterpolation.hpp"
#include <iostream>
#include "clockcycle.h"

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);
    
    Kokkos::initialize(argc, argv);

    {

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    
    // Determine the file name
    std::string x_sourcefilename = "x_source_data.dat";
    std::string y_sourcefilename = "y_source_data.dat";
    std::string x_targetfilename = "x_target_data.dat";
    std::string y_targetfilename = "y_target_data.dat"; 
    // Open the file using MPI I/O
    MPI_File x_source_file;
    MPI_File_open(MPI_COMM_WORLD, x_sourcefilename.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &x_source_file);
    
    MPI_File y_source_file;
    MPI_File_open(MPI_COMM_WORLD, y_sourcefilename.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &y_source_file);
    
    MPI_File x_target_file;
    MPI_File_open(MPI_COMM_WORLD, x_targetfilename.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &x_target_file);

    MPI_File y_target_file;
    MPI_File_open(MPI_COMM_WORLD, y_targetfilename.c_str(), MPI_MODE_RDONLY, MPI_INFO_NULL, &y_target_file);
 
        


    // Calculate the domain size
    const int source_row = 6000;
    const int source_column = 6000;

    const int target_row = 3000;
    const int target_column = 3000;



    int row_per_proc_source = source_row/size;
    int row_per_proc_target = target_row/size;

    int remaining_rows_source = source_row % size;
    int remaining_rows_target = target_row % size;
    

    int buffer_row = 5;

    if (rank == (size -1)){
	row_per_proc_source += remaining_rows_source;
        row_per_proc_target += remaining_rows_target; 	
    }	    

    int source_start_index = rank * row_per_proc_source;
    int target_start_index = rank * row_per_proc_target;

    int total_row_source = row_per_proc_source + 2* buffer_row; 
    

    if (rank == 0 || rank == size - 1){

	   total_row_source -=  buffer_row;

     }

    const int source_size = total_row_source * source_column;
    const int target_size = row_per_proc_target * target_column;

    Kokkos::View<double**, Kokkos::HostSpace> host_xsource_data("x host data", total_row_source, source_column); 
    Kokkos::View<double**, Kokkos::HostSpace> host_ysource_data("y host data", total_row_source, source_column); 
    Kokkos::View<double**, Kokkos::HostSpace> host_xtarget_data("x target data", row_per_proc_target, target_column); 
    Kokkos::View<double**, Kokkos::HostSpace> host_ytarget_data("y target data", row_per_proc_target, target_column); 

    MPI_Offset offset_source = source_start_index * sizeof(double);
    MPI_Offset offset_target = target_start_index * sizeof(double);


    MPI_File_set_view(x_source_file, offset_source , MPI_DOUBLE, MPI_DOUBLE, "native", MPI_INFO_NULL);
    MPI_File_set_view(y_source_file, offset_source , MPI_DOUBLE, MPI_DOUBLE, "native", MPI_INFO_NULL);
    MPI_File_set_view(x_target_file, offset_target, MPI_DOUBLE, MPI_DOUBLE, "native", MPI_INFO_NULL);
    MPI_File_set_view(y_target_file, offset_target, MPI_DOUBLE, MPI_DOUBLE, "native", MPI_INFO_NULL);


    double* xsource_data_ptr = host_xsource_data.data();
    double* ysource_data_ptr = host_ysource_data.data();
    if (rank != 0) {
      xsource_data_ptr += buffer_row * source_column;
      ysource_data_ptr += buffer_row * source_column;
    }


    uint64_t start_read = clock_now();

    MPI_File_read(x_source_file, xsource_data_ptr, row_per_proc_source * source_column, MPI_DOUBLE, MPI_STATUS_IGNORE);
    MPI_File_read(y_source_file, ysource_data_ptr, row_per_proc_source * source_column, MPI_DOUBLE, MPI_STATUS_IGNORE);

    MPI_File_read(x_target_file, host_xtarget_data.data(), row_per_proc_target * target_column, MPI_DOUBLE, MPI_STATUS_IGNORE);
    MPI_File_read(y_target_file, host_ytarget_data.data(), row_per_proc_target * target_column, MPI_DOUBLE, MPI_STATUS_IGNORE);
	
    uint64_t end_read = clock_now();
    uint64_t elapsed_cycles_read = end_read - start_read;     
    // Close the file
    MPI_File_close(&x_source_file);
    MPI_File_close(&y_source_file);
    MPI_File_close(&x_target_file);
    MPI_File_close(&y_target_file);


  // Create buffers for sending and receiving data
   Kokkos::View<double**, Kokkos::HostSpace> send_buffer("send buffer", buffer_row, source_column);
   Kokkos::View<double**, Kokkos::HostSpace> recv_buffer("recv buffer", buffer_row, source_column);

   
   int num_request = 0;
   MPI_Request requests[4];
   MPI_Status statuses[4];
   uint64_t start_comm = clock_now();
   
    // Exchange first five layers and last five layers among ranks
    if (rank != 0) {
        // Send first five rows to the previous rank and receive its last five rows
	MPI_Isend(host_xsource_data.data(), buffer_row * source_column, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &requests[num_request++]);
	MPI_Irecv(recv_buffer.data(), buffer_row * source_column, MPI_DOUBLE, rank - 1, 0, MPI_COMM_WORLD, &requests[num_request++]);
     }

     if (rank != size - 1) {
	 // Send last five rows to the next rank and receive its first five rows
	 MPI_Isend(host_xsource_data.data() + (total_row_source - buffer_row) * source_column, buffer_row * source_column, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &requests[num_request++]);
	 MPI_Irecv(recv_buffer.data(), buffer_row * source_column, MPI_DOUBLE, rank + 1, 0, MPI_COMM_WORLD, &requests[num_request++]);
     }

     // Wait for all non-blocking operations to complete
       MPI_Waitall(num_request, requests, statuses);


     // Copy the received data to the appropriate rows
     for (int i = 0; i < buffer_row; i++){
	 for (int j = 0; j < source_column; j++) {
	     if (rank != 0){
		   host_xsource_data(i, j) = recv_buffer(i, j);
	      }   
		if (rank != size - 1){
		    host_xsource_data(total_row_source - buffer_row + i, j) = recv_buffer(i, j);
	      	}    
	  }
      }    
   
    uint64_t end_comm = clock_now();
    uint64_t elapsed_cycles_comm = end_comm - start_comm;

    HostPoints host_source_data;
    HostPoints host_target_data;
    
    host_source_data.coordinates = HostPointsViewType("host source accumulated data", source_size);
    host_target_data.coordinates = HostPointsViewType("host target accumulated data", target_size);
    int count1 = -1;
    for (int i = 0; i < total_row_source; ++i){
	for (int j = 0; j < source_column; ++j){
		count1 += 1;
		host_source_data.coordinates(count1).x = host_xsource_data(i,j);
	        host_source_data.coordinates(count1).y = host_ysource_data(i,j); 	

	}	
    }	    


    int count2 = -1;
    for (int i = 0; i < row_per_proc_target ; ++i){
	for (int j = 0; j < target_column; ++j){
		count2 += 1;
		host_target_data.coordinates(count2).x = host_xtarget_data(i,j);
	        host_target_data.coordinates(count2).y = host_ytarget_data(i,j); 	

	}	
    }	    
		    

   int nSrc = host_source_data.coordinates.extent(0);
   int nTar = host_target_data.coordinates.extent(0);
 
                                
    HostRealVecView h_source_exact_sol = exact_solution(host_source_data, nSrc);


     DevicePoints device_source_data;
     DevicePoints device_target_data;

    device_source_data.coordinates = Kokkos::create_mirror_view_and_copy(Kokkos::CudaSpace(), host_source_data.coordinates);
     device_target_data.coordinates = Kokkos::create_mirror_view_and_copy(Kokkos::CudaSpace(), host_target_data.coordinates);
 

    RealVecView approx_target_values("Approximate target values", nTar);  
 
    
    
    DeviceRealVecView d_source_exact_sol("Function Values on Device", nSrc);   
 
                               
    Kokkos::deep_copy(d_source_exact_sol, h_source_exact_sol); 
    
    double rho = 25; // square of cut off radius 
    
    uint64_t start_comp = clock_now(); 
    mls_interpolation(device_source_data, device_target_data, d_source_exact_sol, approx_target_values, rho);
    uint64_t end_comp = clock_now();
    uint64_t elapsed_cycles_comp = end_comp - start_comp;

    auto h_target_values = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), approx_target_values);


	// Calculate the offset for this process
     MPI_Offset offset = rank * target_size * sizeof(double);

	// Open the file
     MPI_File output_file;
     MPI_File_open(MPI_COMM_WORLD, "output_data.dat", MPI_MODE_WRONLY | MPI_MODE_CREATE, MPI_INFO_NULL, &output_file);

	// Set the file view
     MPI_File_set_view(output_file, 0, MPI_DOUBLE, MPI_DOUBLE, "native", MPI_INFO_NULL);

	// Write the data
     uint64_t start_write = clock_now();
     MPI_File_write_at_all(output_file, offset, h_target_values.data(), target_size, MPI_DOUBLE, MPI_STATUS_IGNORE);
     uint64_t end_write = clock_now();
     uint64_t elapsed_cycles_write = end_write - start_write;

	// Close the file
	MPI_File_close(&output_file);
	    

    double elapsed_time_read = (double)elapsed_cycles_read / 512e6;
    double elapsed_time_comm = (double)elapsed_cycles_comm / 512e6;
    double elapsed_time_comp = (double)elapsed_cycles_comp / 512e6;
    double elapsed_time_write = (double)elapsed_cycles_write / 512e6;

    if (rank == 0) {
	    printf("Time taken for reading: %f seconds\n", elapsed_time_read);
	    printf("Time taken for communication: %f seconds\n", elapsed_time_comm);
	    printf("Time taken for computation: %f seconds\n", elapsed_time_comp);
	    printf("Time taken for writing: %f seconds\n", elapsed_time_write);
    }

    
     MPI_Barrier(MPI_COMM_WORLD);

    }
    // Finalize Kokkos
    Kokkos::finalize();

    MPI_Finalize();

    return 0;
}

