
// include "MLSCoefficients.hpp"
#include <cstdlib>


// #include <mpi.h>
// #include <Kokkos_Core.hpp>
#include "points.hpp"
#include "MLSCoefficients.hpp"
#include "MLSInterpolation.hpp"
#include "adj_search.hpp"

// #include <iostream>
// #include "clockcycle.h"

using namespace Omega_h;
using namespace pcms;



int main(int argc, char** argv) {
    auto lib = Library(&argc, &argv);
    auto  world = lib.world();
    auto rank = lib.world()->rank();
    auto target_mesh = build_box(world, OMEGA_H_SIMPLEX, 1, 1, 1, 40, 40, 0, false);
    auto source_mesh = build_box(world, OMEGA_H_SIMPLEX, 1, 1, 1, 10, 10, 0, false);
    if (argc != 2){
	fprintf(stderr, "Usage: %s <input mesh>\n", argv[0]);
	return 0;
    }
    
    Real cutoffDistance = std::atof(argv[1]);

    cutoffDistance = cutoffDistance * cutoffDistance;

    const auto dim = target_mesh.dim();

    if (!rank){
	fprintf(stderr, "target mesh <v e f r> %d %d %d %d\n",
	    target_mesh.nglobal_ents(0),
	    target_mesh.nglobal_ents(1),
	    target_mesh.nglobal_ents(2),
	    dim == 3 ? target_mesh.nglobal_ents(3) : 0);
	
	fprintf(stderr, "source mesh <v e f r> %d %d %d %d\n",
	    source_mesh.nglobal_ents(0),
	    source_mesh.nglobal_ents(1),
	    source_mesh.nglobal_ents(2),
	    dim == 3 ? source_mesh.nglobal_ents(3) : 0);
    }

    const auto& target_coordinates = target_mesh.coords();
    const auto& source_coordinates = source_mesh.coords();
   
    SupportResults support =  searchNeighbors(source_mesh, target_mesh, cutoffDistance);
    
    auto approx_target_values = mls_interpolation(source_coordinates, target_coordinates, support, dim, cutoffDistance);


    Write<LO> supports_per_target( target_mesh.nverts(), 0, "number of supports in each target point");
    

    Kokkos::parallel_for(target_mesh.nverts(), KOKKOS_LAMBDA(int i){
	   int start = support.supports_ptr[i];
	   int end = support.supports_ptr[i+1];
	   int num_supports = end - start;
	   supports_per_target[i] = num_supports;
    });

    DevicePoints target_points;
	 
    target_points.coordinates = DevicePointsViewType("Number of local source supports", target_mesh.nverts());
     for (int j = 0; j < target_mesh.nverts(); ++j){
	target_points.coordinates(j).x = target_coordinates[j * dim];
        target_points.coordinates(j).y = target_coordinates[j * dim + 1];

     }

    Write<Real> exact_target_values( target_mesh.nverts(), 0, "exact target values");
			
    Kokkos::parallel_for(target_mesh.nverts(), KOKKOS_LAMBDA(int i){
	    exact_target_values[i] = func(target_points.coordinates(i));
    });
	
    target_mesh.add_tag<Real>(0,"target_field_approx", 1 , approx_target_values);
    target_mesh.add_tag<Real>(0,"target_field_exact", 1 , exact_target_values);
    target_mesh.add_tag<LO>(0,"num_supports", 1 , supports_per_target);
    
    
    vtk::write_parallel("field_values", &target_mesh, dim);

    return 0;
}

