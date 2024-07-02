
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

    if (argc != 4){
	fprintf(stderr, "Usage: %s <input mesh>\n", argv[0]);
	return 0;
    }

    const auto rank = lib.world()->rank();
    const auto inmesh_source = argv[1];
    const auto inmesh_target = argv[2];
    Real cutoffDistance = std::atof(argv[3]);

    cutoffDistance = cutoffDistance * cutoffDistance;

    Mesh source_mesh(&lib);

    Mesh target_mesh(&lib);

    binary::read(inmesh_source, lib.world(), &source_mesh);

    binary::read(inmesh_target, lib.world(), &target_mesh);

    const auto dim = source_mesh.dim();

    if(!rank) {
	fprintf(stderr, "source mesh <v e f r> %d %d %d %d\n",
	    source_mesh.nglobal_ents(0),
	    source_mesh.nglobal_ents(1),
	    source_mesh.nglobal_ents(2),
	    dim == 3 ? source_mesh.nglobal_ents(3) : 0);
	
	fprintf(stderr, "target mesh <v e f r> %d %d %d %d\n",
	    target_mesh.nglobal_ents(0),
	    target_mesh.nglobal_ents(1),
	    target_mesh.nglobal_ents(2),
	    dim == 3 ? target_mesh.nglobal_ents(3) : 0);
    }

    const auto& source_coordinates = source_mesh.coords();
    const auto& target_coordinates = target_mesh.coords();

    
    SupportResults support =  searchNeighbors(source_mesh, target_mesh, cutoffDistance);

    
    auto target_values = mls_interpolation(source_coordinates, target_coordinates, support, dim, cutoffDistance);

    target_mesh.add_tag<Real>(0,"target_field", 1 , target_values);
    
    vtk::write_parallel("approximated field values in target mesh", &target_mesh, dim);

    return 0;
}

