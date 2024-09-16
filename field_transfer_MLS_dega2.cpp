
#include <cstdlib>
#include "MLSInterpolation.hpp"
#include <iostream>

using namespace Omega_h;
using namespace pcms;



int main(int argc, char** argv) {
    auto lib = Library(&argc, &argv);
    auto  world = lib.world();
    auto rank = lib.world()->rank();
    auto mesh = build_box(world, OMEGA_H_SIMPLEX, 1, 1, 1, 10, 10, 0, false);
    if (argc != 3){
	fprintf(stderr, "Usage: %s <cutoff distance> <node id>\n", argv[0]);
	return 0;
    }
    
    Real cutoffDistance = std::atof(argv[1]);
    LO node_id = std::atof(argv[2]); 

    cutoffDistance = cutoffDistance * cutoffDistance;

    const auto dim = mesh.dim();

    if (!rank){
	fprintf(stderr, "mesh <v e f r> %d %d %d %d\n",
	    mesh.nglobal_ents(0),
	    mesh.nglobal_ents(1),
	    mesh.nglobal_ents(2),
	    dim == 3 ? mesh.nglobal_ents(3) : 0);
	
    }

    const auto& target_coordinates = mesh.coords();
    
    const auto& nfaces = mesh.nfaces();                                                                      
              
    Write<Real> source_coordinates(dim * nfaces, 0 , "stores coordinates of cell centroid of each tri element"); 
          
    const auto& faces2nodes = mesh.ask_down(FACE, VERT).ab2b;                                                 
                                                                                                             
    parallel_for("calculate the centroid in each tri element", nfaces, OMEGA_H_LAMBDA(const LO id){          
        const auto current_el_verts = gather_verts<3>(faces2nodes, id);                                      
        const Omega_h::Few<Omega_h::Vector<2>, 3> current_el_vert_coords =                                   
            gather_vectors<3, 2>(target_coordinates, current_el_verts);                                             
        auto centroid = average(current_el_vert_coords);                                                     
        int index = 2 * id;                                                                                  
        source_coordinates[index] = centroid[0];                                                                 
        source_coordinates[index + 1] = centroid[1];                                                             
	
    });
    
   // printf("cell centroid coordinates of cell 10, %f, %f\n", cell_centroids[20], cell_centroids[21]);
    //source_mesh.add_tag<Real>(0,"coordinates_cell_centroids", 1 ,cell_centroids);
   // vtk::write_parallel("values",&source_mesh, dim);
    
    printf("1\n");
    SupportResults support =  searchNeighbors(mesh, cutoffDistance);
    
    printf("2\n");
    auto approx_target_values = mls_interpolation(source_coordinates, target_coordinates, support, dim, cutoffDistance);
   

    printf("3\n");
    auto host_approx_target_values = HostRead<Real>(approx_target_values);


    Write<LO> supports_per_target(mesh.nverts(), 0, "number of supports in each target point");
    

    Kokkos::parallel_for(mesh.nverts(), KOKKOS_LAMBDA(int i){
	   int start = support.supports_ptr[i];
	   int end = support.supports_ptr[i+1];
	   int num_supports = end - start;
	   supports_per_target[i] = num_supports;
    });

    Points target_points;
	 
    target_points.coordinates = PointsViewType("Number of local source supports", mesh.nverts());
     Kokkos::parallel_for("target points", mesh.nverts(), KOKKOS_LAMBDA(int j){
	target_points.coordinates(j).x = target_coordinates[j * dim];
        target_points.coordinates(j).y = target_coordinates[j * dim + 1];

     });

    Write<Real> exact_target_values(mesh.nverts(), 0, "exact target values");
			
    Kokkos::parallel_for(mesh.nverts(), KOKKOS_LAMBDA(int i){
	    exact_target_values[i] = func(target_points.coordinates(i));
    });

    auto host_exact_target_values = HostRead<Real>(exact_target_values); 

    mesh.add_tag<Real>(0,"target_field_approx", 1 , approx_target_values);
    mesh.add_tag<Real>(0,"target_field_exact", 1 , exact_target_values);
    mesh.add_tag<LO>(0,"num_supports", 1 , supports_per_target);
    

    std::cout <<"Exact Values" << "\t" << "Interpolated Values" <<"\n"<< std::endl;
    for (int i = 0; i < host_approx_target_values.size(); ++i){
	std::cout <<host_exact_target_values[i] << "\t" << host_approx_target_values[i] << std::endl;
    }
   
    auto host_supports_ptr = HostRead<LO>(support.supports_ptr);
    auto host_supports_idx = HostRead<LO>(support.supports_idx);
    LO start_point_num = host_supports_ptr[node_id];
    LO end_point_num = host_supports_ptr[node_id + 1];
    const LO num_supports = end_point_num-start_point_num;
    Omega_h::Write<LO> is_inside(nfaces, 0);
    Omega_h::parallel_for(num_supports, OMEGA_H_LAMBDA(int i) {
	auto support_cell = support.supports_idx[start_point_num+i];
	is_inside[support_cell] = 1;
     });
     mesh.add_tag<LO>(2, "supported", 1, is_inside);
     for (LO i = start_point_num; i < end_point_num; ++i){
       cout << host_supports_idx[i] << endl;
    }


    vtk::write_parallel("field_values", &mesh, dim);



    return 0;
}

