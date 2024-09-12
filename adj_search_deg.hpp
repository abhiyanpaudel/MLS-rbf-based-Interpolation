#ifndef ADJ_SEARCH_HPP
#define ADJ_SEARCH_HPP

#include "queue_visited.hpp"
#include <pcms/point_search.h>

using namespace Omega_h;

static constexpr int max_dim = 3;




// TODO change this into span/mdspan
OMEGA_H_INLINE
Real calculateDistance(const Real* p1, const Real* p2, const int dim) {
    Real dx, dy, dz;
    dx = p1[0] - p2[0];
    dy = p1[1] - p2[1];
    if (dim != 3){
      dz = 0.0;
    } else {
      dz = p1[2] - p2[2];
    }
   
    return dx * dx + dy * dy + dz * dz;
}

class FindSupports {

  private:
  Mesh& mesh;
  public:
   
  FindSupports(Mesh& mesh_): mesh(mesh_){};

  void adjBasedSearch(const Real& cutoffDistance, const Write<LO>& supports_ptr, Write<LO>& nSupports, Write<LO>& support_idx);


};

void  FindSupports::adjBasedSearch(const Real& cutoffDistance, const Write<LO>& supports_ptr, Write<LO>& nSupports, Write<LO>& support_idx){   

  // Mesh Info 

    const auto& mesh_coords = mesh.coords();
    const auto& nvertices = mesh.nverts();
    const auto& dim = mesh.dim();
    const auto& nfaces = mesh.nfaces();

    // CSR data structure of adjacent cell information of each vertex in a mesh
    const auto& nodes2faces = mesh.ask_up(VERT, FACE);
    const auto& n2f_ptr = nodes2faces.a2ab;
    const auto& n2f_data = nodes2faces.ab2a;

    Write<Real> cell_centroids(dim * nfaces, 0 , "stores coordinates of cell centroid of each tri element");

    const auto& faces2nodes = mesh.ask_down(FACE, VERT).ab2b;

    parallel_for("calculate the centroid in each tri element", nfaces, OMEGA_H_LAMBDA(const LO id){
	const auto current_el_verts = gather_verts<3>(faces2nodes, id);
	const Omega_h::Few<Omega_h::Vector<2>, 3> current_el_vert_coords =
            gather_vectors<3, 2>(mesh_coords, current_el_verts);
	auto centroid = average(current_el_vert_coords);		
	int index = 2 * id;
	cell_centroids[index] = centroid[0];
	cell_centroids[index + 1] = centroid[1];

    });


    parallel_for(nvertices, OMEGA_H_LAMBDA(const LO id){
      
      queue queue;
      track visited;
     
      Real target_coords[max_dim] ;
      
      Real support_coords[max_dim]; 
      
      for (LO k = 0; k < dim; ++k){
        target_coords[k] = mesh_coords[id * dim + k]; 
      }
     
      LO start_counter;
      
      if (support_idx.size() > 0){
        start_counter = supports_ptr[id]; 
      }
      LO start_ptr = n2f_ptr[id];
      LO end_ptr = n2f_ptr[id + 1];

      int count = 0; 
      // Initialize queue by pushing the vertices in the neighborhood of the given target point 
      
      for (LO i = start_ptr; i < end_ptr; ++i){
         LO cell_id = n2f_data[i];
         visited.push_back(cell_id);
         
         for (LO k = 0; k < dim; ++k){
             support_coords[k] = cell_centroids[cell_id * dim + k];
        }

         Real dist = calculateDistance(target_coords, support_coords, dim);
          if (dist <= cutoffDistance){
              count++;
              queue.push_back(cell_id);
              if (support_idx.size() > 0) {
                LO idx_count = count - 1;
                support_idx[start_counter + idx_count] = cell_id;
              }
          }            
      }
        
          
      while (!queue.isEmpty()){
           LO currentCell = queue.front();
           queue.pop_front();
           LO start = v2v_ptr[currentVertex];
           LO end = v2v_ptr[currentVertex + 1];
           
          for (LO i = start; i < end; ++i){
             auto neighborIndex = v2v_data[i];
           
           // check if neighbor index is already in the queue to be checked
           // TODO refactor this into a function
               
             if (visited.notVisited(neighborIndex)){
                
                visited.push_back(neighborIndex); 
                for (int k = 0; k < dim; ++k){
                  support_coords[k] = sourcePoints_coords[neighborIndex * dim + k];
                }
                
                Real dist = calculateDistance(target_coords, support_coords, dim);
                
		if (dist <= cutoffDistance){
                    count++;
                    queue.push_back(neighborIndex);
                    if (support_idx.size() > 0) {
                      LO idx_count = count - 1;
                      support_idx[start_counter + idx_count] = neighborIndex;
                    }
                }            
              } 
               
          }    
      } // end of while loop
     
      nSupports[id] = count; 
         
    }, "count the number of supports in each target point");


}


struct SupportResults{
    Write<LO> supports_ptr;
    Write<LO> supports_idx;

};    


SupportResults searchNeighbors(Mesh& source_mesh, Mesh& target_mesh, Real& cutoffDistance){
    SupportResults support;
    
    FindSupports search(source_mesh, target_mesh);

    LO nvertices_source = source_mesh.nverts();
    
    LO nvertices_target = target_mesh.nverts();
    
    Write<LO> nSupports(nvertices_target, 0, "number of supports in each target vertex");


    search.adjBasedSearch(cutoffDistance, support.supports_ptr, nSupports, support.supports_idx);

    Kokkos::fence();
    
    support.supports_ptr = Write<LO>(nvertices_target+1, 0, "number of support source vertices in CSR format");

    LO total_supports = 0;

    Kokkos::parallel_scan(nvertices_target, OMEGA_H_LAMBDA(int j, int& update, bool final){
	update += nSupports[j];
	if (final) {
	    support.supports_ptr[j + 1] = update;
	}
	
    }, total_supports);

    Kokkos::fence();

    support.supports_idx = Write<LO>(total_supports, 0, "index of source supports of each target node");

    search.adjBasedSearch(cutoffDistance, support.supports_ptr, nSupports, support.supports_idx);

    return support;

}



#endif



