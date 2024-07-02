#include "adj_search.hpp"
#include <cstdlib>
using namespace Omega_h;
using namespace pcms;
   
int main(int argc, char** argv) {
  
  auto lib = Omega_h::Library(&argc, &argv);
  if(argc!=4) {
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
  LO nvertices_source = source_mesh.nverts();  
  LO nvertices_target = target_mesh.nverts();  
  

  FindSupports search(source_mesh, target_mesh);
  Write<LO> nSupports(nvertices_target, 0, "number of supports in each target vertex");

  Write<LO> supports_idx;
  Write<LO> supports_array;
  
  // first pass counts the size
  search.adjBasedSearch(cutoffDistance, supports_array, nSupports, supports_idx);
  Kokkos::fence();

  supports_array = Write<LO>(nvertices_target+1, 0, "number of support source vertices in CSR format");
 
  LO total_supports = 0;

  Kokkos::parallel_scan(nvertices_target, OMEGA_H_LAMBDA(int j, int& update, bool final){
   update += nSupports[j]; 
   if (final){
      supports_array[j+1] = update;
   }   

  }, total_supports);
  
  
  Kokkos::fence();
   
  supports_idx = Write<LO>(total_supports, 0, "supports index");
  search.adjBasedSearch(cutoffDistance, supports_array, nSupports,  supports_idx);   
  Kokkos::fence();
 
  printf("total supports %d\n",total_supports);
  
  auto host_support_index = HostRead<LO>(supports_array);
  
  auto host_support_values = HostRead<LO>(supports_idx);
  
  printf("The size of supports array is %d\n", host_support_values.size());
  
  printf("The last entry of supports array is %d\n", host_support_index[nvertices_target]);  
  
  LO point_num = 0;
  
  LO start_point_num = host_support_index[point_num];
  
  LO end_point_num = host_support_index[point_num + 1];
  
  cout << "The total supports of " << point_num << " is:" << end_point_num - start_point_num << endl;
  
  const LO num_supports = end_point_num-start_point_num;
  
  Omega_h::Write<LO> is_inside(source_mesh.nverts(), 0);
  
  Omega_h::parallel_for(num_supports, OMEGA_H_LAMBDA(int i) {
    auto support_node = supports_idx[start_point_num+i];
    is_inside[support_node] = 1;
  });
  
  source_mesh.add_tag<LO>(0, "supported", 1, is_inside);
  
  cout <<" supports of node id " << point_num << " are : " << endl;
     for (LO i = start_point_num; i < end_point_num; ++i){ 
       cout << host_support_values[i] << endl;  
    }
  
  Omega_h::vtk::write_parallel("source_mesh_tagged", &source_mesh, source_mesh.dim());

  return 0;
}

