#ifndef MLS_INTERPOLATION_HPP
#define MLS_INTERPOLATION_HPP

#include "points.hpp"
#include "MLSCoefficients.hpp"



void mls_interpolation(DevicePoints device_source_data, DevicePoints device_target_data, DeviceRealVecView d_source_exact_sol, RealVecView approx_target_values , double cutoff_radius){
    
     int nTar = device_target_data.coordinates.extent(0);
     int nSrc = device_source_data.coordinates.extent(0);

     
     RealMatViewRight lower("Lower triangular matrix", 6,6);                                                                                                                                                            
     
     RealMatViewRight forward_matrix("matrix from forward substitution", 6,6);  
       

     RealMatViewRight moment_matrix("moment matrix P^T.phi.p", 6, 6);
 

     RealMatViewRight inverse_matrix("inverse of  P^T.phi.P", 6, 6);


     RealVecView targetMonomialVec("Monomial Vector", 6);
     
     Kokkos::parallel_for("MLS coefficients", team_policy(nTar, Kokkos::AUTO), KOKKOS_LAMBDA(const member_type& team){

     int i = team.league_rank();
      
     TargetToSourceMap target_to_source_map;
      int nsupports = 0;
      Kokkos::parallel_reduce(Kokkos::TeamThreadRange(team, nSrc), [=](int j, int& inner_count) {
         double dist = distance(device_target_data.coordinates(i), device_source_data.coordinates(j));
            if (dist < cutoff_radius) {
                inner_count++;
            }
        },nsupports);
    
    Kokkos::fence();    
  
    target_to_source_map.source_ids = IntVecView("SourceIds", nsupports);

    
        int count = 0;
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nSrc), [&](int j) {
            double dist = distance(device_target_data.coordinates(i), device_source_data.coordinates(j));
            if (dist < cutoff_radius) {
                int idx = Kokkos::atomic_fetch_add(&count,1); 
                target_to_source_map.source_ids(idx-1) = j;
            }
        });
 

    Kokkos::fence();
 
     DevicePoints local_source_points;
     
 
     local_source_points.coordinates = DevicePointsViewType("Number of local source supports", nsupports);

     for (int j = 0; j < nsupports; ++j){
	int source_idx = target_to_source_map.source_ids(j);     
	local_source_points.coordinates(j).x = device_source_data.coordinates(source_idx).x;
	local_source_points.coordinates(j).y = device_source_data.coordinates(source_idx).y;	

     }	     
     
     RealMatViewRight V("Vandermonde matrix initialization", nsupports,6);
      
     RealVecView Phi("Phi Components", nsupports);

     RealMatViewRight Ptphi("PtPhi matrix multiplication", 6, nsupports);   
    
     
     RealMatViewRight resultant_matrix("inverse matrix times ptPhi", 6, nsupports);
     
  
     RealVecView exactSupportValues("exact values within supports", nsupports);
    

     RealVecView result("Kernel vector", nsupports);                                                       
    
        
     Coord target_point;
        
        
     target_point.x = device_target_data.coordinates(i).x;
      
     target_point.y = device_target_data.coordinates(i).y;


     BasisPoly(targetMonomialVec, target_point);
     
    
       Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nsupports), [=](int j){ 
           VandermondeMatrix(V, local_source_points, j);
           team.team_barrier();
           PhiVector (Phi,target_point, local_source_points, j, cutoff_radius);
           team.team_barrier();
           PTphiMatrix(Ptphi, V, Phi, j);
           team.team_barrier();
             
       });

        MatMatMul(team, moment_matrix, Ptphi, V);
        Kokkos::fence(); 
        cholesky_decomposition( team, moment_matrix , lower);   
	Kokkos::fence();
        forward_substitution( team, lower, forward_matrix);
    	Kokkos::fence();
        backward_substitution(team, forward_matrix, lower, inverse_matrix);   
	Kokkos::fence();
        
       MatMatMul( team, resultant_matrix, inverse_matrix, Ptphi); 
       Kokkos::fence();	
       MatVecMul(team, targetMonomialVec, resultant_matrix,result);
       Kokkos::fence(); 
        double tgt_value = 0; 
        dot_product(team, result, exactSupportValues, tgt_value);
        
       if (team.team_rank() == 0){        
           approx_target_values(i) = tgt_value;
        }
   }); 

     
}

#endif 
