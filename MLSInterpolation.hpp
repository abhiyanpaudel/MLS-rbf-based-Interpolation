#ifndef MLS_INTERPOLATION_HPP
#define MLS_INTERPOLATION_HPP

#include "points.hpp"
#include "MLSCoefficients.hpp"
#include "adj_search.hpp"

Write<Real> mls_interpolation(const Reals source_coordinates, const Reals target_coordinates, const SupportResults& support, const LO& dim, const Real& cutoff_radius){
    const auto nvertices_source = source_coordinates.size();    
    const auto nvertices_target = target_coordinates.size();

    Write<Real>  approx_target_values(nvertices_target, 0, "approximated target values");


    Kokkos::parallel_for("MLS coefficients", team_policy(nvertices_target, Kokkos::AUTO), KOKKOS_LAMBDA(const member_type& team){

	int i = team.league_rank();
         
	int start_ptr = support.supports_ptr[i];
	int end_ptr = support.supports_ptr[i+1];


	int nsupports = end_ptr - start_ptr;

	DevicePoints local_source_points;
	 
        local_source_points.coordinates = DevicePointsViewType("Number of local source supports", nsupports);

     for (int j = start_ptr; j < end_ptr; ++j){
	auto index = support.supports_idx[j];
	local_source_points.coordinates(j).x = source_coordinates[index * dim];
        local_source_points.coordinates(j).y = source_coordinates[index * dim + 1];

     }

//	Real lower[6][6];
//	Real forward_matrix[6][6];
//	Real moment_matrix[6][6];
//	Real inverse_matrix[6][6];
//	Real targetMonomialVec[6];
//
	RealMatViewRight lower("Lower triangular matrix", 6,6);                                                                                                                                                            
     
        RealMatViewRight forward_matrix("matrix from forward substitution", 6,6);  
       

        RealMatViewRight moment_matrix("moment matrix P^T.phi.p", 6, 6);
 

        RealMatViewRight inverse_matrix("inverse of  P^T.phi.P", 6, 6);


        RealVecView targetMonomialVec("Monomial Vector", 6);


	
	RealMatViewRight V("Vandermonde matrix initialization", nsupports,6);
   
	RealVecView Phi("Phi Components", nsupports);

	RealMatViewRight Ptphi("PtPhi matrix multiplication", 6, nsupports);   
   
   
	RealMatViewRight resultant_matrix("inverse matrix times ptPhi", 6, nsupports);
   
  
	RealVecView exactSupportValues("exact values within supports", nsupports);
   

	RealVecView result("Kernel vector", nsupports);                                                       
   
    
	Coord target_point;
    
    
	target_point.x = target_coordinates[i*dim];
   
	target_point.y = target_coordinates[i*dim + 1];


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
	Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nsupports), [=](const int i){
	    exactSupportValues(i) = func(local_source_points.coordinates(i));
	});

        double tgt_value = 0; 
        dot_product(team, result, exactSupportValues, tgt_value);
        
       if (team.team_rank() == 0){        
           approx_target_values[i] = tgt_value;
        }
   }); 

     
}

#endif 
