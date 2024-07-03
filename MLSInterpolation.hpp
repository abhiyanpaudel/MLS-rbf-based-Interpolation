#ifndef MLS_INTERPOLATION_HPP
#define MLS_INTERPOLATION_HPP

#include "points.hpp"
#include "MLSCoefficients.hpp"
#include "adj_search.hpp"

Write<Real> mls_interpolation(const Reals source_coordinates, const Reals target_coordinates, const SupportResults& support, const LO& dim, const Real& cutoff_radius){
    const auto nvertices_source = source_coordinates.size()/dim;    
    const auto nvertices_target = target_coordinates.size()/dim;

    Write<Real>  approx_target_values(nvertices_target, 0, "approximated target values");
    team_policy tp(nvertices_target, Kokkos::AUTO);
    Kokkos::parallel_for("MLS coefficients", tp, KOKKOS_LAMBDA(const member_type& team){

	int i = team.league_rank();
	int start_ptr = support.supports_ptr[i];
	int end_ptr = support.supports_ptr[i+1];


	int nsupports = end_ptr - start_ptr;

	DevicePoints local_source_points;
	 
        local_source_points.coordinates = DevicePointsViewType("Number of local source supports", nsupports);
        int count = -1;
for (int j = start_ptr; j < end_ptr; ++j){
	count++;
	auto index = support.supports_idx[j];
	local_source_points.coordinates(count).x = source_coordinates[index * dim];
        local_source_points.coordinates(count).y = source_coordinates[index * dim + 1];

     }
        
	Kokkos::View<double**> lower("Lower triangular matrix", 6,6);                                                                                                                                                            
     
        Kokkos::View<double**> forward_matrix("matrix from forward substitution", 6,6);  
       

        RealMatViewRight moment_matrix("moment matrix P^T.phi.p", 6, 6);
 

        Kokkos::View<double**> inv_mat("inverse of  P^T.phi.P", 6, 6);


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
	});

       
       if (i == 10) {
	   printf("V matrix\n");
	  for (int j = 0; j < V.extent(0); ++j){
	    for (int k = 0; k < V.extent(1); ++k){
        	   printf(" %f ", V(j,k));

	      }	   

	    printf("\n");
	    }

	  } 
	team.team_barrier();
       
       Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nsupports), [=](int j){ 
           PhiVector (Phi,target_point, local_source_points, j, cutoff_radius);
       });
       
       if (i == 10) {
	  for (int k = 0; k < Phi.extent(0); ++k){
	   printf("Phi = %f \n", Phi(k));}

	  } 
	team.team_barrier();
           

       Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nsupports), [=](int j){ 
	   PTphiMatrix(Ptphi, V, Phi, j);        
       });
        
       if (i == 10) {
	   printf("PtPhi matrix\n");
	  for (int j = 0; j < Ptphi.extent(0); ++j){
	    for (int k = 0; k < Ptphi.extent(1); ++k){
        	   printf(" %f ", Ptphi(j,k));

	      }	   

	    printf("\n");
	    }

	  } 
	team.team_barrier();

	MatMatMul(team, moment_matrix, Ptphi, V);
	//team.team_barrier();
        
       if (i == 10) {
	   printf("moment matrix\n");
	  for (int j = 0; j < moment_matrix.extent(0); ++j){
	    for (int k = 0; k < moment_matrix.extent(1); ++k){
        	   printf(" %f ", moment_matrix(j,k));

	      }	   

	    printf("\n");
	    }

	  } 
       
       if (i == 10) {
	   printf("lower matrix\n");
	  for (int j = 0; j < lower.extent(0); ++j){
	    for (int k = 0; k < lower.extent(1); ++k){
        	   printf(" %f ", lower(j,k));

	      }	   

	    printf("\n");
	    }

	  }



       if (i == 10) {
	   printf("forward matrix\n");
	  for (int j = 0; j < forward_matrix.extent(0); ++j){
	    for (int k = 0; k < forward_matrix.extent(1); ++k){
        	   printf(" %f ", forward_matrix(j,k));

	      }	   

	    printf("\n");
	    }

	  } 
        inverse_matrix(team, moment_matrix, lower, forward_matrix, inv_mat);
    
       if (i == 10) {
	   printf("lower matrix\n");
	  for (int j = 0; j < lower.extent(0); ++j){
	    for (int k = 0; k < lower.extent(1); ++k){
        	   printf(" %f ", lower(j,k));

	      }	   

	    printf("\n");
	    }

	  }



       if (i == 10) {
	   printf("forward matrix\n");
	  for (int j = 0; j < forward_matrix.extent(0); ++j){
	    for (int k = 0; k < forward_matrix.extent(1); ++k){
        	   printf(" %f ", forward_matrix(j,k));

	      }	   

	    printf("\n");
	    }

	  } 
         
	team.team_barrier();
       if (i == 10) {
	   printf("inverse matrix\n");
	  for (int j = 0; j < inv_mat.extent(0); ++j){
	    for (int k = 0; k < inv_mat.extent(1); ++k){
        	   printf(" %f ", inv_mat(j,k));

	      }	   

	    printf("\n");
	    }

	  } 
       MatMatMul( team, resultant_matrix, inv_mat, Ptphi); 
	team.team_barrier();
       
       if (i == 10) {
	   printf("matri matrix mult\n");
	  for (int j = 0; j < resultant_matrix.extent(0); ++j){
	    for (int k = 0; k < resultant_matrix.extent(1); ++k){
        	   printf(" %f ", resultant_matrix(j,k));

	      }	   

	    printf("\n");
	    }

	  } 
       MatVecMul(team, targetMonomialVec, resultant_matrix,result);
	team.team_barrier();
	
       if (i == 10) {
	   printf("result matrix\n");
	  for (int j = 0; j < result.extent(0); ++j){
        	   printf(" %f ", result(j));

	      }	   

	    printf("\n");

	  } 
	Kokkos::parallel_for(Kokkos::TeamThreadRange(team, nsupports), [=](const int i){
	    exactSupportValues(i) = func(local_source_points.coordinates(i));
	});

       if (i == 10) {
	   printf("exact values of support\n");
	  for (int j = 0; j < exactSupportValues.extent(0); ++j){
        	   printf(" %f ", exactSupportValues(j));

	      }	   

	    printf("\n");

	  } 
        double tgt_value = 0; 
        dot_product(team, result, exactSupportValues, tgt_value);
	if (i == 10) {printf("target value at %d node = %d", i, tgt_value);}
       if (team.team_rank() == 0){        
           approx_target_values[i] = tgt_value;
        }
   }); 

     printf("target approximated values");
     for (int j = 0; j < approx_target_values.size(); ++j){

	 printf(" %f\n ", approx_target_values[j]);

     }
	return approx_target_values;
}

#endif 
