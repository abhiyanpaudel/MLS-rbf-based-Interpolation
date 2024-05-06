#ifndef POINTS_HPP
#define POINTS_HPP

#include<Kokkos_Core.hpp>
#include<Kokkos_Random.hpp>


struct Coord{
  double x,y;

};


using HostPointsViewType = Kokkos::View<Coord*, Kokkos::HostSpace>;
using DevicePointsViewType = Kokkos::View<Coord*, Kokkos::CudaSpace>;
using HostDevicePointsViewType = Kokkos::View<Coord*, Kokkos::CudaUVMSpace>;
using HostRealVecView = Kokkos::View<double*, Kokkos::HostSpace>;
using DeviceRealVecView = Kokkos::View<double*, Kokkos::CudaUVMSpace>;

using IntVecView = Kokkos::View<int*, Kokkos::CudaSpace>;
using RealVecView = Kokkos::View<double*, Kokkos::CudaSpace>;
struct HostPoints{
  HostPointsViewType coordinates;

};

struct DevicePoints{
  DevicePointsViewType coordinates;

};


struct HostDevicePoints{
  HostDevicePointsViewType coordinates;

};


double func(Coord& p1){
    return p1.x * p1.x + p1.y * p1.y;
}

HostRealVecView exact_solution(HostPoints coord_points, int npts){
    HostRealVecView exact_function_values("Target exact function values",npts);
    //srcPoints.coordinates = HostPointsViewType("Source Points", npts);
    for (int i = 0; i < npts; ++i){
      exact_function_values(i) = func(coord_points.coordinates(i));
    }
    return exact_function_values; 
}

#endif 
