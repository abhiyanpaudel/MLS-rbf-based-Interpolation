#ifndef POINTS_HPP
#define POINTS_HPP

#include<Kokkos_Core.hpp>
#include<Kokkos_Random.hpp>


struct Coord{
  double x,y;

};


#ifdef KOKKOS_ENABLE_CUDA
using DevicePointsViewType = Kokkos::View<Coord*, Kokkos::CudaSpace>;
using HostDevicePointsViewType = Kokkos::View<Coord*, Kokkos::CudaUVMSpace>;
using DeviceRealVecView = Kokkos::View<double*, Kokkos::CudaUVMSpace>;
using IntVecView = Kokkos::View<int*, Kokkos::CudaSpace>;
using RealVecView = Kokkos::View<double*, Kokkos::CudaSpace>;
#else
using DevicePointsViewType = Kokkos::View<Coord*, Kokkos::HostSpace>;
using HostDevicePointsViewType = Kokkos::View<Coord*, Kokkos::HostSpace>;
using DeviceRealVecView = Kokkos::View<double*, Kokkos::HostSpace>;
using IntVecView = Kokkos::View<int*, Kokkos::HostSpace>;
using RealVecView = Kokkos::View<double*, Kokkos::HostSpace>;
#endif

using HostPointsViewType = Kokkos::View<Coord*, Kokkos::HostSpace>;
using HostRealVecView = Kokkos::View<double*, Kokkos::HostSpace>;
struct HostPoints{
  HostPointsViewType coordinates;

};

struct DevicePoints{
  DevicePointsViewType coordinates;

};


struct HostDevicePoints{
  HostDevicePointsViewType coordinates;

};






#endif 
