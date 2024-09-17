export NVCC_WRAPPER_DEFAULT_COMPILER=`which mpicxx`

export DEVICE_ARCH=ADA89

DEPENDENCY_DIR=/lore/hasanm4/wsources/dg2xgcDeps/
DEVICE_ARCH="${DEVICE_ARCH:-ADA89}"

cmake -S . -B build \
      -DCMAKE_INSTALL_PREFIX=/lore/hasanm4/wsources/dg2xgcDeps/build/ADA89/interpolator \
      -DCMAKE_CXX_COMPILER=/lore/mersoj2/laces-software/build/ADA89/kokkos/install/bin/nvcc_wrapper \
      -DCMAKE_C_COMPILER=`which mpicc` 
