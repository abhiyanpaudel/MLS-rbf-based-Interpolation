export NVCC_WRAPPER_DEFAULT_COMPILER=`which mpicxx`

export DEVICE_ARCH=ADA89

DEPENDENCY_DIR="${DEPENDENCY_DIR:-/lore/paudea/build}"
DEVICE_ARCH="${DEVICE_ARCH:-ADA89}"

cmake -S . -B build \
      -DCMAKE_CXX_COMPILER=`which mpicxx` \
      -DCMAKE_C_COMPILER=`which mpicc` \
      -DOmega_h_USE_Kokkos=ON \
      -DOmega_h_USE_CUDA=ON \
      -DOmega_h_ROOT=$DEPENDENCY_DIR/${DEVICE_ARCH}/omega_h/install/ \
      -DKokkos_ROOT=$DEPENDENCY_DIR/${DEVICE_ARCH}/kokkos/install/ \
      -Dpcms_ROOT= \
      -Dperfstubs_DIR=$DEPENDENCY_DIR/perfstubs/install/lib64/cmake/ \
      -DADIOS2_ROOT=$DEPENDENCY_DIR/adios2/install/ \
      -DCMAKE_BUILD_TYPE=Debug
