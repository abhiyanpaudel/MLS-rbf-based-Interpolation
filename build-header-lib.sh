cmake -S . -B build \
      -DCMAKE_INSTALL_PREFIX=/lore/hasanm4/wsources/dg2xgcDeps/build/ADA89/interpolator \
      -DCMAKE_CXX_COMPILER=`which mpicxx` 

cmake --build build --target install
