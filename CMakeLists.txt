project(mfem-cmake-project C CXX)
cmake_minimum_required(VERSION 3.20)

find_package(Omega_h 10 REQUIRED)

find_package(pcms REQUIRED)
add_executable(field_transfer field_transfer_MLS_dega2.cpp)
target_link_libraries(field_transfer PUBLIC Omega_h::omega_h pcms::pcms)
