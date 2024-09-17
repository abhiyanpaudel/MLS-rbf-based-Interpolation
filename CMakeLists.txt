cmake_minimum_required(VERSION 3.20)
project(Interpolator)

# Set version number (optional)
set(VERSION_MAJOR 0)
set(VERSION_MINOR 1)

# Define the list of header files
set(HEADER_FILES
    adj_search_dega2.hpp
    MLSInterpolation.hpp
    points.hpp
    adj_search.hpp
    MLSCoefficients.hpp
    queue_visited.hpp
)

# Specify the install location for the header files
install(FILES ${HEADER_FILES} DESTINATION include/interpolator)

# Optionally export a package configuration
include(CMakePackageConfigHelpers)

# Install the package version file for compatibility
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/interpolatorConfigVersion.cmake"
    VERSION ${VERSION_MAJOR}.${VERSION_MINOR}
    COMPATIBILITY SameMajorVersion
)

install(FILES "${CMAKE_CURRENT_BINARY_DIR}/interpolatorConfigVersion.cmake"
    DESTINATION lib/cmake/interpolator)

# Define the header-only library target
add_library(interpolator INTERFACE)
target_include_directories(interpolator INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include/interpolator>)

# Install the library target
install(TARGETS interpolator
    EXPORT interpolatorTargets
    INCLUDES DESTINATION include/interpolator)

# Generate the interpolatorConfig.cmake file
configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/interpolatorConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/interpolatorConfig.cmake"
    INSTALL_DESTINATION lib/cmake/interpolator
)

# Install the interpolatorConfig.cmake file
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/interpolatorConfig.cmake"
    DESTINATION lib/cmake/interpolator)

# Install the export targets
install(EXPORT interpolatorTargets
    FILE interpolatorTargets.cmake
    NAMESPACE interpolator::
    DESTINATION lib/cmake/interpolator)

