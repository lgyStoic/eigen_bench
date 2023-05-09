cmake_minimum_required (VERSION 3.0)
project (eigen_bench)
 
find_package (Eigen3 3.3 REQUIRED NO_MODULE)
add_subdirectory(./benchmark)

add_executable (abat abat.cpp)
target_link_libraries (abat Eigen3::Eigen benchmark::benchmark)
