cmake_minimum_required (VERSION 3.0)
project (eigen_bench)
 
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
find_package (Eigen3 3.3 REQUIRED NO_MODULE)
add_subdirectory(./benchmark)

add_executable (abat abat.cpp)
target_link_libraries (abat Eigen3::Eigen benchmark::benchmark)

add_executable (solver solver.cpp)
target_link_libraries (solver Eigen3::Eigen benchmark::benchmark)
