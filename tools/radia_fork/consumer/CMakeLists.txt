cmake_minimum_required(VERSION 3.20)
project(ngs_sdk_consumer CXX)
option(CHECK_CUDA "Also link the CUDA SDK consumers (built, not run)" OFF)
find_package(NGSolve CONFIG REQUIRED)       # with USE_CUDA it also finds CUDAToolkit

add_executable(lapack_check lapack_check.cpp)          # CPU, ILP64 LAPACK
target_link_libraries(lapack_check PRIVATE ngsolve)

if(CHECK_CUDA)
  find_package(Matlab REQUIRED COMPONENTS MX_LIBRARY)
  add_executable(cuda_init_check cuda_init_check.cpp)
  target_link_libraries(cuda_init_check PRIVATE ngscudalib)
  matlab_add_mex(NAME ngs_cuda_mex SRC ngs_cuda_mex.cpp LINK_TO ngscudalib)
  matlab_add_mex(NAME ngs_cuda_saxpy_mex SRC ngs_cuda_saxpy_mex.cpp LINK_TO ngscudalib)
endif()
