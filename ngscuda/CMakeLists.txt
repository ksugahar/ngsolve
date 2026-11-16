if(NOT CMAKE_PROJECT_NAME STREQUAL "NGSolve")
  cmake_minimum_required(VERSION 3.18)
  project(ngscuda)
  find_package(NGSolve)
  if (CMAKE_INSTALL_PREFIX_INITIALIZED_TO_DEFAULT)
    set(CMAKE_INSTALL_PREFIX "${NGSOLVE_INSTALL_DIR}" CACHE PATH "Install directory" FORCE)
  endif()
  set(lib_name ngscudalib_local)
else()
  set(lib_name ngscudalib)
endif()

if(WIN32)
  # ngscudalib is its own DLL: it imports ngsolve.dll's symbols instead of
  # re-declaring them as its own exports (which breaks data members)
  remove_definitions(-DNGS_EXPORTS)
endif()

# host compiler only: the kernels are compiled at run time by nvrtc
find_package(CUDAToolkit REQUIRED)

include_directories(BEFORE ${CMAKE_CURRENT_SOURCE_DIR})

add_library(${lib_name} SHARED
    cuda_linalg.cpp cuda_ngstd.cpp cuda_ngbla.hpp
    cuda_profiler.cpp cuda_device.cpp
)

target_link_libraries(${lib_name} PUBLIC CUDA::cudart CUDA::nvrtc CUDA::cuda_driver ngsolve $<BUILD_INTERFACE:netgen_python>)

# optional sparse Cholesky through NVIDIA cuDSS: headers only at build time,
# the library is loaded at run time from an absolute path (no link dependency)
option(USE_CUDSS "GPU sparse Cholesky through cuDSS (headers from CUDSS_INCLUDE_DIR)" OFF)
if(USE_CUDSS)
  if(NOT EXISTS "${CUDSS_INCLUDE_DIR}/cudss.h")
    message(FATAL_ERROR "USE_CUDSS=ON needs CUDSS_INCLUDE_DIR with cudss.h, got '${CUDSS_INCLUDE_DIR}'")
  endif()
  target_sources(${lib_name} PRIVATE cudss_cholesky.cpp)
  target_include_directories(${lib_name} PRIVATE ${CUDSS_INCLUDE_DIR})
  target_compile_definitions(${lib_name} PUBLIC NGS_USE_CUDSS)
  target_link_libraries(${lib_name} PRIVATE ${CMAKE_DL_LIBS})
  install(FILES cudss_cholesky.hpp DESTINATION ${NGSOLVE_INSTALL_DIR_INCLUDE} COMPONENT ngsolve_devel)
endif()
set_target_properties(${lib_name} PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(CMAKE_PROJECT_NAME STREQUAL "NGSolve")
  install( TARGETS ${lib_name} ${ngs_install_dir} )
endif()

if(NETGEN_USE_PYTHON)
    find_package(Python3 COMPONENTS Interpreter Development)
    Python3_add_library(_ngscuda python_ngscuda.cpp)
    target_link_libraries(_ngscuda PUBLIC ${lib_name})
    set_target_properties(_ngscuda PROPERTIES INSTALL_RPATH "${NETGEN_RPATH_TOKEN}/../${NETGEN_PYTHON_RPATH}")
    install(TARGETS _ngscuda DESTINATION ${NGSOLVE_INSTALL_DIR_PYTHON}/ngsolve COMPONENT ngsolve)
endif(NETGEN_USE_PYTHON)

install( FILES
        cuda_api.hpp
        cuda_linalg.hpp
        cuda_profiler.hpp
        cuda_core.hpp
        cuda_ngstd.hpp
        cuda_ngbla.hpp
        cuda_device.hpp
        DESTINATION ${NGSOLVE_INSTALL_DIR_INCLUDE}
        COMPONENT ngsolve_devel
)

