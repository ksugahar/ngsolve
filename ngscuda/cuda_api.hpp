#ifndef FILE_CUDA_API
#define FILE_CUDA_API

// Export macro of ngscudalib. On Windows it is a DLL of its own, used by
// MEX and other C++ consumers that call InitCUDA / InitCudaDevice /
// InitCuLinalg themselves (the Python module is not part of the SDK).
#include <core/ngcore_api.hpp>

#if defined(ngscudalib_EXPORTS) || defined(ngscudalib_local_EXPORTS)
  #define NGS_CUDA_API NGCORE_API_EXPORT
#else
  #define NGS_CUDA_API NGCORE_API_IMPORT
#endif

#endif
