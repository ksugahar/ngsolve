#ifndef FILE_GPU_API
#define FILE_GPU_API

// Export macro for the backend-independent GPU interface (gpuwrapper,
// gpukernel). On Windows these symbols live in ngsolve.dll and are used by
// separate backend libraries (ngscudalib) and by consumers of the SDK.
#include <core/ngcore_api.hpp>

#ifdef NGS_EXPORTS
  #define NGS_GPU_API NGCORE_API_EXPORT
#else
  #define NGS_GPU_API NGCORE_API_IMPORT
#endif

#endif
