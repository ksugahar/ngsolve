/*********************************************************************/
/* File:   templates.cpp                                             */
/* Author: Joachim Schoeberl                                         */
/* Date:   25. Mar. 2000                                             */
/*********************************************************************/

#ifdef USE_MKL
#include <mkl.h>
#endif // USE_MKL

#include <ngstd.hpp>
#include <ngsolve_version.hpp>
#include <netgen_version.hpp>
#include <core/mpi_wrapper.hpp>

namespace ngstd
{
  ngcore::NG_MPI_Comm ngs_comm;

  // int printmessage_importance = 5;
  // bool NGSOStream :: glob_active = true;
  const string ngsolve_version = NGSOLVE_VERSION;


#ifdef USE_MKL
  // Lazy initialization to avoid calling MKL functions during DllMain
  // MKL functions cannot be safely called during static initialization on Windows
  static bool mkl_initialized = false;
  static int mkl_max_threads_value = 1;

  int get_mkl_max_threads() {
      if (!mkl_initialized) {
          mkl_max_threads_value = mkl_get_max_threads();
          mkl_set_num_threads(1);
          mkl_initialized = true;
      }
      return mkl_max_threads_value;
  }

  // For backward compatibility, provide the variable as a function call
  // Note: Code that uses mkl_max_threads directly should be updated to call get_mkl_max_threads()
  int mkl_max_threads = 1;  // Default value, actual value obtained via get_mkl_max_threads()
#endif // USE_MKL

  // All static initialization disabled for Windows DllMain compatibility
  // TODO: Move initialization to a runtime function called from Python __init__.py
  // static bool dummy = [] ()
  // {
  //   ngcore::SetLibraryVersion("ngsolve", NGSOLVE_VERSION);
  //   return true;
  // }();
}
