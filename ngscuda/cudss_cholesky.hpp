#ifndef FILE_CUDSS_CHOLESKY_HPP
#define FILE_CUDSS_CHOLESKY_HPP

/*
  Sparse Cholesky of the free-dof block A_FF on the GPU through NVIDIA cuDSS
  (built with USE_CUDSS). FP64, one right-hand side.

  cuDSS is not linked: LoadCuDSS resolves it from an absolute path at run time
  (Python passes the pinned pip package, a MEX its own path), so ngscuda loads
  without it and a missing library is an error at construction.

  Two entries:
   - from a host SparseMatrix: the pattern and values are uploaded once
     (a host-assembled pilot, not a device-origin route);
   - from caller-owned device CSR buffers (zero-based, int32 indices): only the
     pattern (symbolic metadata) is read back, values never leave the device.

  Contract (see RADIA_FORK.md, P10):
   - storage Full must be exactly mirrored on the free block (equal finite
     values, an absent entry counts as 0); Lower/Upper are one triangle;
   - device data is ready on the default queue, or by the event passed
     (waited on the host), and is not changed while a call runs;
   - every call returns after the device work is complete;
   - MultAdd: y += s * scatter(A_FF^{-1} gather(x)), constrained entries of y
     untouched; x, y are DeviceVector<double>, current on the device, x != y;
   - a factorization reporting a non-positive minor, a non-finite result or a
     true residual above the limit raises and leaves the factor unusable.
*/

#include <la.hpp>
#include <cuda_runtime.h>
#include "cuda_api.hpp"

namespace ngla
{
  enum class CuDSSStorage { Full, Lower, Upper };

  // Load cuDSS from an absolute path (idempotent for the same path; another path
  // or another already-loaded module of the same name raises). Returns a report
  // with the loaded file, the version and the cuBLAS module it resolved to.
  NGS_CUDA_API std::string LoadCuDSS (const std::string & abs_path, int major, int minor, int patch);

  class NGS_CUDA_API CuDSSCholesky : public BaseMatrix
  {
    struct Impl;
    unique_ptr<Impl> impl;
  public:
    CuDSSCholesky (const SparseMatrixTM<double> & mat, shared_ptr<BitArray> freedofs,
                   CuDSSStorage storage, double residual_limit = 1e-6);
    CuDSSCholesky (size_t n, size_t nnz, uintptr_t dev_rowptr, uintptr_t dev_colind,
                   uintptr_t dev_values, CuDSSStorage storage, shared_ptr<BitArray> freedofs,
                   cudaEvent_t ready = nullptr, double residual_limit = 1e-6);
    ~CuDSSCholesky () override;

    // same pattern (compared entry by entry) and new values; refactorization only
    void Update (const SparseMatrixTM<double> & mat);
    void Update (uintptr_t dev_rowptr, uintptr_t dev_colind, uintptr_t dev_values,
                 cudaEvent_t ready = nullptr);

    void Solve (double s, const BaseVector & x, BaseVector & y, cudaEvent_t ready) const;

    int VHeight() const override;
    int VWidth() const override;
    bool IsComplex() const override { return false; }
    void MultAdd (double s, const BaseVector & x, BaseVector & y) const override
    { Solve (s, x, y, nullptr); }
    void MultTransAdd (double s, const BaseVector & x, BaseVector & y) const override
    { Solve (s, x, y, nullptr); }
    VecFormat RowFormat () const override;
    VecFormat ColFormat () const override;

    // timings and transfer ledger of the last calls, as a json object
    std::string Report () const;
    // test hook: skip the queue synchronisation before reading device data
    void SetSkipQueueSync (bool skip);
  };
}

#endif
