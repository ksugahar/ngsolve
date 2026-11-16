// MEX entry: initialise ngscuda from MATLAB (no Python); returns HasDevice().
#include <cuda_linalg.hpp>
#include <gpuwrapper.hpp>
#include "mex.h"
void mexFunction(int nlhs, mxArray *plhs[], int, const mxArray *[])
{
  ngs_cuda::InitCUDA(0);
  ngs_cuda::InitCudaDevice();
  ngla::InitCuLinalg();
  plhs[0] = mxCreateLogicalScalar(ngs_gpu::HasDevice());
}
