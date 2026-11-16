// MEX entry: run a double-precision saxpy kernel on the GPU through the
// ngs_gpu interface (nvrtc-compiled source, device buffers, H2D/D2H) without
// Python. y = ngs_cuda_saxpy_mex(a, x, y) returns a*x + y computed on the device,
// with the shape of y.
#include <cuda_linalg.hpp>
#include <gpuwrapper.hpp>
#include "mex.h"
#include <climits>

static bool IsRealDenseDouble (const mxArray *m)
{
  return mxIsDouble(m) && !mxIsComplex(m) && !mxIsSparse(m);
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
  // every argument check happens before the GPU is touched
  if (nrhs != 3 || nlhs > 1)
    mexErrMsgIdAndTxt("ngs:saxpy", "usage: y = ngs_cuda_saxpy_mex(a, x, y)");
  for (int k = 0; k < 3; k++)
    if (!IsRealDenseDouble(prhs[k]))
      mexErrMsgIdAndTxt("ngs:saxpy", "argument %d must be a real, full (not sparse) double array", k+1);
  if (mxGetNumberOfElements(prhs[0]) != 1)
    mexErrMsgIdAndTxt("ngs:saxpy", "a must be a scalar");
  const size_t numel = mxGetNumberOfElements(prhs[1]);
  if (numel != mxGetNumberOfElements(prhs[2]))
    mexErrMsgIdAndTxt("ngs:saxpy", "x and y must have the same number of elements");
  if (numel > size_t(INT_MAX))
    mexErrMsgIdAndTxt("ngs:saxpy", "%zu elements exceed the kernel's int index range", numel);

  plhs[0] = mxDuplicateArray(prhs[2]);   // keeps the shape of y, any number of dimensions
  if (numel == 0)
    return;                              // nothing to launch

  try
    {
      ngs_cuda::InitCUDA(0);
      ngs_cuda::InitCudaDevice();
      ngla::InitCuLinalg();
      auto dev = ngs_gpu::GetDevice();
      if (!dev)
        mexErrMsgIdAndTxt("ngs:saxpy", "no GPU backend registered");

      const double a = mxGetScalar(prhs[0]);
      const int n = int(numel);
      const char *src = R"(
extern "C" __global__ void saxpy(const double* x, double* y, double a, int n)
{
  int id = blockIdx.x*blockDim.x + threadIdx.x;
  if (id < n) y[id] = a*x[id] + y[id];
}
)";
      auto kernel = dev->CompileSource(src)->GetKernel("saxpy");
      auto queue = dev->DefaultQueue();
      auto x = dev->NewBuffer<double>(n);
      auto y = dev->NewBuffer<double>(n);
      x.H2D(mxGetPr(prhs[1]), n);
      y.H2D(mxGetPr(plhs[0]), n);
      const unsigned tg = 256;
      const unsigned groups = unsigned((numel + tg - 1) / tg);   // numel <= INT_MAX: no overflow
      queue->Launch(kernel, {groups}, {tg}, {x, y, a, n});
      queue->Finish();
      y.D2H(mxGetPr(plhs[0]), n);
    }
  catch (const std::exception &e)
    {
      mexErrMsgIdAndTxt("ngs:saxpy", "%s", e.what());
    }
}
