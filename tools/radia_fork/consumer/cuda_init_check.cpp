// Registers the ngscuda backend without Python and reports the device.
#include <cuda_linalg.hpp>
#include <gpuwrapper.hpp>
#include <cstdio>
int main()
{
  ngs_cuda::InitCUDA(1);
  ngs_cuda::InitCudaDevice();
  ngla::InitCuLinalg();
  std::printf("{\"has_device\": %s}\n", ngs_gpu::HasDevice() ? "true" : "false");
}
