#include <core/python_ngcore.hpp>

#include "../ngstd/python_ngstd.hpp"
#include "cuda_linalg.hpp"
#include "cuda_device.hpp"
#ifdef NGS_USE_CUDSS
#include "cudss_cholesky.hpp"
#endif

// TODO: always use ngs_cuda?
using namespace ngbla;
using namespace ngla;
using namespace ngs_cuda;

PYBIND11_MODULE(_ngscuda, m) {

  InitCUDA(1);
  InitCudaDevice();      // register as ngs_gpu backend
  InitCuLinalg();

  m.def("InitCuLinalg", &InitCuLinalg, "Initializing cuda linalg.");
  
  m.def("CreateDevMatrix", [] (BaseMatrix &mat)
          {
            return CreateDevMatrix(mat);
          });
    



#ifdef NGS_USE_CUDSS
  py::enum_<CuDSSStorage> (m, "CuDSSStorage")
    .value("Full", CuDSSStorage::Full).value("Lower", CuDSSStorage::Lower).value("Upper", CuDSSStorage::Upper);
  m.def("_LoadCuDSS", &LoadCuDSS, py::arg("path"), py::arg("major"), py::arg("minor"), py::arg("patch"));
  py::class_<CuDSSCholesky, shared_ptr<CuDSSCholesky>, BaseMatrix> (m, "CuDSSCholesky",
      "sparse Cholesky of the free-dof block on the GPU through cuDSS (fp64, SPD); "
      "load the library with ngsolve.ngscuda.LoadCuDSS() first")
    .def(py::init([] (shared_ptr<BaseMatrix> mat, shared_ptr<BitArray> freedofs, CuDSSStorage storage, double limit)
                  {
                    auto sm = dynamic_pointer_cast<SparseMatrixTM<double>> (mat);
                    if (!sm) throw Exception ("CuDSSCholesky: expected a real sparse matrix");
                    return make_shared<CuDSSCholesky> (*sm, freedofs, storage, limit);
                  }), py::arg("mat"), py::arg("freedofs"), py::arg("storage"), py::arg("residual_limit") = 1e-6)
    .def(py::init([] (size_t n, size_t nnz, uintptr_t rowptr, uintptr_t colind, uintptr_t values,
                      CuDSSStorage storage, shared_ptr<BitArray> freedofs, uintptr_t ready, double limit)
                  {
                    return make_shared<CuDSSCholesky> (n, nnz, rowptr, colind, values, storage, freedofs,
                                                       reinterpret_cast<cudaEvent_t>(ready), limit);
                  }), py::arg("n"), py::arg("nnz"), py::arg("rowptr"), py::arg("colind"), py::arg("values"),
         py::arg("storage"), py::arg("freedofs"), py::arg("ready_event") = 0, py::arg("residual_limit") = 1e-6)
    .def("Update", [] (CuDSSCholesky & self, shared_ptr<BaseMatrix> mat)
         {
           auto sm = dynamic_pointer_cast<SparseMatrixTM<double>> (mat);
           if (!sm) throw Exception ("CuDSSCholesky.Update: expected a real sparse matrix");
           self.Update (*sm);
         }, py::arg("mat"))
    .def("Update", [] (CuDSSCholesky & self, uintptr_t rowptr, uintptr_t colind, uintptr_t values, uintptr_t ready)
         { self.Update (rowptr, colind, values, reinterpret_cast<cudaEvent_t>(ready)); },
         py::arg("rowptr"), py::arg("colind"), py::arg("values"), py::arg("ready_event") = 0)
    .def("Solve", [] (CuDSSCholesky & self, double s, const BaseVector & x, BaseVector & y, uintptr_t ready)
         { self.Solve (s, x, y, reinterpret_cast<cudaEvent_t>(ready)); },
         py::arg("s"), py::arg("x"), py::arg("y"), py::arg("ready_event") = 0)
    .def("Report", &CuDSSCholesky::Report)
    .def("_SetSkipQueueSync", &CuDSSCholesky::SetSkipQueueSync);
#endif



  // ExportDemo(m);
}

