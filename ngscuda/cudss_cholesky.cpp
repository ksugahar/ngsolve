// Sparse Cholesky of the free-dof block on the GPU through NVIDIA cuDSS; see cudss_cholesky.hpp.

#include <la.hpp>
#include <gpukernel.hpp>
#include <algorithm>
#include <climits>
#include <cmath>
#include <chrono>
#include <mutex>
#include <sstream>
#include "cudss_cholesky.hpp"
#include <cudss.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#include <climits>
#include <cstdlib>
#endif

namespace ngla
{
  using namespace ngs_gpu;

  // ---------------------------------------------------------------- loader

  namespace
  {
    struct CuDSSApi
    {
      decltype(&cudssCreate) Create = nullptr;
      decltype(&cudssDestroy) Destroy = nullptr;
      decltype(&cudssSetStream) SetStream = nullptr;
      decltype(&cudssConfigCreate) ConfigCreate = nullptr;
      decltype(&cudssConfigDestroy) ConfigDestroy = nullptr;
      decltype(&cudssDataCreate) DataCreate = nullptr;
      decltype(&cudssDataDestroy) DataDestroy = nullptr;
      decltype(&cudssDataGet) DataGet = nullptr;
      decltype(&cudssExecute) Execute = nullptr;
      decltype(&cudssMatrixCreateCsr) MatrixCreateCsr = nullptr;
      decltype(&cudssMatrixCreateDn) MatrixCreateDn = nullptr;
      decltype(&cudssMatrixDestroy) MatrixDestroy = nullptr;
      decltype(&cudssMatrixSetValues) MatrixSetValues = nullptr;
      decltype(&cudssGetProperty) GetProperty = nullptr;
      std::string path, report;
    };

    std::mutex api_mutex;
    CuDSSApi * api = nullptr;

#ifdef _WIN32
    std::string ModulePath (HMODULE h)
    {
      wchar_t buf[32768];
      DWORD n = GetModuleFileNameW (h, buf, 32768);
      if (n == 0 || n >= 32768) throw Exception ("cuDSS: GetModuleFileNameW failed");
      int len = WideCharToMultiByte (CP_UTF8, 0, buf, int(n), nullptr, 0, nullptr, nullptr);
      std::string s(len, '\0');
      WideCharToMultiByte (CP_UTF8, 0, buf, int(n), s.data(), len, nullptr, nullptr);
      return s;
    }
    std::wstring Wide (const std::string & s)
    {
      int len = MultiByteToWideChar (CP_UTF8, 0, s.data(), int(s.size()), nullptr, 0);
      std::wstring w(len, L'\0');
      MultiByteToWideChar (CP_UTF8, 0, s.data(), int(s.size()), w.data(), len);
      return w;
    }
    bool SameFile (const std::string & a, const std::string & b)
    {
      // compare the files themselves (case, 8.3 names, \\?\ prefixes)
      auto open = [] (const std::string & p) {
        return CreateFileW (Wide(p).c_str(), 0, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                            nullptr, OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS, nullptr); };
      HANDLE ha = open(a), hb = open(b);
      bool same = false;
      BY_HANDLE_FILE_INFORMATION ia, ib;
      if (ha != INVALID_HANDLE_VALUE && hb != INVALID_HANDLE_VALUE &&
          GetFileInformationByHandle (ha, &ia) && GetFileInformationByHandle (hb, &ib))
        same = ia.dwVolumeSerialNumber == ib.dwVolumeSerialNumber &&
          ia.nFileIndexHigh == ib.nFileIndexHigh && ia.nFileIndexLow == ib.nFileIndexLow;
      if (ha != INVALID_HANDLE_VALUE) CloseHandle (ha);
      if (hb != INVALID_HANDLE_VALUE) CloseHandle (hb);
      return same;
    }
#endif

    const CuDSSApi & Api ()
    {
      std::lock_guard<std::mutex> lock(api_mutex);
      if (!api)
        throw Exception ("cuDSS is not loaded: call LoadCuDSS with the absolute path of the pinned library first");
      return *api;
    }

    void Check (cudssStatus_t st, const char * what)
    {
      if (st != CUDSS_STATUS_SUCCESS)
        throw Exception (std::string("cuDSS: ") + what + " failed, status " + ToString(int(st)));
    }
    void CheckCuda (cudaError_t err, const char * what)
    {
      if (err != cudaSuccess)
        throw Exception (std::string("cuDSS route: ") + what + ": " + cudaGetErrorString(err));
    }
  }

  std::string LoadCuDSS (const std::string & path, int major, int minor, int patch)
  {
    std::lock_guard<std::mutex> lock(api_mutex);
    if (api)
      {
        if (api->path != path)
          throw Exception ("cuDSS is already loaded from " + api->path + ", not " + path);
        return api->report;
      }
    auto a = make_unique<CuDSSApi>();
    a->path = path;
    std::ostringstream rep;
#ifdef _WIN32
    if (HMODULE pre = GetModuleHandleW (L"cudss64_0.dll"))
      if (!SameFile (ModulePath(pre), path))
        throw Exception ("cuDSS: another cudss64_0.dll is already loaded: " + ModulePath(pre));
    HMODULE h = LoadLibraryExW (Wide(path).c_str(), nullptr,
                                LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    if (!h) throw Exception ("cuDSS: cannot load " + path + " (error " + ToString(int(GetLastError())) + ")");
    std::string loaded = ModulePath(h);
    if (!SameFile (loaded, path))
      throw Exception ("cuDSS: loaded " + loaded + " instead of " + path);
    rep << "{\"cudss\": \"" << loaded << "\"";
    if (HMODULE cb = GetModuleHandleW (L"cublas64_12.dll"))
      rep << ", \"cublas\": \"" << ModulePath(cb) << "\"";
    if (HMODULE cl = GetModuleHandleW (L"cublasLt64_12.dll"))
      rep << ", \"cublasLt\": \"" << ModulePath(cl) << "\"";
    auto sym = [h] (const char * name) {
      auto p = GetProcAddress (h, name);
      if (!p) throw Exception (std::string("cuDSS: missing entry point ") + name);
      return (void*)p; };
#else
    void * h = dlopen (path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!h) throw Exception ("cuDSS: cannot load " + path + ": " + dlerror());
    auto sym = [h] (const char * name) {
      void * p = dlsym (h, name);
      if (!p) throw Exception (std::string("cuDSS: missing entry point ") + name);
      return p; };
    Dl_info info;
    if (!dladdr (sym("cudssCreate"), &info) || !info.dli_fname)
      throw Exception ("cuDSS: cannot identify the loaded library");
    char r1[PATH_MAX], r2[PATH_MAX];
    if (!realpath (info.dli_fname, r1) || !realpath (path.c_str(), r2) || std::string(r1) != r2)
      throw Exception (std::string("cuDSS: loaded ") + info.dli_fname + " instead of " + path);
    rep << "{\"cudss\": \"" << r1 << "\"";
#endif
#define CUDSS_SYM(field, name) a->field = reinterpret_cast<decltype(a->field)>(sym(#name))
    CUDSS_SYM(Create, cudssCreate); CUDSS_SYM(Destroy, cudssDestroy);
    CUDSS_SYM(SetStream, cudssSetStream);
    CUDSS_SYM(ConfigCreate, cudssConfigCreate); CUDSS_SYM(ConfigDestroy, cudssConfigDestroy);
    CUDSS_SYM(DataCreate, cudssDataCreate); CUDSS_SYM(DataDestroy, cudssDataDestroy);
    CUDSS_SYM(DataGet, cudssDataGet); CUDSS_SYM(Execute, cudssExecute);
    CUDSS_SYM(MatrixCreateCsr, cudssMatrixCreateCsr); CUDSS_SYM(MatrixCreateDn, cudssMatrixCreateDn);
    CUDSS_SYM(MatrixDestroy, cudssMatrixDestroy); CUDSS_SYM(MatrixSetValues, cudssMatrixSetValues);
    CUDSS_SYM(GetProperty, cudssGetProperty);
#undef CUDSS_SYM
    int v[3];
    Check (a->GetProperty (MAJOR_VERSION, &v[0]), "GetProperty");
    Check (a->GetProperty (MINOR_VERSION, &v[1]), "GetProperty");
    Check (a->GetProperty (PATCH_LEVEL, &v[2]), "GetProperty");
    if (v[0] != major || v[1] != minor || v[2] != patch)
      throw Exception ("cuDSS: " + path + " is version " + ToString(v[0]) + "." + ToString(v[1]) + "." +
                       ToString(v[2]) + ", the pin is " + ToString(major) + "." + ToString(minor) + "." + ToString(patch));
    rep << ", \"version\": \"" << v[0] << "." << v[1] << "." << v[2] << "\"}";
    a->report = rep.str();
    api = a.release();     // the library stays loaded for the process
    return api->report;
  }

  // ---------------------------------------------------------------- kernels

  namespace
  {
    const char * kernel_source = R"(
      // dst[k] = src[map[k]]
      KERNEL(cd_gather, GLOBAL_IN(double,src), GLOBAL_IN(int,map), GLOBAL(double,dst), VALUE(int,n))
      { int k = GLOBAL_ID_X; if (k < n) dst[k] = src[map[k]]; }

      // flag 1: a mirrored pair differs (an absent mirror is -1 and counts as 0); flag 2: non-finite
      KERNEL(cd_check, GLOBAL_IN(double,val), GLOBAL_IN(int,entry), GLOBAL_IN(int,mirror),
                       GLOBAL_ATOMIC(int,flag), VALUE(int,n))
      {
        int k = GLOBAL_ID_X;
        if (k >= n) return;
        double a = val[entry[k]];
        if (!isfinite(a)) { atomicOr (&flag[0], 2); return; }
        int m = mirror[k];
        double b = (m >= 0) ? val[m] : 0.0;
        if (!(a == b)) atomicOr (&flag[0], 1);
      }

      // xf[i] = x[idx[i]]
      KERNEL(cd_gather_vec, GLOBAL_IN(double,x), GLOBAL_IN(int,idx), GLOBAL(double,xf), VALUE(int,n))
      { int i = GLOBAL_ID_X; if (i < n) xf[i] = x[idx[i]]; }

      // y[idx[i]] += s*xf[i]
      KERNEL(cd_scatter_add, GLOBAL_IN(double,xf), GLOBAL_IN(int,idx), GLOBAL(double,y),
                             VALUE(double,s), VALUE(int,n))
      { int i = GLOBAL_ID_X; if (i < n) y[idx[i]] += s*xf[i]; }

      // r = b, then r -= A x with A given by its lower triangle (symmetric)
      KERNEL(cd_copy, GLOBAL_IN(double,b), GLOBAL(double,r), VALUE(int,n))
      { int i = GLOBAL_ID_X; if (i < n) r[i] = b[i]; }
      KERNEL(cd_res_lower, GLOBAL_IN(int,rowptr), GLOBAL_IN(int,col), GLOBAL_IN(double,val),
                           GLOBAL_IN(double,x), GLOBAL_ATOMIC(double,r), VALUE(int,n))
      {
        int row = GLOBAL_ID_X;
        if (row >= n) return;
        double xr = x[row], sum = 0;
        for (int j = rowptr[row]; j < rowptr[row+1]; j++)
          {
            int c = col[j];
            sum += val[j]*x[c];
            if (c != row) ATOMIC_ADD (&r[c], -val[j]*xr);
          }
        ATOMIC_ADD (&r[row], -sum);
      }

      // out[0] += sum v[i]^2 (a non-finite entry makes the sum non-finite)
      KERNEL(cd_sumsq, GLOBAL_IN(double,v), GLOBAL_ATOMIC(double,out), VALUE(int,n))
      {
        int i = GLOBAL_ID_X;
        double s = (i < n) ? v[i]*v[i] : 0.0;
        for (int off = 16; off > 0; off /= 2) s += __shfl_down_sync (0xffffffff, s, off);
        if ((threadIdx.x & 31) == 0 && s != 0.0) ATOMIC_ADD (&out[0], s);
      }
    )";

    struct Kernels
    {
      shared_ptr<Device> device;
      shared_ptr<Queue> queue;
      shared_ptr<Library> lib;
      shared_ptr<Kernel> gather, check, gather_vec, scatter_add, copy, res_lower, sumsq;
      Kernels (shared_ptr<Device> dev) : device(dev)
      {
        if (!device->HasFloat64()) throw Exception ("CuDSSCholesky: the device has no fp64");
        lib = device->CompileSource (string(code_gpukernel) + kernel_source);
        gather = lib->GetKernel ("cd_gather"); check = lib->GetKernel ("cd_check");
        gather_vec = lib->GetKernel ("cd_gather_vec"); scatter_add = lib->GetKernel ("cd_scatter_add");
        copy = lib->GetKernel ("cd_copy"); res_lower = lib->GetKernel ("cd_res_lower");
        sumsq = lib->GetKernel ("cd_sumsq");
        queue = device->DefaultQueue();
      }
      void Run (Kernel & k, size_t n, std::vector<KernelArg> args) const
      {
        if (n == 0) return;
        queue->Launch (k, Dim3((n+255)/256), Dim3(256), std::move(args));
      }
    };

    double Seconds (std::chrono::steady_clock::time_point t0)
    { return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(); }
  }

  // ---------------------------------------------------------------- object

  struct CuDSSCholesky::Impl
  {
    size_t n = 0;                // full size
    int nf = 0, nnzf = 0;        // free block, its lower triangle
    CuDSSStorage storage;
    double residual_limit;
    bool skip_queue_sync = false;
    bool factor_valid = false;
    std::mutex mtx;
    unique_ptr<Kernels> k;

    // host symbolic data of the analysed pattern (for Update comparison)
    Array<int> rowptr_in, col_in;           // input CSR pattern (full size)
    // device data
    TypedBuffer<int> d_free, d_map, d_entry, d_mirror, d_rowptr, d_col, d_flag;
    TypedBuffer<double> d_vals_in, d_vals, d_b, d_x, d_r, d_norm;
    uintptr_t ext_values = 0;               // caller-owned values (constructor B)
    size_t ncheck = 0;

    cudaStream_t stream = nullptr;
    cudssHandle_t handle = nullptr;
    cudssConfig_t config = nullptr;
    cudssData_t data = nullptr;
    cudssMatrix_t A = nullptr, B = nullptr, X = nullptr;

    // ledger
    double t_setup = 0, t_analysis = 0, t_factor = 0, t_solve = 0, last_residual = 0;
    size_t up_bytes = 0, down_bytes = 0, nfactor = 0, nsolve = 0;

    ~Impl ()
    {
      if (api)
        {
          if (A) api->MatrixDestroy (A);
          if (B) api->MatrixDestroy (B);
          if (X) api->MatrixDestroy (X);
          if (data) api->DataDestroy (handle, data);
          if (config) api->ConfigDestroy (config);
          if (handle) api->Destroy (handle);
        }
      if (stream) cudaStreamDestroy (stream);
    }

    void QueueSync () { if (!skip_queue_sync) k->queue->Finish(); }

    // symbolic part on the host from the input pattern: free index map, the
    // lower triangle of the free block with a map into the input values, and
    // the mirror table for the Full symmetry check
    void Symbolic (FlatArray<int> rowptr, FlatArray<int> col, const BitArray & free)
    {
      if (free.Size() != n) throw Exception ("CuDSSCholesky: freedofs has size " + ToString(free.Size()) + ", the matrix " + ToString(n));
      Array<int> compact(n); compact = -1;
      Array<int> freeidx;
      for (size_t i = 0; i < n; i++)
        if (free.Test(i)) { compact[i] = int(freeidx.Size()); freeidx.Append(int(i)); }
      nf = int(freeidx.Size());

      auto find = [&] (int r, int c) -> int {        // entry (r,c) of the input, -1 if absent
        auto b = col.begin() + rowptr[r], e = col.begin() + rowptr[r+1];
        auto it = std::lower_bound (b, e, c);
        return (it != e && *it == c) ? int(it - col.begin()) : -1; };

      // validate the pattern rows: sorted, unique, in range, stored triangle
      for (size_t r = 0; r < n; r++)
        {
          if (rowptr[r+1] < rowptr[r]) throw Exception ("CuDSSCholesky: row offsets decrease at row " + ToString(r));
          for (int j = rowptr[r]; j < rowptr[r+1]; j++)
            {
              int c = col[j];
              if (c < 0 || size_t(c) >= n) throw Exception ("CuDSSCholesky: column index out of range in row " + ToString(r));
              if (j > rowptr[r] && c <= col[j-1]) throw Exception ("CuDSSCholesky: column indices not sorted/unique in row " + ToString(r));
              if (storage == CuDSSStorage::Lower && size_t(c) > r) throw Exception ("CuDSSCholesky: storage Lower has an entry above the diagonal in row " + ToString(r));
              if (storage == CuDSSStorage::Upper && size_t(c) < r) throw Exception ("CuDSSCholesky: storage Upper has an entry below the diagonal in row " + ToString(r));
            }
        }

      // lower triangle of A_FF in compact numbering, by rows; map[k] = input entry
      Array<int> rp(nf+1), cl, map, entry, mirror;
      rp[0] = 0;
      if (storage == CuDSSStorage::Upper)
        {
          // transpose the upper triangle: count per compact column first
          Array<int> cnt(nf); cnt = 0;
          for (size_t r = 0; r < n; r++)
            if (compact[r] >= 0)
              for (int j = rowptr[r]; j < rowptr[r+1]; j++)
                if (compact[col[j]] >= 0) cnt[compact[col[j]]]++;
          for (int i = 0; i < nf; i++) rp[i+1] = rp[i] + cnt[i];
          cl.SetSize(rp[nf]); map.SetSize(rp[nf]);
          Array<int> pos(nf); for (int i = 0; i < nf; i++) pos[i] = rp[i];
          for (size_t r = 0; r < n; r++)                  // rows ascending => columns sorted
            if (compact[r] >= 0)
              for (int j = rowptr[r]; j < rowptr[r+1]; j++)
                if (int cc = compact[col[j]]; cc >= 0)
                  { cl[pos[cc]] = compact[r]; map[pos[cc]++] = j; }
        }
      else
        for (int i = 0; i < nf; i++)
          {
            int r = freeidx[i];
            for (int j = rowptr[r]; j < rowptr[r+1]; j++)
              {
                int cc = compact[col[j]];
                if (cc < 0) continue;
                if (cc <= i) { cl.Append(cc); map.Append(j); }
                else          // Full: an upper entry is checked against its mirror
                  { entry.Append(j); mirror.Append(find(col[j], r)); }
              }
            rp[i+1] = int(cl.Size());
          }
      if (storage == CuDSSStorage::Full)     // lower entries whose mirror is absent must be 0
        for (int i = 0; i < nf; i++)
          for (int jj = rp[i]; jj < rp[i+1]; jj++)
            if (cl[jj] != i)
              {
                int m = find (freeidx[cl[jj]], freeidx[i]);
                if (m < 0) { entry.Append(map[jj]); mirror.Append(-1); }
              }
      // every free row needs its diagonal
      for (int i = 0; i < nf; i++)
        if (rp[i+1] == rp[i] || cl[rp[i+1]-1] != i)
          throw Exception ("CuDSSCholesky: no stored diagonal in free row " + ToString(freeidx[i]));
      nnzf = rp[nf];

      auto & dev = *k->device;
      auto up = [&] (TypedBuffer<int> & b, FlatArray<int> a) {
        b = dev.NewBuffer<int> (std::max<size_t>(a.Size(),1));
        if (a.Size()) b.H2D (a.Data(), a.Size());
        up_bytes += a.Size()*sizeof(int); };
      up (d_free, freeidx); up (d_rowptr, rp); up (d_col, cl); up (d_map, map);
      up (d_entry, entry); up (d_mirror, mirror);
      ncheck = entry.Size();
      d_flag = dev.NewBuffer<int>(1);
      d_vals = dev.NewBuffer<double> (std::max(nnzf,1));
      d_b = dev.NewBuffer<double> (std::max(nf,1)); d_x = dev.NewBuffer<double> (std::max(nf,1));
      d_r = dev.NewBuffer<double> (std::max(nf,1)); d_norm = dev.NewBuffer<double> (2);
    }

    // cuDSS objects on the free block; analysis once per pattern
    void Setup ()
    {
      auto & a = Api();
      CheckCuda (cudaStreamCreateWithFlags (&stream, cudaStreamNonBlocking), "cudaStreamCreate");
      Check (a.Create (&handle), "Create");
      Check (a.SetStream (handle, stream), "SetStream");
      Check (a.ConfigCreate (&config), "ConfigCreate");
      Check (a.DataCreate (handle, &data), "DataCreate");
      Check (a.MatrixCreateCsr (&A, nf, nf, nnzf, (void*)d_rowptr.Raw()->DevicePtr(), nullptr,
                                (void*)d_col.Raw()->DevicePtr(), (void*)d_vals.Raw()->DevicePtr(),
                                CUDA_R_32I, CUDA_R_64F, CUDSS_MTYPE_SPD, CUDSS_MVIEW_LOWER,
                                CUDSS_BASE_ZERO), "MatrixCreateCsr");
      Check (a.MatrixCreateDn (&B, nf, 1, nf, (void*)d_b.Raw()->DevicePtr(), CUDA_R_64F,
                               CUDSS_LAYOUT_COL_MAJOR), "MatrixCreateDn");
      Check (a.MatrixCreateDn (&X, nf, 1, nf, (void*)d_x.Raw()->DevicePtr(), CUDA_R_64F,
                               CUDSS_LAYOUT_COL_MAJOR), "MatrixCreateDn");
      // the scratch belongs to this object's device context
      cudaPointerAttributes attr;
      CheckCuda (cudaPointerGetAttributes (&attr, (void*)d_vals.Raw()->DevicePtr()), "cudaPointerGetAttributes");
      int cur; CheckCuda (cudaGetDevice (&cur), "cudaGetDevice");
      if (attr.type != cudaMemoryTypeDevice || attr.device != cur)
        throw Exception ("CuDSSCholesky: the ngscuda buffers are not device memory of the current CUDA device");
    }

    // values of the input (d_vals_in, full input numbering) -> checks -> d_vals -> factorization
    void Factor (bool first)
    {
      factor_valid = false;
      auto & a = Api();
      int flag = 0;
      d_flag.H2D (&flag, 1);
      k->Run (*k->check, ncheck, { d_vals_in, d_entry, d_mirror, d_flag, int(ncheck) });
      k->Run (*k->gather, nnzf, { d_vals_in, d_map, d_vals, nnzf });
      double vs2 = 0;                       // every admitted value finite
      d_norm.H2D (&vs2, 1);
      k->Run (*k->sumsq, nnzf, { d_vals, d_norm, nnzf });
      k->queue->Finish();
      d_flag.D2H (&flag, 1);
      d_norm.D2H (&vs2, 1);
      if ((flag & 2) || !std::isfinite(vs2)) throw Exception ("CuDSSCholesky: non-finite matrix entry in the free block");
      if (flag & 1) throw Exception ("CuDSSCholesky: storage Full is not exactly symmetric on the free block");
      if (nf == 0) { factor_valid = true; return; }
      auto t0 = std::chrono::steady_clock::now();
      if (first)
        {
          Check (a.Execute (handle, CUDSS_PHASE_ANALYSIS, config, data, A, X, B), "analysis");
          CheckCuda (cudaStreamSynchronize (stream), "analysis");
          t_analysis = Seconds(t0); t0 = std::chrono::steady_clock::now();
        }
      Check (a.Execute (handle, first ? CUDSS_PHASE_FACTORIZATION : CUDSS_PHASE_REFACTORIZATION,
                        config, data, A, X, B), "factorization");
      CheckCuda (cudaStreamSynchronize (stream), "factorization");
      int info = 0; size_t written = 0;
      Check (a.DataGet (handle, data, CUDSS_DATA_INFO, &info, sizeof(info), &written), "DataGet(INFO)");
      CheckCuda (cudaStreamSynchronize (stream), "DataGet(INFO)");
      if (info != 0)
        throw Exception ("CuDSSCholesky: factorization reports info = " + ToString(info) +
                         " (matrix not positive definite or singular)");
      t_factor = Seconds(t0);
      nfactor++;
      factor_valid = true;
    }

    void LoadHostValues (const SparseMatrixTM<double> & mat)
    {
      auto fv = mat.AsVector().FV<double>();
      if (!d_vals_in || d_vals_in.Size() < fv.Size())
        d_vals_in = k->device->NewBuffer<double> (std::max<size_t>(fv.Size(),1));
      if (fv.Size()) d_vals_in.H2D (fv.Data(), fv.Size());
      up_bytes += fv.Size()*sizeof(double);
    }

    void LoadDeviceValues (uintptr_t values, size_t nnz)
    {
      if (!d_vals_in || d_vals_in.Size() < nnz)
        d_vals_in = k->device->NewBuffer<double> (std::max<size_t>(nnz,1));
      if (nnz)
        {
          CheckCuda (cudaMemcpyAsync ((void*)d_vals_in.Raw()->DevicePtr(), (const void*)values,
                                      nnz*sizeof(double), cudaMemcpyDeviceToDevice, stream ? stream : nullptr),
                     "device values copy");
          CheckCuda (stream ? cudaStreamSynchronize (stream) : cudaDeviceSynchronize(), "device values copy");
        }
    }

    static Array<int> HostPattern (FlatArray<size_t> firsti, size_t n)
    {
      if (firsti[n] > size_t(INT_MAX)) throw Exception ("CuDSSCholesky: nnz exceeds int32");
      Array<int> rp(n+1);
      for (size_t i = 0; i <= n; i++) rp[i] = int(firsti[i]);
      return rp;
    }

    void DevicePattern (size_t nnz, uintptr_t rowptr, uintptr_t colind, Array<int> & rp, Array<int> & cl)
    {
      rp.SetSize (n+1); cl.SetSize (nnz);
      CheckCuda (cudaMemcpy (rp.Data(), (const void*)rowptr, (n+1)*sizeof(int), cudaMemcpyDeviceToHost), "pattern read");
      if (nnz) CheckCuda (cudaMemcpy (cl.Data(), (const void*)colind, nnz*sizeof(int), cudaMemcpyDeviceToHost), "pattern read");
      down_bytes += (n+1+nnz)*sizeof(int);
      if (rp[0] != 0 || size_t(rp[n]) != nnz) throw Exception ("CuDSSCholesky: row offsets do not span nnz");
    }

    static void WaitReady (cudaEvent_t ready)
    { if (ready) CheckCuda (cudaEventSynchronize (ready), "ready event"); }
  };

  static void CheckSizes (size_t n, size_t nnz)
  {
    if (n > size_t(INT_MAX) || nnz > size_t(INT_MAX))
      throw Exception ("CuDSSCholesky: n = " + ToString(n) + ", nnz = " + ToString(nnz) + " exceed int32");
  }

  CuDSSCholesky :: CuDSSCholesky (const SparseMatrixTM<double> & mat, shared_ptr<BitArray> freedofs,
                                  CuDSSStorage storage, double residual_limit)
    : impl (make_unique<Impl>())
  {
    auto t0 = std::chrono::steady_clock::now();
    auto & I = *impl;
    if (mat.Height() != mat.Width()) throw Exception ("CuDSSCholesky: matrix is not square");
    if (!freedofs) throw Exception ("CuDSSCholesky: freedofs required");
    Api();                                    // loaded before anything is allocated
    I.n = mat.Height(); I.storage = storage; I.residual_limit = residual_limit;
    CheckSizes (I.n, mat.NZE());
    I.k = make_unique<Kernels> (GetGpuDevice());
    I.rowptr_in = Impl::HostPattern (mat.GetFirstArray(), I.n);
    I.col_in = Array<int> (mat.GetColIndices());
    I.Symbolic (I.rowptr_in, I.col_in, *freedofs);
    I.Setup ();
    I.LoadHostValues (mat);
    I.t_setup = Seconds(t0);
    I.Factor (true);
  }

  CuDSSCholesky :: CuDSSCholesky (size_t n, size_t nnz, uintptr_t dev_rowptr, uintptr_t dev_colind,
                                  uintptr_t dev_values, CuDSSStorage storage, shared_ptr<BitArray> freedofs,
                                  cudaEvent_t ready, double residual_limit)
    : impl (make_unique<Impl>())
  {
    auto t0 = std::chrono::steady_clock::now();
    auto & I = *impl;
    if (!freedofs) throw Exception ("CuDSSCholesky: freedofs required");
    if (!dev_rowptr || (nnz && (!dev_colind || !dev_values))) throw Exception ("CuDSSCholesky: null device pointer");
    Api();
    CheckSizes (n, nnz);
    I.n = n; I.storage = storage; I.residual_limit = residual_limit;
    I.k = make_unique<Kernels> (GetGpuDevice());
    Impl::WaitReady (ready);
    I.QueueSync ();
    I.DevicePattern (nnz, dev_rowptr, dev_colind, I.rowptr_in, I.col_in);
    I.Symbolic (I.rowptr_in, I.col_in, *freedofs);
    I.Setup ();
    I.LoadDeviceValues (dev_values, nnz);
    I.t_setup = Seconds(t0);
    I.Factor (true);
  }

  CuDSSCholesky :: ~CuDSSCholesky () = default;

  void CuDSSCholesky :: Update (const SparseMatrixTM<double> & mat)
  {
    auto & I = *impl;
    std::lock_guard<std::mutex> lock(I.mtx);
    I.factor_valid = false;
    if (size_t(mat.Height()) != I.n || mat.NZE() != size_t(I.col_in.Size()))
      throw Exception ("CuDSSCholesky::Update: the pattern changed (size or nnz)");
    auto rp = Impl::HostPattern (mat.GetFirstArray(), I.n);
    auto cl = mat.GetColIndices();
    for (size_t i = 0; i <= I.n; i++)
      if (rp[i] != I.rowptr_in[i]) throw Exception ("CuDSSCholesky::Update: the pattern changed at row " + ToString(i));
    for (size_t j = 0; j < cl.Size(); j++)
      if (cl[j] != I.col_in[j]) throw Exception ("CuDSSCholesky::Update: the pattern changed at entry " + ToString(j));
    I.LoadHostValues (mat);
    I.Factor (false);
  }

  void CuDSSCholesky :: Update (uintptr_t dev_rowptr, uintptr_t dev_colind, uintptr_t dev_values, cudaEvent_t ready)
  {
    auto & I = *impl;
    std::lock_guard<std::mutex> lock(I.mtx);
    I.factor_valid = false;
    Impl::WaitReady (ready);
    I.QueueSync ();
    Array<int> rp, cl;
    I.DevicePattern (I.col_in.Size(), dev_rowptr, dev_colind, rp, cl);
    for (size_t i = 0; i <= I.n; i++)
      if (rp[i] != I.rowptr_in[i]) throw Exception ("CuDSSCholesky::Update: the pattern changed at row " + ToString(i));
    for (size_t j = 0; j < cl.Size(); j++)
      if (cl[j] != I.col_in[j]) throw Exception ("CuDSSCholesky::Update: the pattern changed at entry " + ToString(j));
    I.LoadDeviceValues (dev_values, I.col_in.Size());
    I.Factor (false);
  }

  void CuDSSCholesky :: Solve (double s, const BaseVector & x, BaseVector & y, cudaEvent_t ready) const
  {
    auto & I = *impl;
    std::lock_guard<std::mutex> lock(I.mtx);
    if (!I.factor_valid) throw Exception ("CuDSSCholesky: no valid factorization (a previous factorization or update failed)");
    auto dx = dynamic_cast<const DeviceVector<double>*> (&x);
    auto dy = dynamic_cast<const DeviceVector<double>*> (&y);
    if (!dx || !dy) throw Exception ("CuDSSCholesky: x and y must be DeviceVector<double> (no host vectors)");
    if (dx->Size() != I.n || dy->Size() != I.n) throw Exception ("CuDSSCholesky: vector size mismatch");
    if (!dx->IsDevUptodate() || !dy->IsDevUptodate())
      throw Exception ("CuDSSCholesky: x or y is not current on the device (no implicit upload)");
    if (&x == &y || (dx->DevBufferRO() == dy->DevBufferRO() && dx->DevOffset() == dy->DevOffset()))
      throw Exception ("CuDSSCholesky: x and y must not alias");
    Impl::WaitReady (ready);
    I.QueueSync ();
    if (I.nf == 0) return;
    auto t0 = std::chrono::steady_clock::now();

    // b_F = x_F; zero right-hand side => zero solution, y unchanged
    double nb2 = 0;
    I.k->Run (*I.k->gather_vec, I.nf, { dx->DevArgRO(), I.d_free, I.d_b, I.nf });
    I.d_norm.H2D (&nb2, 1);
    I.k->Run (*I.k->sumsq, I.nf, { I.d_b, I.d_norm, I.nf });
    I.k->queue->Finish();
    I.d_norm.D2H (&nb2, 1);
    if (!std::isfinite(nb2)) throw Exception ("CuDSSCholesky: non-finite right-hand side");
    if (nb2 == 0) { I.nsolve++; return; }

    auto & a = Api();
    Check (a.Execute (I.handle, CUDSS_PHASE_SOLVE, I.config, I.data, I.A, I.X, I.B), "solve");
    CheckCuda (cudaStreamSynchronize (I.stream), "solve");

    // true residual of the admitted free block, before y is touched
    double nr2 = 0;
    I.d_norm.H2D (&nr2, 1);
    I.k->Run (*I.k->copy, I.nf, { I.d_b, I.d_r, I.nf });
    I.k->Run (*I.k->res_lower, I.nf, { I.d_rowptr, I.d_col, I.d_vals, I.d_x, I.d_r, I.nf });
    I.k->Run (*I.k->sumsq, I.nf, { I.d_r, I.d_norm, I.nf });
    I.k->queue->Finish();
    I.d_norm.D2H (&nr2, 1);
    double rel = std::sqrt (nr2 / nb2);
    I.last_residual = rel;
    if (!std::isfinite(rel)) throw Exception ("CuDSSCholesky: non-finite solution");
    if (rel > I.residual_limit)
      throw Exception ("CuDSSCholesky: true residual " + ToString(rel) + " above the limit " + ToString(I.residual_limit));

    I.k->Run (*I.k->scatter_add, I.nf, { I.d_x, I.d_free, dy->DevArgRW(), s, I.nf });
    I.k->queue->Finish();
    I.t_solve = Seconds(t0);
    I.nsolve++;
  }

  int CuDSSCholesky :: VHeight() const { return int(impl->n); }
  int CuDSSCholesky :: VWidth() const { return int(impl->n); }
  VecFormat CuDSSCholesky :: RowFormat () const { return DeviceVectorFormat<double> (impl->n, MemType::Device); }
  VecFormat CuDSSCholesky :: ColFormat () const { return DeviceVectorFormat<double> (impl->n, MemType::Device); }
  void CuDSSCholesky :: SetSkipQueueSync (bool skip) { impl->skip_queue_sync = skip; }

  std::string CuDSSCholesky :: Report () const
  {
    auto & I = *impl;
    std::ostringstream o;
    o << "{\"n\": " << I.n << ", \"nfree\": " << I.nf << ", \"nnz_lower_free\": " << I.nnzf
      << ", \"storage\": " << int(I.storage)
      << ", \"t_setup\": " << I.t_setup << ", \"t_analysis\": " << I.t_analysis
      << ", \"t_factor\": " << I.t_factor << ", \"t_solve\": " << I.t_solve
      << ", \"factorizations\": " << I.nfactor << ", \"solves\": " << I.nsolve
      << ", \"last_rel_residual\": " << I.last_residual
      << ", \"host_to_device_bytes\": " << I.up_bytes << ", \"device_to_host_bytes\": " << I.down_bytes
      << ", \"library\": " << (api ? api->report : std::string("null")) << "}";
    return o.str();
  }
}
