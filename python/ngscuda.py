import ctypes
import os
import sys
import importlib.resources as r

from ngsolve import config as _config

if not _config.USE_CUDA:
    raise ImportError("ngscuda was imported, but CUDA support is not enabled in this build of NGSolve.")

# loaded libraries and DLL-directory registrations stay alive for the whole process
_cuda_handles = []

def preload_cuda_libs():
    if sys.platform.startswith("win"):
        # nvrtc loads its builtins with a plain LoadLibrary by base name; a module
        # already loaded under that name is used, so preload it by absolute path
        libs = [
            ("nvidia.cuda_runtime", "bin/cudart64_12.dll"),
            ("nvidia.cuda_nvrtc", "bin/nvrtc-builtins64_128.dll"),
            ("nvidia.cuda_nvrtc", "bin/nvrtc64_120_0.dll"),
        ]
    else:
        libs = [
            ("nvidia.cuda_runtime", "lib/libcudart.so.12"),
            ("nvidia.nvjitlink", "lib/libnvJitLink.so.12"),
            # _ngscuda and libngscudalib link libnvrtc.so.12; nvrtc loads its builtins by name
            ("nvidia.cuda_nvrtc", "lib/libnvrtc-builtins.so.12.8"),
            ("nvidia.cuda_nvrtc", "lib/libnvrtc.so.12"),
        ]

    for pkg, rel in libs:
        path = str(r.files(pkg).joinpath(rel))
        if not os.path.isfile(path):
            raise ImportError(f"ngscuda: {path} from the pinned {pkg} package is missing")
        if sys.platform.startswith("win"):
            _cuda_handles.append(os.add_dll_directory(os.path.dirname(path)))
        _cuda_handles.append(ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL))


if _config.is_python_package:
    preload_cuda_libs()

from ngsolve._ngscuda import *


# cuDSS (optional, USE_CUDSS builds): loaded on request from the pinned pip packages
# by absolute path, with its cuBLAS first; ngscuda itself does not depend on it
_CUDSS_VERSION = (0, 8, 0)

def LoadCuDSS():
    """Load the pinned cuDSS library (and the cuBLAS it needs) by absolute path.
    Returns a json report of the loaded files and version; raises if anything
    is missing, another copy is loaded, or the version differs from the pin."""
    import ngsolve._ngscuda as _m
    if not hasattr(_m, "_LoadCuDSS"):
        raise RuntimeError("this ngsolve build has no cuDSS support (USE_CUDSS=OFF)")
    if sys.platform.startswith("win"):
        deps = [("nvidia.cublas", "bin/cublasLt64_12.dll"), ("nvidia.cublas", "bin/cublas64_12.dll")]
        lib = ("nvidia.cu12", "bin/cudss64_0.dll")
    else:
        deps = [("nvidia.cublas", "lib/libcublasLt.so.12"), ("nvidia.cublas", "lib/libcublas.so.12")]
        lib = ("nvidia.cu12", "lib/libcudss.so.0")
    paths = []
    for pkg, rel in deps + [lib]:
        try:
            path = str(r.files(pkg).joinpath(rel))
        except ModuleNotFoundError:
            raise RuntimeError(f"cuDSS: the pinned package providing {pkg} is not installed")
        if not os.path.isfile(path):
            raise RuntimeError(f"cuDSS: {path} from the pinned {pkg} package is missing")
        paths.append(path)
    for path in paths[:-1]:
        if sys.platform.startswith("win"):
            _cuda_handles.append(os.add_dll_directory(os.path.dirname(path)))
        _cuda_handles.append(ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL))
    return _m._LoadCuDSS(paths[-1], *_CUDSS_VERSION)
