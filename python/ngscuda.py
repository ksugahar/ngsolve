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
