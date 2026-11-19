"""Plain `import ngsolve, netgen` must map no CUDA library; CUDA loads only via ngsolve.ngscuda.

Writes the record to the file argument and exits non-zero if any CUDA library is mapped.
"""
import json
import re
import sys

import netgen  # noqa: F401
import ngsolve
import ngsolve.solvers  # noqa: F401

pattern = re.compile(r"lib(cuda|cudart|nvrtc|nvJitLink|cudss|cublas|cublasLt|cusparse)[^/]*\.so")
mapped = sorted({line.split()[-1] for line in open("/proc/self/maps")
                 if len(line.split()) >= 6 and pattern.search(line.split()[-1])})
record = {"ngsolve": ngsolve.__file__, "cuda_libraries_mapped": mapped, "passed": not mapped}
json.dump(record, open(sys.argv[1], "w"), indent=2)
if mapped:
    sys.exit(f"plain import mapped CUDA libraries: {mapped}")
