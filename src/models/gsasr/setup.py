from setuptools import setup
from torch.utils import cpp_extension
import os

# Resolve paths relative to this file, not the caller's cwd, so
# `pip install -e .` / `uv pip install -e .` works from anywhere.
this_dir = os.path.dirname(os.path.abspath(__file__))
folder = os.path.join(this_dir, "gs_cuda_dmax")

setup(
    name="gscuda",
    ext_modules=[
        cpp_extension.CUDAExtension(
            name="gscuda",
            sources=[
                os.path.join(folder, "gs.cu"),
                os.path.join(folder, "gswrapper.cpp"),
            ],
            include_dirs=[folder],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math"],
            },
        )
    ],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
)