import numpy
from Cython.Build import build_ext
from setuptools import Extension, setup

extensions = [
    Extension(
        "pdist_aggregation",
        sources=["_argkmin_fast.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=[
            "-fopenmp",
            "-O3",
            "-ftree-vectorize",
        ],
        extra_link_args=["-fopenmp"],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
    )
]

setup(
    name="pdist_aggregation",
    cmdclass={"build_ext": build_ext},
    version="0.1",
    ext_modules=extensions,
    install_requires=[
        # When using make, those should be installed in the env directly.
        # Especially cython 3 (in pre-release) needs to be installed.
        "cython",
        "numpy>=1.20",
        "setuptools>=18.0",
        "scikit-learn>=0.24.2",
    ],
    python_requires=">=3.6",
)
