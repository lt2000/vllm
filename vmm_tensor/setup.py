from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name="vmm_tensor",
    version="0.1.0",
    packages=["vmm_tensor"],
    ext_modules=[
        CUDAExtension(
            name="vmm_tensor._C",
            sources=["vmm_tensor_refactor.cpp"],
            libraries=["cuda"],
            library_dirs=["/usr/local/cuda/lib64/stubs"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
