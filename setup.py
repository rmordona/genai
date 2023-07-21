import os
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
import sys
import setuptools
#import subprocess

# Run make to build the C++ project
def build_cpp_project():
    subprocess.check_call(['make'])

class BuildExt(build_ext):
    def build_extensions(self):
        self.extensions[0].extra_compile_args.append("-std=c++11")
        self.extensions[0].include_dirs.append("/usr/local/pkg/homebrew/lib/python3.7/site-packages/pybind11/include")
        super().build_extensions()

ext_modules = [
    Extension(
        "genai",
        sources=["src/embeddings.cpp", "src/tokenmodel.cpp", "src/distributed.cpp","src/operators.cpp", "src/transformer.cpp", "src/topology.cpp", "src/model.cpp", "src/genai.cpp"],
        include_dirs=['src', '/usr/local/pkg/homebrew/lib/python3.7/site-packages/pybind11/include',
                      '/usr/local/pkg/homebrew/Cellar/open-mpi/4.1.5/include',
                      '/usr/local/pkg/homebrew/Cellar/openblas/0.3.10/include',
                      '/usr/local/pkg/homebrew/Cellar/gcc/13.1.0/lib/gcc/current/gcc/x86_64-apple-darwin18/13/include',
                      '/usr/local/include/eigen3',
                      '/usr/local/pkg/homebrew/Cellar/libmemcached/1.0.18_2/include',
                      '/usr/local/pkg/homebrew/Cellar/openssl/1.0.2s/include',
                      '/usr/local/pkg/homebrew/opt/sqlite/include',
                      '/usr/local/pkg/homebrew/Cellar/utf8cpp/3.2.3/include',
                      '/usr/local/pkg/homebrew/Cellar/zeromq/4.3.4/include'],
        extra_compile_args = ['-fopenmp', '-Wall', '-fpermissive', '-fPIC', '-shared', '-mavx', '-mfma', '-DEIGEN_USE_BLAS'],
        extra_link_args=['-L/usr/local/pkg/homebrew/Cellar/open-mpi/4.1.5/lib', '-L/usr/local/pkg/homebrew/Cellar/zeromq/4.3.4/lib', '-fopenmp'],
        libraries=['cblas', 'mpi', 'zmq', 'memcached', 'ssl','crypto', 'sqlite3'],
        library_dirs=['/usr/local/pkg/homebrew/Cellar/open-mpi/4.1.5/lib','/usr/local/pkg/homebrew/Cellar/zeromq/4.3.4/lib',
                      '/usr/local/pkg/homebrew/opt/openssl/lib/',
                      '/usr/local/pkg/homebrew/opt/sqlite/lib',
                      '/usr/local/pkg/homebrew/Cellar/libmemcached/1.0.18_2/lib'],
        language="c++",
        py_limited_api=True
    ),
]

# Call the build_cpp_project function before building the extension
# build_cpp_project()

# Setup configuration
setuptools.setup(
    name="genai",
    version="0.1.0",
    author="Raymond Ordona",
    author_email="rmordona@gmail.com",
    description="A Python interface for your C++ module",
    long_description="Some long description",
    long_description_content_type="text/markdown",
    ext_modules=ext_modules,
    packages=setuptools.find_packages(),
    cmdclass={"build_ext": BuildExt},
#    install_requires=[
#           'open-mpi>=4.1.5',
#           'openblas>=0.3.10',
#           'zeromq>=4.3.4',
#           'python>=3.7',
#           'memcached>=1.0.18_2'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ],
)
