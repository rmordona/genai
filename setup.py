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

        compiler = self.compiler.compiler_so[0]

        # Ensure ccache is used for caching compiled objects
        if "ccache" not in compiler:
            self.compiler.compiler_so.insert(0, "ccache")

        # Add required compiler flags
        for ext in self.extensions:
            ext.extra_compile_args.extend(["-std=gnu++17", "-O3"])
            ext.include_dirs.append("/usr/local/Cellar/pybind11/2.13.6_1/include")


        #self.extensions[0].extra_compile_args.append("-std=gnu++17")

        #self.extensions[0].include_dirs.append("/usr/local/Cellar/pybind11/2.13.6_1/include")



        super().build_extensions()

ext_modules = [
    Extension(
        "genai",
        sources=["src/scraper.cpp", "src/operators.cpp", "src/embeddings.cpp", "src/tokenmodel.cpp",  "src/convolution.cpp",
         	 "src/distributed.cpp", "src/transformer.cpp", "src/topology.cpp", "src/model.cpp", "src/recurrent.cpp", "src/genai.cpp"],
        include_dirs=['src', 
                    '/usr/local/Cellar/gcc/14.2.0_1/lib/gcc/current/gcc/x86_64-apple-darwin22/14/include',
                    '/usr/local/Cellar/eigen/3.4.0_1/include/eigen3',
                    '/usr/local/Cellar/open-mpi/5.0.6/include',
                    '/usr/local/Cellar/openblas/0.3.28/include',
                    '/usr/local/Cellar/pybind11/2.13.6_1/include',
                    '/usr/local/Cellar/fmt/11.1.3/include',
                    '/usr/local/Cellar/spdlog/1.15.0_1/include',
                    '/usr/local/Cellar/libxml2/2.13.5/include',
                    '/usr/local/opt/sqlite/include',
                    '/usr/local/Cellar/openssl@3/3.4.0/include',
                    '/usr/local/Cellar/utf8cpp/3.2.3/include',
                    '/usr/local/Cellar/zeromq/4.3.5_1/include',
                    '/usr/local/Cellar/libmemcached/1.0.18_2/include'],
        extra_compile_args = ['-DEIGEN_USE_BLAS', '-DFMT_HEADER_ONLY', '-DENABLE_DEBUG', 
				'-DENABLE_TRACE', '-DENABLE_WARNING', '-DENABLE_INFO', '-DENABLE_ERROR', '-DERROR_CRITICAL', '-g'],
        #extra_compile_args = ['-DEIGEN_USE_BLAS', '-DFMT_HEADER_ONLY'],
        extra_link_args = ['-fopenmp', '-Wall', '-fpermissive', '-fPIC', '-mavx', '-mfma' , '-DFMT_HEADER_ONLY', '-DENABLE_INFO', '-g'],
        #extra_link_args = ['-fopenmp', '-Wall', '-fpermissive', '-fPIC', '-mavx', '-mfma' , '-DFMT_HEADER_ONLY'],
        libraries=['cblas', 'mpi', 'zmq', 'memcached', 'ssl','crypto', 'sqlite3', 'fmt', 'xml2', 'curl' ],
        library_dirs=['/usr/local/Cellar/open-mpi/5.0.6/lib',
                    '/usr/local/Cellar/openblas/0.3.28/lib',
                     '/usr/local/Cellar/pybind11/2.13.6_1/lib',
                     '/usr/local/Cellar/fmt/11.1.3/lib',
                     '/usr/local/Cellar/spdlog/1.15.0_1/lib',
                     '/usr/local/Cellar/libxml2/2.13.5/lib',
                     '/usr/local/opt/sqlite/lib',
                     '/usr/local/Cellar/openssl@3/3.4.0/lib',
                     '/usr/local/Cellar/utf8cpp/3.2.3/lib',
                     '/usr/local/Cellar/zeromq/4.3.5_1/lib',
                     '/usr/local/Cellar/libmemcached/1.0.18_2/lib'],
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
