export CC=/usr/local/Cellar/gcc/14.2.0_1/bin/gcc-14
export CXX=/usr/local/Cellar/gcc/14.2.0_1/bin/g++-14
#export CC=/usr/local/Cellar/gcc/13.1.0/bin/gcc-13
#export CXX=/usr/local/Cellar/gcc/13.1.0/bin/g++-13
#export CC=/usr/local/Cellar/gcc/13.1.0/bin/x86_64-apple-darwin22-gcc-13
#export CXX=/usr/local/Cellar/gcc/13.1.0/bin/x86_64-apple-darwin22-g++-13
#rm -rf  /Users/raymondordona/Documents/genai/framework_cpp/build
#make clean
#cmake -DCMAKE_BUILD_TYPE=Release -Dpybind11_DIR=/usr/local/pkg/homebrew/lib/python3.7/site-packages/pybind11/share/cmake/pybind11
#make
#CXX=/usr/local/Cellar/gcc/13.1.0/bin/x86_64-apple-darwin22-g++-13 /usr/local/bin/python3 setup.py build
#CC=/usr/local/Cellar/gcc/13.1.0/bin/gcc-13 CXX=/usr/local/Cellar/gcc/13.1.0/bin/g++-13 /usr/local/bin/python3.11 setup.py build


python3.11 setup.py build_ext --inplace

CC=/usr/local/Cellar/gcc/14.2.0_1/bin/gcc-14 CXX=/usr/local/Cellar/gcc/14.2.0_1/bin/g++-14 /usr/local/bin/python3.11 setup.py build

# in conjunction with /usr/local/bin/python3.11

/usr/local/bin/pip3.11 uninstall genai
/usr/local/bin/pip3.11 install .
