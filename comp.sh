export CC=/usr/local/pkg/homebrew/Cellar/gcc/13.1.0/bin/g++-13
export CXX=/usr/local/pkg/homebrew/Cellar/gcc/13.1.0/bin/g++-13
#rm -rf  /Users/raymondordona/Documents/genai/framework_cpp/build
#make clean
#cmake -DCMAKE_BUILD_TYPE=Release -Dpybind11_DIR=/usr/local/pkg/homebrew/lib/python3.7/site-packages/pybind11/share/cmake/pybind11
#make
CXX=/usr/local/pkg/homebrew/Cellar/gcc/13.1.0/bin/g++-13 python3 setup.py build

pip3 uninstall genai
pip3 install .
