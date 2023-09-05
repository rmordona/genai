#include <iostream>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <string>
#include <cmath>
#include <type_traits>
#include <cblas.h>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>
#include <memory>
#include <queue>
#include <thread>
#include <mutex>
#include <ctime>

#include <algorithm>

template <class T>
using aitensor = Eigen::Tensor<T,3,Eigen::RowMajor>;

template <class T>
using aitensor2 = Eigen::Tensor<T,2,Eigen::RowMajor>;

template <class T>
using aimatrix = Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>;

class BaseOperator {
private:
public:

};

template <class T>
class Linear : public BaseOperator {
private:
     aitensor<T> input_data;

    int batch_size;
    int input_size;
    int embedding_size;

    int W = 0;
public:

 const aitensor<T> forward(const aitensor<T>& input_data);

};


template <class T>
const aitensor<T> Linear<T>::forward(const aitensor<T>& input_data) { 

     this->input_data = input_data;

     this->batch_size = input_data.dimension(0);
     this->input_size = input_data.dimension(1);
     this->embedding_size = input_data.dimension(2);

     aitensor<T> output_data(this->batch_size, this->input_size, this->W); // BxNxW
     aitensor2<T> input;

     for (int i = 0; i < this->batch_size; ++i) {
        input = this->input_data.chip(i, 0);
     }
     return output_data;
}

template class Linear<float>;  // Instantiate with float
template class Linear<double>;  // Instantiate with double

int main() {

   aitensor<double> aloha(5,3,4);

   int batch_size = aloha.dimension(0);
//   int input_size = aloha.dimension(1;
 //  int embedding_size = aloha.dimension(2);

   for (int i = 0; i < batch_size; i++) {
       aitensor2<double> input = aloha.chip(i, 0);
   }

}
