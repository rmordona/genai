/*
 * GENAI - A distributed framework that can help build LLMs
 *
 * Copyright (c) 2023 Raymod Michael O. Ordona
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 * Author: Raymond Michael O. Ordona
 *
 */

/******************************************************************************************************************************************
* Design considerations:
* 
* Designing an ideal structure for a neural network in C++ depends on various factors, including the specific requirements, 
* complexity of the network, and the desired level of flexibility. However, here are some key components and considerations 
* for building an efficient and modular neural network structure in C++:
* 
* Layer Abstraction: Define a base layer class that represents a generic layer in the neural network. This class should have 
* common methods like forward, backward, and updateWeights for performing forward propagation, backpropagation, and weight updates.
*
* Different Layer Implementations: Implement various types of layers as derived classes, such as fully connected layers, convolutional 
* layers, recurrent layers, activation layers, etc. Each layer implementation should inherit from the base layer class and provide 
* specific implementations for the required methods.
*
* Network Architecture: Define a network class that represents the overall neural network structure. It should encapsulate multiple 
* layers and provide methods for adding layers, connecting them, and performing forward and backward passes.
*
* Input and Output Handling: Consider how input data will be fed to the network and how output predictions will be obtained. Design 
* appropriate interfaces for loading and preprocessing input data and handling the output predictions.
*
* Training and Optimization: Implement training algorithms like stochastic gradient descent (SGD), mini-batch gradient descent, or more 
* advanced optimization methods. Include methods for setting up training parameters, initializing weights, and updating them during training.
* 
* Serialization and Persistence: Provide methods for saving and loading trained models to disk, allowing you to reuse or deploy the trained 
* network without retraining.
* 
* Integration with Libraries: Utilize existing C++ libraries for efficient numerical computations, such as BLAS libraries for matrix 
* operations or CUDA for GPU acceleration. Incorporate these libraries into your network implementation to improve performance.
* 
* Modularity and Flexibility: Design the structure in a modular and flexible way, allowing easy extension and customization. Consider 
* incorporating design patterns like Factory, Builder, or Strategy to enhance flexibility and reusability.
*
* System Design:
* Model Parallelism: Given the massive size of the model, it might be necessary to partition the parameters across multiple GPUs or devices 
* to enable efficient training. Model parallelism techniques, such as pipeline parallelism or model slicing, could be employed.
* 
* Distributed Training: Training a model with 175 billion parameters would likely require distributing the workload across multiple machines. 
* Distributed training frameworks like TensorFlow's Distributed Strategy or PyTorch's DistributedDataParallel can help parallelize the computations 
* and synchronize gradients across machines.
* 
* Cluster Infrastructure: A large-scale model would require a powerful cluster infrastructure with high-performance GPUs or specialized hardware 
* like TPUs. The number of machines and GPUs would depend on the specific requirements and scalability needs.
*
* Data Pipeline and Preprocessing: Handling large datasets efficiently is crucial. Designing a robust and scalable data pipeline that can 
* preprocess and load data in parallel is essential for training such models. Techniques like distributed data loading and data sharding can be employed.
*
* Model Architecture: The specific architecture of the model would depend on the task it aims to solve. For natural language processing tasks, 
* architectures like transformers have been successful. However, with 175 billion parameters, the architecture might involve complex variations, 
* deep hierarchies, and advanced attention mechanisms.
*
* Parameter Server or All-Reduce Approach: Coordinating the model parameters across different devices or machines can be done using a parameter 
* server architecture or an all-reduce approach. Both approaches have their trade-offs in terms of communication overhead and synchronization efficiency.
*
* Deployment Considerations: Deploying a large-scale model requires careful engineering and optimization to ensure efficient inference. 
* Techniques like model pruning, quantization, or specialized hardware deployment (e.g., using TensorRT or ONNX Runtime) might be considered to 
* improve inference speed and resource utilization.
*
* Monitoring and Scalability: Monitoring the training process, tracking model performance, and ensuring scalability are critical. Tools like 
* TensorFlow Extended (TFX) or Kubeflow can help with managing large-scale training pipelines and monitoring system metrics.
* 
* It's important to note that the above considerations provide a broad overview and may vary depending on the specific requirements, constraints, 
* and resources available. Designing a system for a model with 175 billion parameters is a highly complex task that requires deep expertise in 
* distributed systems, machine learning, and infrastructure design.
* 
* Attention Mechanism Considerations:
* Advanced attention mechanisms refer to more sophisticated and enhanced variations of the traditional attention mechanism. These variations 
* aim to address specific challenges or improve the effectiveness of attention-based models in different tasks. Here are a few examples:
* 
* Multi-head Attention: Instead of using a single attention mechanism, multi-head attention employs multiple parallel attention mechanisms 
* operating on different subspaces of the input. This allows the model to capture different types of information and learn more diverse representations.
*
* Self-Attention with Masking: In certain tasks, such as machine translation or language generation, it is essential to mask certain 
* positions in the input sequence to prevent the model from attending to future or unseen information. Masked self-attention ensures that 
* attention is only applied to valid positions in the sequence, taking into account the autoregressive nature of the task.
* 
* Relative Positional Encodings: Positional encodings are often used in attention mechanisms to incorporate positional information into the model. 
* Advanced attention mechanisms introduce relative positional encodings that capture relative distances or relationships between positions, 
* enabling the model to better understand the sequential structure of the input.
*
* Sparse Attention: In models with a large number of parameters, computing attention weights for all possible pairwise interactions can be 
* computationally expensive. Sparse attention mechanisms aim to reduce this computational burden by approximating the attention weights only for 
* a subset of the elements, typically based on their proximity or relevance to each other.
*
* Structured Attention: Traditional attention mechanisms operate on sequential or tabular data. However, in some domains, such as graphs or 
* images, attention needs to be applied to structured data. Advanced attention mechanisms adapt the attention mechanism to incorporate the 
* structural properties of the data, enabling the model to capture dependencies and relationships in a structured manner.
*
* Parameter server Approach (Distributed Training Strategies)
* Parameter server and all-reduce are two commonly used distributed training strategies in deep learning. Here's a comparison between 
* the two approaches:
* 
* Communication Pattern:
* 
* Parameter Server: In the parameter server approach, the model parameters are divided and stored on separate parameter servers. During 
* training, workers (e.g., GPUs or machines) communicate with the parameter servers to read and update the parameters.
* All-reduce: In the all-reduce approach, all workers participate in collective communication operations to synchronize their model parameters. 
* Each worker computes gradients locally, and all workers collectively reduce and update the gradients to ensure consistency across all replicas.
* Communication Overhead:
*
* Parameter Server: The parameter server approach involves communication between workers and parameter servers during parameter read and update 
* operations. The frequency of communication can be higher compared to all-reduce, especially when there are a large number of parameter servers 
* or when parameter updates are frequent.
* All-reduce: All-reduce involves communication among all workers during the gradient reduction step. The frequency of communication is 
* typically lower compared to parameter server, as workers exchange gradients periodically rather than after each update.
* Scalability:
* 
* Parameter Server: Parameter server architectures can scale to a large number of workers, as each worker communicates with a subset of parameter 
* servers. However, the scalability is limited by the communication bandwidth and latency between workers and parameter servers.
* All-reduce: All-reduce can scale efficiently to a large number of workers, as all workers participate in the collective communication. 
* The scalability of all-reduce depends on the network topology and the efficiency of the collective communication implementation.
* Fault Tolerance:

* Parameter Server: Parameter server architectures may suffer from single points of failure if a parameter server becomes unavailable. 
* Fault tolerance can be achieved by replicating the parameter servers or implementing backup mechanisms.
* All-reduce: All-reduce is inherently fault-tolerant as it relies on collective communication among all workers. If a worker fails, the 
* remaining workers can continue the training process.
* Memory and Storage Requirements:
* 
* Parameter Server: Parameter server architectures require storage for the model parameters on the parameter servers. The storage requirements 
* depend on the size of the model and the number of parameter servers.
* All-reduce: All-reduce requires memory for storing gradients during the reduction step. The memory requirements depend on the size of the 
* model and the number of workers.
* Both parameter server and all-reduce approaches have their strengths and weaknesses, and the choice depends on various factors such as the 
* size of the model, the number of workers, communication overhead, and fault tolerance requirements. In recent years, all-reduce has gained 
* popularity due to its scalability, fault tolerance, and efficient utilization of resources in distributed deep learning training.
*
* In terms of All-reduce approach:
* All-reduce algorithms, such as ring-based or tree-based algorithms, have demonstrated good scalability and efficiency in synchronous 
* gradient aggregation, reducing the communication overhead compared to parameter server architectures. This has made all-reduce more 
* attractive for large-scale distributed training, especially in scenarios with a large number of workers or when training large models.
* 
* However, it's important to note that the field of deep learning and distributed training is rapidly evolving. New techniques, frameworks, 
* and approaches continue to emerge, and the choice of distributed training strategy may vary depending on the specific requirements and 
* constraints of the training task.
* 
* To have the most up-to-date information on the current trends and practices in distributed deep learning training, I would recommend 
* referring to recent research papers, industry practices, and consulting with experts in the field.
*
* Ring-based and tree-based algorithms are two common approaches used in distributed computing, including in the context of all-reduce 
* operations in distributed deep learning. Here's a brief comparison of the two:
*
* Ring-based Algorithm:
*
* In a ring-based algorithm, the workers are arranged in a logical ring structure.
* The data is passed sequentially from one worker to the next in a circular manner until it reaches all the workers.
* Each worker performs a local reduction operation on the data it receives and then forwards the result to the next worker.
* The process continues until the data has been reduced by all workers and returned to the original sender.
* Ring-based algorithms are relatively simple to implement and have low latency but may suffer from load imbalance if the computation or communication 
* times vary significantly between workers.
*
* Tree-based Algorithm:
* 
* In a tree-based algorithm, the workers are organized in a hierarchical tree structure.
* The data is aggregated in a hierarchical manner, starting from the leaf nodes and moving up to the root node.
* Each node in the tree combines the data from its child nodes and performs a reduction operation.
* The process continues recursively until the root node receives the final reduced data.
* Tree-based algorithms can provide better load balancing compared to ring-based algorithms as the data aggregation happens in a 
* hierarchical structure.
* However, they may introduce higher latency due to additional communication steps involved in traversing the tree structure.
* The choice between ring-based and tree-based algorithms depends on various factors, such as the number of workers, the communication infrastructure, 
* and the characteristics of the training workload. Both algorithms have their strengths and weaknesses, and their performance can vary based on 
* the specific system and workload conditions.
* 
* It's worth noting that there are also other variations and optimizations of all-reduce algorithms, such as recursive doubling, butterfly, and more, 
* which aim to improve performance in different contexts. The choice of the most suitable algorithm often requires experimentation and benchmarking 
* on the target system to find the optimal configuration for a given distributed training task.
* 
* Here is a list of some commonly used variations and optimizations of all-reduce algorithms:
* 
* Ring-Based Algorithms: Traditional ring-based algorithms are widely used and serve as the baseline for many other algorithms.
* 
* Tree-Based Algorithms: Tree-based algorithms, such as binomial tree, k-ary tree, or hypercube-based tree, provide better load balancing and 
* reduced communication steps compared to ring-based algorithms.
* 
* Recursive Doubling: Recursive doubling algorithms leverage the binary representation of the rank to perform reduction operations in a hierarchical 
* manner, effectively reducing the number of communication steps.
* 
* Butterfly Algorithm: The butterfly algorithm uses a combination of butterfly networks and hypercube networks to achieve reduced latency and 
* improved bandwidth utilization.
* 
* AllGather: AllGather is an extension of all-reduce that collects the input data from all workers onto every worker, rather than performing a 
* reduction operation. It is commonly used for gathering statistics or exchanging information across all workers.
* 
* AllReduce-Multi: AllReduce-Multi algorithms allow simultaneous communication of multiple smaller messages instead of a single large message, which 
* can improve performance in certain scenarios, especially when dealing with heterogeneous network environments.
* 
* Gradient Compression: Gradient compression techniques, such as top-K sparsification or quantization, can be applied to reduce the communication 
* bandwidth and latency during the all-reduce operation while still maintaining reasonable model accuracy.
* 
* Ring All-Reduce with All-Gather: This approach combines the ring-based all-reduce with an all-gather operation to reduce the overall communication 
* time, especially when the number of workers is large.
*
* Gradient Accumulation: Gradient accumulation techniques allow workers to accumulate gradients over multiple iterations before performing the 
* all-reduce operation, reducing the frequency of communication and potentially improving scalability.
* 
* Asynchronous All-Reduce: Asynchronous algorithms, such as asynchronous decentralized parallel stochastic gradient descent (A-DePSGD), relax 
* the synchronization requirements and overlap communication with computation to improve overall training throughput.
* 
* These are just a few examples of the variations and optimizations available for all-reduce algorithms. The choice of which algorithm to use 
* depends on factors such as network topology, system characteristics, workload, and communication patterns, and it often requires careful 
* experimentation and benchmarking to identify the best approach for a specific distributed training scenario.
*
* In terms of Training distribution:
* Distributing training across multiple workers in a C++ codebase involves partitioning the data and model parameters, performing computations 
* on each worker, and synchronizing the updates to keep the model consistent. Here's a high-level overview of how training can be broken 
* down and distributed across workers:
* 
* Data Partitioning: Split the training data into multiple shards or subsets, where each worker is responsible for processing a different portion 
* of the data. The data can be partitioned based on samples, batches, or other appropriate criteria.
* 
* Model Replication: Replicate the model parameters on each worker. This ensures that each worker has a copy of the complete model for performing 
* computations independently.
* 
* Forward Pass: Each worker performs a forward pass on its local data subset using the replicated model. The forward pass computes the predictions 
* and loss for the local data.
* 
* Backward Pass and Gradient Computation: After the forward pass, each worker computes the gradients of the model parameters with respect to the 
* local data subset. The gradients can be computed using techniques like backpropagation or automatic differentiation.
* 
* Gradient Aggregation: The computed gradients from each worker need to be aggregated to obtain a global gradient. This can be done using various 
* aggregation algorithms, such as the All-Reduce algorithm, where gradients are exchanged and combined across workers to compute the average or sum 
* of gradients.
* 
* Parameter Update: Once the global gradient is obtained, each worker updates its local copy of the model parameters using an optimization algorithm, 
* such as stochastic gradient descent (SGD) or Adam. The updates can be applied asynchronously or synchronously based on the distributed training strategy.
* 
* Synchronization: If training is performed asynchronously, it is necessary to periodically synchronize the model parameters across workers to maintain 
* consistency. Synchronization can be done by broadcasting the updated model parameters from a designated worker to other workers.
*
* Iterative Training: The above steps are repeated for multiple iterations or epochs until the desired convergence or training criteria are met. Each 
* iteration involves data partitioning, forward and backward passes, gradient aggregation, parameter updates, and synchronization.
*
* It's important to note that the implementation details of distributed training in C++ may vary depending on the specific framework or library being 
* used. Popular frameworks like TensorFlow or PyTorch provide built-in support for distributed training with their own APIs and abstractions. 
* These frameworks handle the underlying communication, synchronization, and parameter updates across workers, allowing you to focus more on 
* defining the model and training process.
*
* When implementing distributed training in C++, you may need to utilize distributed computing libraries, such as MPI (Message Passing Interface) or 
* specialized distributed frameworks like Horovod, to facilitate inter-worker communication and coordination. These libraries provide functions and 
* utilities for message passing, collective operations, and distributed training patterns.
* 
* Overall, the process of distributing training across workers in C++ involves partitioning data, replicating the model parameters, performing 
* computations on each worker, aggregating gradients, updating parameters, and ensuring synchronization to achieve distributed training and collaboration.
******************************************************************************************************************************************/
#pragma once
#include <iostream>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <string>
#include <cmath>
#include <type_traits>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <cblas.h>
#include <omp.h>
#include <mpi.h>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <unsupported/Eigen/CXX11/Tensor>
#include <memory>
#include <queue>
#include <thread>
#include <mutex>
#include <ctime>
#include <chrono>
#include <stdexcept>
#include <algorithm>
#include <functional>
#include <curl/curl.h>
#include <map>

/**************************************************************************************************
*  Define the Types to be used for data.
*  A good example:
*      Eigen::Tensor<double, 3, Eigen::RowMajor> tensor(3, 2, 4);
*    tensor.setValues({
*       {
*        {1.0, 2.0, 3.0, 4.0},
*        {5.0, 6.0, 7.0, 8.0}
*       },
*       {
*        {21.0, 22.0, 23.0, 24.0},
*        {25.0, 26.0, 27.0, 28.0}
*       },
*       {
*        {31.0, 32.0, 33.0, 34.0},
*        {35.0, 36.0, 37.0, 38.0}
*       }
*    });
**************************************************************************************************/


template <class T>
using aitensor3 = Eigen::Tensor<T,3,Eigen::RowMajor>;

template <class T>
using aitensor2 = Eigen::Tensor<T,2,Eigen::RowMajor>;

template <class T>
using aimatrix = Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor>;

template <class T>
using aivector = Eigen::Vector<T, Eigen::Dynamic>;   // Defaults to ColumnMajor  (vertical/column)

template <class T>
using airowvector = Eigen::RowVector<T, Eigen::Dynamic>;  // (horizontal/row)

template <class T>
using aiscalar = T; // Eigen::Tensor<T, 0>; // (scalar)

template <class T>
using aitensor = std::vector<aimatrix<T>>;

/*************************************************************************************************
// Helper class for Exceptions
**************************************************************************************************/

#ifndef AIEXCEPTION_H
#define AIEXCEPTION_H

class AIException : public std::exception {
public:
    AIException(const std::string& message) : message_(message) {}

    // Overriding the what() function for customized message.
    const char* what() const noexcept override {
        return message_.c_str();
    }

private:
    std::string message_;
};

#endif
 
/**************************************************************************************************
  Helper Functions shared by other classes
**************************************************************************************************/

void log_msg(const std::string& text);

double inf();

/*
void print_string(const std::string& text, bool printNextLine);

void print_double(double value, bool printNextLine);

std::string scalar_to_string(const float& value);

std::string scalar_to_string(const double& value);
*/

double* allocate_matrix(ssize_t rows, ssize_t cols);

namespace py = pybind11;
using namespace py::literals;

#ifndef UTF8ANDSHA_H
#define UTF8ANDSHA_H

#include <utf8cpp/utf8.h>
#include <sstream>
#include <openssl/evp.h>
#include <openssl/sha.h>
#include <iomanip>
#include <fstream>

/*
std::string wstringToUtf8(const std::wstring& wstr);

std::wstring utf8ToWstring(const std::string& utf8Str);

// Function to serialize std::wstring to UTF-8 encoded std::vector<unsigned char>
std::vector<unsigned char> serializeWString(const std::wstring& wstr);

// Function to deserialize UTF-8 encoded std::vector<unsigned char> back to std::wstring
std::wstring deserializeWString(const std::vector<unsigned char>& bytes);

// Function to hash a given word token using SHA256
std::string sha256(const std::wstring& str);
*/


void log_msg(const std::string& text);
 
double inf();

void print_string(const std::string& text, bool printNextLine);

void print_double(double value, bool printNextLine);

std::string scalar_to_string(const float& value);

std::string scalar_to_string(const double& value);

std::string wstringToUtf8(const std::wstring& wstr);

std::wstring utf8ToWstring(const std::string& utf8Str);

// Function to calculate SHA-256 and return it as a string
std::string sha256(const std::wstring& data);


/* sha256Context is deprecated 
// Function to hash a given word token using SHA256
std::string sha256(const std::wstring& str) {
    std::vector<unsigned char> bytes = serializeWString(str);
    unsigned char hash[SHA256_DIGEST_LENGTH];
    SHA256_CTX sha256Context;

    if (SHA256_Init(&sha256Context) != 1) {
        // Handle initialization error
        return "";
    }

    if (SHA256_Update(&sha256Context, bytes.data(), bytes.size()) != 1) {
        // Handle update error
        return "";
    }

    if (SHA256_Final(hash, &sha256Context) != 1) {
        // Handle finalization error
        return "";
    }

    std::stringstream ss;
    ss << std::hex << std::setfill('0');
    for (int i = 0; i < SHA256_DIGEST_LENGTH; ++i) {
        ss << std::setw(2) << static_cast<int>(hash[i]);
    }

    return ss.str();
}
*/
 
#endif

#include "logger.h"

#ifndef OUTPUTFORMATTER_H
#define OUTPUTFORMATTER_H

class PythonStream {
public:
    PythonStream() {}

    template <typename T>
    PythonStream& operator<<(const T& value) {
        buffer_ << value;
        return *this;
    }

    PythonStream& operator<<(std::ostream& (*func)(std::ostream&)) {
        if (func == static_cast<std::ostream& (*)(std::ostream&)>(std::endl)) {
            py::print(buffer_.str());
            buffer_.str("");
        }
        return *this;
    }

private:
    std::stringstream buffer_;
};

// Declare an extern instance of PythonStream
extern PythonStream py_cout;

#endif


#ifndef BASEOPERATORS_H
#define BASEOPERATORS_H

template <class T>
struct OperationParams {
    aimatrix<T> weights; // MxW
    airowvector<T> biases;  // 1xW
};

// Add more metrics if necessary. 
// Then create API functions in Metrics Class under operators.h/operators.cpp
template <class T>
struct PerfMetrics {
    bool isprecision = false; // is precision required in training?
    bool isrecall = false;    // is recall required in training?
    bool isf1score = false;   // is f1score required in training?
    bool isaucroc = false;     // is aucroc required in training?
    T precision;   
    T recall;   
    T f1score; 
    T aucroc;  
};

/******* Initiate Operation param template */

template struct OperationParams<float>;   // Instantiate with float
template struct OperationParams<double>; // Instantiate with float

enum class OperType {
    LINEAR,
    BATCHNORM,
    LAYERNORM,
    REDUCT,
    ACTIVATE,
    MASK,
    DROPOUT,
    SCALE,
    ATTENTION
};

enum class ReductionType {
    SUM,
    AVG,
    MAX,
    MIN,
    ARGMAX,
    ARGMIN,
    MATMUL,
    MUL,
    CONCAT
};

enum class ActivationType {
    SIGMOID,
    TANH,
    RELU,
    LEAKYRELU,
    GELU,
    SOFTMAX
};

/*****************************************************************************************************
* ConvertData
*  Function to convert data from python array to C++ objects (tensor, matrix)
*****************************************************************************************************/
class ConvertData {
public:

    template <class T>
    static aitensor<T> totensor(const py::array_t<T>& parray) {

        log_info("Converting data ...");

        int ndim = parray.ndim();
        py::buffer_info buffer_info = parray.request();

        log_info( "Received buffer info:" );
        log_detail( "Format: {0}", buffer_info.format );
        log_detail( "Item size: {0}", buffer_info.itemsize );
        log_detail( "Size: {0}", buffer_info.size );
        log_detail( "Dimension: {0}", ndim );

        std::vector<ssize_t> shape = buffer_info.shape;
        // extract data and shape of input array
        T* dataPtr = static_cast<T *>(buffer_info.ptr);

        ssize_t dim0, dim1, dim2;

        if (ndim == 2) {
            dim0 = 1;        // Batch Size
            dim1 = shape[0]; // Input Size
            dim2 = shape[1]; // Parameter / Embedding Size
            
        } else
        if (ndim == 3) {
            dim0 = shape[0]; // Batch Size
            dim1 = shape[1]; // Input Size
            dim2 = shape[2]; // Parameter / Embedding Size        
        } else {
            throw AIException(" Incorrect data dimension (Use 2D or 3D only)...");
        } 

        log_detail( "Size: {:d} {:d} {:d}", dim0, dim1,  dim2 );

        aitensor<T> eigenMatrices;
        eigenMatrices.reserve(dim0);

        for (int i = 0; i < dim0; ++i) {
            aimatrix<T> eigenMatrix(dim1, dim2);
            std::memcpy(eigenMatrix.data(), &dataPtr[i * dim1 * dim2], dim1 * dim2 * sizeof(T));
            eigenMatrices.push_back(eigenMatrix);
        }

        return eigenMatrices;
    }

    template <class T>
    static py::array_t<T> topyarray(const aitensor<T>& matrices) {
        // Determine the shape and size of the NumPy array
        size_t num_matrices = matrices.size();
        size_t matrix_rows = matrices[0].rows();
        size_t matrix_cols = matrices[0].cols();

        // Create a NumPy array with the same shape
        auto result = py::array_t<T>({num_matrices, matrix_rows, matrix_cols});
        auto buffer_info = result.request();
        T* ptr = static_cast<T*>(buffer_info.ptr);

        // Copy data from Eigen matrices to NumPy array
        for (size_t i = 0; i < num_matrices; i++) {
            Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(ptr, matrix_rows, matrix_cols) = matrices[i].template cast<T>();
            ptr += matrix_rows * matrix_cols;
        }

        return result;
    }

    template <class T>
    static py::array_t<T> topyarray(const aimatrix<T>& matrix) {
        // Determine the shape and size of the NumPy array

        log_detail("Converting 1 ...");
        size_t matrix_rows = matrix.rows();
        size_t matrix_cols = matrix.cols();

       log_detail("Converting 2 ...");

        // Create a NumPy array with the same shape
        auto result = py::array_t<T>({matrix_rows, matrix_cols});
        auto buffer_info = result.request();
        T* ptr = static_cast<T*>(buffer_info.ptr);

       log_detail("Converting 3 ...");

        // Copy data from Eigen matrices to NumPy array
        Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(ptr, matrix_rows, matrix_cols) = matrix.template cast<T>();

       log_detail("Converting 4 ...");

        return result;
    }


};

/*****************************************************************************************************
* Base Operators
*  Linear, BatchNorm, LayerNorm, Reduct classes derive from BaseOperator class.
*****************************************************************************************************/
class BaseOperator {
    private:
       // ActivationType otype;
    public:
    virtual void forwardPass() = 0;
    virtual void backwardPass() = 0;

    static aimatrix<float> standardize(const aimatrix<float>& input_data) {
        // Calculate the mean and standard deviation along each column
        aivector<float> mean = input_data.colwise().mean();
        aivector<float> stdDev = ((input_data.rowwise() - mean.transpose()).array().square().colwise().sum() / (input_data.rows() - 1)).sqrt();

        // Standardize the matrix by subtracting the mean and dividing by the standard deviation
        aimatrix<float> standard = (input_data.rowwise() - mean.transpose()).array().rowwise() / stdDev.transpose().array();

        return standard;
    }

    static aimatrix<double> standardize(const aimatrix<double>& input_data) {
        // Calculate the mean and standard deviation along each column
        aivector<double> mean = input_data.colwise().mean();
        aivector<double> stdDev = ((input_data.rowwise() - mean.transpose()).array().square().colwise().sum() / (input_data.rows() - 1)).sqrt();

        // Standardize the matrix by subtracting the mean and dividing by the standard deviation
        aimatrix<double> standard = (input_data.rowwise() - mean.transpose()).array().rowwise() / stdDev.transpose().array();

        return standard;
    }

    static aimatrix<float> matmul(const aimatrix<float>& A, const aimatrix<float>& B) {
        float alpha = 1.0f;
        float beta = 0.0f;

        // Use the list of variables if not using Eigen::RowMajor (meaning, default is Column Major)
        //int M = A.rows();
        //int N = B.cols();
        //int K = A.cols();
        //int lda = A.rows();  // leading dimension of A.
        //int ldb = A.cols();  // leading dimension of B.
        //int ldc = A.rows();  // leading dimension of C.

        // Use the list of variables if using Eigen::RowMajor
        int M = A.rows();
        int N = B.cols();
        int K = A.cols();
        int lda = A.cols();  // leading dimension of A.
        int ldb = B.cols();  // leading dimension of B.
        int ldc = B.cols();  // leading dimension of C.

        aimatrix<float> C(M, N);

        // Default:
        // Here we assume the following:
        //  A = MxK,  B = KxN,    C = MxN. 
        // Therefore for a Column Major (Vertical), lda = M, ldb = K, ldc = M. Those represent the length of rows per matrix.
        // cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A.data(), lda, B.data(), ldb, beta, C.data(), ldc);

        // For Eigen::RowMajor, use bellow instead:
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A.data(), lda, B.data(), ldb, beta, C.data(), ldc);

        return C;
    }

    static aimatrix<double> matmul(const aimatrix<double>& A, const aimatrix<double>& B) {
        double alpha = 1.0;
        double beta = 0.0;

        // Use the list of variables if not using Eigen::RowMajor (meaning, default is Column Major)
        //int M = A.rows();
        //int N = B.cols();
        //int K = A.cols();
        //int lda = A.rows();  // leading dimension of A.
        //int ldb = A.cols();  // leading dimension of B.
        //int ldc = A.rows();  // leading dimension of C.

        // Use the list of variables if using Eigen::RowMajor

        int M = A.rows();
        int N = B.cols();
        int K = A.cols();
        int lda = A.cols();  // leading dimension of A.
        int ldb = B.cols();  // leading dimension of B.
        int ldc = B.cols();  // leading dimension of C.

        aimatrix<double> C(M, N);

        // Default:
        // Here we assume the following:
        //  A = MxK,  B = KxN,    C = MxN. 
        // Therefore for a Column Major (Vertical), lda = M, ldb = K, ldc = M. Those represent the length of rows per matrix.
        // cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A.data(), lda, B.data(), ldb, beta, C.data(), ldc);

        // For Eigen::RowMajor, use bellow instead:
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A.data(), lda, B.data(), ldb, beta, C.data(), ldc);

        return C;
    }

    // Apply the sigmoid function element-wise to a matrix
    template <class T>
    static aimatrix<T> sigmoid(const aimatrix<T>& output_data) {
        return 1.0 / (1.0 + (-output_data.array()).exp());
    }

    /*****************************************************************************************************
    * So, the gradient of the sigmoid function with respect to its input (z) can be expressed as follows:
    * dy/dz = propagated_gradient * sigmoid(z) * (1 - sigmoid(z))
    * Or if output (y) is cached, where y = sigmoid(z)
    * then we use the output such that:
    * dy/dz = propagated_gradient * y * (1 - y)
    *****************************************************************************************************/
    template <class T>
    static aimatrix<T> sigmoidGradient(const aimatrix<T>& gradients, const aimatrix<T>& output) {
        aimatrix<T> dInput = gradients.array() * output.array() * ( 1 - output.array());
        return dInput;
    }

    // Apply the tanh function element-wise to a matrix
    template <class T>
    static aimatrix<T> tanh(const aimatrix<T>& output) {
        return output.array().tanh();
    }

    /*****************************************************************************************************
    * So, the gradient of the tanh function with respect to its input (z) can be expressed as follows:
    * dy/dz = propagated_gradient * (1 - tanh(z)^2)
    * Or if output (y) is cached, where y = tanh(z)
    * then we use the output such that:
    * dy/dz = propagated_gradient * y * (1 - y^2)
    *****************************************************************************************************/
    template <class T>
    static aimatrix<T> tanhGradient(const aimatrix<T>& gradients, const aimatrix<T>& output) {
        aimatrix<T> dInput =  gradients.array() *  ( 1 - output.array().pow(2));
        return dInput;
    }

    // Define softmax function
    template <class T>
    static aimatrix<T> softmax(const aimatrix<T>& X) {
        aimatrix<T> S(X.rows(), X.cols());

        for (int row = 0; row < X.rows(); ++row) {
            aivector<T>  row_vector = X.row(row);
            double row_max = row_vector.maxCoeff();
            aivector<T> exp_values = (row_vector - row_max * aivector<T> ::Ones(row_vector.size())).array().exp();
            double sum_exp_values = exp_values.sum();
            S.row(row) = exp_values / sum_exp_values;
        }

        return S;
    }

    template <class T>
    static aimatrix<T> softmax1(const aimatrix<T>& output) {

        // Find the maximum value in each column of x
        // Required to handle large numbers.
        aivector<T> maxVal = output.rowwise().maxCoeff();

        // Subtract the maximum value from each element in each column of x and compute the exponential
        aimatrix<T> expmat =  ((output.colwise() - maxVal)).array().exp();

        // Compute the sum of exponential values for each row
        aivector<T> sumexp =  expmat.rowwise().sum();

        // Compute the softmax probabilities by dividing each element in each row by the sum of exponential values
        aimatrix<T> softmat =  expmat.array() / sumexp.replicate(1, expmat.cols()).array();

        // Return the computed softmax probabilities 
        return softmat;
    }

    /****************************************************************************************************
     * The gradient of the softmax function with respect to its input (z) can be expressed as follows:
     * dy_i/dz_j = propagated_gradient * (softmax(z_i) * (1 - softmax(z_j))) for i = j
     * dy_i/dz_j = propagated_gradient * (-softmax(z_i) * softmax(z_j)) for i != j
     * Or if output (y) is cached, where y = softmax(z)
     * then we use the output such that:
     * dy_i/dz_j = propagated_gradient * (y_i * (1 - y_j)) for i = j
     * dy_i/dz_j = propagated_gradient * (-y_i * y_j) for i != j
     ****************************************************************************************************/

    // Calculate gradient with respect to X (dS/dX)
    template <class T>
    static aimatrix<T> softmaxGradient(const aimatrix<T>& gradients, const aimatrix<T>& output_data) {
        aimatrix<T> J(output_data.rows(), output_data.cols());

        for (int row = 0; row < output_data.rows(); ++row) {
            for (int col = 0; col < output_data.cols(); ++col) {
                T S_i = output_data(row, col);
                J(row, col) = S_i * (gradients(row, col) - output_data(row, col) * gradients.row(row).sum());
            }
        }

        return J;
    }

    template <class T>
    static aimatrix<T> softmaxGradient1(const aimatrix<T>& gradients, const aimatrix<T>& output_data) {
        int N = gradients.rows();  // Number of samples
        int M = gradients.cols();  // Number of classes

        aimatrix<T> J(N, M);
        J.setZero();

            for (int i = 0; i < N; ++i) {
            for (int j = 0; j < M; ++j) {
                for (int k = 0; k < M; ++k) {
                    if (k == j) {
                      //  dInput(i, j) += gradients(i, j) * output_data(i, j) * (1 - output_data(i, j));
                        J(i, j) +=  output_data(i, j) * (1 - output_data(i, j));
                    } else {
                      // dInput(i, j) -= gradients(i, j) * output_data(i, j) * output_data(i, k);
                       J(i, j) -=  output_data(i, j) * output_data(i, k);
                    }
                }
            }
        }

        aimatrix<T> dInput = J * gradients;

        return dInput;
    }

    template <class T>
    static void xavierInitMatrix(aimatrix<T>& weights) {
        T scale = std::sqrt(6.0 / (weights.rows() + weights.cols()));
        weights.setRandom();
        weights *= scale;
    }

    template <class T>
    static void xavierInitVector(aivector<T>& biases) {
        T scale = std::sqrt(6.0 / (biases.cols() + biases.rows()));
        biases.setRandom();
        biases *= scale;
    }

    template <class T>
    static void xavierInitRowVector(airowvector<T>& biases) {
        T scale = std::sqrt(6.0 / (biases.cols() + biases.rows()));
        biases.setRandom();
        biases *= scale;
    }

    template <class T>
    static void heInitMatrix(aimatrix<T>& weights) {
        T scale = std::sqrt(2.0 / weights.rows());
        weights.setRandom();
        weights *= scale;
    }

    template <class T>
    static void heInitVector(aivector<T>& biases) {
        T scale = std::sqrt(2.0 / biases.rows());
        biases.setRandom();
        biases *= scale;
    }

    template <class T>
    static void heInitRowVector(airowvector<T>& biases) {
        T scale = std::sqrt(2.0 / biases.cols());
        biases.setRandom();
        biases *= scale;
    }

    template <class T>
    static void uniformInitialization(const aimatrix<T>& weights, T minVal, T maxVal) {
        weights = Eigen::MatrixXd::Random(weights.rows(), weights.cols());
        weights = (weights.array() * (maxVal - minVal)) + minVal;
    }

    template <class T>
    static void normalInitialization(const aimatrix<T>& weights, T mean, T stdDev) {
        weights = Eigen::MatrixXd::Random(weights.rows(), weights.cols());
        weights = (weights.array() * stdDev) + mean;
    }

    template <class T>
    static void zeroInitialization(const aimatrix<T>& weights) {
        weights.setZero();
    }

};

#endif

#ifndef DISTRIBUTED_KVS_H
#define DISTRIBUTED_KVS_H

#include <libmemcached/memcached.hpp> // For libmemcached functions
#include <libmemcached/util.h> // For libketama functions
#include "zmq.hpp"

class DistributedKVStore {
private:

    struct Node {
        std::string identifier; // IP address or hostname of the node
        memcached_st* memc;     // libmemcached connection to the node
    };

    // ZeroMQ context and sockets
    zmq::context_t zmqContext;
    zmq::socket_t publisherSocket; // Publisher socket to send replication messages
    zmq::socket_t queuePushSocket; // Push socket for the queue node to receive replication messages
    zmq::socket_t queuePullSocket; // Pull socket for the queue node to distribute replication messages
    std::vector<Node> nodes;
    // Thread for handling replication on subscriber nodes
    std::thread subscriberThread;

    memcached_st* memc; // Memcached client object

    std::mutex mtx;  // Mutex for protecting data during replication

    // Local data store for the distributed key-value store
    // Outer map handles node identifiers. 
    // Inner map handles key-value paris.
    std::unordered_map<std::string, std::unordered_map<std::string, std::string>> data;


    // Replication thread function for subscriber nodes
    void replicationSubscriberThread();



public:
    // Constructor to initialize nodes
    DistributedKVStore(const std::vector<std::string>& nodeIdentifiers);
   ~DistributedKVStore();

    // Functions for put, get, and remove operations
    void put(const std::string& key, const std::string& value);

    size_t getServerIndexForKey(const std::string& key);

    std::string get(const std::string& key);

    void remove(const std::string& key);

    void addNode(const std::string& nodeIdentifier);

    void removeNode(const std::string& nodeIdentifier);

    std::string getNodeForKey(const std::string& key);

    // If you need a simple and direct replication mechanism and are already relying on memcached for replication, 
    // the first function with libmemcached might be a good fit.
    void replicateDataMC(const std::string& nodeIdentifier, const std::string& key, const std::string& value);
    // If you want more flexibility in communication patterns and are open to using additional libraries like ZeroMQ, 
    // the second function with ZeroMQ REQ-REP might be suitable.
    void replicateDataZMQReqRep(const std::string& nodeIdentifier, const std::string& key, const std::string& value);
    // If you need efficient broadcasting of replication messages to multiple nodes, the third function with 
    // ZeroMQ PUB-SUB might be a better option.
    void replicateDataZMQPubSub(size_t sourceNodeIndex, const std::string& key, const std::string& value);

    void handleNetworkPartition(const std::string& failedNodeIdentifier);

    // Helper functions for recovery strategies, data consistency, and fault tolerance
    void recoverFromNodeFailure(const std::string& failedNodeIdentifier);

    void redistributeData(const std::string& fromNode, const std::string& toNode);

    void ensureDataConsistency();

    bool pingNode(const std::string& nodeIdentifier);

    void handleFaults();

    void startServer();

};

#endif
