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
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <cblas.h>
#include <omp.h>
#include <mpi.h>
#include <Eigen/Dense>
#include <Eigen/Core>
#include <memory>
#include <queue>
#include <thread>
#include <mutex>
#include <ctime>
#include <chrono>

#include <algorithm>
#include <map>



/**************************************************************************************************
  Helper Functions shared by other classes
**************************************************************************************************/

void log_msg(const std::string& text);

double inf();

void print_string(const std::string& text, bool printNextLine);

void print_double(double value, bool printNextLine);

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

std::string wstringToUtf8(const std::wstring& wstr);

std::wstring utf8ToWstring(const std::string& utf8Str);

// Function to serialize std::wstring to UTF-8 encoded std::vector<unsigned char>
std::vector<unsigned char> serializeWString(const std::wstring& wstr);

// Function to deserialize UTF-8 encoded std::vector<unsigned char> back to std::wstring
std::wstring deserializeWString(const std::vector<unsigned char>& bytes);

// Function to hash a given word token using SHA256
std::string sha256(const std::wstring& str);

#endif

#ifndef OPERATORS_H
#define OPERATORS_H



struct OperationParams {
    Eigen::MatrixXd weights; // MxW
    Eigen::RowVectorXd biases;  // 1xW
};

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
    MUL
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
* Base Operators
*  Linear, BatchNorm, LayerNorm, Reduct classes derive from BaseOperator class.
*****************************************************************************************************/
class BaseOperator {
    private:
        OperType otype;
    public:
    virtual void forwardPass() = 0;
    virtual void backwardPass() = 0;

    // Perform Matrix Multiplication. 
    static Eigen::MatrixXd matmul(Eigen::MatrixXd A, Eigen::MatrixXd B) {
        double alpha = 1.0;
        double beta = 0.0;
        int M = A.rows();
        int N = B.cols();
        int K = A.cols();
        int lda = M;  // leading dimension of A.
        int ldb = K;  // leading dimension of B.
        int ldc = M;  // leading dimension of C.

        Eigen::MatrixXd C(M, N);

        // Default:
        // Here we assume the following:
        //  A = MxK,  B = KxN,    C = MxN. 
        // Therefore for a Column Major (Vertical), lda = M, ldb = K, ldc = M. Those represent the length of rows per matrix.
        cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, M, N, K, alpha, A.data(), lda, B.data(), ldb, beta, C.data(), ldc);

        return C;
    }

    static Eigen::MatrixXd softmax(const Eigen::MatrixXd& input_data) {

        std::cout << "Softmax forward ...\n";
        std::cout << input_data << "\n";

        // Find the maximum value in each column of x
        // Required to handle large numbers.
        double maxVal = input_data.maxCoeff();

        std::cout << "maxVal maxCoeff forward ...\n";
        std::cout << maxVal << "\n";

        // Subtract the maximum value from each element in each column of x and compute the exponential
        Eigen::MatrixXd expX = (input_data.array().colwise() - Eigen::ArrayXd::Constant(input_data.cols(), maxVal)).exp();

        std::cout << "expX forward ...\n";
        std::cout << expX << "\n";

        // Compute the sum of exponential values for each row
        Eigen::VectorXd sumExp = expX.rowwise().sum();

        std::cout << "sumExp forward ...\n";
        std::cout << sumExp << "\n";

        // Compute the softmax probabilities by dividing each element in each row by the sum of exponential values
        Eigen::MatrixXd softmax = expX.array() / sumExp.replicate(1,2).array();

        std::cout << "softmax forward ...\n";
        std::cout << softmax << "\n";

        // Return the computed softmax probabilities 
        return softmax;
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
    static Eigen::MatrixXd softmaxGradient(const Eigen::MatrixXd& gradients, const Eigen::MatrixXd& output_data) {
        int N = gradients.rows();  // Number of samples
        int M = gradients.cols();  // Number of classes

        Eigen::MatrixXd dInput(N, M);
        dInput.setZero();

        std::cout << "Softmax Gradient \n";
        std::cout << "N:" << N << "\n";
        std::cout << "M:" << M << "\n";

        std::cout << gradients << "\n";

        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < M; ++j) {
                for (int k = 0; k < M; ++k) {
                    if (k == j) {
                        dInput(i, j) += gradients(i, k) * output_data(i, j) * (1 - output_data(i, j));
                    } else {
                        dInput(i, j) -= gradients(i, k) * output_data(i, j) * output_data(i, k);
                    }
                }
            }
        }
        return dInput;
    }

    static void xavierInitialization(Eigen::MatrixXd& weights) {
        double scale = std::sqrt(6.0 / (weights.rows() + weights.cols()));
        weights.setRandom();
        weights *= scale;
    }

    static void xavierInitialization(Eigen::VectorXd& biases) {
        double scale = std::sqrt(6.0 / (biases.cols() + biases.rows()));
        biases.setRandom();
        biases *= scale;
    }

    static void xavierInitialization(Eigen::RowVectorXd& biases) {
        double scale = std::sqrt(6.0 / (biases.cols() + biases.rows()));
        biases.setRandom();
        biases *= scale;
    }

    static void heInitialization(Eigen::MatrixXd& weights) {
        double scale = std::sqrt(2.0 / weights.rows());
        weights.setRandom();
        weights *= scale;
    }

    static void heInitialization(Eigen::VectorXd& biases) {
        double scale = std::sqrt(2.0 / biases.rows());
        biases.setRandom();
        biases *= scale;
    }

    static void heInitialization(Eigen::RowVectorXd& biases) {
        double scale = std::sqrt(2.0 / biases.cols());
        biases.setRandom();
        biases *= scale;
    }

    static void uniformInitialization(Eigen::MatrixXd& weights, double minVal, double maxVal) {
        weights = Eigen::MatrixXd::Random(weights.rows(), weights.cols());
        weights = (weights.array() * (maxVal - minVal)) + minVal;
    }

    static void normalInitialization(Eigen::MatrixXd& weights, double mean, double stdDev) {
        weights = Eigen::MatrixXd::Random(weights.rows(), weights.cols());
        weights = (weights.array() * stdDev) + mean;
    }

    static void zeroInitialization(Eigen::MatrixXd& weights) {
        weights.setZero();
    }

};

class Optimizer : public BaseOperator {
private:
    std::string optimizertype = "adam";
    double learningRate = 0.001;
    Eigen::MatrixXd moments;
    Eigen::MatrixXd velocity;
    Eigen::MatrixXd rho;
    Eigen::MatrixXd rms;
    Eigen::MatrixXd accum;
    Eigen::MatrixXd nu;
public:

    Optimizer(const std::string& optimizertype, double& learningRate) {
        this->optimizertype = optimizertype;
        this->learningRate = learningRate; 
        moments.setZero();  
        velocity.setZero();  
        rho.setZero();   
        rms.setZero();   
        accum.setZero();  
        nu.setZero();  
    }

    // SGD optimizer with optional step decay
    void sgd(Eigen::MatrixXd& weights, Eigen::MatrixXd& gradients, int currentEpoch = 0,
                    bool useStepDecay = false, double decayRateStep = 0.1, int decayStep = 0);

    // Momentum optimizer with optional step decay
    void momentum(Eigen::MatrixXd& weights, Eigen::MatrixXd& gradients, int currentEpoch = 0,
                    double momentumRate = 0.9,
                    bool useStepDecay = false, double decayRateStep = 0.1,  int decayStep = 0);

    // Adam optimizer with optional step decay
    void adam(Eigen::MatrixXd& weights, Eigen::MatrixXd& gradients, int currentEpoch = 0,
                    double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8,
                    bool useStepDecay = false, double decayRateStep = 0.1,  int decayStep = 0);

    // RMSprop optimizer with optional step decay
    void rmsprop(Eigen::MatrixXd& weights, Eigen::MatrixXd& gradients, int currentEpoch = 0,
                    double rho = 0.9, double epsilon = 1e-8,
                    bool useStepDecay = false, double decayRateStep = 0.1,  int decayStep = 0);

    // Adagrad optimizer with optional step decay
    void adagrad(Eigen::MatrixXd& weights, Eigen::MatrixXd& gradients, int currentEpoch = 0,
                    double epsilon = 1e-8,
                    bool useStepDecay = false, double decayRateStep = 0.1,  int decayStep = 0);

    // Adamax optimizer with optional step decay
    void adamax(Eigen::MatrixXd& weights, Eigen::MatrixXd& gradients, int currentEpoch = 0, 
                    double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8,
                    bool useStepDecay = false, double decayRateStep = 0.1, int decayStep = 0);

    // Nadam optimizer with optional step decay
    void nadam(Eigen::MatrixXd& weights, Eigen::MatrixXd& gradients, int currentEpoch = 0, 
                    double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8,
                    bool useStepDecay = false, double decayRateStep = 0.1, int decayStep = 0);

    // Step decay for learning rate
    void stepDecay(double& learningRate, double decayRate, int currentEpoch, int decayStep);

    void forwardPass() {}
    void backwardPass() {}
};

class Linear : public BaseOperator {
private:

    Eigen::MatrixXd input_data; // NxW samples, by using & 
    OperationParams parameters; // inputs to next forward-wise Nodes
    OperationParams gradients;  // inputs to next backward-wise Nodes   (gradients with respect to weights & biases)

    int M = 0; // number of features (embedding vector size)
    int W = 0; // number of weights 
    bool bias = true; // Use bias by default.

    Optimizer* opt_weights = nullptr; // for optimizer
    Optimizer* opt_biases = nullptr; // for optimizer

public:
    Linear(int size, bool bias = true)  {
        this->W = size;
        this->bias = bias;
        print_string("Linear operation ...", true);
    }

    // This assumes that the input is defined with NxM dimensionality.
    // Therefore the size of the parameters and thus gradients will be based on MxW where W is the number of weights to use;
    // The output will have NxW dimension.
    void setInitialWeights(int M);

    OperationParams getParameters() const;

    OperationParams getGradients() const;

    // While the parameter weight has dimension MxW,  the resulting transformation has dimension of NxW.
    // We only need the M dimension from an NxM input to generate parameter matrix.
    // where weights is NxW and bias is W.
    Eigen::MatrixXd linearTransform(const Eigen::MatrixXd& input_data);

    Eigen::MatrixXd forward(Eigen::MatrixXd& input_data);

    void gradient_Wrt_Weight_Bias(Eigen::MatrixXd& gradient);

    Eigen::MatrixXd gradient_Wrt_Input(Eigen::MatrixXd& gradient);

    // Leave the gradients as is. They are cached in the Node. 
    // They will be used to update the parameters in next parallel operations.
    // the dInput is the gradient we propagate to source Nodes in the graph;
    // while the parameter gradients get cached to be used to update the parameters later.
    Eigen::MatrixXd backward(Eigen::MatrixXd& gradients);

    void updateParameters(std::string& optimizertype, double& learningRate, int& iter);

    void forwardPass() {}
    void backwardPass() {}

};

class BatchNorm : public BaseOperator {
private:
    // Across W dimension
    Eigen::RowVectorXd scale; // (gamma) 
    Eigen::RowVectorXd shift; // (beta)
    Eigen::RowVectorXd dScale; // gradient for scale (gamma)
    Eigen::RowVectorXd dShift; // gradient for the shift (beta)

    // Cached data for backpropagation.
    Eigen::MatrixXd input_data; // NxW samples 
    Eigen::MatrixXd normalizedInput; // NxW samples 
    Eigen::MatrixXd minusMean;
    Eigen::RowVectorXd batchStdDev;
    int M = 0;

    double epsilon=1e-8;

    Optimizer* opt_scale = nullptr; // for optimizer
    Optimizer* opt_shift = nullptr; // for optimizer

public:
    BatchNorm(int size) {
      // initialize gradients for next iteration.
        dScale.setZero();
        dShift.setZero();
    }

    // This assumes that the input is defined with NxM dimensionality.
    // Therefore the size of the parameters and thus gradients will be based on MxW where W is the number of weights to use.
    void setInitialWeights(int M);

    Eigen::MatrixXd normalize(const Eigen::MatrixXd& input_data);

    void setScale(const Eigen::VectorXd& newScale);

    void setShift(const Eigen::VectorXd& newShift);

    Eigen::MatrixXd forward(Eigen::MatrixXd& input_data);

    // Leave the gradients as is for the scale and shift. They are cached in the Node. 
    // They will be used to update the parameters in next parallel operations.
    // the dInput is the gradient we propagate to source Nodes in the graph;
    // while the parameter gradients get cached to be used to update the parameters later.
    Eigen::MatrixXd backward(const Eigen::MatrixXd& gradients);

    void updateParameters(std::string& optimizertype, double& learningRate, int& iter);

    void forwardPass() {}
    void backwardPass() {}
};

class LayerNorm : public BaseOperator {
private:
    Eigen::VectorXd scale; // (gamma)
    Eigen::VectorXd shift; // (beta)
    Eigen::VectorXd dScale; // gradient for the scale (gamma)
    Eigen::VectorXd dShift; // gradient for the shift (beta)

    // Cached data for backpropagation.
    Eigen::MatrixXd input_data; // NxW samples 
    Eigen::MatrixXd normalizedInput; // NxW samples 
    Eigen::MatrixXd minusMean;
    Eigen::VectorXd layerStdDev;
    int N = 0;

    double epsilon=1e-8;

    Optimizer* opt_scale = nullptr; // for optimizer
    Optimizer* opt_shift = nullptr; // for optimizer

public:
    LayerNorm(int size) {
      // initialize gradients for next iteration.
        dScale.setZero();
        dShift.setZero();
    }

    // This assumes that the input is defined with NxM dimensionality.
    // Therefore the size of the parameters and thus gradients will be based on MxW where W is the number of weights to use.
    void setInitialWeights(int N);

    Eigen::MatrixXd normalize(const Eigen::MatrixXd& input_data);

    void setScale(const Eigen::VectorXd& newScale);

    void setShift(const Eigen::VectorXd& newShift);

    Eigen::MatrixXd forward(Eigen::MatrixXd& input_data);

    // Leave the gradients as is for the scale and shift. They are cached in the Node. 
    // They will be used to update the parameters in next parallel operations.
    // the dInput is the gradient we propagate to source Nodes in the graph;
    // while the parameter gradients get cached to be used to update the parameters later.
    Eigen::MatrixXd backward(const Eigen::MatrixXd& gradients);

    void updateParameters(std::string& optimizertype, double& learningRate, int& iter);

    void forwardPass() {}
    void backwardPass() {}
};

class Activation : public BaseOperator {
private:

    // Cached data for backpropagation.
    Eigen::MatrixXd input_data; // NxW samples 

    std::string activationtype = "leakyrelu";
    float alpha = 0.01; // for leakyReLU
public:

    Activation(const std::string& activationtype = "leakyrelu") {
        this->activationtype = activationtype;
    }

    Activation(const std::string& activationtype = "leakyrelu", const float alpha=0.01) {
        this->activationtype = activationtype;
        this->alpha = alpha;
    }

    // Here, instead of using the term logits, let's just use x.
    Eigen::MatrixXd sigmoid(const Eigen::MatrixXd& x);

    /*****************************************************************************************************
     * So, the gradient of the sigmoid function with respect to its input (z) can be expressed as follows:
     * dy/dz = propagated_gradient * sigmoid(z) * (1 - sigmoid(z))
     * Or if output (y) is cached, where y = sigmoid(z)
     * then we use the output such that:
     * dy/dz = propagated_gradient * y * (1 - y)
     *****************************************************************************************************/
    Eigen::MatrixXd sigmoidGradient(const Eigen::MatrixXd& gradients, const Eigen::MatrixXd& output_data);

    Eigen::MatrixXd tanh(const Eigen::MatrixXd& x);

    /*****************************************************************************************************
     * So, the gradient of the tanh function with respect to its input (z) can be expressed as follows:
     * dy/dz = propagated_gradient * (1 - tanh(z)^2)
     * Or if output (y) is cached, where y = tanh(z)
     * then we use the output such that:
     * dy/dz = propagated_gradient * y * (1 - y^2)
     *****************************************************************************************************/
    Eigen::MatrixXd tanhGradient(const Eigen::MatrixXd& gradients, const Eigen::MatrixXd& output_data);

    Eigen::MatrixXd relu(const Eigen::MatrixXd& x);

    /*****************************************************************************************************
     * So, the gradient of the ReLU function with respect to its input (z) can be expressed as follows:
     * dy/dz = propagated_gradient * 1 for z > 0
     * dy/dz = propagated_gradient * 0 for z <= 0
     *****************************************************************************************************/
    Eigen::MatrixXd reluGradient(const Eigen::MatrixXd& gradients);  

    Eigen::MatrixXd leakyReLU(const Eigen::MatrixXd& x, float alpha);

    /*****************************************************************************************************
     * So, the gradient of the LeakyReLU function with respect to its input (z) can be expressed as follows:
     * dy/dz = propagated_gradient * 1 for z > 0
     * dy/dz = propagated_gradient * alpha for z <= 0
    *****************************************************************************************************/
    Eigen::MatrixXd leakyReluGradient(const Eigen::MatrixXd& gradients);

    Eigen::MatrixXd gelu(const Eigen::MatrixXd& x);

    /*****************************************************************************************************
     * Gelu Gradient ...
    *****************************************************************************************************/
    Eigen::MatrixXd geluGradient(const Eigen::MatrixXd& gradients);

    Eigen::MatrixXd computeActivation(const Eigen::MatrixXd& input_data);

    Eigen::MatrixXd computeGradient(const Eigen::MatrixXd& gradients, const Eigen::MatrixXd& output_data);

    Eigen::MatrixXd forward(Eigen::MatrixXd& input_data);

    Eigen::MatrixXd backward(const Eigen::MatrixXd& gradient, const Eigen::MatrixXd& output_data);

    void forwardPass() {}
    void backwardPass() {}

};

/*****************************************************************************************************
* Base Loss Functions
*****************************************************************************************************/
class Loss : public BaseOperator {
private:
    std::string losstype = "mse";
public:

    Loss(const std::string& losstype = "mse") {
        this->losstype = losstype;
    }

    // Mean Squared Error. Returns 1x1 matrix (scalar)
    Eigen::MatrixXd mse(const Eigen::MatrixXd& predicted, const Eigen::MatrixXd& target);

    Eigen::MatrixXd mseGradient(const Eigen::MatrixXd& predicted, const Eigen::MatrixXd& target);

    // Binary Cross Entropy.  Returns 1x1 matrix (scalar)
    Eigen::MatrixXd bce(const Eigen::MatrixXd& predicted, const Eigen::MatrixXd& target);

    Eigen::MatrixXd bceGradient(const Eigen::MatrixXd& predicted, const Eigen::MatrixXd& target);
    // For Loss Categorial Cross Entropy. Usually, we use Softmax.
    // If predicted and target has NxC dimension where C is number of classes, then result will be Nx1.
    Eigen::MatrixXd cce(const Eigen::MatrixXd& predicted, const Eigen::MatrixXd& target);

    Eigen::MatrixXd cceGradient(const Eigen::MatrixXd& predicted, const Eigen::MatrixXd& target);

    // For Support Vectors (not necessarily for Neural)
    Eigen::MatrixXd hingeLoss(const Eigen::MatrixXd& predicted, const Eigen::MatrixXd& target);

    Eigen::MatrixXd hingeLossGradient(const Eigen::MatrixXd& predicted, const Eigen::MatrixXd& target);

    Eigen::MatrixXd computeLoss(const Eigen::MatrixXd& predicted, const Eigen::MatrixXd& target);

    Eigen::MatrixXd computeGradients(const Eigen::MatrixXd& predicted, const Eigen::MatrixXd& target);

    void forwardPass() {}
    void backwardPass() {}
};

class Dropout: public BaseOperator {
public:
    Dropout(int size) {
        // Initialize scaling and shifting parameters
    }

    void forwardPass() {}
    void backwardPass() {}
};

#endif

#ifndef TRANSFORMER_H
#define TRANSFORMER_H

class Attention : public BaseOperator {
private:

    Eigen::MatrixXd input_data; // NxW samples, by using & 
    Linear* Q  = nullptr;
    Linear* K  = nullptr;
    Linear* V  = nullptr;
    Linear* Wo = nullptr; // this extra weight matrix will align the output dimension the same as the input.

    Eigen::MatrixXd Qout;
    Eigen::MatrixXd Kout;
    Eigen::MatrixXd Vout;

    Eigen::MatrixXd QKsoft;
    Eigen::MatrixXd QKsoftV;

    int N = 0;  // number of samples
    int M = 0;  // number of features (embedding vector size)
    int W = 0;  // number of weights (or number of features)
    int H = 1;  // number of heads
    int Dk = 0; // number of dimensions per head (M/H)

    bool bias = false;

public:
    Attention(int heads = 1, int size = 3, bool bias = false)  {
        this->W = size;
        this->H = heads;
        this->bias = bias;
        print_string("Attention operation ...", true);
    }

    // While the parameter weight has dimension MxW,  the resulting transformation has dimension of NxW.
    // We only need the M dimension from an NxM input to generate parameter matrix.
    // where weights is NxW and bias is W.
    Eigen::MatrixXd forward(Eigen::MatrixXd& input_data);

    // Leave the gradients as is. They are cached in the Node. 
    // They will be used to update the parameters in next parallel operations.
    // the dInput is the gradient we propagate to source Nodes in the graph;
    // while the parameter gradients get cached to be used to update the parameters later.
    Eigen::MatrixXd backward(Eigen::MatrixXd& gradients);

    void updateParameters(std::string& optimizertype, double& learningRate, int& iter);

    void forwardPass() {}
    void backwardPass() {}

};

/*****************************************************************************************************
* Base Multi-Head Attention Layer
*****************************************************************************************************/
class MultiAttention : public BaseOperator {
private:
    Eigen::MatrixXd input_data; // NxW samples, by using & 
    std::vector<Attention*> M1;

    bool bias = true;

    int N = 0;  // number of samples
    int M = 0;  // number of features (embedding vector size)
    int W = 0;  // number of weights (or number of features)
    int H = 1;  // number of heads
    int Dk = 0; // number of dimensions per head (M/H)
    int split = 0; // number of values in an array to jump.

    std::string activationtype = "leakyrelu";
    float alpha = 0.01; // for leakyReLU

public:
    MultiAttention(int heads = 3, int size = 3, bool bias = false)  {
        this->W = size;
        this->H = heads;
        this->bias = bias;
        // M1.setZero();
        print_string("MultiAttention operation ...", true);
    }

    // While the parameter weight has dimension MxW,  the resulting transformation has dimension of NxW.
    // We only need the M dimension from an NxM input to generate parameter matrix.
    // where weights is NxW and bias is W.
    Eigen::MatrixXd forward(Eigen::MatrixXd& input_data);

    // Leave the gradients as is. They are cached in the Node. 
    // They will be used to update the parameters in next parallel operations.
    // the dInput is the gradient we propagate to source Nodes in the graph;
    // while the parameter gradients get cached to be used to update the parameters later.
    Eigen::MatrixXd backward(Eigen::MatrixXd& gradients);

    void updateParameters(std::string& optimizertype, double& learningRate, int& iter);

    void forwardPass() {}
    void backwardPass() {}
};

/*****************************************************************************************************
* Base FeedForward  Layer
*****************************************************************************************************/
class FeedForward : public BaseOperator {
private:
    Eigen::MatrixXd input_data; // NxW samples, by using & 
    Linear* L1 = nullptr;
    Linear* L2 = nullptr; // required to align the dimension similar to the input
    Activation* A1 = nullptr;

    Eigen::MatrixXd L1out; // Cache output for use by activation backprop

    bool bias = true;
    int W = 0;
    int N = 0;
    int M = 0;

    std::string activationtype = "leakyrelu";
    float alpha = 0.01; // for leakyReLU

public:

    FeedForward(int size = 3, bool bias = true, const std::string& activationtype = "leakyrelu") {
        this->activationtype = activationtype;
        this->W = size;
        this->bias = bias;
        print_string("FeedForward operation ...", true);
    }

    FeedForward(int size = 3, bool bias = true, const std::string& activationtype = "leakyrelu", const float alpha=0.01) {
        this->activationtype = activationtype;
        this->alpha = alpha;
        this->W = size;
        this->bias = bias;
        print_string("FeedForward operation ...", true);
    }

    // While the parameter weight has dimension MxW,  the resulting transformation has dimension of NxW.
    // We only need the M dimension from an NxM input to generate parameter matrix.
    // where weights is NxW and bias is W.
    Eigen::MatrixXd forward(Eigen::MatrixXd& input_data);

    // Leave the gradients as is. They are cached in the Node. 
    // They will be used to update the parameters in next parallel operations.
    // the dInput is the gradient we propagate to source Nodes in the graph;
    // while the parameter gradients get cached to be used to update the parameters later.
    Eigen::MatrixXd backward(Eigen::MatrixXd& gradients);

    void updateParameters(std::string& optimizertype, double& learningRate, int& iter);

    void forwardPass() {}
    void backwardPass() {}
};

/*****************************************************************************************************
* Base Encoder  Layer
*****************************************************************************************************/
class Encoder : public BaseOperator {
private:
    Eigen::MatrixXd input_data; // NxW samples, by using & 
    MultiAttention* M1 = nullptr;
    LayerNorm* LN1 = nullptr;
    FeedForward* F1 = nullptr;
    LayerNorm* LN2 = nullptr;

    Eigen::MatrixXd M1out; // Cache output for use by attention backprop
    Eigen::MatrixXd F1out; // Cache output for use by feedforward backprop
    Eigen::MatrixXd LN1out; // Cache output for use by feedforward backprop

    bool bias = true;
    int H = 0;
    int W = 0;
    int N = 0;

    std::string activationtype = "leakyrelu";
    float alpha = 0.01; // for leakyReLU

public:

    Encoder(int heads = 1, int size = 3, bool bias = true, const std::string& activationtype = "leakyrelu") {
        this->activationtype = activationtype;
        this->W = size;
        this->bias = bias;
        this->H = heads;
        print_string("Encoder operation ...", true);
    }

    Encoder(int heads = 1, int size = 3, bool bias = true, const std::string& activationtype = "leakyrelu", const float alpha=0.01) {
        this->activationtype = activationtype;
        this->alpha = alpha;
        this->W = size;
        this->bias = bias;
        this->H = heads;
        print_string("Encoder operation ...", true);
    }

    // While the parameter weight has dimension MxW,  the resulting transformation has dimension of NxW.
    // We only need the M dimension from an NxM input to generate parameter matrix.
    // where weights is NxW and bias is W.
    Eigen::MatrixXd forward(Eigen::MatrixXd& input_data);

    // Leave the gradients as is. They are cached in the Node. 
    // They will be used to update the parameters in next parallel operations.
    // the dInput is the gradient we propagate to source Nodes in the graph;
    // while the parameter gradients get cached to be used to update the parameters later.
    Eigen::MatrixXd backward(Eigen::MatrixXd& gradients);

    void updateParameters(std::string& optimizertype, double& learningRate, int& iter);

    void forwardPass() {}
    void backwardPass() {}
};

class Reduction : public BaseOperator {
private:
    Eigen::MatrixXd input_data;
    std::string reducttype = "add";

public:
    Reduction(const std::string& reducttype = "add") {
        this->reducttype = reducttype;
    }

    std::string getType();

    void forwardPass() {}
    void backwardPass() {}

};

#endif

#ifndef TOPOLOGY_H
#define TOPOLOGY_H

enum class NodeType {
    Input,
    Hidden,
    Output
};

class Node {
private:
    int id;
   // Graph* graph; // Pointer to the graph that the node belongs to
    std::unordered_set<Node*> outputs;
    std::unordered_set<Node*> inputs;
    std::vector<std::shared_ptr<BaseOperator>> operations;
    Eigen::MatrixXd input_data;
    Eigen::MatrixXd output_data;
    Eigen::MatrixXd dInput;
    // Reduction* reduce_op = nullptr;
    ssize_t repeat = 1;
    std::string reduce = "add";


public:
    std::string name;
    NodeType type;

    Node(const std::string& name, NodeType type, const py::array_t<double>& embedding = {})
        : name(name), type(type) {
        if (embedding.size() != 0) {
            setData(embedding);
        } else {
            dInput.setZero();   
            input_data.setZero();  
            output_data.setZero();
        }
    }

    std::string getName();

    NodeType nodeType();

    // The input is assumed to have NxM where N=number of samples, M=embedding vector size
    // This allows to compute for the output size,  MxW where W is the number of weights (features) to use.
    void setData(py::array_t<double> embedding);

    Eigen::MatrixXd getInput();

    Eigen::MatrixXd getOutput();

    void addInput(Node* input);

    void addOutput(Node* output);

    std::unordered_set<Node*> getOutputs();

    std::unordered_set<Node*> getInputs();

    Node& setOperations(std::vector<std::shared_ptr<BaseOperator>>& operations);

    void setReduction(std::string& reducttype);

    void sequential(ssize_t repeat);

    void parallel(ssize_t repeat, std::string& reduce);

    Eigen::MatrixXd aggregateData(Eigen::MatrixXd& input_data);

    void setGradients(Eigen::MatrixXd gradients);

    void propagateGradients(Eigen::MatrixXd& gradients);

    // Because of Kahn Algorithm done (see Graph), this function runs forward pass only to 
    // nodes whose source nodes are already processed.
    Eigen::MatrixXd forwardPass();

    void backwardPass();

    void updateParameters(std::string& optimizertype, double& learningRate, int& iter);


};

class Connection {
private:
    Node* source;
    Node* destination;

public:
    Connection(Node* sourceNode, Node* destinationNode) : source(sourceNode), destination(destinationNode) {}

    Node* getSource();

    Node* getDestination();

    void forwardPass();

    Eigen::MatrixXd backwardPass(Eigen::MatrixXd& gradients);

};


class Graph {
private:
    std::vector<Node*> nodes;
    std::vector<Connection*> connections;
    std::unordered_map<Node*, int> indegree;
    std::unordered_map<Node*, int> outdegree;
    Eigen::MatrixXd input_data;
    Eigen::MatrixXd predicted;
    Eigen::MatrixXd target;
    Eigen::MatrixXd loss;

public:

    // Create a node with three arguments: name, type, and initial values
    Node* createNode(const std::string& name, NodeType type, const py::array_t<double>& embedding);

    // Create a node with two arguments: name and type (no initial values)
    Node* createNode(const std::string& name, NodeType type);

    void connect(Node* from, Node* to);

    void connect(Node* from, Node* to, std::vector<std::shared_ptr<BaseOperator>>& operations);

    void connect(std::vector<Node*> from_nodes, Node* to);

    void connect(std::vector<Node*> from_nodes, Node* to, std::vector<std::shared_ptr<BaseOperator>>& operations);

    void addConnection(Connection* connection);

    std::vector<Node*> getNodes();

    // Perform the Kahn's Algorithm by Arthur B. Khan based on his 1962 paper, "Topological Sorting of Large Networks"
    Eigen::MatrixXd forwardPropagation();

    Eigen::MatrixXd backwardPropagation(Eigen::MatrixXd& gradients);

    Eigen::MatrixXd computeLoss(std::string losstype, Eigen::MatrixXd& predicted, const Eigen::MatrixXd& target);

    Eigen::MatrixXd computeGradients(std::string losstype, Eigen::MatrixXd& predicted, const Eigen::MatrixXd& target);

    void updateParameters(std::string& optimizertype, double& learningRate, int& iter);

    void nextBatch();

    const std::unordered_map<Node*, int>& getIndegree() const;

};

#endif

#ifndef TRANSFORMER_MODEL_H
#define TRANSFORMER_MODEL_H

class BaseModel {
private:
    Graph* graph;
    Eigen::MatrixXd target;
    Eigen::MatrixXd predicted;
    Eigen::MatrixXd loss;
    Eigen::MatrixXd gradients;
    std::string losstype = "mse";
    std::string optimizertype = "adam";
    double learningRate = 0.01;
    int itermax = 1;
public:
    BaseModel(const std::string& losstype = "mse", const std::string& optimizertype = "adam",
          const double learningRate = 0.01, const int itermax = 1) {
        this->losstype = losstype;
        this->optimizertype = optimizertype;
        this->learningRate = learningRate;
        this->itermax = itermax;
    }

    void setGraph(Graph* graph);

    void setLoss(std::string& losstype);

    // The input is assumed to have NxM where N=number of samples, M=embedding vector size
    // This allows to compute for the output size,  MxW where W is the number of weights (features) to use.
    void setTarget(py::array_t<double> target);

    Eigen::MatrixXd getTarget();

    void useCrossEntropy();

    void train(std::string& losstype, std::string& optimizertype, double learnrate = 0.01, int itermax = 1);

};

#endif

#ifndef EMBEDDINGS_H
#define EMBEDDINGS_H

#include <sqlite3.h>


class Embeddings {
private:
    std::unordered_map<std::wstring, int> vocab;
    Eigen::MatrixXd wordEmbeddings;
    Eigen::VectorXd wordBiases;
    int vocabSize = 0;
    int embeddingSize = 5;

    const std::string dbFileName = "data.db";
    sqlite3* db;

    struct Record {

        // For the Vocabulary
        std::wstring token;
        int frequency;
        int tokenIndex;

        // For the Embedding
        std::string hashKey;
        Eigen::VectorXd vectorValue;
        double bias;
    };

    // Create the token hash-to-index mapping and index-to-token mapping
    std::unordered_map<std::string, int> tokenHashToIndex;
    // std::vector<std::wstring> indexToToken;

public:

    Embeddings() {  initializeVectorDB();  }

    // Initialize Vector DB
    void initializeVectorDB();

    // Seed Vector DB
    void seedVectorDB(std::unordered_map<std::wstring, int>& vocabulary);

    // Function to create simple SQLite DB
    void createVectorDB();

    // Function to create the vector table
    void createVectorTable();

    // Function to create the vocabulary table
    void createVocabularyTable();

    // Function to update a record into the vocabulary table
    void saveVocabulary(const Record& record);

    // Function to insert a record into the vector table
    void saveEmbeddings(const Record& record);

    // Function to retrieve an embedding from the database based on the hash key
    bool retrieveEmbeddings(const std::string& hashKey, Record& record);

    // Function to retrieve a record from the database based on the hash key
    // TODO: To use some kind of memcache
    bool retrieveVocabulary(const std::wstring& token, Record& record);

    // Function to retrieve a record from the database based on the hash key
    // TODO: To use some kind of memcache
    bool isInVocabulary(const std::wstring& token);

    bool isInVocabulary(const std::wstring& token, Record& record);

    // Function to get the vocabulary
    const std::unordered_map<std::wstring, int>& getVocabulary() const { return this->vocab; }

    // Function to seed the Vector and Vocabulary Tables
    void initializeVectorandVocabMetadata(std::unordered_map<std::wstring, int>& vocabulary, int embeddingSize);

    // Cross Reference Vocabulary
    void crossReferenceVocabularyinDBandCache(std::unordered_map<std::wstring, int>& vocabulary);

    // Function to fetch the embeddings for tokens in the current corpus from the vector database
    void prefetchVocabularyToCache(const std::vector<std::vector<std::wstring>>& corpus);

    // Function to fetch the embeddings for tokens in the cached vocabulary (instead of corpus) from the vector database to cache
    void prefetchEmbeddingsToCache();

    // Update Embeddings in the Database
    void updateEmbeddingsInDatabase(const Eigen::MatrixXd& wordEmbeddings,
                                    const Eigen::VectorXd& wordBiases);

    // Get Vocab Size and Embedding Size
    int getVocabSize() { return this->wordEmbeddings.rows(); }
    int getEmbeddingSize() { return this->wordEmbeddings.cols(); }

    // Get Embeddings and indcies
    Eigen::MatrixXd& getWordEmbeddings() { return this->wordEmbeddings; }
    Eigen::VectorXd& getWordBiases() { return this->wordBiases; }
    std::unordered_map<std::string, int>& getTokenIndex() { return this->tokenHashToIndex; }


};

#endif

#ifndef TOKENIZER_MODEL_H
#define TOKENIZER_MODEL_H

#include <sqlite3.h>

struct TrieNode {
    std::unordered_map<wchar_t, TrieNode*> children;
    bool isEndOfToken;
};

class TokenModel {
private:

    std::string losstype = "mse";
    std::string optimizertype = "adagrad";
    float learningRate = 0.01;
    int maxIterations = 1;
    float regularization = 1.0;
    double clipThreshold = 5.0;

public:

    Embeddings* embeddings;

    std::unordered_map<std::wstring, int> vocab;
    Eigen::MatrixXd wordEmbeddings;
    Eigen::VectorXd wordBiases;
    int vocabSize = 0;
    int embeddingSize = 5;

    bool resetVocab = false;

    // Create the token hash-to-index mapping and index-to-token mapping
    // std::unordered_map<std::string, int> tokenHashToIndex;
    // std::vector<std::wstring> indexToToken;

    // Dynamic Embeddings Data Structure (Hash Table)
    // std::unordered_map<std::string, Eigen::VectorXd> dynamicEmbeddings;

    struct Record {

        // For the Vocabulary
        std::wstring token;
        int frequency;
        int tokenIndex;

        // For the Embedding
        std::string hashKey;
        Eigen::VectorXd vectorValue;

    };

    TokenModel(const std::string& losstype = "mse", const std::string& optimizertype = "adagrad",
          const double learningRate = 0.01, float regularization = 1.0,
          const int maxIterations = 1,  double clipThreshold = 5.0) {

        this->losstype = losstype;
        this->optimizertype = optimizertype;
        this->learningRate = learningRate;
        this->maxIterations = maxIterations;
        this->clipThreshold = clipThreshold;
    }

    // Function to print the vocabulary
    void printVocabulary(int rows);

    // Function to print the word embedding
    void printWordEmbeddings(int rows);

    // Helper function to split string into words.
    std::vector<std::wstring> splitString(const std::wstring& str);

    // Tokenize new sentences. The tokens are compared against the vocabulary to generate the final tokens.
    // The final tokens are then used to generate the initial word embeddings for later training (e.g. trainGloVe())
    std::vector<std::wstring> tokenize(const std::wstring& sentence);

    // Overload function to perform the same tokenization but for multiple sentences.
    std::vector<std::vector<std::wstring>> tokenize(const std::vector<std::wstring>& sentences);

    // Now train a GloVe model
     void trainGloVe(std::vector<std::wstring>& sentences, int batchSize = 2, 
                    float learningRate = 0.01, int maxIteration = 1);

};

class BPETokenizer : public TokenModel {
private:
    TrieNode* root;
public:
    BPETokenizer() {
        root = new TrieNode();
    }

    // Build or Reset Vocabulary.
    void buildVocabulary(const std::vector<std::wstring>& sentences, int numMerges);

    // Helper function to determine if suffix exists in a string.
    bool endsWith(const std::wstring& str, const std::wstring& suffix);

    // Tokenize the corpus. The result is fed to the mergeTokens to construct the vocabulary.
    std::vector<std::wstring> tokenizeCorpus(const std::vector<std::wstring>& corpus);

    // Part of Byte Pair Encoding is to merge tokens that have the highest frequency.
    void mergeTokens(std::vector<std::wstring>& tokens, int numMerges);

    // Pretrain BPE Tokenizer
    void pretrain(const std::vector<std::wstring>& sentences, int numMerges,  int embeddingSize);

    // Train BPE Tokenizer
    void train(const std::vector<std::wstring>& sentences, int numMerges);

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