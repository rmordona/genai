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
#include "genai.h"
#include "logger.h"
#include "topology.h"
#include "model.h"
#include "tokenmodel.h"
#include "scraper.h"

// Define PY as the static member instance of PythonStream
PythonStream py_cout;

void log_msg(const std::string& text) {
    std::cout << text << std::endl;
}
 
double inf() {
    return std::numeric_limits<double>::infinity();
}

void print_string(const std::string& text, bool printNextLine) {
    // std::string xyz = "";

    if (printNextLine) {
        py::print(text);
    } else {
        py::print(text, py::arg("end") = "");
    }
}

void print_double(double value, bool printNextLine) {
    if (printNextLine) {
        py::print(value);
    } else {
        py::print(value, py::arg("end") = "");
    }
}

std::string scalar_to_string(const float& value) {
      return std::to_string(value);
}

std::string scalar_to_string(const double& value) {
      return std::to_string(value);
}

std::string wstringToUtf8(const std::wstring& wstr) {
    std::string utf8Str;
    utf8::utf16to8(wstr.begin(), wstr.end(), std::back_inserter(utf8Str));
    return utf8Str;
}

std::wstring utf8ToWstring(const std::string& utf8Str) {
    std::wstring wstr;
    utf8::utf8to16(utf8Str.begin(), utf8Str.end(), std::back_inserter(wstr));
    return wstr;
}

// Function to serialize std::wstring to UTF-8 encoded std::vector<unsigned char>
std::vector<unsigned char> serializeWString(const std::wstring& wstr) {
    std::vector<unsigned char> bytes;
    bytes.reserve(wstr.size() * 4); // Reserve enough space for UTF-8 encoding (up to 4 bytes per character)

    for (const wchar_t& wc : wstr) {
        if (wc <= 0x7F) {
            bytes.push_back(static_cast<unsigned char>(wc));
        } else if (wc <= 0x7FF) {
            bytes.push_back(0xC0 | ((wc >> 6) & 0x1F));
            bytes.push_back(0x80 | (wc & 0x3F));
        } else if (wc <= 0xFFFF) {
            bytes.push_back(0xE0 | ((wc >> 12) & 0x0F));
            bytes.push_back(0x80 | ((wc >> 6) & 0x3F));
            bytes.push_back(0x80 | (wc & 0x3F));
        } else {
            bytes.push_back(0xF0 | ((wc >> 18) & 0x07));
            bytes.push_back(0x80 | ((wc >> 12) & 0x3F));
            bytes.push_back(0x80 | ((wc >> 6) & 0x3F));
            bytes.push_back(0x80 | (wc & 0x3F));
        }
    }

    return bytes;
}

// Function to deserialize UTF-8 encoded std::vector<unsigned char> back to std::wstring
std::wstring deserializeWString(const std::vector<unsigned char>& bytes) {
    std::wstring wstr;
    wstr.reserve(bytes.size()); // Reserve enough space for the wide characters

    for (size_t i = 0; i < bytes.size();) {
        wchar_t wc;
        if ((bytes[i] & 0x80) == 0x00) {
            wc = static_cast<wchar_t>(bytes[i]);
            i += 1;
        } else if ((bytes[i] & 0xE0) == 0xC0) {
            wc = static_cast<wchar_t>((bytes[i] & 0x1F) << 6 | (bytes[i + 1] & 0x3F));
            i += 2;
        } else if ((bytes[i] & 0xF0) == 0xE0) {
            wc = static_cast<wchar_t>((bytes[i] & 0x0F) << 12 | (bytes[i + 1] & 0x3F) << 6 | (bytes[i + 2] & 0x3F));
            i += 3;
        } else if ((bytes[i] & 0xF8) == 0xF0) {
            wc = static_cast<wchar_t>((bytes[i] & 0x07) << 18 | (bytes[i + 1] & 0x3F) << 12 | (bytes[i + 2] & 0x3F) << 6 | (bytes[i + 3] & 0x3F));
            i += 4;
        } else {
            // Invalid UTF-8 sequence, handle the error if needed
            break;
        }

        wstr.push_back(wc);
    }

    return wstr;
}

// Function to calculate SHA-256 and return it as a string
std::string sha256(const std::wstring& data) {
    EVP_MD_CTX* mdctx;
    const EVP_MD* md;
    unsigned int md_len;
    unsigned char digest[EVP_MAX_MD_SIZE]; // To store the resulting digest

    // Convert the wide string to a byte sequence (unsigned char array)
    std::string data_bytes(reinterpret_cast<const char*>(data.c_str()), data.length() * sizeof(wchar_t));

    // Initialize the EVP context
    mdctx = EVP_MD_CTX_new();

    // Choose the digest algorithm (EVP_sha256 for SHA-256)
    md = EVP_sha256();

    // Initialize the digest calculation
    EVP_DigestInit_ex(mdctx, md, NULL);

    // Perform the digest calculation
    EVP_DigestUpdate(mdctx, reinterpret_cast<const unsigned char*>(data_bytes.c_str()), data_bytes.length());

    // Finalize the digest and store the result in 'digest'
    EVP_DigestFinal_ex(mdctx, digest, &md_len);

    // Clean up the EVP context
    EVP_MD_CTX_free(mdctx);

    std::stringstream ss;
    ss << std::hex << std::setfill('0');
    for (unsigned int i = 0; i < md_len; ++i) {
        ss << std::setw(2) << static_cast<int>(digest[i]);
    }

    return ss.str();
}


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

double* allocate_matrix(ssize_t rows, ssize_t cols) {
    // Allocate memory for the matrix
    double* matrix = (double*)malloc(rows * cols * sizeof(double));
    return matrix;
}

void process_array1(py::array_t<double> inputArray) {
    // Access the underlying NumPy array data
    py::buffer_info bufInfo = inputArray.request();
    double* dataPtr = static_cast<double*>(bufInfo.ptr);

    // Access the shape and strides of the array
    std::vector<size_t> shape(bufInfo.shape.begin(), bufInfo.shape.end());
    std::vector<size_t> strides(bufInfo.strides.begin(), bufInfo.strides.end());

    // Iterate over the array elements
    for (size_t i = 0; i < shape[0]; ++i) {
        for (size_t j = 0; j < shape[1]; ++j) {
            // Access the element at (i, j)
            double value = dataPtr[i * strides[0] + j * strides[1]];
            // Process the element as needed
            // ...
            print_double(value, true);
        }
    }
}

void process_array(py::array_t<double> inputArray) {
    py::print("Processing array:");

    py::buffer_info bufInfo = inputArray.request();

    // Get the shape of the array
    std::vector<size_t> shape(bufInfo.shape.begin(), bufInfo.shape.end());

    // Get the shape of the array
    // std::vector<size_t> shape = bufInfo.shape;

    // Determine the number of dimensions
    size_t numDimensions = shape.size();

    // Print the dimensions
    py::print("Number of dimensions:", numDimensions);
    py::print("Shape:", shape);

}

void process_matrix(py::array_t<double> inputMatrix) {
    py::print("Processing matrix:");

    auto bufInfo = inputMatrix.request();
    double* dataPtr = static_cast<double*>(bufInfo.ptr);

    ssize_t rows = bufInfo.shape[0];
    ssize_t cols = bufInfo.shape[1];

    for (ssize_t i = 0; i < rows; ++i) {
        for (ssize_t j = 0; j < cols; ++j) {
            double value = dataPtr[i * cols + j];
            // py::print(value);
            print_double(value, false);
            print_string(" ", false);
        }
        print_string("", true);
    }
}

py::array_t<double>  matmul(py::array_t<double> A, py::array_t<double> B) {
    py::print("Processing matrix:");

    auto bufInfoA = A.request();
    double* matA = static_cast<double*>(bufInfoA.ptr);

    auto bufInfoB = B.request();
    double* matB = static_cast<double*>(bufInfoB.ptr);

    ssize_t rows = bufInfoA.shape[0];
    ssize_t cols = bufInfoA.shape[1];

    for (ssize_t i = 0; i < rows; ++i) {
        for (ssize_t j = 0; j < cols; ++j) {
            double value = matA[i * cols + j];
            print_double(value, false);
            print_string(" ", false);
        }
        print_string("", true);
    }

    for (ssize_t i = 0; i < rows; ++i) {
        for (ssize_t j = 0; j < cols; ++j) {
            double value = matB[i * cols + j];
            print_double(value, false);
            print_string(" ", false);
        }
        print_string("", true);
    }

    int rows_a = rows;
    int cols_a = cols;
    int cols_b = cols;
  
    // Create a new NumPy array with the same shape as the matrix
    // This actually allocates memory.
    py::array_t<double> result({rows, cols});

    // Get a pointer to the underlying data of the NumPy array
    double* matC = result.mutable_data();

    // otherwise, allocate manually
    // double* matC = allocate_matrix(rows, cols);

    float alpha = 1.0;
    float beta  = 0.0;
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, rows_a, cols_b, cols_a, alpha,
                (double *) matA, cols_a,
                (double *) matB, cols_b, beta,
                (double *) matC, cols_b);

    return result;
}
 
LOGGER* ai_log;

PYBIND11_MODULE(genai, m) {
    m.doc() = "Example C++ module for Python";

    ai_log = new LOGGER();

    log_tag("GENAI");
    log_info( "******************************************************************************************" );
    log_info( "*************************** GENAI Module Loading *****************************************")
    log_info( "******************************************************************************************" );

    py::enum_<NodeType>(m, "NodeType")
        .value("Input", NodeType::Input)
        .value("Hidden", NodeType::Hidden)
        .value("Output", NodeType::Output)
        .value("Generic", NodeType::Generic)
        .export_values();

    py::enum_<RNNType>(m, "RNNtype")
        .value("MANY_TO_ONE", RNNType::MANY_TO_ONE)
        .value("ONE_TO_MANY", RNNType::ONE_TO_MANY)
        .value("MANY_TO_MANY", RNNType::MANY_TO_MANY)
        .export_values();

    py::enum_<ReductionType>(m, "ReductionType")
        .value("SUM", ReductionType::SUM)
        .value("AVG", ReductionType::AVG)
        .value("MIN", ReductionType::MIN)
        .value("MAX", ReductionType::MAX)
        .value("ARGMIN", ReductionType::ARGMIN)
        .value("ARGMAX", ReductionType::ARGMAX)
        .value("MATMUL", ReductionType::MATMUL)
        .value("MUL", ReductionType::MUL)
        .export_values();

      py::enum_<ActivationType>(m, "ActivationType")
        .value("SIGMOID", ActivationType::SIGMOID)
        .value("TANH", ActivationType::TANH)
        .value("LEAKYRELU", ActivationType::LEAKYRELU)
        .value("GELU", ActivationType::GELU)
        .value("SOFTMAX", ActivationType::SOFTMAX)
        .export_values();
   
    py::class_<SampleClass, std::shared_ptr<SampleClass>>(m, "SampleClass")
        .def(py::init<double>())
        .def(py::init<float>());


    py::class_<ModelNode, std::shared_ptr<ModelNode>>(m, "Node")
        .def("setOperations", (void (ModelNode::*)(std::vector<std::shared_ptr<BaseOperator>>&)) &ModelNode::setOperations)
        .def("setData", (void (ModelNode::*)(const py::array_t<double>&, const bool)) &ModelNode::setDataDouble, 
                py::arg("data"), py::arg("normalize") = false, "Function with double argument")
        .def("setData", (void (ModelNode::*)(const py::array_t<float>&, const bool)) &ModelNode::setDataFloat, 
                py::arg("data"), py::arg("normalize") = false, "Function with float argument")
            .def("setDecoderData", (void (ModelNode::*)(const py::array_t<double>&, const bool)) &ModelNode::setDecoderDataDouble, 
                py::arg("data"), py::arg("normalize") = false, "Function with double argument")
        .def("setDecoderData", (void (ModelNode::*)(const py::array_t<float>&, const bool)) &ModelNode::setDecoderDataFloat, 
                py::arg("data"), py::arg("normalize") = false, "Function with float argument");

    
    py::class_<BaseOperator, std::shared_ptr<BaseOperator>>(m, "BaseOperator");
    py::class_<ModelLinear, BaseOperator, std::shared_ptr<ModelLinear>>(m, "Dense")
        .def(py::init<int, bool>(), py::arg("size") = 0, py::arg("bias") = true);
    py::class_<ModelBatchNorm, BaseOperator, std::shared_ptr<ModelBatchNorm>>(m, "BatchNorm")
        .def(py::init<>());
    py::class_<ModelLayerNorm, BaseOperator, std::shared_ptr<ModelLayerNorm>>(m, "LayerNorm")
        .def(py::init<>()); 
    py::class_<ModelReduction, BaseOperator, std::shared_ptr<ModelReduction>>(m, "Reduction")
        .def(py::init<const std::string&>(), py::arg("type") = "sum");  
    py::class_<ModelActivation, BaseOperator, std::shared_ptr<ModelActivation>>(m, "Activation")
        .def(py::init<const std::string&, const float>(), py::arg("type") = "relu", py::arg("alpha") = 0.01);
    py::class_<ModelDropout, BaseOperator, std::shared_ptr<ModelDropout>>(m, "Dropout")
        .def(py::init<const float>(), py::arg("probability") = 0.05);
    py::class_<ModelFlatten, BaseOperator, std::shared_ptr<ModelFlatten>>(m, "Flatten")
        .def(py::init<>());
    py::class_<ModelConvolution, BaseOperator, std::shared_ptr<ModelConvolution>>(m, "Convolution")
        .def(py::init<const int, const int, const int, const int, bool>(), py::arg("kernel_size") = 2,
        py::arg("stride") = 1, py::arg("padding") =1, py::arg("dilation") = 1, py::arg("bias") = true);
    py::class_<ModelAttention, BaseOperator, std::shared_ptr<ModelAttention>>(m, "Attention")
        .def(py::init<int, bool, bool>(), py::arg("size") = 3, py::arg("bias") = false, py::arg("masked") = false); 
    py::class_<ModelFeedForward, BaseOperator, std::shared_ptr<ModelFeedForward>>(m, "FeedForward")
        .def(py::init<int, bool, const std::string&, const float>(), 
                py::arg("size") = 3, py::arg("bias") = true,
                py::arg("type") = "relu", py::arg("alpha") = 0.01);
    py::class_<ModelEncoder, BaseOperator, std::shared_ptr<ModelEncoder>>(m, "Encoder")
        .def(py::init<int, int, bool, const std::string&, const float>(), 
                py::arg("heads") = 1,
                py::arg("size") = 3, 
                py::arg("bias") = true,
                py::arg("type") = "relu", 
                py::arg("alpha") = 0.01);
    py::class_<ModelDecoder, BaseOperator, std::shared_ptr<ModelDecoder>>(m, "Decoder")
        .def(py::init<int, int, bool, const std::string&, const float>(), 
                py::arg("heads") = 1,
                py::arg("size") = 3, 
                py::arg("bias") = true,
                py::arg("type") = "relu", 
                py::arg("alpha") = 0.01);
    py::class_<ModelRNN, BaseOperator, std::shared_ptr<ModelRNN>>(m, "RNN")
        .def(py::init<int, int, int, int, bool, RNNType>(), 
                py::arg("hidden_size") = 1,
                py::arg("output_size") = 1,
                py::arg("output_sequence_length") = 0, 
                py::arg("num_layers") = 1, 
                py::arg("bidirectional") = true,
                py::arg("rnntype") = RNNType::MANY_TO_MANY);

    py::class_<ModelLSTM, BaseOperator, std::shared_ptr<ModelLSTM>>(m, "LSTM")
        .def(py::init<int, int, int, int, bool, RNNType>(), 
                py::arg("hidden_size") = 1,
                py::arg("output_size") = 1, 
                py::arg("output_sequence_length") = 0,
                py::arg("num_layers") = 1, 
                py::arg("bidirectional") = true,
                py::arg("rnntype") = RNNType::MANY_TO_MANY);

    py::class_<ModelGRU, BaseOperator, std::shared_ptr<ModelGRU>>(m, "GRU")
        .def(py::init<int, int, int, int, bool, RNNType>(), 
                py::arg("hidden_size") = 1,
                py::arg("output_size") = 1,
                py::arg("output_sequence_length") = 0,  
                py::arg("num_layers") = 1, 
                py::arg("bidirectional") = true,
                py::arg("rnntype") = RNNType::MANY_TO_MANY);
 
    py::class_<Model>(m, "Model")
        .def(py::init<const std::string&, const std::string&, const double, const int, const std::string&>(), 
                py::arg("losstype") = "mse", py::arg("optimizertype") = "adam",
                py::arg("learning_rate") = 0.01, py::arg("max_epoch") = 1, py::arg("datatype") = "float")
        .def("addNode", (std::shared_ptr<ModelNode> (Model::*)(const std::string&, NodeType)) &Model::addNode,
                  py::arg("name"),  py::arg("nodetype"), "Add Node To Graph")
        .def("connect", (void (Model::*)(std::shared_ptr<ModelNode>,std::shared_ptr<ModelNode>)) &Model::connect, "Connects this node to another node")
        .def("connect", (void (Model::*)(std::vector<std::shared_ptr<ModelNode>>, std::shared_ptr<ModelNode>)) &Model::connect, "Connects this node from multiple nodes")
        .def("connect", (void (Model::*)(std::shared_ptr<ModelNode>, std::vector<std::shared_ptr<ModelNode>>)) &Model::connect, "Connects this node to multiple nodes")
        .def("setTarget", (void (Model::*)(const py::array_t<double>&)) &Model::setTargetDouble, py::arg("data"), "Function with double argument")
        .def("setTarget", (void (Model::*)(const py::array_t<float>&)) &Model::setTargetFloat, py::arg("data"), "Function with float argument")
        .def("getPredictions", (py::array_t<double> (Model::*)()) &Model::getPredictionsDouble, "Function with double argument")
        .def("getPredictions", (py::array_t<float> (Model::*)()) &Model::getPredictionsFloat, "Function with float argument")
        .def("train", &Model::train, py::arg("loss") = "mse",  
                py::arg("metrics"),  py::arg("optimizer") = "adam", py::arg("learn_rate") = 0.01, 
                py::arg("max_epoch")=1, "Training a model")
        .def("generateDotFormat", (std::string (Model::*)(bool, bool)) &Model::generateDotFormat,
                py::arg("operators") = true, py::arg("weights") = true);
     
    // Definitions for TokenModel APIs
    py::class_<TokenModel>(m, "TokenModel")
        .def(py::init<const std::string&, const std::string&>(), py::arg("tokenizer") = "bpetokenizer", py::arg("datatype") = "float")
        .def("tokenize",  (std::vector<std::wstring> (TokenModel::*)(const std::wstring&)) &TokenModel::tokenize, 
                "Tokenize a Sentence")
        .def("tokenize",  (std::vector<std::vector<std::wstring>> (TokenModel::*)(const std::vector<std::wstring>&)) &TokenModel::tokenize, 
                "Tokenize a set of Sentences")
        .def("preload", (void (TokenModel::*)(const std::vector<std::wstring>&, int, int)) &TokenModel::preload, 
            py::arg("corpus"), py::arg("merges") = 2, py::arg("size") = 5, "Train a BPE tokenizer")
        .def("merge", (void (TokenModel::*)(const std::vector<std::wstring>&, int)) &TokenModel::merge, 
            py::arg("corpus"), py::arg("merges"), "Merge Tokens from new Corpus")
        .def("train", (void (TokenModel::*)(const std::vector<std::wstring>&, int, const std::string&, const std::string&,
                             double, int, double, double)) &TokenModel::train,
            py::arg("corpus"),  py::arg("batchsize"), py::arg("losstype") = "mse", py::arg("optimizertype") = "adam",
            py::arg("learningrate") = 0.01, py::arg("maxiteration") = 1, 
            py::arg("clipthreshold"), py::arg("regularization"), "Train Word Embedding using GloVe");

    // Definitions for Scraper APIs
    py::class_<Scraper>(m, "Scraper")
        .def(py::init<>())
        .def("crawl", (void (Scraper::*)(std::string&, int)) &Scraper::crawl, py::arg("url"), py::arg("depth") = 0, "Simple crawler");
  
    // Define function to print hello
    m.def("print_string", &print_string, "Print 'string'");
    m.def("print_double", &print_double, "Print 'double'");
    m.def("process_array", &process_array, "Process a NumPy array");
    m.def("process_matrix", &process_matrix, "Process a NumPy array");
    m.def("matmul", &matmul, "Matrix Multiplication a NumPy array");

    // Definitions for URLFrontier APIs
    py::class_<URLFrontier>(m, "URLFrontier")
        .def(py::init<int, const std::string&>(), py::arg("max_urls"), py::arg("queue_address"))
        .def("enqueue",  (void (URLFrontier::*)(const std::string&, int)) &URLFrontier::enqueue, 
                py::arg("url"), py::arg("priority_percentage"))
        .def("start_worker_threads", (void (URLFrontier::*)()) &URLFrontier::startWorkerThread);

    // Set std::cout precision display
    std::cout.precision(12);

 
}
