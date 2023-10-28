/*
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

#include "operators.h"
#include "transformer.h"
#include "recurrent.h"
#include "convolution.h"

#ifndef TOPOLOGY_H
#define TOPOLOGY_H

enum class NodeType {
    Input,
    Hidden,
    Output,
    Generic
};


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

};

/*****************************************************************************************************
* Node
*****************************************************************************************************/
template <class T>
class Node {
private:
    std::string name = "";
    NodeType ntype;

    int id;
    std::unordered_set<Node<T>*> outputs;
    std::unordered_set<Node<T>*> inputs;
    std::vector<BaseOperator*> operations;
    aitensor<T> input_data;
    aitensor<T> decoder_data;
    aitensor<T> output_data;
    aitensor<T> gradients;
    ssize_t repeat = 1;
    std::string reduce = "add";

    // If Node has other input sources, count the number of sources.
    T suminputs = 0.0;

public:

    Node(std::string name, NodeType ntype) : name(name), ntype(ntype)  {
        log_detail( "**** Node [{0}] instance created ****", name );
    }

    ~Node() {
    }

    std::string getName() { return this->name; }

    NodeType nodeType() { return this->ntype; }

    // The input is assumed to have BxNxM where B=Batch, N=number of samples, M=embedding vector size
    // This allows to compute for the output size,  MxW where W is the number of weights (features) to use.
    void setData(const py::array_t<T>& data, const bool normalize);

    // Invoked by Graph.setData from Model.setData class
    void setData(const aitensor<T> data, const bool normalize);

    void setDecoderData(const py::array_t<T>& data, const bool normalize);

    // Invoked by Graph.setData from Model.setData class
    void setDecoderData(const aitensor<T> data, const bool normalize);

    // Let's handle Tensors
    //void setDataTensor(const py::array_t<T>& embedding);

    const aitensor<T>& getInput();

    const aitensor<T>& getOutput();

    void addInput(Node<T>* input, Node<T>* output);

    void addOutput(Node<T>* output, Node<T>* input);

    std::unordered_set<Node<T>*> getOutputs();

    std::unordered_set<Node<T>*> getInputs();

    Node<T>* setOperations(std::vector<BaseOperator*>& operations);

    void setReduction(std::string& reducttype);

    void sequential(ssize_t repeat);

    void parallel(ssize_t repeat, std::string& reduce);

    const aitensor<T> aggregateData(const aitensor<T>& input_data);

    void setGradients(const aitensor<T>&  gradients);

    void propagateGradients(const aitensor<T>&  gradients);

    // Because of Kahn Algorithm done (see Graph), this function runs forward pass only to 
    // nodes whose source nodes are already processed.
    void forwardPass();

    void backwardPass();

    void updateParameters(std::string& optimizertype, T& learningRate, int& iter);

    std::string generateDotFormat(bool operators = false, bool weights = false);


};

template <class T>
class Connection {
private:
    Node<T>* source;
    Node<T>* destination;

public:
    Connection(Node<T>* sourceNode, Node<T>* destinationNode) : source(sourceNode), destination(destinationNode) {}

    Node<T>* getSource();

    Node<T>* getDestination();

    void forwardPass();

    const aitensor<T> backwardPass(const aitensor<T>& gradients);

};

template <class T>
class Graph {
private:
    std::vector<Node<T>*> nodes;
    std::vector<std::shared_ptr<Connection<T>>> connections;
    std::unordered_map<Node<T>*, int> indegree;
    std::unordered_map<Node<T>*, int> outdegree;
    aitensor<T> input_data;
    aitensor<T> predicted;
    aitensor<T> target;
    aiscalar<T> loss;
    aiscalar<T> metrics;

    Loss<T>* lossobj;
    Metrics<T>* metricsobj;

public:

    ~Graph() { // release nodes.
       // for (auto& node : nodes) {
         //   delete node;
      //  }
    }

    Node<T>* findNode(const std::string& nodename);

    void setData(const std::string& nodename, const aitensor<T>& data, const bool normalize);

    void setData(const std::string& nodename, const py::array_t<T>& data, const bool normalize);

    void setDecoderData(const std::string& nodename, const aitensor<T>& data, const bool normalize);

    void setDecoderData(const std::string& nodename, const py::array_t<T>& data, const bool normalize);

    void setOperations(const std::string& nodename, std::vector<BaseOperator*> operations);

    const std::vector<Node<T>*> getNodes() { return this->nodes; }

    // Create a node with two arguments: name and type (no initial values)
    Node<T>* createNode(const std::string& name, NodeType type);

    void connect(std::string from_name, std::string to_name);

    void connect(Node<T>* from, Node<T>* to);

    void connect(Node<T>* from, Node<T>* to, std::vector<BaseOperator*>& operations);

    void connect(std::vector<Node<T>*> from_nodes, Node<T>* to);

    void connect(Node<T>* from, std::vector<Node<T>*> to_nodes);

    void connect(std::vector<Node<T>*> from_nodes, Node<T>* to, std::vector<BaseOperator*>& operations);

    void addConnection(std::shared_ptr<Connection<T>> connection);

    // Perform the Kahn's Algorithm by Arthur B. Khan based on his 1962 paper, "Topological Sorting of Large Networks"
    const aitensor<T> forwardPropagation();

    const aitensor<T> backwardPropagation(const aitensor<T>& gradients);

    const aiscalar<T> computeLoss(const std::string& losstype, const aitensor<T>& predicted, const aitensor<T>& target);

    const aitensor<T> computeGradients(const std::string& losstype, const aitensor<T>& predicted, const aitensor<T>& target);

    const PerfMetrics<T> computeMetrics(const std::vector<std::string>& metricstype, const aitensor<T>& predicted, const aitensor<T>& target);

    void updateParameters(std::string& optimizertype, T& learningRate, int& iter);

    void nextBatch();

    const std::unordered_map<Node<T>*, int>& getIndegree() const;

    std::string generateDotFormat(bool operators = false, bool weights = false);

};

#endif
