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

#include "logger.h"
#include "operators.h"

#ifndef TOPOLOGY_H
#define TOPOLOGY_H

enum class NodeType {
    Input,
    Hidden,
    Output
};

/*****************************************************************************************************
* Node
*****************************************************************************************************/
template <class T>
class Node {
private:
    int id;
   // Graph* graph; // Pointer to the graph that the node belongs to
    std::unordered_set<std::shared_ptr<Node<T>>> outputs;
    std::unordered_set<std::shared_ptr<Node<T>>> inputs;
    std::vector<std::shared_ptr<BaseOperator>> operations;
    aitensor<T> input_data;
    aitensor<T> output_data;
    aitensor<T> gradients;
    ssize_t repeat = 1;
    std::string reduce = "add";

    // Handles Tensor
    aitensor<T> input_data_tensor;
    std::vector<aitensor<T>> dInput_vector;
    bool tensor = false;

    // If Node has other input sources, count the number of sources.
    aiscalar<T> suminputs = 0.0;

public:
    std::string name;
    NodeType type;

    Node(const std::string& name, NodeType type, const py::array_t<T>& embedding = {})
        : name(name), type(type) {
        if (embedding.size() != 0) {
         //   setData(embedding);
        } else {
            gradients.setZero();   
            input_data.setZero();  
            output_data.setZero();
        }
        log_info( "**** Node instance created ****" );
    }

    std::string getName();

    NodeType nodeType();

    // The input is assumed to have BxNxM where B=Batch, N=number of samples, M=embedding vector size
    // This allows to compute for the output size,  MxW where W is the number of weights (features) to use.
    void setData(const py::array_t<T>& embedding);

    // Let's handle Tensors
    //void setDataTensor(const py::array_t<T>& embedding);

    const aitensor<T>& getInput();

    const aitensor<T>& getOutput();

    void addInput(std::shared_ptr<Node<T>> input, std::shared_ptr<Node<T>> output);

    void addOutput(std::shared_ptr<Node<T>> output, std::shared_ptr<Node<T>> input);

    std::unordered_set<std::shared_ptr<Node<T>>> getOutputs();

    std::unordered_set<std::shared_ptr<Node<T>>> getInputs();

    std::shared_ptr<Node<T>> setOperations(std::vector<std::shared_ptr<BaseOperator>>& operations);

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

    std::string generateDotFormat();


};

template <class T>
class Connection {
private:
    std::shared_ptr<Node<T>> source;
    std::shared_ptr<Node<T>> destination;

public:
    Connection(std::shared_ptr<Node<T>> sourceNode, std::shared_ptr<Node<T>> destinationNode) : source(sourceNode), destination(destinationNode) {}

    std::shared_ptr<Node<T>> getSource();

    std::shared_ptr<Node<T>> getDestination();

    void forwardPass();

    const aitensor<T> backwardPass(const aitensor<T>& gradients);

};

template <class T>
class Graph {
private:
    std::vector<std::shared_ptr<Node<T>>> nodes;
    std::vector<std::shared_ptr<Connection<T>>> connections;
    std::unordered_map<std::shared_ptr<Node<T>>, int> indegree;
    std::unordered_map<std::shared_ptr<Node<T>>, int> outdegree;
    aitensor<T> input_data;
    aitensor<T> predicted;
    aitensor<T> target;
    aiscalar<T> loss;

    Loss<T>* lossobj;

public:

    // Create a node with three arguments: name, type, and initial values
    // Node<T>* createNode(const std::string& name, NodeType type, const py::array_t<T>& embedding);

    // Create a node with two arguments: name and type (no initial values)
    std::shared_ptr<Node<T>> createNode(const std::string& name, NodeType type);

    void connect(std::shared_ptr<Node<T>> from, std::shared_ptr<Node<T>> to);

    void connect(std::shared_ptr<Node<T>> from, std::shared_ptr<Node<T>> to, std::vector<std::shared_ptr<BaseOperator>>& operations);

    void connect(std::vector<std::shared_ptr<Node<T>>> from_nodes, std::shared_ptr<Node<T>> to);

    void connect(std::vector<std::shared_ptr<Node<T>>> from_nodes, std::shared_ptr<Node<T>> to, std::vector<std::shared_ptr<BaseOperator>>& operations);

    void addConnection(std::shared_ptr<Connection<T>> connection);

    std::vector<std::shared_ptr<Node<T>>> getNodes();

    // Perform the Kahn's Algorithm by Arthur B. Khan based on his 1962 paper, "Topological Sorting of Large Networks"
    const aitensor<T> forwardPropagation();

    const aitensor<T> backwardPropagation(const aitensor<T>& gradients);

    const aiscalar<T> computeLoss(std::string losstype, const aitensor<T>& predicted, const aitensor<T>& target);

    const aitensor<T> computeGradients(const aitensor<T>& predicted, const aitensor<T>& target);

    void updateParameters(std::string& optimizertype, T& learningRate, int& iter);

    void nextBatch();

    const std::unordered_map<std::shared_ptr<Node<T>>, int>& getIndegree() const;

    std::string generateDotFormat();

};

#endif
