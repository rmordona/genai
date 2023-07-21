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
 *
*/
#include "genai.h"

namespace py = pybind11;
using namespace py::literals;


/********************************************************************************************
* NodeFactory
********************************************************************************************/

std::string Node::getName() {
    return this->name;
}

NodeType Node::nodeType() {
    return this->type;
}

// The input is assumed to have NxM where N=number of samples, M=embedding vector size
// This allows to compute for the output size,  MxW where W is the number of weights (features) to use.
void Node::setData(py::array_t<double> embedding) {

    std::cout << "create Node 1 for node " << this->name << " ...\n";
    std::cout << embedding << "\n";

    // Convert values to C++ array
    py::buffer_info values_info = embedding.request();
    double* data = static_cast<double*>(values_info.ptr);
    int v_rows = values_info.shape[0]; // N
    int v_cols = values_info.shape[1]; // M
    // Convert a py::array_t row-major order to an Eigen::MatrixXd column-major order.
    this->input_data = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(data, v_rows, v_cols);

    std::cout << "create Node 2 for node " << this->name << " ...\n";
    std::cout << input_data << "\n";
}

Eigen::MatrixXd Node::getInput() {
    return input_data;
}

Eigen::MatrixXd Node::getOutput() {
    return output_data;
}

void Node::addInput(Node* input) {
    inputs.insert(input);
    input->outputs.insert(this);
}

void Node::addOutput(Node* output) {
    outputs.insert(output);
    output->inputs.insert(this);
}

std::unordered_set<Node*> Node::getOutputs() {
    return outputs;
}

std::unordered_set<Node*> Node::getInputs() {
    return inputs;
}

Node& Node::setOperations(std::vector<std::shared_ptr<BaseOperator>>& operations) {
    this->operations = operations;
    return *this;
}

void Node::setReduction(std::string& reducttype) {
    this->reduce = reducttype;
}

void Node::sequential(ssize_t repeat) {
    if (repeat > 1) {
        this->repeat = repeat;
    }
}

void Node::parallel(ssize_t repeat, std::string& reduce) {
    if (repeat > 1) {
        this->repeat = repeat;
    }
    this->reduce = reduce;
}


Eigen::MatrixXd Node::aggregateData(Eigen::MatrixXd& input_data) {
    // start with any input_data we keep.
    Eigen::MatrixXd output = input_data;
    // Now add any output data from any source Node that connects to this node.
    std::cout << "Aggregating Data ...\n";

    if (inputs.size() == 0) {
        return input_data;
    }

    std::cout << "Getting size of node input: " << output.size() << " \n";

    for (Node* node : inputs) {
        Eigen::MatrixXd outputx = node->getOutput();
        std::cout << "Getting size of node [" << node->getName() << "] output: " << outputx.size() << " \n";
        std::cout << outputx << "\n";
        if (output.size() == 0) {
            output = outputx;
            continue;
        }
        if (reduce == "add" || reduce == "avg") {
            std::cout << "Aggregating Data by add or average ...\n";
            // change to use more efficient element-wise function which uses GPU/TPU.
            output = output.array() + outputx.array();
        } else
        if (reduce == "mul") {
            std::cout << "Aggregating Data by mul ...\n";
            // change to use more efficient element-wise function which uses GPU/TPU.
            output = output.array() * outputx.array();
        } else
        if (reduce == "matmul") {
            std::cout << "Aggregating Data by matmul ...\n";
            // uses cblas_dgemm
            output = BaseOperator::matmul(output, outputx);
        }

    }
    if (reduce == "avg") {
        // change to use more efficient element-wise function which uses GPU/TPU.
        std::cout << "Aggregating Data by average ...\n";
        output = output.array() / inputs.size();
    }
    std::cout << "Aggregated ...\n";
    std::cout << output << "\n";
    return output;
}

void Node::setGradients(Eigen::MatrixXd gradients) {
    dInput = gradients;
}

void Node::propagateGradients(Eigen::MatrixXd& gradients) {
    // Now handle all other gradients for other inputs.
    if (inputs.size() != 0)
    for (Node* node : inputs) {
        if (reduce == "add") {
            node->setGradients(gradients);
        } else
        if (reduce == "avg") {
            node->setGradients(gradients.array() / outputs.size());
        } else
        if (reduce == "mul") {
            // change to use more efficient element-wise function which uses GPU/TPU.
            Eigen::MatrixXd dInput = gradients;
            for (Node* nodex : outputs) {
                if (nodex->getName() != node->getName()) {
                    dInput = dInput.array() * nodex->getOutput().array();
                }
            }
            node->setGradients(dInput);
        } else
        if (reduce == "matmul") {
            // change to use more efficient element-wise function which uses GPU/TPU.
            Eigen::MatrixXd dInput = gradients;
            for (Node* nodex : outputs) {
                if (nodex->getName() != node->getName()) {
                    dInput = BaseOperator::matmul(dInput, nodex->getOutput());
                }
                dInput = dInput.transpose();
            }
            node->setGradients(dInput);
        }
    }
}

// Because of Kahn Algorithm done (see Graph), this function runs forward pass only to 
// nodes whose source nodes are already processed.
Eigen::MatrixXd Node::forwardPass() {
    // Propagate forward data to connected nodes
    int size = operations.size();
    print_string(name + " forward pass ...", true);
    std::cout << " operation size: " << size << "\n";

    // See if we can perform reduction.
    Eigen::MatrixXd output = aggregateData(input_data);

    for (const auto& op : operations ) {
            // Check the dynamic type of the object using dynamic_cast
        if (auto linear = std::dynamic_pointer_cast<Linear>(op)) {
            print_string("Linear object", true);
            output = linear->forward(output);
            std::cout << output << std::endl;
        } else
        if (auto batchnorm = std::dynamic_pointer_cast<BatchNorm>(op)) {
            print_string("Batchnorm object", true);
            output = batchnorm->forward(output);
            std::cout << output << std::endl;
        } else            
        if (auto layernorm = std::dynamic_pointer_cast<LayerNorm>(op)) {
            print_string("Layernorm object", true);
            output = layernorm->forward(output);
            std::cout << output << std::endl;
        } else           
        if (auto activate = std::dynamic_pointer_cast<Activation>(op)) {
            print_string("Activate object", true);
            output = activate->forward(output);
            std::cout << output << std::endl;
        } else           
        if (auto attention = std::dynamic_pointer_cast<Attention>(op)) {
            print_string("Attention object", true);
            output = attention->forward(output);
            std::cout << output << std::endl;
        } else           
        if (auto feedforward = std::dynamic_pointer_cast<FeedForward>(op)) {
            print_string("FeedForward object", true);
            output = feedforward->forward(output);
            std::cout << output << std::endl;
        } else           
        if (auto encoder = std::dynamic_pointer_cast<Encoder>(op)) {
            print_string("Encoder object", true);
            output = encoder->forward(output);
            std::cout << output << std::endl;
        }
    }
    this->output_data = output;
    return this->output_data;
}

void Node::backwardPass() {
    // Propagate backward gradients to connected nodes
    int size = operations.size();
    print_string(name + " backward pass ...", true);
    std::cout << " operation size: " << size << "\n";

    // Create a copy of the original vector
    std::vector<std::shared_ptr<BaseOperator>> reversedOperations = operations;

    // Reverse the elements in the copied vector
    std::reverse(reversedOperations.begin(), reversedOperations.end());

    // Here, dInput is assumed to have already been propagated
    // through setGradients or propagatGradients.
    for (const auto& op : reversedOperations ) {
        if (auto linear = std::dynamic_pointer_cast<Linear>(op)) {
            print_string("Linear object", true);
            dInput = linear->backward(dInput);
            std::cout << dInput << std::endl;
        } else
        if (auto batchnorm = std::dynamic_pointer_cast<BatchNorm>(op)) {
            print_string("Batchnorm object", true);
            dInput = batchnorm->backward(dInput);
            std::cout << dInput << std::endl;
        } else            
        if (auto layernorm = std::dynamic_pointer_cast<LayerNorm>(op)) {
            print_string("Layernorm object", true);
            dInput = layernorm->backward(dInput);
            std::cout << dInput << std::endl;
        } else           
        if (auto activate = std::dynamic_pointer_cast<Activation>(op)) {
            print_string("Activate object", true);
            dInput = activate->backward(dInput, this->output_data);
            std::cout << dInput << std::endl;
        } else           
        if (auto attention = std::dynamic_pointer_cast<Attention>(op)) {
            print_string("Attention object", true);
            dInput = attention->backward(dInput);
            std::cout << dInput << std::endl;
        } else           
        if (auto feedforward = std::dynamic_pointer_cast<FeedForward>(op)) {
            print_string("FeedForward object", true);
            dInput = feedforward->backward(dInput);
            std::cout << dInput << std::endl;
        } else           
        if (auto encoder = std::dynamic_pointer_cast<Encoder>(op)) {
            print_string("Encoder object", true);
            dInput = encoder->backward(dInput);
            std::cout << dInput << std::endl;
        } 
    }

    // Propagate gradients to next nodes.
    std::cout << "This node: " << this->getName() << " propagating ...\n";
    std::cout << dInput << "\n";
    propagateGradients(dInput);

    // Reinitialize dInput for next EPOCH, as long as parameter gradients have been preserved.
    dInput.setZero();

}


void Node::updateParameters(std::string& optimizertype, double& learningRate, int& iter) {
    std::cout << "************************************ Node: " << this->getName() << " ...\n";
    for (const auto& op : operations ) {
            // Check the dynamic type of the object using dynamic_cast
        if (auto linear = std::dynamic_pointer_cast<Linear>(op)) {
            print_string("Linear object", true);
            linear->updateParameters(optimizertype, learningRate, iter);
        } else
        if (auto batchnorm = std::dynamic_pointer_cast<BatchNorm>(op)) {
            print_string("Batchnorm object", true);
            batchnorm->updateParameters(optimizertype, learningRate, iter);
        } else            
        if (auto layernorm = std::dynamic_pointer_cast<LayerNorm>(op)) {
            print_string("Layernorm object", true);
            layernorm->updateParameters(optimizertype, learningRate, iter);
        } else            
        if (auto attention = std::dynamic_pointer_cast<Attention>(op)) {
            print_string("Attention object", true);
            attention->updateParameters(optimizertype, learningRate, iter);
        } else            
        if (auto feedforward = std::dynamic_pointer_cast<FeedForward>(op)) {
            print_string("FeedForward object", true);
            feedforward->updateParameters(optimizertype, learningRate, iter);
        } else            
        if (auto encoder = std::dynamic_pointer_cast<Encoder>(op)) {
            print_string("Encoder object", true);
            encoder->updateParameters(optimizertype, learningRate, iter);
        }
    }
}




/*****************************************************************************************************
* Base Connection Functions
*****************************************************************************************************/

Node* Connection::getSource() {
    return source;
}

Node* Connection::getDestination() {
    return destination;
}

void Connection::forwardPass() {
}

Eigen::MatrixXd Connection::backwardPass(Eigen::MatrixXd& gradients) {
    return gradients;
}


/*****************************************************************************************************
* Base Graph Functions
*****************************************************************************************************/

// Create a node with three arguments: name, type, and initial values
Node* Graph::createNode(const std::string& name, NodeType type, const py::array_t<double>& embedding) {
    Node* node = new Node(name, type, embedding);
    nodes.push_back(node);
    return node;
}

// Create a node with two arguments: name and type (no initial values)
Node* Graph::createNode(const std::string& name, NodeType type) {
    Node* node = new Node(name, type);
    nodes.push_back(node);
    return node;
}

void Graph::connect(Node* from, Node* to) {
    addConnection(new Connection(from, to));
}

void Graph::connect(Node* from, Node* to, std::vector<std::shared_ptr<BaseOperator>>& operations) {
    to->setOperations(operations);
    addConnection(new Connection(from, to));
}

void Graph::connect(std::vector<Node*> from_nodes, Node* to) {
    for (Node* from : from_nodes) {
        addConnection(new Connection(from, to));
    }
}

void Graph::connect(std::vector<Node*> from_nodes, Node* to, std::vector<std::shared_ptr<BaseOperator>>& operations) {
    to->setOperations(operations);
    for (Node* from : from_nodes) {
        addConnection(new Connection(from, to));
    }
}

void Graph::addConnection(Connection* connection) {
    connections.push_back(connection);
    Node* from = connection->getSource();
    Node* to = connection->getDestination();
    outdegree[from]++;
    indegree[to]++;

    to->addInput(from);
    from->addOutput(to);

    std::cout << "Adding Connection\n";
    std::cout << "Nodes size:" << nodes.size() << "\n";
}

std::vector<Node*> Graph::getNodes() {
    return nodes;
}

// Perform the Kahn's Algorithm by Arthur B. Khan based on his 1962 paper, "Topological Sorting of Large Networks"
Eigen::MatrixXd Graph::forwardPropagation() {
    std::queue<Node*> q;
    std::unordered_map<Node*, int> indegree_(indegree); 

    std::cout << "Entering Forward Pass.\n";

    std::cout << "Size:" << getNodes().size() << "\n";

    for (Node* node : nodes) {
        if (indegree[node] == 0) {
            q.push(node);
        }
    }

    std::cout << "Collecting nodes for the queue.\n";
    std::cout << "Size of queue: " << q.size() << "\n";

    Eigen::MatrixXd output(0, 0);

    std::cout << "Entering Queue Loop.\n";

    while (!q.empty()) {
        Node* node = q.front();
        q.pop();

        // Eigen::MatrixXd input_data = node->getInput();

        std::cout << "*** Entering forward pass for " << node->name << " **** \n";

            // output = node->forwardPass(input_data);     
        output = node->forwardPass();     

        std::cout << "Processing completed for node [" << node->name << "]\n";
        // std::cout << output << std::endl;

        for (Connection* connection : connections) {
            if (connection->getSource() == node) {

                //std::cout << "Processing completed for connection.\n";
                //connection->forwardPass();
        
                // Propagate gradient to source node
                Node* dstNode = connection->getDestination();

                indegree_[dstNode]--;
                if (indegree_[dstNode] == 0) {
                    q.push(dstNode);
                }
            }
        }
    }

    std::cout << "Forward Processing completed for all nodes.\n";

    return output;
}

Eigen::MatrixXd Graph::backwardPropagation(Eigen::MatrixXd& gradients) {
    std::queue<Node*> q;
    std::unordered_map<Node*, int> outdegree_(outdegree); 

    std::cout << "\n\nEntering Backward Pass ***************************************\n";

    std::cout << "Size:" << getNodes().size() << "\n";

    for (Node* node : nodes) {
        if (outdegree[node] == 0) {
            node->setGradients(gradients); // gradients with respect to loss function.
            q.push(node);
        }
    }

    std::cout << "Collecting nodes for the queue.\n";
    std::cout << "Size of queue: " << q.size() << "\n";

    std::cout << "Entering Queue Loop.\n";

    while (!q.empty()) {
        Node* node = q.front();
        q.pop();

        std::cout << "*** Entering backward pass for " << node->name << " **** \n";

        // Compute gradients for the node
        node->backwardPass();

        std::cout << "Processing completed for node [" << node->name << "]\n";

        for (Connection* connection : connections) {
            if (connection->getDestination() == node) {

                //std::cout << "Processing completed for connection.\n";
                //Eigen::MatrixXd srcGradients = connection->backwardPass(gradients);

                // Propagate gradient to source node
                Node* srcNode = connection->getSource();

                outdegree_[srcNode]--;
                if (outdegree_[srcNode] == 0) {
                    q.push(srcNode);
                }
            }
        }
    }

    std::cout << "Backward Processing completed for all nodes.\n";

    return gradients;
}

Eigen::MatrixXd Graph::computeLoss(std::string losstype, Eigen::MatrixXd& predicted, const Eigen::MatrixXd& target) {
    Loss* obj = new Loss(losstype);
    Eigen::MatrixXd loss = obj->computeLoss(predicted, target);
    std::cout << "Loss calculated ... \n";
    std::cout << loss << "\n";
    return loss;
}

Eigen::MatrixXd Graph::computeGradients(std::string losstype, Eigen::MatrixXd& predicted, const Eigen::MatrixXd& target) {
    Loss* obj = new Loss(losstype);
    Eigen::MatrixXd gradients = obj->computeGradients(predicted, target);
    std::cout << "Loss Gradient calculated ... \n";
    // std::cout << gradients << "\n";
    return gradients;
}

void Graph::updateParameters(std::string& optimizertype, double& learningRate, int& iter) {
    for (Node* node: nodes) {
        node->updateParameters(optimizertype, learningRate, iter);
    }
}

void Graph::nextBatch() {
    for (Node* node: nodes) {
        if (node->nodeType() == NodeType::Input)  {

        }
    }
}

const std::unordered_map<Node*, int>& Graph::getIndegree() const {
    return indegree;
}

// };

