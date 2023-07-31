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

    log_info("=================");
    log_info( "Setting Data ..." );

    // Convert values to C++ array
    py::buffer_info values_info = embedding.request();
    double* data = static_cast<double*>(values_info.ptr);
    int v_rows = values_info.shape[0]; // N
    int v_cols = values_info.shape[1]; // M
    // Convert a py::array_t row-major order to an Eigen::MatrixXd column-major order.
    this->input_data = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(data, v_rows, v_cols);

    log_detail( "Input Data For Node Name: {0}", this->name );
    log_matrix( input_data );
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

    log_info( "====================" );
    log_info( "Aggregating Data ..." );

    if (inputs.size() == 0) {
        return input_data;
    }

    log_detail( "Getting size of node input: {0}", output.size() );

    for (Node* node : inputs) {
        Eigen::MatrixXd outputx = node->getOutput();
        log_detail( "Getting size of node [{0}] output: {1}", node->getName(), outputx.size() );
        // std::cout << outputx << "\n";
        if (output.size() == 0) {
            output = outputx;
            continue;
        }
        if (reduce == "add" || reduce == "avg") {
            log_detail( "Aggregating Data by add or average ..." );
            // change to use more efficient element-wise function which uses GPU/TPU.
            output = output.array() + outputx.array();
        } else
        if (reduce == "mul") {
            log_detail( "Aggregating Data by mul ..." );
            // change to use more efficient element-wise function which uses GPU/TPU.
            output = output.array() * outputx.array();
        } else
        if (reduce == "matmul") {
            log_detail(  "Aggregating Data by matmul ..." );
            // uses cblas_dgemm
            output = BaseOperator::matmul(output, outputx);
        }

    }
    if (reduce == "avg") {
        // change to use more efficient element-wise function which uses GPU/TPU.
        log_detail( "Aggregating Data by average ..." );
        output = output.array() / inputs.size();
    }

    log_detail( "Aggregated output:" );
    log_matrix( output );

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

    log_info( "**************************************" );
    log_info( "***      Node Forward Pass  **********" );
    log_info( "**************************************" );

    log_detail("Node: {0} Size: {1}", name, size);

    // See if we can perform reduction.
    Eigen::MatrixXd output = aggregateData(input_data);

    for (const auto& op : operations ) {
            // Check the dynamic type of the object using dynamic_cast
        if (auto linear = std::dynamic_pointer_cast<Linear>(op)) {
            log_detail("Node [{0}] Linear Operation  (Forward Pass) Size: {1}", name, size);
            output = linear->forward(output);
            log_matrix( output );
        } else
        if (auto batchnorm = std::dynamic_pointer_cast<BatchNorm>(op)) {
            log_detail("Node [{0}] Batch Normal Operation (Forward Pass)", name );
            output = batchnorm->forward(output);
            log_matrix( output );
        } else            
        if (auto layernorm = std::dynamic_pointer_cast<LayerNorm>(op)) {
            log_detail("Node [{0}] Layer Normal Operation (Forward Pass)", name );
            output = layernorm->forward(output);
            log_matrix( output );
        } else           
        if (auto activate = std::dynamic_pointer_cast<Activation>(op)) {
            log_detail("Node [{0}] Activation Operation (Forward Pass)", name );
            output = activate->forward(output);
            log_matrix( output );
        } else           
        if (auto attention = std::dynamic_pointer_cast<Attention>(op)) {
            log_detail("Node [{0}] Attention Operation (Forward Pass)", name );
            output = attention->forward(output);
            log_matrix( output );
        } else           
        if (auto feedforward = std::dynamic_pointer_cast<FeedForward>(op)) {
            log_detail("Node [{0}] FeedForward Operation (Forward Pass)", name );
            output = feedforward->forward(output);
            log_matrix( output );
        } else           
        if (auto encoder = std::dynamic_pointer_cast<Encoder>(op)) {
            log_detail("Node [{0}] Encoder Operation (Forward Pass)", name );
            output = encoder->forward(output);
            log_matrix( output );
        }
    }
    this->output_data = output;
    return this->output_data;
}

void Node::backwardPass() {
    // Propagate backward gradients to connected nodes
    int size = operations.size();

    log_info( "**************************************" );
    log_info( "***     Node Backward Pass  **********" );
    log_info( "**************************************" );

    log_detail("Node: {0} Size: {1}", name, size);

    // Create a copy of the original vector
    std::vector<std::shared_ptr<BaseOperator>> reversedOperations = operations;

    // Reverse the elements in the copied vector
    std::reverse(reversedOperations.begin(), reversedOperations.end());

    // Here, dInput is assumed to have already been propagated
    // through setGradients or propagatGradients.
    for (const auto& op : reversedOperations ) {
        if (auto linear = std::dynamic_pointer_cast<Linear>(op)) {
            log_detail("Node [{0}] Linear Operation (Backward Pass)", name );
            dInput = linear->backward(dInput);
            log_matrix( dInput );
        } else
        if (auto batchnorm = std::dynamic_pointer_cast<BatchNorm>(op)) {
            log_detail("Node [{0}] Batch Normal Operation (Backward Pass)", name );
            dInput = batchnorm->backward(dInput);
            log_matrix( dInput );
        } else            
        if (auto layernorm = std::dynamic_pointer_cast<LayerNorm>(op)) {
            log_detail("Node [{0}] Layer Normal Operation (Backward Pass)", name );
            dInput = layernorm->backward(dInput);
            log_matrix( dInput );
        } else           
        if (auto activate = std::dynamic_pointer_cast<Activation>(op)) {
            log_detail("Node [{0}] Activation Operation (Backward Pass)", name );
            dInput = activate->backward(dInput, this->output_data);
            log_matrix( dInput );
        } else           
        if (auto attention = std::dynamic_pointer_cast<Attention>(op)) {
            log_detail("Node [{0}] Attention Operation (Backward Pass)", name );
            dInput = attention->backward(dInput);
            log_matrix( dInput );
        } else           
        if (auto feedforward = std::dynamic_pointer_cast<FeedForward>(op)) {
            log_detail("Node [{0}] Feedforward Operation (Backward Pass)", name );
            dInput = feedforward->backward(dInput);
            log_matrix( dInput );
        } else           
        if (auto encoder = std::dynamic_pointer_cast<Encoder>(op)) {
            log_detail("Node [{0}] Encoder Operation (Backward Pass)", name );
            dInput = encoder->backward(dInput);
            log_matrix( dInput );
        } 
    }

    // Propagate gradients to next nodes.

    log_detail("Node [{0}] Propagating Gradient", name );
    log_matrix( dInput );

    propagateGradients(dInput);

    // Reinitialize dInput for next EPOCH, as long as parameter gradients have been preserved.
    dInput.setZero();

}


void Node::updateParameters(std::string& optimizertype, double& learningRate, int& iter) {

    log_info( "*****************************************" );
    log_info( "***     Node Parameter Update  **********" );
    log_info( "*****************************************" );

    log_detail("Node: {0}", name );

    for (const auto& op : operations ) {
            // Check the dynamic type of the object using dynamic_cast
        if (auto linear = std::dynamic_pointer_cast<Linear>(op)) {
            log_detail("Node [{0}] Linear Operation (Update Params)", name );
            linear->updateParameters(optimizertype, learningRate, iter);
        } else
        if (auto batchnorm = std::dynamic_pointer_cast<BatchNorm>(op)) {
            log_detail("Node [{0}] Barch Normal Operation (Update Params)", name );
            batchnorm->updateParameters(optimizertype, learningRate, iter);
        } else            
        if (auto layernorm = std::dynamic_pointer_cast<LayerNorm>(op)) {
            log_detail("Node [{0}] Layer Normal Operation (Update Params)", name );
            layernorm->updateParameters(optimizertype, learningRate, iter);
        } else            
        if (auto attention = std::dynamic_pointer_cast<Attention>(op)) {
            log_detail("Node [{0}] Attention Operation (Update Params)", name );
            attention->updateParameters(optimizertype, learningRate, iter);
        } else            
        if (auto feedforward = std::dynamic_pointer_cast<FeedForward>(op)) {
            log_detail("Node [{0}] Feedforward Operation (Update Params)", name );
            feedforward->updateParameters(optimizertype, learningRate, iter);
        } else            
        if (auto encoder = std::dynamic_pointer_cast<Encoder>(op)) {
            log_detail("Node [{0}] Encoder Operation (Update Params)", name );
            encoder->updateParameters(optimizertype, learningRate, iter);
        } 
    }
}

std::string removeSpace(Node* node) {
    std::string s = node->getName();
    s.erase(std::remove(s.begin(), s.end(), ' '), s.end());
    return s;
}

std::string replaceSpace(Node* node) {
    std::string nodename = node->getName();
    std::replace(nodename.begin(), nodename.end(), ' ', '_');
    return nodename;
}

std::string Node::generateDotFormat() {
    std::string nodename = replaceSpace(this);
    std::string nodelabel = nodename + "_label";
    std::string dot_ = "";
    int cnt = 0; 
    for (const auto& op : operations ) {
            // Check the dynamic type of the object using dynamic_cast
        if (auto linear = std::dynamic_pointer_cast<Linear>(op)) {
            dot_ +=  linear->generateDotFormat();
        } else
        if (auto batchnorm = std::dynamic_pointer_cast<BatchNorm>(op)) {
            dot_ +=  batchnorm->generateDotFormat();
        } else            
        if (auto layernorm = std::dynamic_pointer_cast<LayerNorm>(op)) {
            dot_ += layernorm->generateDotFormat();
        } else               
        if (auto activation = std::dynamic_pointer_cast<Activation>(op)) {
            dot_ += activation->generateDotFormat();
        } else
        if (auto attention = std::dynamic_pointer_cast<Attention>(op)) {
            dot_ += attention->generateDotFormat();
        } else            
        if (auto feedforward = std::dynamic_pointer_cast<FeedForward>(op)) {
            dot_ += feedforward->generateDotFormat();
        } else            
        if (auto encoder = std::dynamic_pointer_cast<Encoder>(op)) {
            dot_ += encoder->generateDotFormat();
        } 
        if (++cnt < (int) operations.size()) { dot_ += "|"; }
    }
    dot_ = nodelabel + " [shape=record, label=\"" + dot_ + "\"]; ";
    dot_ += nodename + "->" + nodelabel + ";";
    return dot_;
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

    log_info( "***************************************" );
    log_info( "***    Graph Add Connection  **********" );
    log_info( "***************************************" );

    connections.push_back(connection);
    Node* from = connection->getSource();
    Node* to = connection->getDestination();
    outdegree[from]++;
    indegree[to]++;

    to->addInput(from);
    from->addOutput(to);

    log_detail( "Source Node: {0}", from->getName() );
    log_detail( "Destination Node: {0}", to->getName() );
    log_detail( "Number of Nodes: {:d}", nodes.size() );
}

std::vector<Node*> Graph::getNodes() {
    return nodes;
}

// Perform the Kahn's Algorithm by Arthur B. Khan based on his 1962 paper, "Topological Sorting of Large Networks"
Eigen::MatrixXd Graph::forwardPropagation() {

    std::queue<Node*> q;
    std::unordered_map<Node*, int> indegree_(indegree); 

    log_info( "***************************************" );
    log_info( "*****    Graph Forward Pass  **********" );
    log_info( "***************************************" );

    log_detail( "Number of Nodes: {0}", getNodes().size() );

    for (Node* node : nodes) {
        if (indegree[node] == 0) {
            q.push(node);
        }
    }

    log_detail( "Graph: Collecting nodes for the queue." );
    log_detail( "Size of queue: {0}",  q.size() );

    Eigen::MatrixXd output(0, 0);

    log_detail(  "Graph: Entering Queue Loop." );

    while (!q.empty()) {
        Node* node = q.front();
        q.pop();

        log_detail( "*** Graph: Entering forward pass for {0} ****", node->name );

        output = node->forwardPass();     

        for (Connection* connection : this->connections) {
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

    log_detail( "Graph: forward pass completed for all nodes." );

    return output;
}

Eigen::MatrixXd Graph::backwardPropagation(Eigen::MatrixXd& gradients) {
    std::queue<Node*> q;
    std::unordered_map<Node*, int> outdegree_(outdegree); 

    log_info( "***************************************" );
    log_info( "*****    Graph Backward Pass  *********" );
    log_info( "***************************************" );

    log_detail( "Number of Nodes:", getNodes().size() );

    for (Node* node : nodes) {
        if (outdegree[node] == 0) {
            node->setGradients(gradients); // gradients with respect to loss function.
            q.push(node);
        }
    }

    log_detail( "Collecting nodes for the queue." );
    log_detail( "Size of queue: {0}", q.size() );

    log_detail( "Entering Queue Loop." );

    while (!q.empty()) {
        Node* node = q.front();
        q.pop();

        log_detail( "*** Graph: Entering backward pass for {0} ****", node->name );

        // Compute gradients for the node
        node->backwardPass();

        for (Connection* connection : this->connections) {
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

    log_detail( "Graph: backward pass completed for all nodes." );

    return gradients;
}

Eigen::MatrixXd Graph::computeLoss(std::string losstype, Eigen::MatrixXd& predicted, const Eigen::MatrixXd& target) {

    log_info( "***************************************************" );
    log_info( "*****    Graph: Processing Loss Function  *********" );
    log_info( "***************************************************" );

    Loss* obj = new Loss(losstype);
    Eigen::MatrixXd loss = obj->computeLoss(predicted, target);

    log_detail( "Loss calculated: " );
    log_matrix( loss );

    return loss;
}

Eigen::MatrixXd Graph::computeGradients(std::string losstype, Eigen::MatrixXd& predicted, const Eigen::MatrixXd& target) {

    log_info( "*********************************************" );
    log_info( "*****    Graph: Processing Gradient *********" );
    log_info( "*********************************************" );

    Loss* obj = new Loss(losstype);
    Eigen::MatrixXd gradients = obj->computeGradients(predicted, target);

    log_detail( "Loss Gradient calculated ..." );
    log_matrix( gradients );

    return gradients;
}

void Graph::updateParameters(std::string& optimizertype, double& learningRate, int& iter) {

    log_info( "*******************************************************" );
    log_info( "*****    Graph: Processing Parameter Update ***********" );
    log_info( "*******************************************************" );

    for (Node* node: nodes) {
        node->updateParameters(optimizertype, learningRate, iter);
    }

    log_detail( "Graph: parameter update completed for all nodes." );

}

std::string Graph::generateDotFormat() {
    std::string dot = 
        "digraph G {  node [shape=circle]; rankdir=LR; ";

    for (Node* node: nodes) {
        dot += removeSpace(node) + "; ";
    }

    for (Connection* connection : this->connections) {
        Node *source = connection->getSource();
        Node *destination = connection->getDestination();
        std::string edge = removeSpace(source) + "->" + removeSpace(destination) + ";";
        dot += edge;
    }

    for (Node* node: nodes) {
        dot += node->generateDotFormat();
    }

    dot += "}";
    return dot;
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



