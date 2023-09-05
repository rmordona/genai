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
#include "recurrent.h"

namespace py = pybind11;
using namespace py::literals;


/********************************************************************************************
* NodeFactory
********************************************************************************************/

template <class T>
std::string Node<T>::getName() {
    return this->name;
}

template <class T>
NodeType Node<T>::nodeType() {
    return this->type;
}

// The input is assumed to have NxM where N=number of samples, M=embedding vector size
// This allows to compute for the output size,  MxW where W is the number of weights (features) to use.
template <class T>
void Node<T>::setData(const py::array_t<T>& input_data) {

    log_info("=================");
    log_info( "Setting Data ..." );

    // request a buffer descriptor from Python
    py::buffer_info buffer_info = input_data.request();

    // extract data an shape of input array
    T* data = static_cast<T*>(buffer_info.ptr);

    int dim0 = buffer_info.shape[0]; // N
    int dim1 = buffer_info.shape[1]; // M
    // Convert a py::array_t row-major order to an Eigen::MatrixXd column-major order.
    this->input_data = Eigen::Map<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(data, dim0, dim1);

    log_detail( "Input Data For Node Name: {0}", this->name );
    log_matrix( this->input_data );
}

// The input is assumed to have NxM where N=number of samples, M=embedding vector size
// This allows to compute for the output size,  MxW where W is the number of weights (features) to use.
template <class T>
void Node<T>::setDataTensor(const py::array_t<T>& input_data) {

    log_info("=================");
    log_info( "Setting Data (Tensor) ..." );

    log_detail( "Input Data For Node Name: {0}", this->name );

    // request a buffer descriptor from Python
    py::buffer_info buffer_info = input_data.request();

    // extract data an shape of input array
    T* data = static_cast<T *>(buffer_info.ptr);
    std::vector<ssize_t> shape = buffer_info.shape;

    // Get the dimensions of the py::array_t
    ssize_t dim0 = shape[0]; // Batch Size
    ssize_t dim1 = shape[1]; // Input Size
    ssize_t dim2 = shape[2]; // Parameter / Embedding Size

    // Create an Eigen::Map to map the raw data to an Eigen::Tensor
    // Eigen::Map<Eigen::Tensor<T,3>> tensor_map(data, dim0, dim1, dim2);

    aitensor<T> tensor(dim0, dim1, dim2);

    // auto ptr = static_cast<T*>(info.ptr);
    for (int i = 0; i < dim0; ++i) {
        for (int j = 0; j < dim1; ++j) {
            for (int k = 0; k < dim2; ++k) {
                tensor(i, j, k) = *data++;
            }
        }
    }

    this->tensor = true;

    this->input_data = tensor;

}

template <class T>
const aitensor<T>& Node<T>::getInput() {
    return input_data;
}

template <class T>
const aitensor<T>& Node<T>::getOutput() {
    return output_data;
}

template <class T>
void Node<T>::addInput(Node<T>* input) {
    inputs.insert(input);
    input->outputs.insert(this);
}

template <class T>
void Node<T>::addOutput(Node<T>* output) {
    outputs.insert(output);
    output->inputs.insert(this);
}

template <class T>
std::unordered_set<Node<T>*> Node<T>::getOutputs() {
    return outputs;
}

template <class T>
std::unordered_set<Node<T>*> Node<T>::getInputs() {
    return inputs;
}

template <class T>
Node<T>& Node<T>::setOperations(std::vector<std::shared_ptr<BaseOperator>>& operations) {
    this->operations = operations;
    return *this;
}

template <class T>
void Node<T>::setReduction(std::string& reducttype) {
    this->reduce = reducttype;
}

template <class T>
void Node<T>::sequential(ssize_t repeat) {
    if (repeat > 1) {
        this->repeat = repeat;
    }
}

template <class T>
void Node<T>::parallel(ssize_t repeat, std::string& reduce) {
    if (repeat > 1) {
        this->repeat = repeat;
    }
    this->reduce = reduce;
}

template <class T>
const aitensor<T> Node<T>::aggregateData(const aitensor<T>& input_data) {
    // start with any input_data we keep.
    aitensor<T> aggregate_input = input_data;
    // Now add any output data from any source Node that connects to this node.

    log_info( "====================" );
    log_info( "Aggregating Data ..." );

    // if (inputs.size() == 0) {
    //    return input_data;
    // }

    log_detail( "Getting size of node input: {0}", input_data.size() );
    aiscalar<T> suminputs = 0.0;
    for (Node* node : this->inputs) { // Check if we have nodes that are feeding this node.
        aitensor<T> external_input = node->getOutput();
        log_detail( "Getting size of node [{0}] output: {1}", node->getName(), external_input.size() );
        if (aggregate_input.size() == 0) {
            aggregate_input = external_input;
            continue; // See if there are more external inputs.
        }
        if (reduce == "add" || reduce == "avg") {
            log_detail( "Aggregating Data by add or average ..." );
            // change to use more efficient element-wise function which uses GPU/TPU.
            aggregate_input = aggregate_input + external_input;
            suminputs = suminputs + 1.0;
        } else
        if (reduce == "mul") {
            log_detail( "Aggregating Data by mul ..." );
            // change to use more efficient element-wise function which uses GPU/TPU.
            aggregate_input = aggregate_input * external_input;
        } else
        if (reduce == "concat") { // Assume adding two batches. Requires dimension 1 and 2 are all the same.

                int o1_dim0 = aggregate_input.dimension(0);
                int o1_dim1 = aggregate_input.dimension(1);
                int o1_dim2 = aggregate_input.dimension(2);
                int o2_dim0 = external_input.dimension(0);
                int o2_dim1 = external_input.dimension(1);
                int o2_dim2 = external_input.dimension(2);

                if (o1_dim1 != o2_dim1 || o1_dim2 != o2_dim2) {
                    std::cerr << "Error concatenating two batches of different dimensions." << std::endl;
                }

                aitensor<T> concatOutput(aggregate_input.dimension(0) + external_input.dimension(0), aggregate_input.dimension(1), aggregate_input.dimension(2));

                Eigen::array<Eigen::Index, 3> starting;
                Eigen::array<Eigen::Index, 3> ending;

                // Fill up concatOutput with output data.
                starting = {0, 0, 0};
                ending   = {o1_dim0, o1_dim1, o1_dim2};
                concatOutput.slice(starting, ending) = external_input;

                // Fill up concatOutput with outputx data.
                starting = {o1_dim0, o1_dim1, o1_dim2};
                ending   = {o1_dim0 + o2_dim0, o1_dim1 + o2_dim1, o1_dim2 + o2_dim2};
                concatOutput.slice(starting, ending) = external_input;

                aggregate_input = concatOutput;
        }

    }
    if (reduce == "avg") {
        // change to use more efficient element-wise function which uses GPU/TPU.
        log_detail( "Aggregating Data by average ..." );
        aggregate_input = aggregate_input / (aiscalar<T>) (suminputs);
        this->suminputs = suminputs;
    }

    log_detail( "Aggregated output:" );
    log_matrix( aggregate_input );

    return aggregate_input;
}

template <class T>
void Node<T>::setGradients(const aitensor<T>& gradients) {
    this->gradients = gradients;
}

template <class T>
void Node<T>::propagateGradients(const aitensor<T>& gradients) {
    // Now handle all other gradients for other inputs.
    if (inputs.size() != 0)
    for (Node* node : this->inputs) { // These are input nodes that connect to this node.
        if (reduce == "add") {
            node->setGradients(gradients);
        } else
        if (reduce == "avg") {
            node->setGradients(gradients / (aiscalar<T>) (this->suminputs));
        } else
        if (reduce == "mul") {
            // change to use more efficient element-wise function which uses GPU/TPU.
            aitensor<T> dInput = gradients;
            for (Node* nodex : outputs) {
                if (nodex->getName() != node->getName()) {
                    dInput = dInput * nodex->getOutput();
                }
            }
            node->setGradients(dInput);
        } 
    }
}
 
// Because of Kahn Algorithm done (see Graph), this function runs forward pass only to 
// nodes whose source nodes are already processed.
template <class T>
void Node<T>::forwardPass() {
    // Propagate forward data to connected nodes
    int size = operations.size();

    log_info( "**************************************" );
    log_info( "***      Node Forward Pass  **********" );
    log_info( "**************************************" );

    log_detail("Node: {0} Size: {1}", name, size);

    // See if we can perform reduction.
    aitensor<T> output = aggregateData(this->input_data); // see Node.setData

    // If we are dealing with 3D
    // aitensor<T> output_tensor = this->input_data_tensor; // see Node.setDataTensor

    for (const auto& op : operations ) {
            // Check the dynamic type of the object using dynamic_cast
        if (auto linear = std::dynamic_pointer_cast<Linear<T>>(op)) {
            log_detail("Node [{0}] Linear Operation  (Forward Pass) Size: {1}", name, size);
            output = linear->forward(output);
            log_matrix( output );
        } else
        if (auto batchnorm = std::dynamic_pointer_cast<BatchNorm<T>>(op)) {
            log_detail("Node [{0}] Batch Normal Operation (Forward Pass)", name );
            output = batchnorm->forward(output);
            log_matrix( output );
        } else
        if (auto layernorm = std::dynamic_pointer_cast<LayerNorm<T>>(op)) {
            log_detail("Node [{0}] Layer Normal Operation (Forward Pass)", name );
            output = layernorm->forward(output);
            log_matrix( output );
        } else
        if (auto activate = std::dynamic_pointer_cast<Activation<T>>(op)) {
            log_detail("Node [{0}] Activation Operation (Forward Pass)", name );
            output = activate->forward(output);
            log_matrix( output );
        } else
        if (auto attention = std::dynamic_pointer_cast<Attention<T>>(op)) {
            log_detail("Node [{0}] Attention Operation (Forward Pass)", name );
            output = attention->forward(output);
            log_matrix( output );
        } else
        if (auto feedforward = std::dynamic_pointer_cast<FeedForward<T>>(op)) {
            log_detail("Node [{0}] FeedForward Operation (Forward Pass)", name );
            output = feedforward->forward(output);
            log_matrix( output );
        } else
        if (auto encoder = std::dynamic_pointer_cast<Encoder<T>>(op)) {
            log_detail("Node [{0}] Encoder Operation (Forward Pass)", name );
            output = encoder->forward(output);
            log_matrix( output );
        } else
        if (auto rnn = std::dynamic_pointer_cast<RNN<T>>(op)) {
            log_detail("Node [{0}] RNN Operation (Forward Pass)", name );
            output = rnn->forward(output);
            log_matrix( output );
        } else
        if (auto lstm = std::dynamic_pointer_cast<LSTM<T>>(op)) {
            log_detail("Node [{0}] LSTM Operation (Forward Pass)", name );
            output = lstm->forward(output);
            log_matrix( output );
        } else
        if (auto gru = std::dynamic_pointer_cast<GRU<T>>(op)) {
            log_detail("Node [{0}] gru Operation (Forward Pass)", name );
            output = gru->forward(output);
            log_matrix( output );
        }
    }

    this->output_data = output; // Cache output for input of connecting nodes.
}

template <class T>
void Node<T>::backwardPass() {
    // Propagate backward gradients to connected nodes
    int size = operations.size();

    log_info( "**************************************" );
    log_info( "***     Node Backward Pass  **********" );
    log_info( "**************************************" );

    log_detail("Node: {0} Size: {1}", name, size);

    // Create a copy of the original vector
    std::vector<std::shared_ptr<BaseOperator>> reversedOperations = operations;

    // If we are dealing with 3D
    aitensor<T> dInput = this->gradients; // initially generated through Graph.backwardPropagation()

    // Reverse the elements in the copied vector
    std::reverse(reversedOperations.begin(), reversedOperations.end());

    // Here, dInput is assumed to have already been propagated
    // through setGradients or propagatGradients.
    for (const auto& op : reversedOperations ) {
        if (auto linear = std::dynamic_pointer_cast<Linear<T>>(op)) {
            log_detail("Node [{0}] Linear Operation (Backward Pass)", name );
            dInput = linear->backward(dInput);
            log_matrix( dInput );
        } else
        if (auto batchnorm = std::dynamic_pointer_cast<BatchNorm<T>>(op)) {
            log_detail("Node [{0}] Batch Normal Operation (Backward Pass)", name );
            dInput = batchnorm->backward(dInput);
            log_matrix( dInput );
        } else            
        if (auto layernorm = std::dynamic_pointer_cast<LayerNorm<T>>(op)) {
            log_detail("Node [{0}] Layer Normal Operation (Backward Pass)", name );
            dInput = layernorm->backward(dInput);
            log_matrix( dInput );
        } else           
        if (auto activate = std::dynamic_pointer_cast<Activation<T>>(op)) {
            log_detail("Node [{0}] Activation Operation (Backward Pass)", name );
            dInput = activate->backward(dInput);
            log_matrix( dInput );
        } else           
        if (auto attention = std::dynamic_pointer_cast<Attention<T>>(op)) {
            log_detail("Node [{0}] Attention Operation (Backward Pass)", name );
            dInput = attention->backward(dInput);
            log_matrix( dInput );
        } else           
        if (auto feedforward = std::dynamic_pointer_cast<FeedForward<T>>(op)) {
            log_detail("Node [{0}] Feedforward Operation (Backward Pass)", name );
            dInput = feedforward->backward(dInput);
            log_matrix( dInput );
        } else           
        if (auto encoder = std::dynamic_pointer_cast<Encoder<T>>(op)) {
            log_detail("Node [{0}] Encoder Operation (Backward Pass)", name );
            dInput = encoder->backward(dInput);
            log_matrix( dInput );
        } else           
        if (auto rnn = std::dynamic_pointer_cast<RNN<T>>(op)) {
            log_detail("Node [{0}] Encoder Operation (Backward Pass)", name );
            dInput = rnn->backward(dInput);
            // log_matrix( dInput );
        } else           
        if (auto lstm = std::dynamic_pointer_cast<LSTM<T>>(op)) {
            log_detail("Node [{0}] Encoder Operation (Backward Pass)", name );
            dInput = lstm->backward(dInput);
            // log_matrix( dInput );
        } else           
        if (auto gru = std::dynamic_pointer_cast<GRU<T>>(op)) {
            log_detail("Node [{0}] Encoder Operation (Backward Pass)", name );
            dInput = gru->backward(dInput);
            // log_matrix( dInput );
        } 
    }

    // Propagate gradients to next nodes.

    log_detail("Node [{0}] Propagating Gradient", name );
    log_matrix( dInput );

    propagateGradients(dInput); // Let's make sure other input nodes get the gradiens.

}

template <class T>
void Node<T>::updateParameters(std::string& optimizertype, T& learningRate, int& iter) {

    log_info( "*****************************************" );
    log_info( "***     Node Parameter Update  **********" );
    log_info( "*****************************************" );

    log_detail("Node: {0}", name );

    for (const auto& op : operations ) {
            // Check the dynamic type of the object using dynamic_cast
        if (auto linear = std::dynamic_pointer_cast<Linear<T>>(op)) {
            log_detail("Node [{0}] Linear Operation (Update Params)", name );
            linear->updateParameters(optimizertype, learningRate, iter);
        } else
        if (auto batchnorm = std::dynamic_pointer_cast<BatchNorm<T>>(op)) {
            log_detail("Node [{0}] Barch Normal Operation (Update Params)", name );
            batchnorm->updateParameters(optimizertype, learningRate, iter);
        } else            
        if (auto layernorm = std::dynamic_pointer_cast<LayerNorm<T>>(op)) {
            log_detail("Node [{0}] Layer Normal Operation (Update Params)", name );
            layernorm->updateParameters(optimizertype, learningRate, iter);
        } else            
        if (auto attention = std::dynamic_pointer_cast<Attention<T>>(op)) {
            log_detail("Node [{0}] Attention Operation (Update Params)", name );
            attention->updateParameters(optimizertype, learningRate, iter);
        } else            
        if (auto feedforward = std::dynamic_pointer_cast<FeedForward<T>>(op)) {
            log_detail("Node [{0}] Feedforward Operation (Update Params)", name );
            feedforward->updateParameters(optimizertype, learningRate, iter);
        } else            
        if (auto encoder = std::dynamic_pointer_cast<Encoder<T>>(op)) {
            log_detail("Node [{0}] Encoder Operation (Update Params)", name );
            encoder->updateParameters(optimizertype, learningRate, iter);
        }  else            
        if (auto rnn = std::dynamic_pointer_cast<RNN<T>>(op)) {
            log_detail("Node [{0}] Encoder Operation (Update Params)", name );
            rnn->updateParameters(optimizertype, learningRate, iter);
        }  else            
        if (auto lstm = std::dynamic_pointer_cast<LSTM<T>>(op)) {
            log_detail("Node [{0}] Encoder Operation (Update Params)", name );
            lstm->updateParameters(optimizertype, learningRate, iter);
        }  else            
        if (auto gru = std::dynamic_pointer_cast<GRU<T>>(op)) {
            log_detail("Node [{0}] Encoder Operation (Update Params)", name );
            gru->updateParameters(optimizertype, learningRate, iter);
        } 
    }
}

template <class T>
std::string removeSpace(Node<T>* node) {
    std::string s = node->getName();
    s.erase(std::remove(s.begin(), s.end(), ' '), s.end());
    return s;
}

template <class T>
std::string replaceSpace(Node<T>* node) {
    std::string nodename = node->getName();
    std::replace(nodename.begin(), nodename.end(), ' ', '_');
    return nodename;
}

template <class T>
std::string Node<T>::generateDotFormat() {
    std::string nodename = replaceSpace(this);
    std::string nodelabel = nodename + "_label";
    std::string dot_ = "";
    int cnt = 0; 
    for (const auto& op : operations ) {
            // Check the dynamic type of the object using dynamic_cast
        if (auto linear = std::dynamic_pointer_cast<Linear<T>>(op)) {
            dot_ +=  linear->generateDotFormat();
        } else
        if (auto batchnorm = std::dynamic_pointer_cast<BatchNorm<T>>(op)) {
            dot_ +=  batchnorm->generateDotFormat();
        } else            
        if (auto layernorm = std::dynamic_pointer_cast<LayerNorm<T>>(op)) {
            dot_ += layernorm->generateDotFormat();
        } else               
        if (auto activation = std::dynamic_pointer_cast<Activation<T>>(op)) {
            dot_ += activation->generateDotFormat();
        } else
        if (auto attention = std::dynamic_pointer_cast<Attention<T>>(op)) {
            dot_ += attention->generateDotFormat();
        } else            
        if (auto feedforward = std::dynamic_pointer_cast<FeedForward<T>>(op)) {
            dot_ += feedforward->generateDotFormat();
        } else            
        if (auto encoder = std::dynamic_pointer_cast<Encoder<T>>(op)) {
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

template <class T>
Node<T>* Connection<T>::getSource() {
    return source;
}

template <class T>
Node<T>* Connection<T>::getDestination() {
    return destination;
}

template <class T>
void Connection<T>::forwardPass() {
}

template <class T>
const aitensor<T> Connection<T>::backwardPass(const aitensor<T>& gradients) {
    return gradients;
}

/*****************************************************************************************************
* Base Graph Functions
*****************************************************************************************************/

// Create a node with three arguments: name, type, and initial values
/*
template <class T>
Node<T>* Graph<T>::createNode(const std::string& name, NodeType type, const py::array_t<T>& embedding) {
    Node<T>* node = new Node<T>(name, type, embedding);
    nodes.push_back(node);
    return node;
}
*/

// Create a node with two arguments: name and type (no initial values)
template <class T>
Node<T>* Graph<T>::createNode(const std::string& name, NodeType type) {
    Node<T>* node = new Node<T>(name, type);
    nodes.push_back(node);
    return node;
}

template <class T>
void Graph<T>::connect(Node<T>* from, Node<T>* to) {
    addConnection(new Connection<T>(from, to));
}

template <class T>
void Graph<T>::connect(Node<T>* from, Node<T>* to, std::vector<std::shared_ptr<BaseOperator>>& operations) {
    to->setOperations(operations);
    addConnection(new Connection<T>(from, to));
}

template <class T>
void Graph<T>::connect(std::vector<Node<T>*> from_nodes, Node<T>* to) {
    for (Node<T>* from : from_nodes) {
        addConnection(new Connection<T>(from, to));
    }
}

template <class T>
void Graph<T>::connect(std::vector<Node<T>*> from_nodes, Node<T>* to, std::vector<std::shared_ptr<BaseOperator>>& operations) {
    to->setOperations(operations);
    for (Node<T>* from : from_nodes) {
        addConnection(new Connection<T>(from, to));
    }
}

template <class T>
void Graph<T>::addConnection(Connection<T>* connection) {

    log_info( "***************************************" );
    log_info( "***    Graph Add Connection  **********" );
    log_info( "***************************************" );

    connections.push_back(connection);
    Node<T>* from = connection->getSource();
    Node<T>* to = connection->getDestination();
    outdegree[from]++;
    indegree[to]++;

    to->addInput(from);
    from->addOutput(to);

    log_detail( "Source Node: {0}", from->getName() );
    log_detail( "Destination Node: {0}", to->getName() );
    log_detail( "Number of Nodes: {:d}", nodes.size() );
}

template <class T>
std::vector<Node<T>*> Graph<T>::getNodes() {
    return nodes;
}

// Perform the Kahn's Algorithm by Arthur B. Khan based on his 1962 paper, "Topological Sorting of Large Networks"
template <class T>
const aitensor<T>& Graph<T>::forwardPropagation() {

    std::queue<Node<T>*> q;
    std::unordered_map<Node<T>*, int> indegree_(indegree); 

    log_info( "***************************************" );
    log_info( "*****    Graph Forward Pass  **********" );
    log_info( "***************************************" );

    log_detail( "Number of Nodes: {0}", getNodes().size() );

    for (Node<T>* node : nodes) {
        if (indegree[node] == 0) {
            q.push(node);
        }
    }

    log_detail( "Graph: Collecting nodes for the queue." );
    log_detail( "Size of queue: {0}",  q.size() );

    const aitensor<T>& output(0, 0, 0);

    log_detail(  "Graph: Entering Queue Loop." );

    while (!q.empty()) {
        Node<T>* node = q.front();
        q.pop();

        log_detail( "*** Graph: Entering forward pass for {0} ****", node->name );

        output = node->forwardPass();     

        for (Connection<T>* connection : this->connections) {
            if (connection->getSource() == node) {

                //std::cout << "Processing completed for connection.\n";
                //connection->forwardPass();
        
                // Propagate gradient to source node
                Node<T>* dstNode = connection->getDestination();

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

template <class T>
const aitensor<T>& Graph<T>::backwardPropagation(const aitensor<T>& gradients) {
    std::queue<Node<T>*> q;
    std::unordered_map<Node<T>*, int> outdegree_(outdegree); 

    log_info( "***************************************" );
    log_info( "*****    Graph Backward Pass  *********" );
    log_info( "***************************************" );

    log_detail( "Number of Nodes:", getNodes().size() );

    for (Node<T>* node : nodes) {
        if (outdegree[node] == 0) {
            node->setGradients(gradients); // gradients with respect to loss function.
            q.push(node);
        }
    }

    log_detail( "Collecting nodes for the queue." );
    log_detail( "Size of queue: {0}", q.size() );

    log_detail( "Entering Queue Loop." );

    while (!q.empty()) {
        Node<T>* node = q.front();
        q.pop();

        log_detail( "*** Graph: Entering backward pass for {0} ****", node->name );

        // Compute gradients for the node
        node->backwardPass();

        for (Connection<T>* connection : this->connections) {
            if (connection->getDestination() == node) {

                //std::cout << "Processing completed for connection.\n";
                //Eigen::MatrixXd srcGradients = connection->backwardPass(gradients);

                // Propagate gradient to source node
                Node<T>* srcNode = connection->getSource();

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

template <class T>
const aimatrix<T>& Graph<T>::computeLoss(std::string losstype, const aitensor<T>& predicted, const aitensor<T>& target) {

    log_info( "***************************************************" );
    log_info( "*****    Graph: Processing Loss Function  *********" );
    log_info( "***************************************************" );

    this->lossobj = new Loss<T>(losstype);

    aimatrix<T> loss = this->lossobj->computeLoss(predicted, target);

    log_detail( "Loss calculated: " );
    log_matrix( loss );

    return loss;
}

template <class T>
const aitensor<T>& Graph<T>::computeGradients(const aitensor<T>& predicted, const aitensor<T>& target) {

    log_info( "*********************************************" );
    log_info( "*****    Graph: Processing Gradient *********" );
    log_info( "*********************************************" );

    aitensor<T> gradients = this->lossobj->computeGradients(predicted, target);

    log_detail( "Loss Gradient calculated ..." );
    log_matrix( gradients );

    return gradients;
}

template <class T>
void Graph<T>::updateParameters(std::string& optimizertype, T& learningRate, int& iter) {

    log_info( "*******************************************************" );
    log_info( "*****    Graph: Processing Parameter Update ***********" );
    log_info( "*******************************************************" );

    for (Node<T>* node: nodes) {
        node->updateParameters(optimizertype, learningRate, iter);
    }

    log_detail( "Graph: parameter update completed for all nodes." );

}

template <class T>
std::string Graph<T>::generateDotFormat() {
    std::string dot = 
        "digraph G {  node [shape=circle]; rankdir=LR; ";

    for (Node<T>* node: nodes) {
        dot += removeSpace(node) + "; ";
    }

    for (Connection<T>* connection : this->connections) {
        Node<T> *source = connection->getSource();
        Node<T> *destination = connection->getDestination();
        std::string edge = removeSpace(source) + "->" + removeSpace(destination) + ";";
        dot += edge;
    }

    for (Node<T>* node: nodes) {
        dot += node->generateDotFormat();
    }

    dot += "}";
    return dot;
}

template <class T>
void Graph<T>::nextBatch() {
    for (Node<T>* node: nodes) {
        if (node->nodeType() == NodeType::Input)  {

        }
    }
}

template <class T>
const std::unordered_map<Node<T>*, int>& Graph<T>::getIndegree() const {
    return indegree;
}



