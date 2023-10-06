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
#include "logger.h"
#include "recurrent.h"
#include "topology.h"

namespace py = pybind11;
using namespace py::literals;

/********************************************************************************************
* NodeFactory
********************************************************************************************/

// The input is assumed to have NxM where N=number of samples, M=embedding vector size
// This allows to compute for the output size,  MxW where W is the number of weights (features) to use.
template <class T>
void Node<T>::setData(const py::array_t<T>& data, const bool normalize) {
    log_detail("Node: [{0}] Setting Data of Size: {1}", this->getName());
    this->input_data = ConvertData::totensor(data);
    if (normalize == true) {
        int input_size = this->input_data.size();
        for (int i=0; i < input_size; i++) {
            this->input_data.at(i).array() = BaseOperator::standardize(this->input_data.at(i));
        }   
    }
}
 
template <class T>
void Node<T>::setData(const aitensor<T> data, const bool normalize) {
    this->input_data =  data;
    if (normalize == true) {
        int input_size = this->input_data.size();
        for (int i=0; i < input_size; i++) {
            this->input_data.at(i).array() = BaseOperator::standardize(this->input_data.at(i));
        }   
    }
}

template <class T>
const aitensor<T>& Node<T>::getInput() {
    return this->input_data;
}

template <class T>
const aitensor<T>& Node<T>::getOutput() {
    return this->output_data;
}

template <class T>
void Node<T>::addInput(Node<T>* input, Node<T>* output) {
    inputs.insert(input);
    input->outputs.insert(output); // struggling to use enable_from_this()
}

template <class T>
void Node<T>::addOutput(Node<T>* output, Node<T>* input) {
    outputs.insert(output);
    output->inputs.insert(input);  // struggling to use enable_from_this()
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
Node<T>* Node<T>::setOperations(std::vector<BaseOperator*>& operations) {
    log_detail("Node: [{0}] Setting Operation of Size: {1}", this->getName(), operations.size());
    this->operations = operations;
    return (Node<T>*) this;
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

    if (inputs.size() == 0) {
        return input_data;
    }
 
    log_detail( "Getting size of node input: {0}", input_data.size() );
    aiscalar<T> suminputs = 0.0;
    for (auto& node : this->inputs) { // Check if we have nodes that are feeding this node.
        aitensor<T> external_input = node->getOutput();
        log_detail( "Getting size of node [{0}] output: {1}", node->getName(), external_input.size() );
        if (aggregate_input.size() == 0) {
            aggregate_input = external_input;
            continue; // See if there are more external inputs.
        }
        if (reduce == "add" || reduce == "avg") {
            log_detail( "Aggregating Data by add or average ..." );
            for (int i = 0; i < (int) external_input.size(); i++) {
                aggregate_input.at(i) += external_input.at(i);
            }
            suminputs = suminputs + 1.0;
        } else
        if (reduce == "mul") {
            log_detail( "Aggregating Data by mul ..." );
            for (int i = 0; i < (int) external_input.size(); i++) {
                aggregate_input.at(i) = BaseOperator::matmul(aggregate_input.at(i), external_input.at(i));
            }
            // aggregate_input = aggregate_input * external_input;
        } else
        if (reduce == "concat") { // Assume adding two batches. Requires dimension 1 and 2 are all the same.
            for (int i = 0; i < (int) external_input.size(); i++) {
                aggregate_input.push_back( external_input.at(i) );
            }
        }

    }
    if (reduce == "avg") {
        log_detail( "Aggregating Data by average ..." );
        for (int i = 0; i < (int) aggregate_input.size(); i++) {
            aggregate_input.at(i) = aggregate_input.at(i) / suminputs;
        }
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
    for (auto& node : this->inputs) { // These are input nodes that connect to this node.
        if (reduce == "add") {
            node->setGradients(gradients);
        } else
        if (reduce == "avg") {
            aitensor<T> dInput = gradients;
            for (int i=0; i < (int) gradients.size(); i++) {
                dInput.at(i) = dInput.at(i) /  this->suminputs;
            }
            node->setGradients(dInput);
        } else
        if (reduce == "mul") {
            // change to use more efficient element-wise function which uses GPU/TPU.
            aitensor<T> dInput = gradients;
            int nbatch = dInput.size();
            for (auto& nodex : this->outputs) {
                if (nodex->getName() != node->getName()) {
                    aitensor<T> output = nodex->getOutput();
                    for (int i=0; i < nbatch; i++) {
                        dInput.at(i) = dInput.at(i) * output.at(i);
                    }
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

    std::string name = this->getName();

    log_info( "**************************************" );
    log_info( "***      Node Forward Pass  **********" );
    log_info( "**************************************" );

    log_detail("Node forward: ({0}) Operation Size: {1}", name, size);

    // See if we can perform reduction.
    aitensor<T> output = aggregateData(this->input_data); // see Node.setData

    for (const auto& op : operations ) {
        // Check the dynamic type of the object using dynamic_cast
        // if (auto linear = std::dynamic_pointer_cast<Linear<T>>(op)) {
        if (Linear<T>* linear = dynamic_cast<Linear<T>*>(op)) {
            log_detail("Node [{0}] Linear Operation  (Forward Pass) Size: {1}", name, size);
            output = linear->forward(output); 
            log_info("Returned Linear Forward pass with the below output ...");
            log_matrix( output );
        } else
        //if (auto batchnorm = std::dynamic_pointer_cast<BatchNorm<T>>(op)) {
        if (BatchNorm<T>* batchnorm = dynamic_cast<BatchNorm<T>*>(op)) {
            log_detail("Node [{0}] Batch Normal Operation (Forward Pass)", name );
            output = batchnorm->forward(output);
            log_info("Returned Batch Normal pass with the below output ...");
            log_matrix( output );
        } else
        //if (auto layernorm = std::dynamic_pointer_cast<LayerNorm<T>>(op)) {
        if (LayerNorm<T>* layernorm = dynamic_cast<LayerNorm<T>*>(op)) {
            log_detail("Node [{0}] Layer Normal Operation (Forward Pass)", name );
            output = layernorm->forward(output);
            log_info("Returned Layer Normal pass with the below output ...");
            log_matrix( output );
        } else
        //if (auto activate = std::dynamic_pointer_cast<Activation<T>>(op)) {
        if (Activation<T>* activate = dynamic_cast<Activation<T>*>(op)) {
            log_detail("Node [{0}] Activation Operation (Forward Pass)", name );
            output = activate->forward(output);
            log_info("Returned Activation pass with the below output ...");
            log_matrix( output );
        } else
        //if (auto attention = std::dynamic_pointer_cast<Attention<T>>(op)) {
        if (Attention<T>* attention = dynamic_cast<Attention<T>*>(op)) {
            log_detail("Node [{0}] Attention Operation (Forward Pass)", name );
            output = attention->forward(output);
            log_info("Returned Attention pass with the below output ...");
            log_matrix( output );
        } else
        //if (auto feedforward = std::dynamic_pointer_cast<FeedForward<T>>(op)) {
        if (FeedForward<T>* feedforward = dynamic_cast<FeedForward<T>*>(op)) {
            log_detail("Node [{0}] FeedForward Operation (Forward Pass)", name );
            output = feedforward->forward(output);
            log_info("Returned FeedForward pass with the below output ...");
            log_matrix( output );
        } else
        //if (auto encoder = std::dynamic_pointer_cast<Encoder<T>>(op)) {
        if (Encoder<T>* encoder = dynamic_cast<Encoder<T>*>(op)) {
            log_detail("Node [{0}] Encoder Operation (Forward Pass)", name );
            output = encoder->forward(output);
            log_info("Returned Encoder pass with the below output ...");
            log_matrix( output );
        } else
        //if (auto rnn = std::dynamic_pointer_cast<RNN<T>>(op)) {
        if (RNN<T>* rnn = dynamic_cast<RNN<T>*>(op)) {
            log_detail("Node [{0}] RNN Operation (Forward Pass)", name );
            output = rnn->forward(output);
            log_info("Returned RNN pass with the below output ...");
            log_matrix( output );
        } else
        //if (auto lstm = std::dynamic_pointer_cast<LSTM<T>>(op)) {
        if (LSTM<T>* lstm = dynamic_cast<LSTM<T>*>(op)) {
            log_detail("Node [{0}] LSTM Operation (Forward Pass)", name );
            output = lstm->forward(output);
            log_info("Returned LSTM pass with the below output ...");
            log_matrix( output );
        } else
        //if (auto gru = std::dynamic_pointer_cast<GRU<T>>(op)) {
        if (GRU<T>* gru = dynamic_cast<GRU<T>*>(op)) {
            log_detail("Node [{0}] GRU Operation (Forward Pass)", name );
            output = gru->forward(output);
            log_info("Returned GRU with the below output ...");
            log_matrix( output );
        }
    }

    this->output_data = output; // Cache output for input of connecting nodes.
}

template <class T>
void Node<T>::backwardPass() {
    // Propagate backward gradients to connected nodes
    int size = operations.size();

    std::string name = this->getName();

    log_info( "**************************************" );
    log_info( "***     Node Backward Pass  **********" );
    log_info( "**************************************" );

    log_detail("Node: {0} Size: {1}", name, size);

    // Create a copy of the original vector
    std::vector<BaseOperator*> reversedOperations = operations;

    // If we are dealing with 3D
    aitensor<T> dInput = this->gradients; // initially generated through Graph.backwardPropagation()

    // Reverse the elements in the copied vector
    std::reverse(reversedOperations.begin(), reversedOperations.end());

    // Here, dInput is assumed to have already been propagated
    // through setGradients or propagatGradients.
    for (const auto& op : reversedOperations ) {
        // if (auto linear = std::dynamic_pointer_cast<Linear<T>>(op)) {
        if (Linear<T>* linear = dynamic_cast<Linear<T>*>(op)) {
            log_detail("Node [{0}] Linear Operation (Backward Pass)", name );
            dInput = linear->backward(dInput);
            log_matrix( dInput );
        } else
        //if (auto batchnorm = std::dynamic_pointer_cast<BatchNorm<T>>(op)) {
        if (BatchNorm<T>* batchnorm = dynamic_cast<BatchNorm<T>*>(op)) {
            log_detail("Node [{0}] Batch Normal Operation (Backward Pass)", name );
            dInput = batchnorm->backward(dInput);
            log_matrix( dInput );
        } else            
        //if (auto layernorm = std::dynamic_pointer_cast<LayerNorm<T>>(op)) {
        if (LayerNorm<T>* layernorm = dynamic_cast<LayerNorm<T>*>(op)) {
            log_detail("Node [{0}] Layer Normal Operation (Backward Pass)", name );
            dInput = layernorm->backward(dInput);
            log_matrix( dInput );
        } else           
        //if (auto activate = std::dynamic_pointer_cast<Activation<T>>(op)) {
        if (Activation<T>* activate = dynamic_cast<Activation<T>*>(op)) {
            log_detail("Node [{0}] Activation Operation (Backward Pass)", name );
            dInput = activate->backward(dInput);
            log_matrix( dInput );
        } else        
        //if (auto attention = std::dynamic_pointer_cast<Attention<T>>(op)) {
        if (Attention<T>* attention = dynamic_cast<Attention<T>*>(op)) {
            log_detail("Node [{0}] Attention Operation (Backward Pass)", name );
            dInput = attention->backward(dInput);
            log_matrix( dInput );
        } else           
        //if (auto feedforward = std::dynamic_pointer_cast<FeedForward<T>>(op)) {
        if (FeedForward<T>* feedforward = dynamic_cast<FeedForward<T>*>(op)) {
            log_detail("Node [{0}] Feedforward Operation (Backward Pass)", name );
            dInput = feedforward->backward(dInput);
            log_matrix( dInput );
        } else           
        //if (auto encoder = std::dynamic_pointer_cast<Encoder<T>>(op)) {
        if (Encoder<T>* encoder = dynamic_cast<Encoder<T>*>(op)) {
            log_detail("Node [{0}] Encoder Operation (Backward Pass)", name );
            dInput = encoder->backward(dInput);
            log_matrix( dInput );
        } else       
        //if (auto rnn = std::dynamic_pointer_cast<RNN<T>>(op)) {
        if (RNN<T>* rnn = dynamic_cast<RNN<T>*>(op)) {
            log_detail("Node [{0}] RNN Operation (Backward Pass)", name );
            dInput = rnn->backward(dInput);
            log_matrix( dInput );
        } else           
        //if (auto lstm = std::dynamic_pointer_cast<LSTM<T>>(op)) {
        if (LSTM<T>* lstm = dynamic_cast<LSTM<T>*>(op)) {
            log_detail("Node [{0}] LSTM Operation (Backward Pass)", name );
            dInput = lstm->backward(dInput);
            log_matrix( dInput );
        } else           
        //if (auto gru = std::dynamic_pointer_cast<GRU<T>>(op)) {
        if (GRU<T>* gru = dynamic_cast<GRU<T>*>(op)) {
            log_detail("Node [{0}] GRU Operation (Backward Pass)", name );
            dInput = gru->backward(dInput);
            log_matrix( dInput );
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

    std::string name = this->getName();

    log_detail("Node: {0}", name );

    for (const auto& op : operations ) {
        // Check the dynamic type of the object using dynamic_cast
        //if (auto linear = std::dynamic_pointer_cast<Linear<T>>(op)) {
        if (Linear<T>* linear = dynamic_cast<Linear<T>*>(op)) {
            log_detail("Node [{0}] Linear Operation (Update Params)", name );
            linear->updateParameters(optimizertype, learningRate, iter);
        } else
        //if (auto batchnorm = std::dynamic_pointer_cast<BatchNorm<T>>(op)) {
        if (BatchNorm<T>* batchnorm = dynamic_cast<BatchNorm<T>*>(op)) {
            log_detail("Node [{0}] Batch Normal Operation (Update Params)", name );
            batchnorm->updateParameters(optimizertype, learningRate, iter);
        } else            
        //if (auto layernorm = std::dynamic_pointer_cast<LayerNorm<T>>(op)) {
        if (LayerNorm<T>* layernorm = dynamic_cast<LayerNorm<T>*>(op)) {
            log_detail("Node [{0}] Layer Normal Operation (Update Params)", name );
            layernorm->updateParameters(optimizertype, learningRate, iter);
        } else      
        //if (auto attention = std::dynamic_pointer_cast<Attention<T>>(op)) {
        if (Attention<T>* attention = dynamic_cast<Attention<T>*>(op)) {
            log_detail("Node [{0}] Attention Operation (Update Params)", name );
            attention->updateParameters(optimizertype, learningRate, iter);
        } else            
        //if (auto feedforward = std::dynamic_pointer_cast<FeedForward<T>>(op)) {
        if (FeedForward<T>* feedforward = dynamic_cast<FeedForward<T>*>(op)) {
            log_detail("Node [{0}] Feedforward Operation (Update Params)", name );
            feedforward->updateParameters(optimizertype, learningRate, iter);
        } else            
        //if (auto encoder = std::dynamic_pointer_cast<Encoder<T>>(op)) {
        if (Encoder<T>* encoder = dynamic_cast<Encoder<T>*>(op)) {
            log_detail("Node [{0}] Encoder Operation (Update Params)", name );
            encoder->updateParameters(optimizertype, learningRate, iter);
        }  else         
        //if (auto rnn = std::dynamic_pointer_cast<RNN<T>>(op)) {
        if (RNN<T>* rnn = dynamic_cast<RNN<T>*>(op)) {
            log_detail("Node [{0}] RNN Operation (Update Params)", name );
            rnn->updateParameters(optimizertype, learningRate, iter);
        }  else            
        //if (auto lstm = std::dynamic_pointer_cast<LSTM<T>>(op)) {
        if (LSTM<T>* lstm = dynamic_cast<LSTM<T>*>(op)) {
            log_detail("Node [{0}] LSTM Operation (Update Params)", name );
            lstm->updateParameters(optimizertype, learningRate, iter);
        }  else            
        //if (auto gru = std::dynamic_pointer_cast<GRU<T>>(op)) {
        if (GRU<T>* gru = dynamic_cast<GRU<T>*>(op)) {
            log_detail("Node [{0}] GRU Operation (Update Params)", name );
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
    std::string nodename = replaceSpace((Node<T>*) this);
    std::string nodelabel = nodename + "_label";
    std::string dot_ = "";
    int cnt = 0; 
    for (const auto& op : operations ) {
            // Check the dynamic type of the object using dynamic_cast
        //if (auto linear = std::dynamic_pointer_cast<Linear<T>>(op)) {
        if (Linear<T>* linear = dynamic_cast<Linear<T>*>(op)) {
            dot_ +=  linear->generateDotFormat();
        } else
        //if (auto batchnorm = std::dynamic_pointer_cast<BatchNorm<T>>(op)) {
        if (BatchNorm<T>* batchnorm = dynamic_cast<BatchNorm<T>*>(op)) {
            dot_ +=  batchnorm->generateDotFormat();
        } else            
        //if (auto layernorm = std::dynamic_pointer_cast<LayerNorm<T>>(op)) {
        if (LayerNorm<T>* layernorm = dynamic_cast<LayerNorm<T>*>(op)) {
            dot_ += layernorm->generateDotFormat();
        } else               
        //if (auto activation = std::dynamic_pointer_cast<Activation<T>>(op)) {
        if (Activation<T>* activation = dynamic_cast<Activation<T>*>(op)) {
            dot_ += activation->generateDotFormat();
        }  else
        //if (auto attention = std::dynamic_pointer_cast<Attention<T>>(op)) {
        if (Attention<T>* attention = dynamic_cast<Attention<T>*>(op)) {
            dot_ += attention->generateDotFormat();
        } else            
        //if (auto feedforward = std::dynamic_pointer_cast<FeedForward<T>>(op)) {
        if (FeedForward<T>* feedforward = dynamic_cast<FeedForward<T>*>(op)) {
            dot_ += feedforward->generateDotFormat();
        } else            
        //if (auto encoder = std::dynamic_pointer_cast<Encoder<T>>(op)) {
        if (Encoder<T>* encoder = dynamic_cast<Encoder<T>*>(op)) {
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
*************t****************************************************************************************/

template <class T>
Node<T>* Graph<T>::findNode(const std::string& nodename) {
    for (const auto& node : this->getNodes()) {
        // auto pnode = std::dynamic_pointer_cast<Node<T>>(node);
        if (nodename == node->getName()) {
            return node;
        }
    }
    return nullptr;
}

// Create a node with two arguments: name and type (no initial values)
template <class T>
Node<T>* Graph<T>::createNode(const std::string& name, NodeType type) {
    std::cout << "Creating Node 1 ..." << std::endl;
    Node<T>* node = new Node<T>(name, type); 
    std::cout << "Creating Node 2 ..." << std::endl;
    this->nodes.push_back(node);
    std::cout << "Creating Node 3 ..." << std::endl;
    return node;
}

template <class T>
void Graph<T>::setData(const std::string& nodename, const py::array_t<T>& input_data, const bool normalize) {
    Node<T>* node = this->findNode(nodename);
    if (node != nullptr) {
        node->setData(input_data, normalize);
    }
}

template <class T>
void Graph<T>::setData(const std::string& nodename, const aitensor<T>& input_data, const bool normalize) {
    Node<T>* node = this->findNode(nodename);
    if (node != nullptr) {
        node->setData(input_data, normalize);
    }
}

template <class T>
void Graph<T>::setOperations(const std::string& nodename, std::vector<BaseOperator*> operations) {
    Node<T>* node = this->findNode(nodename);
    if (node != nullptr) {
        node->setOperations(operations);
    }
}

template <class T>
void Graph<T>::connect(std::string from_name, std::string to_name) {
    Node<T>* from = this->findNode(from_name);
    Node<T>* to = this->findNode(to_name);
    addConnection(std::make_unique<Connection<T>>(from, to));
}

template <class T>
void Graph<T>::connect(Node<T>* from, Node<T>* to) {
    addConnection(std::make_unique<Connection<T>>(from, to));
}

template <class T>
void Graph<T>::connect(Node<T>* from, Node<T>* to, std::vector<BaseOperator*>& operations) {
    to->setOperations(operations);
    addConnection(std::make_unique<Connection<T>>(from, to));
}

template <class T>
void Graph<T>::connect(std::vector<Node<T>*> from_nodes, Node<T>* to) {
    for (auto& from : from_nodes) {
        addConnection(std::make_unique<Connection<T>>(from, to));
    }
}

template <class T>
void Graph<T>::connect(std::vector<Node<T>*> from_nodes, Node<T>* to, std::vector<BaseOperator*>& operations) {
    to->setOperations(operations);
    for (auto& from : from_nodes) {
        addConnection(std::make_unique<Connection<T>>(from, to));
    }
}
 
template <class T>
void Graph<T>::addConnection(std::shared_ptr<Connection<T>> connection) {

    log_info( "***************************************" );
    log_info( "***    Graph Add Connection  **********" );
    log_info( "***************************************" );

    connections.push_back(connection);
    Node<T>* from = connection->getSource();
    Node<T>* to = connection->getDestination();
    outdegree[from]++;
    indegree[to]++;
 
    to->addInput(from, to);
    from->addOutput(to, from);

    log_detail( "Source Node: ({0}), Destination Node: ({1})", from->getName(), to->getName());
    log_detail( "Number of Nodes: {:d}", this->getNodes().size() );
}

// Perform the Kahn's Algorithm by Arthur B. Khan based on his 1962 paper, "Topological Sorting of Large Networks"
template <class T>
const aitensor<T> Graph<T>::forwardPropagation() {

    log_detail( "Entered forward pass in Graph 1 ..." );

    std::queue<Node<T>*> q;

    log_detail( "Entered forward pass in Graph 2 ..." );

    std::unordered_map<Node<T>*, int> indegree_(indegree); 

    log_detail( "Iitialized done pass in Graph ..." );

    log_info( "***************************************" );
    log_info( "*****    Graph Forward Pass  **********" );
    log_info( "***************************************" );

    log_detail( "Number of Nodes: {0}", this->getNodes().size() );

    for (const auto& node : this->getNodes()) {
        if (indegree[node] == 0) {
            q.push(node);
        }
    }
 
    log_detail( "Graph: Collecting nodes for the queue." );
    log_detail( "Size of queue and Graph: {0} {1}",  q.size(), this->getNodes().size() );
    log_detail( "Graph: Entering Queue Loop." );

    aitensor<T> output;
 
    std::cout << "Entering queue ...." << std::endl;
    while (!q.empty()) {

    std::cout << "Entered 1 ...." << std::endl;

        const auto& node = q.front();

    std::cout << "Entered 2 ...." << std::endl;

        std::string nodename = node->getName();

    std::cout << "Entered 3 ...." << std::endl;

    std::cout << "*** Graph: Entering forward pass for {0} ***" << nodename << std::endl;

        log_detail( "*** Graph: Entering forward pass for {0} ***", nodename );
 
        // Perform the forward pass. 
        node->forwardPass();     

        // The last output becomes the final prediction.
        output = node->getOutput();

        q.pop();

        for (auto connection : this->connections) {
            if (connection->getSource()->getName() == nodename) {

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
const aitensor<T> Graph<T>::backwardPropagation(const aitensor<T>& gradients) {
    std::queue<Node<T>*> q;
    std::unordered_map<Node<T>*, int> outdegree_(outdegree); 

    log_info( "***************************************" );
    log_info( "*****    Graph Backward Pass  *********" );
    log_info( "***************************************" );

    log_detail( "Number of Nodes:", getNodes().size() );

    for (const auto& node : nodes) {
        if (outdegree[node] == 0) {
            node->setGradients(gradients); // gradients with respect to loss function.
            q.push(node);
        }
    }

    log_detail( "Collecting nodes for the queue." );
    log_detail( "Size of queue and size of Graph: {0} {1}", q.size(), this->getNodes().size() );

    log_detail( "Entering Queue Loop." );

    while (!q.empty()) {

        const auto& node = q.front();

        std::string nodename = node->getName();

        log_detail( "*** Graph: Entering backward pass for {0} ****", nodename );

        // Compute gradients for the node
        node->backwardPass();
    
        q.pop();

        for (auto& connection : this->connections) {
            if (connection->getDestination()->getName() == nodename) {

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
const aiscalar<T> Graph<T>::computeLoss(std::string losstype, const aitensor<T>& predicted, const aitensor<T>& target) {

    log_info( "***************************************************" );
    log_info( "*****    Graph: Processing Loss Function  *********" );
    log_info( "***************************************************" );

    std::cout << "Entering Graph computeLoss 1 ..." << std::endl;

    this->lossobj = new Loss<T>(losstype);

    std::cout << "Entering Graph computeLoss 2 ..." << std::endl;

    aiscalar<T> loss = this->lossobj->computeLoss(predicted, target);

    log_detail( "Loss calculated: " );
    log_scalar( loss );

    return loss;
}

template <class T>
const aitensor<T> Graph<T>::computeGradients(const aitensor<T>& predicted, const aitensor<T>& target) {

    log_info( "*********************************************" );
    log_info( "*****    Graph: Processing Gradient *********" );
    log_info( "*********************************************" );

    log_detail("Predicted:");
    log_matrix(predicted);

    log_detail("Target:");
    log_matrix(target);

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

    for (auto& node: nodes) {
        node->updateParameters(optimizertype, learningRate, iter);
    }

    log_detail( "Graph: parameter update completed for all nodes." );

}

template <class T>
std::string Graph<T>::generateDotFormat() {
    std::string dot = 
        "digraph G {  node [shape=circle]; rankdir=LR; ";

    for (auto& node: nodes) {
        dot += removeSpace(node) + "; ";
    }

    for (auto& connection : this->connections) {
        Node<T>* source = connection->getSource();
        Node<T>* destination = connection->getDestination();
        std::string edge = removeSpace(source) + "->" + removeSpace(destination) + ";";
        dot += edge;
    }

    for (auto& node: nodes) {
        dot += node->generateDotFormat();
    }

    dot += "}";
    return dot;
}

template <class T>
void Graph<T>::nextBatch() {
    for (auto& node: nodes) {
        if (node->nodeType() == NodeType::Input)  {

        }
    }
}

template <class T>
const std::unordered_map<Node<T>*, int>& Graph<T>::getIndegree() const {
    return indegree;
}

/************ Graph / Network initialize templates ************/

template class Node<float>;  // Instantiate with float
template class Node<double>;  // Instantiate with double

template class Graph<float>;  // Instantiate with float
template class Graph<double>;  // Instantiate with double



