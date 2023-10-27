/*
 * GENAI - A framework that can help build LLMs
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
  
#include "genai.h"
#include "topology.h"
#include "model.h"

namespace py = pybind11;
using namespace py::literals;

template <class T>
void BaseModel<T>::setGraph(std::shared_ptr<Graph<T>>  graph) {

    // auto mygraph = dynamic_cast<Graph<T>*>(graph);
    // this->graph = mygraph;
    this->graph = graph;
    log_detail( "SetGraph ... Node count: {:d}", this->graph->getNodes().size() );
}

template <class T>
std::shared_ptr<Graph<T>> BaseModel<T>::getGraph( ) {
    return this->graph;
}

template <class T>
void BaseModel<T>::setLoss(std::string& losstype) {
    this->losstype = losstype;
}

// The output is assumed to have BxNxM where B=batch/sequence size, N=number of samples, M=embedding vector size
// This allows to compute for the output size,  MxW where W is the number of weights (features) to use.
template <class T>
void BaseModel<T>::setTarget(const py::array_t<T>& target) {
    this->target = ConvertData::totensor(target);
}

template <class T>
aitensor<T> BaseModel<T>::getTarget() {
    return this->target;
}

template <class T>
void BaseModel<T>::useCrossEntropy() {

}

template <class T>
void BaseModel<T>::train(std::string& losstype, std::vector<std::string>& metricstype, std::string& optimizertype, const T learningRate , const int max_epoch) {

    // Initialize MPI
    //MPI_Init(NULL, NULL);

    this->losstype = losstype;
    this->optimizertype = optimizertype;
    this->learningRate = learningRate;

    log_info( "******************************************************************************************" );
    log_info( "********************************* Start Training *****************************************")
    log_info( "******************************************************************************************" );
    log_detail( "Number of Graph Nodes: {:d}", this->graph->getNodes().size() );

    aiscalar<T> epsilon = 1e-3;
    aiscalar<T> old_loss = inf();

    std::cout << "Setting Training Time ..." << std::endl;

    auto start_time = std::chrono::system_clock::now();

    std::cout << "Starting Iteration ..." << std::endl;
 
    for (int iter = 1; iter <= max_epoch; iter++) {

        log_detail( "<<<<<<<<<<<<<<<<<<<<<<<<< Process batch (iteration {:d})  >>>>>>>>>>>>>>>>>>>>>>>>>>", (iter) );
        this->graph->nextBatch();

        log_detail( "Entering Forward Propagation ..." );
        this->predicted = this->graph->forwardPropagation();

        log_detail( "Predicted Result: Tensor Size {0}", this->predicted.size() );
        log_matrix( this->predicted );

        log_detail( "Computing Loss ..." );
        this->loss = this->graph->computeLoss(this->losstype, this->predicted, this->target); 

        log_detail( "Computing Loss Gradient ..." );
        this->gradients = this->graph->computeGradients(this->predicted, this->target);
        log_matrix( this->gradients );

        log_detail( "Entering Backward Propagation ..." );
        this->gradients = this->graph->backwardPropagation(this->gradients); 

        log_detail( "Updating Parameters ..." );
        this->graph->updateParameters(this->optimizertype, this->learningRate, iter);

        log_detail( "Calculate Performance Metrics ...");
        this->metrics = this->graph->computeMetrics(this->metricstype, this->predicted, this->target);

        // Calculate Time, then display loss
        auto end_time = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end_time - start_time;
        std::time_t next_time = std::chrono::system_clock::to_time_t(end_time);
        start_time = end_time;
        py_cout << "Epoch " << iter << "/" << max_epoch << " ...";
        py_cout << "Loss: " << this->loss;
        // py_cout << "Accuracy: " << this->metrics;
        py_cout << " ... elapsed " <<  elapsed_seconds.count();
        py_cout << " at " << std::ctime(&next_time) << std::endl;

        // Also, log the result if Logging INFO is enabled
        log_detail( "Epoch {}/{} ... Loss: {:8.5f} ... Accuracy: {:8.5f} ... Elapsed {} at {}", iter, max_epoch, 
                this->loss, this->metrics, elapsed_seconds.count(), std::ctime(&next_time) );

        if (abs(old_loss - this->loss) <= epsilon) break;

    }

    log_detail( "Training done ..." );

    // Finalize MPI
    //MPI_Finalize();

}
  
/**************************************************************************************************
* ModelNode::setDataFloat and setDecoderDataFloat
* Temporarily store np.array (passed as dtype = np.float32) to a double pointer (this->input_fdata).
* Upon training entry, the double pointer will be transformed to an aitensor and handed over
* to the Node class.
**************************************************************************************************/
void ModelNode::setDataFloat(const py::array_t<float>& data, const bool normalize) {
    try {
        if (datatype == "double") {
            throw AIException("Precision used in data is 'float' but the model uses 'double' ...");
        }

        this->input_fdata = ConvertData::totensor(data);
        this->normalize   = normalize;

    } catch (const AIException& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    } catch (const std::exception& e) {
        // Catch standard exceptions
        std::cerr << "Standard Error: " << e.what() << " at " << __LINE__ << std::endl;
    } catch (...) {
        // Catch all other exceptions
        std::cerr << "Unknown Error:" << std::endl;
    }
}

void ModelNode::setDecoderDataFloat(const py::array_t<float>& data, const bool normalize) {
    try {
        if (datatype == "double") {
            throw AIException("Precision used in data is 'float' but the model uses 'double' ...");
        }

        this->decoder_fdata = ConvertData::totensor(data);
        this->normalize   = normalize;

    } catch (const AIException& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    } catch (const std::exception& e) {
        // Catch standard exceptions
        std::cerr << "Standard Error: " << e.what() << " at " << __LINE__ << std::endl;
    } catch (...) {
        // Catch all other exceptions
        std::cerr << "Unknown Error:" << std::endl;
    }
}

/**************************************************************************************************
* MModelNode::setDataDouble and setDecoderDataDouble
* Temporarily store np.array (passed as dtype = np.float64) to a double pointer (this->input_ddata).
* Upon training entry, the double pointer will be transformed to an aitensor and handed over
* to the Node class.
**************************************************************************************************/
void ModelNode::setDataDouble(const py::array_t<double>& data, bool const normalize) {

    try {
        if (datatype == "float") {
            throw AIException("Precision used in data is 'double' but the model uses 'float' ...");
        }

        this->input_ddata = ConvertData::totensor(data);
        this->normalize   = normalize;

    } catch (const AIException& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    } catch (const std::exception& e) {
        // Catch standard exceptions
        std::cerr << "Standard Error: " << e.what() << " at " << __LINE__ << std::endl;
    } catch (...) {
        // Catch all other exceptions
        std::cerr << "Unknown Error:" << std::endl;
    }
}

void ModelNode::setDecoderDataDouble(const py::array_t<double>& data, bool const normalize) {

    try {
        if (datatype == "float") {
            throw AIException("Precision used in data is 'double' but the model uses 'float' ...");
        }

        this->decoder_ddata = ConvertData::totensor(data);
        this->normalize   = normalize;

    } catch (const AIException& e) {
        std::cerr << "Error: " << e.what() << std::endl;
    } catch (const std::exception& e) {
        // Catch standard exceptions
        std::cerr << "Standard Error: " << e.what() << " at " << __LINE__ << std::endl;
    } catch (...) {
        // Catch all other exceptions
        std::cerr << "Unknown Error:" << std::endl;
    }
}

/**************************************************************************************************
* ModelNode::setOperations
* Captures operations for the node and temporarily caches into the ModelNode instance. Later, it will
* be transfered to the main Node class as part of preparation for Model.train().
**************************************************************************************************/
void ModelNode::setOperations(std::vector<std::shared_ptr<BaseOperator>>& operations) {
    // Let's now seed with operations

    for (auto& op : operations) {
        // Check the dynamic type of the object using dynamic_cast
        if (auto linear = std::dynamic_pointer_cast<ModelLinear>(op)) {
            if (datatype == "float") {
                Linear<float>* newop = new Linear<float>(
                                                            linear->getSize(), 
                                                            linear->getBias()
                                                        );
                this->operations.push_back(newop);
            } else 
            if (datatype == "double") {
                Linear<double>* newop = new Linear<double>(
                                                            linear->getSize(), 
                                                            linear->getBias()
                                                        );
                this->operations.push_back(newop);
            }
        } else
        if (auto batchnorm = std::dynamic_pointer_cast<ModelBatchNorm>(op)) {
            if (datatype == "float") {
                BatchNorm<float>* newop = new BatchNorm<float>();
                this->operations.push_back(newop);
            } else 
            if (datatype == "double") {
                BatchNorm<double>* newop = new BatchNorm<double>();
                this->operations.push_back(newop);
            }
        } else
        if (auto layernorm = std::dynamic_pointer_cast<ModelLayerNorm>(op)) {
            if (datatype == "float") {
                LayerNorm<float>* newop = new LayerNorm<float>();
                this->operations.push_back(newop);
            } else 
            if (datatype == "double") {
                LayerNorm<double>* newop = new LayerNorm<double>();
                this->operations.push_back(newop);
            } 
        } else
        if (auto activate = std::dynamic_pointer_cast<ModelActivation>(op)) {
            if (datatype == "float") {
                Activation<float>* newop = new Activation<float>(
                                                            activate->getActivationType(), 
                                                            activate->getAlpha()
                                                        );
                this->operations.push_back(newop);
            } else 
            if (datatype == "double") {
                Activation<double>* newop = new Activation<double>(
                                                            activate->getActivationType(), 
                                                            activate->getAlpha()
                                                        );
                this->operations.push_back(newop);
            }
        }  else  
        if (auto dropout = std::dynamic_pointer_cast<ModelDropout>(op)) {
            if (datatype == "float") {
                Dropout<float>* newop = new Dropout<float>(
                                                            dropout->getProbability()
                                                        );
                this->operations.push_back(newop);
            } else 
            if (datatype == "double") {
                Dropout<double>* newop = new Dropout<double>(
                                                            dropout->getProbability()
                                                        );
                this->operations.push_back(newop);
            }
        }  else
        if (auto flatten = std::dynamic_pointer_cast<ModelFlatten>(op)) {
            if (datatype == "float") {
                Flatten<float>* newop = new Flatten<float>();
                this->operations.push_back(newop);
            } else 
            if (datatype == "double") {
                Flatten<double>* newop = new Flatten<double>();
                this->operations.push_back(newop);
            }
        }  else
        if (auto convolution = std::dynamic_pointer_cast<ModelConvolution>(op)) {
            if (datatype == "float") {
                Convolution<float>* newop = new Convolution<float>(
                                                            convolution->getKernelSize(),
                                                            convolution->getStride(),
                                                            convolution->getPadding(),
                                                            convolution->getDilation(),
                                                            convolution->getBias()                                                                                                                                                                                                                                             
                                                        );
                this->operations.push_back(newop);
            } else 
            if (datatype == "double") {
                Convolution<double>* newop = new Convolution<double>(
                                                            convolution->getKernelSize(),
                                                            convolution->getStride(),
                                                            convolution->getPadding(),
                                                            convolution->getDilation(),
                                                            convolution->getBias()   
                                                        );
                this->operations.push_back(newop);
            }
        }  else  // Transformer Component
        if (auto attention = std::dynamic_pointer_cast<ModelAttention>(op)) {
            if (datatype == "float") {
                Attention<float>* newop = new Attention<float>(
                                                            attention->getSize(),
                                                            attention->getBias()
                                                        );
                this->operations.push_back(newop);
            } else 
            if (datatype == "double") {
                Attention<double>* newop = new Attention<double>(
                                                            attention->getSize(),
                                                            attention->getBias(),
                                                            attention->getMasked()
                                                        );
                this->operations.push_back(newop);
            }
        } else  // Transformer Component 
        if (auto feedforward = std::dynamic_pointer_cast<ModelFeedForward>(op)) {
            if (datatype == "float") {
                FeedForward<float>* newop = new FeedForward<float>(
                                                            feedforward->getSize(),
                                                            feedforward->getBias(),
                                                            feedforward->getActivationType(),
                                                            feedforward->getAlpha()
                                                        );
                this->operations.push_back(newop);
            } else 
            if (datatype == "double") {
                FeedForward<double>* newop = new FeedForward<double>(
                                                            feedforward->getSize(),
                                                            feedforward->getBias(),
                                                            feedforward->getActivationType(),
                                                            feedforward->getAlpha()
                                                        );
                this->operations.push_back(newop);
            }
        } else   // Transformer Component
        if (auto encoder = std::dynamic_pointer_cast<ModelEncoder>(op)) {
            if (datatype == "float") {
                Encoder<float>* newop = new Encoder<float>(
                                                            encoder->getHead(),
                                                            encoder->getSize(),
                                                            encoder->getBias(),
                                                            encoder->getActivationType(),
                                                            encoder->getAlpha()
                                                        );
                this->operations.push_back(newop);
            } else 
            if (datatype == "double") {
                Encoder<double>* newop = new Encoder<double>(
                                                            encoder->getHead(),
                                                            encoder->getSize(),
                                                            encoder->getBias(),
                                                            encoder->getActivationType(),
                                                            encoder->getAlpha()
                                                        );
                this->operations.push_back(newop);
            }
        } else   // Transformer Component
        if (auto decoder = std::dynamic_pointer_cast<ModelDecoder>(op)) {
            if (datatype == "float") {
                Decoder<float>* newop = new Decoder<float>(
                                                            decoder->getHead(),
                                                            decoder->getSize(),
                                                            decoder->getBias(),
                                                            decoder->getActivationType(),
                                                            decoder->getAlpha()
                                                        );
                this->operations.push_back(newop);
            } else 
            if (datatype == "double") {
                Decoder<double>* newop = new Decoder<double>(
                                                            decoder->getHead(),
                                                            decoder->getSize(),
                                                            decoder->getBias(),
                                                            decoder->getActivationType(),
                                                            decoder->getAlpha()
                                                        );
                this->operations.push_back(newop);
            }
        }  else   // Recurrent Network Component
        if (auto rnn = std::dynamic_pointer_cast<ModelRNN>(op)) {
            if (datatype == "float") {
                RNN<float>* newop = new RNN<float>(         rnn->getHiddenSize(),
                                                            rnn->getOuputSize(),
                                                            rnn->getOutputSequenceLength(), // For One-TO-Many Scenario
                                                            rnn->getNumLayers(),
                                                            rnn->getBiDirection(),
                                                            rnn->getRNNType()
                                                        );
                this->operations.push_back(newop);
            } else 
            if (datatype == "double") {
                RNN<double>* newop = new RNN<double>(       rnn->getHiddenSize(),
                                                            rnn->getOuputSize(),
                                                            rnn->getOutputSequenceLength(), // For One-TO-Many Scenario
                                                            rnn->getNumLayers(),
                                                            rnn->getBiDirection(),
                                                            rnn->getRNNType()
                                                        );
                this->operations.push_back(newop);
            }
        } else   // Recurrent Network Component
        if (auto lstm = std::dynamic_pointer_cast<ModelLSTM>(op)) {
            if (datatype == "float") {
                LSTM<float>* newop = new LSTM<float>(       lstm->getHiddenSize(),
                                                            lstm->getOuputSize(),
                                                            lstm->getOutputSequenceLength(), // For One-TO-Many Scenario
                                                            lstm->getNumLayers(),
                                                            lstm->getBiDirection(),
                                                            lstm->getRNNType()
                                                        );
                this->operations.push_back(newop);
            } else 
            if (datatype == "double") {
                LSTM<double>* newop = new LSTM<double>(     lstm->getHiddenSize(),
                                                            lstm->getOuputSize(),
                                                            lstm->getOutputSequenceLength(), // For One-TO-Many Scenario
                                                            lstm->getNumLayers(),
                                                            lstm->getBiDirection(),
                                                            lstm->getRNNType()
                                                        );
                this->operations.push_back(newop);
            }
        } else   // Recurrent Network Component
        if (auto gru = std::dynamic_pointer_cast<ModelGRU>(op)) {
            if (datatype == "float") {
                GRU<float>* newop = new GRU<float>(         gru->getHiddenSize(),
                                                            gru->getOuputSize(),
                                                            gru->getOutputSequenceLength(), // For One-TO-Many Scenario
                                                            gru->getNumLayers(),
                                                            gru->getBiDirection(),
                                                            gru->getRNNType()
                                                        );
                this->operations.push_back(newop);
            } else 
            if (datatype == "double") {
                GRU<double>* newop = new GRU<double>(       gru->getHiddenSize(),
                                                            gru->getOuputSize(),
                                                            gru->getOutputSequenceLength(), // For One-TO-Many Scenario
                                                            gru->getNumLayers(),
                                                            gru->getBiDirection(),
                                                            gru->getRNNType()
                                                        );
                this->operations.push_back(newop);
            }
        } 
    }
}

/************************************************************************************************
* Model::Model
* This is the Model Constructor Implementation - Note that this is only a meta model.
* The actual model is the BaseModel Class.
*************************************************************************************************/
Model::Model(const std::string& losstype, const std::string& optimizertype, 
        const double learningRate, const int max_epoch, const std::string& datatype) {
    this->losstype = losstype;
    this->optimizertype = optimizertype;
    this->learningRate = learningRate;
    this->max_epoch = max_epoch;
    this->datatype = datatype;
    if (datatype == "float") {
        std::shared_ptr<BaseModel<float>> bmodelf = std::make_shared<BaseModel<float>>(losstype, optimizertype, static_cast<float>(learningRate), max_epoch);
        std::shared_ptr<Graph<float>> graphXf = std::make_unique<Graph<float>>();
        bmodelf->setGraph(graphXf);
        this->modelXf = bmodelf;
    } else if (datatype == "double") {
        std::shared_ptr<BaseModel<double>> bmodeld = std::make_shared<BaseModel<double>>(losstype, optimizertype, static_cast<double>(learningRate), max_epoch);
        std::shared_ptr<Graph<double>> graphXd = std::make_unique<Graph<double>>();
        bmodeld->setGraph(graphXd);
        this->modelXd = bmodeld;
    } else {
        throw std::invalid_argument("Unsupported datatype");
    }
    std::cout << "Got here :" << datatype << " learningRate " << learningRate << std::endl;
}

/************************************************************************************************
* Model::addNode
* First, we create a meta node to support python API. Alongside, we also create the node 
* and register to the graph.
*************************************************************************************************/
std::shared_ptr<ModelNode> Model::addNode(std::string name, NodeType ntype) {
    if (datatype == "float") {
        std::cout << "Entering add Node ..." << std::endl;
        if (!isNode(name)) { 
              std::cout << "Entering add Node 1..." << std::endl;
            std::shared_ptr<ModelNode> node = std::make_shared<ModelNode>(name, ntype, datatype);
              std::cout << "Entering add Node 2..." << std::endl;
            this->modelXf->getGraph()->createNode(name, ntype);
              std::cout << "Entering add Node 3 ..." << std::endl;
            this->nodes.push_back(node);
            std::cout << "Entering add Node 4 ..." << std::endl;
            return node;
        } else {
            std::cerr << "Node already exists. Use another name ..." << std::endl;
        }
    } else if (datatype == "double") {
        if (!isNode(name)) {
            std::shared_ptr<ModelNode> node = std::make_shared<ModelNode>(name, ntype, datatype);
            this->modelXd->getGraph()->createNode(name, ntype);
            this->nodes.push_back(node);
            return node;
        } else {
            std::cerr << "Node already exists. Use another name ..." << std::endl;
        }
    } else {
        throw std::invalid_argument("Unsupported datatype");
    }
    return nullptr;
}

/************************************************************************************************
* Model::connect
* Connect nodes in the graph.
*************************************************************************************************/
void Model::connect(std::shared_ptr<ModelNode> from,std::shared_ptr<ModelNode> to) {
    try {
        if (from == nullptr) {
        throw AIException(" Source node is missing ...");
        }
        if (to == nullptr) {
        throw AIException(" Target node is missing ...");
        }
        if (datatype == "float") {
            modelXf->getGraph()->connect(from->getName(), to->getName());
        } else
        if (datatype == "double") {
            modelXd->getGraph()->connect(from->getName(), to->getName());
        }
    } catch (const AIException& e) {
        std::cerr << "(Model:connect) Error: " << e.what() << std::endl;
    } catch (const std::exception& e) {
        // Catch standard exceptions
        std::cerr << "(Model:connect) Standard Error: " << e.what() << " at " << __LINE__ << std::endl;
    } catch (...) {
        // Catch all other exceptions
        std::cerr << "(Model:connect) Unknown Error:" << std::endl;
    }
}

void Model::connect(std::vector<std::shared_ptr<ModelNode>> from_nodes, std::shared_ptr<ModelNode> to) {
    for (auto& from : from_nodes) {
        this->connect(from, to);
    }
}

void Model::connect(std::shared_ptr<ModelNode> from, std::vector<std::shared_ptr<ModelNode>> to_nodes) {
    for (auto& to : to_nodes) {
        this->connect(from, to);
    }
}

/************************************************************************************************
* Model::seedNodes
* Here, we begin to fill the nodes with data and operations as assigned to them.
*************************************************************************************************/
void Model::seedNodes() {
    for (auto& node: nodes) {
        // First, let's seed with data
        ssize_t size = node->getDataSize();

        // First, let's seed with decoder data (For Transformer Decoders)
        ssize_t dsize = node->getDecoderDataSize();

        // set Node Input Data
        if (size != 0) {
            if (datatype == "float") {
                modelXf->getGraph()->setData(node->getName(), node->getDataFloat(), node->getNormalize());
            } else
            if (datatype == "double") {
                modelXd->getGraph()->setData(node->getName(), node->getDataDouble(), node->getNormalize());
            }
        }

        // set Node Decoder Data (For Transformer Decoders)
        if (dsize != 0) {
            if (datatype == "float") {
                modelXf->getGraph()->setDecoderData(node->getName(), node->getDecoderDataFloat(), node->getNormalize());
            } else
            if (datatype == "double") {
                modelXd->getGraph()->setDecoderData(node->getName(), node->getDecoderDataDouble(), node->getNormalize());
            }
        }


        std::cout << "(Operations) Node: " << node->getName() << std::endl;
        if (datatype == "float") {
            modelXf->getGraph()->setOperations(node->getName(), node->getOperations());
        } else
        if (datatype == "double") {
            modelXd->getGraph()->setOperations(node->getName(), node->getOperations());
        }
    } 
}

/************************************************************************************************
* Model::setTargetFlat
* We use modelXf.setTarget to convert the python array to aitensor<float> and store
* the tensor inside the model.
*************************************************************************************************/
void Model::setTargetFloat(const py::array_t<float>& target) {
    try {
        if (datatype == "double") {
            throw AIException("Precision used in target data is 'float' but the model uses 'double' ...");
        }
        modelXf->setTarget(target);

    } catch (const AIException& e) {
        std::cerr << "(Model:setTargetFloat) Error: " << e.what() << std::endl;
    } catch (const std::exception& e) {
        // Catch standard exceptions
        std::cerr << "(Model:setTargetFloat) Standard Error: " << e.what() << " at " << __LINE__ << std::endl;
    } catch (...) {
        // Catch all other exceptions
        std::cerr << "(Model:setTargetFloat) Unknown Error:" << std::endl;
    }
}

/************************************************************************************************
* Model::setTargetDouble
* We use modelXd.setTarget to convert the python array to aitensor<double> and store
* the tensor inside the model.
*************************************************************************************************/
void Model::setTargetDouble(const py::array_t<double>& target) {
    try {
        if (datatype == "float") {
            throw AIException("Precision used in target data is 'double' but the model uses 'float' ...");
        }
        modelXd->setTarget(target);

    } catch (const AIException& e) {
        std::cerr << "(Model:setTargetDouble) Error: " << e.what() << std::endl;
    } catch (const std::exception& e) {
        // Catch standard exceptions
        std::cerr << "(Model:setTargetDouble)Standard Error: " << e.what() << " at " << __LINE__ << std::endl;
    } catch (...) {
        // Catch all other exceptions
        std::cerr << "(Model:setTargetDouble) Unknown Error:" << std::endl;
    }
}

/************************************************************************************************
* Model::compile
* This is where configuration begins.
*************************************************************************************************/
 
/************************************************************************************************
* Model::generateDotFormat
* Show the graph of the neural network in Dot Format
*************************************************************************************************/
std::string Model::generateDotFormat(bool operators, bool weights) {
        if (datatype == "float") {
            return this->modelXf->getGraph()->generateDotFormat(operators, weights);
        } else
        if (datatype == "double") {
            return this->modelXd->getGraph()->generateDotFormat(operators, weights);
        }
        return "none";
}

/************************************************************************************************
* Model::train
* This is where training begins. We train the actual model by passing hyperparameters.
*************************************************************************************************/
void Model::train(std::string& losstype, std::vector<std::string>& metricstype, std::string& optimizertype, double learningRate,  int max_epoch) {
    try {
        this->seedNodes();
        if (datatype == "float") {
            this->modelXf->train(losstype, metricstype, optimizertype, static_cast<float>(learningRate), max_epoch);
        } else
        if (datatype == "double") {
            this->modelXd->train(losstype, metricstype, optimizertype, static_cast<double>(learningRate), max_epoch);
        }
    } catch (const AIException& e) {
        std::cerr << "(Model::train) Error: " << e.what() << std::endl;
    } catch (const std::exception& e) {
        // Catch standard exceptions
        std::cerr << "(Model:train) Standard Error: " << e.what() << " at " << __LINE__ << std::endl;
    } catch (...) {
        // Catch all other exceptions
        std::cerr << "(Model:train) Unknown Error:" << std::endl;
    }
}

/*
void trainModel(const py::list& repeatSequentially, const py::list& repeatCounts) {

    std::vector<bool> repeatSeq;
    std::vector<int> repeatCnt;

    for (const auto& item : repeatSequentially) {
        repeatSeq.push_back(py::cast<bool>(item));
    }

    for (const auto& item : repeatCounts) {
        repeatCnt.push_back(py::cast<int>(item));
    }

    std::vector<Node*> nodes = this->getNodes();

    // Sort nodes topologically
    std::vector<Node*> sortedNodes = topologicalSort(nodes);

    // Group sorted nodes into sequential regions
    std::vector<std::vector<Node*>> regions = groupNodes(sortedNodes);

    // Repeat regions based on repeatSequentially and repeatCounts
    std::vector<std::vector<Node*>> repeatedRegions = repeatRegions(regions, repeatSeq, repeatCnt);

        // Initialize MPI
    MPI_Init(NULL, NULL);

    // Execute Pipeline
    //executePipelineParallel(repeatedRegions);

    executePipelineParallel();

    // Perform data exchange between regions based on connections
    // performDataExchange(graph, repeatedRegions);

    // Gather gradients from all regions
    // std::vector<double> aggregatedGradients(nodes.size());
    // gatherGradients(repeatedRegions, aggregatedGradients);

    // Aggregate gradients and perform parameter update
    // aggregateGradients(aggregatedGradients);
    // performParameterUpdate(nodes);

    // Finalize MPI
    MPI_Finalize();

}
*/
