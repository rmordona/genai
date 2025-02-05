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

#include "topology.h"
#include <cstdlib>  // For std::srand

#ifndef BASEMODEL_H
#define BASEMODEL_H


/*************************************************************************************************
 * BaseModel is the actual structure we use to build graphs, nodes, connections, and the
 * underlying operations needed to train a model.
 *************************************************************************************************/
template <class T>
class BaseModel {
private:
    std::shared_ptr<Graph<T>> graph;
    aitensor<T> target;
    // aitensor<T> batch_output;
    aitensor<T> gradients;
    aiscalar<T> loss;
    PerfMetrics<T> metrics;
    std::string losstype = "mse";
    std::string optimizertype = "adam";
    int max_epoch = 1;
    T learningRate = 0.01;
    bool useStepDecay = false;
    float decayRate = 0.90;
    int batch_size  = 10;

    std::vector<std::string> metricstype;

    int start_index = 0;  // start index of batch

public:
    BaseModel(int seed) { 
        if (seed != 0) {
            std::srand(seed);
        }
    }

    void setGraph(std::shared_ptr<Graph<T>>  graph);

    std::shared_ptr<Graph<T>> getGraph();

    void setLoss(std::string& losstype);

    // The input is assumed to have NxM where N=number of samples, M=embedding vector size
    // This allows to compute for the output size,  MxW where W is the number of weights (features) to use.
    void setTarget(const py::array_t<T>& target, const bool normalize);

    aitensor<T> getTarget();

    void useCrossEntropy();

    std::vector<float> train(std::string& losstype, std::vector<std::string>& metricstype, std::string& optimizertype, 
            int batch_size = 10, 
            const int max_epoch = 1, const T learn_rate = 0.01, const bool use_step_decay = false, const T decay_rate = 0.90);

    aitensor<T> predict(int sequence_length = 0);

    void test() {}

};

/************ Base Model initialize templates ************/

template class BaseModel<float>;  // Instantiate with float
template class BaseModel<double>;  // Instantiate with double

/************************************************************************************************
* We use ModelNode class as a meta model only for use  as entry point for python API.
* The actual model is the Node class.
*************************************************************************************************/
class ModelNode {
private:
    std::string name;
    NodeType ntype;
    DataType datatype = DataType::float32;
    std::vector<BaseOperator*> operations;
    aitensor<float> input_fdata;
    aitensor<double> input_ddata;

    aitensor<float> decoder_fdata;  // (For Transformer Decoders)
    aitensor<double> decoder_ddata; // (For Transformer Decoders)

    aitensor<float> encoder_fdata;  // (For Transformer Decoders, in the case there's no encoder )
    aitensor<double> encoder_ddata; // (For Transformer Decoders, in the case there's no encoder )

    bool normalize = false;
    bool positional = false;

public: 
    ModelNode(std::string name, NodeType ntype, DataType datatype) { 
        this->name = name; 
        this->ntype = ntype;
        this->datatype = datatype;
    }

    std::string getName() { return this->name; }

    NodeType getNodeType() { return this->ntype; }

    void setData(const py::array& data, const bool normalize, const bool positional);

    // void setDataFloat(const py::array_t<float>& data, const bool normalize, const bool positional);

    // void setDataDouble(const py::array_t<double>& data, const bool normalize, const bool positional);

    void setDecoderData(const py::array& data, const bool normalize, const bool positional);

    // void setDecoderDataFloat(const py::array_t<float>& data, const bool normalize, const bool positional);

    // void setDecoderDataDouble(const py::array_t<double>& data, const bool normalize, const bool positional);

    void setEncoderData(const py::array& data, const bool normalize, const bool positional);

    // void setEncoderDataFloat(const py::array_t<float>& data, const bool normalize, const bool positional);

    // void setEncoderDataDouble(const py::array_t<double>& data, const bool normalize, const bool positional);

    bool getNormalize() { return this->normalize; }

    bool getPositional() { return this->positional; }

    ssize_t getDataSize() { 
        if (this->datatype == DataType::float32) {
            return this->input_fdata.size(); 
        } else 
        if (this->datatype == DataType::float64) {
            return this->input_ddata.size(); 
        }
        return 0;
    }

    ssize_t getDecoderDataSize() { 
        if (this->datatype == DataType::float32) {
            return this->decoder_fdata.size(); 
        } else 
        if (this->datatype == DataType::float64) {
            return this->decoder_ddata.size(); 
        }
        return 0;
    }

    ssize_t getEncoderDataSize() { 
        if (this->datatype == DataType::float32) {
            return this->encoder_fdata.size(); 
        } else 
        if (this->datatype == DataType::float64) {
            return this->encoder_ddata.size(); 
        }
        return 0;
    }

    aitensor<float> getDataFloat() { return this->input_fdata; }
    aitensor<double> getDataDouble() { return this->input_ddata; }

    aitensor<float> getDecoderDataFloat() { return this->decoder_fdata; }
    aitensor<double> getDecoderDataDouble() { return this->decoder_ddata; }

    aitensor<float> getEncoderDataFloat() { return this->encoder_fdata; }
    aitensor<double> getEncoderDataDouble() { return this->encoder_ddata; }

    void setOperations(std::vector<std::shared_ptr<BaseOperator>>& operations); // accept as ModelNode operations

    std::vector<BaseOperator*> getOperations() { return this->operations; }    // return as Node operations

};

/************************************************************************************************
* We use Model class as a meta model only for use  as entry point for python API.
* The actual model is the BaseModel class.
* All other meta models include the Model operations.
*************************************************************************************************/
class Model {
private:
    DataType datatype = DataType::float32;
    int seed = 0;
    std::shared_ptr<Graph<float>> graphXf;
    std::shared_ptr<Graph<double>> graphXd;
    std::shared_ptr<BaseModel<float>> modelXf;
    std::shared_ptr<BaseModel<double>> modelXd;
    std::vector<std::shared_ptr<ModelNode>> nodes;

    // Iterate through the vector to search for the name
    bool isNode(const std::string& target) {
        for (auto& node : nodes) {
            if (node->getName() == target) {
                return true;
            }
        }
        return false;
    }

public:
    Model(DataType dtype = DataType::float32, int seed = 2017);

    std::shared_ptr<ModelNode> addNode(std::string name, NodeType ntype);

    // void unpackData();

    // void unpackOperations();

    void setTarget(const py::array& target, const bool normalize);

    void connect(std::shared_ptr<ModelNode> from, std::shared_ptr<ModelNode> to);

    void connect(std::vector<std::shared_ptr<ModelNode>> from_nodes, std::shared_ptr<ModelNode> to);

    void connect(std::shared_ptr<ModelNode> from, std::vector<std::shared_ptr<ModelNode>> to_nodes);

    void seedNodes(bool setOps = false);

    py::array predict(int sequence_length = 0);

    std::vector<float> train(std::string& losstype, std::vector<std::string>& metricstype, std::string& optimizertype, int batch_size = 10, int max_epoch = 1,
                    double learningRate = 0.01, bool useStepDecay = false, double decayRate = 0.90);

    Topology generateDotFormat(bool operators = false, bool weights = false);

    //py::array_t<double> process_xarray(py::array_t<double> arr);

    py::array process_array(py::array arr);

    //py::array process_xtype(py::array arr);

    //py::array array_with_dtype(py::dtype dtype);

};
 
/************************************************************************************************
* We use ModelLinear class as a meta model only for use  as entry point for python API.
* The actual model is the Linear class.
*************************************************************************************************/
class ModelLinear : public BaseOperator {
private:
    int W = 0; // number of weights 
    bool bias = true; // Use bias by default.
public: 
    ModelLinear(int size, bool bias = true)  {
        this->W = size;
        this->bias = bias;
    }
    int getSize() { return  this->W; }
    bool getBias() { return this->bias; }

    void forwardPass() {}
    void backwardPass() {}
};

/************************************************************************************************
* We use ModelBatchNorm class as a meta model only for use  as entry point for python API.
* The actual model is the BatchNorm  class.
*************************************************************************************************/
class ModelBatchNorm : public BaseOperator {
private:
public: 
    ModelBatchNorm() {}
    void forwardPass() {}
    void backwardPass() {}
};

/************************************************************************************************
* We use ModelLayerNorm class as a meta model only for use  as entry point for python API.
* The actual model is the LayerNorm  class.
*************************************************************************************************/
class ModelLayerNorm : public BaseOperator {
private:
public: 
    ModelLayerNorm() {}
    void forwardPass() {}
    void backwardPass() {}
};

/************************************************************************************************
* We use ModelActivation class as a meta model only for use  as entry point for python API.
* The actual model is the Activation  class.
*************************************************************************************************/
class ModelActivation : public BaseOperator {
private:
    std::string activationtype = "leakyrelu";
    float alpha = 0.01;
public: 
    ModelActivation(const std::string& activationtype = "leakyrelu") {
        this->activationtype = activationtype;
    }
    ModelActivation(const std::string& activationtype = "leakyrelu", const float alpha=0.01) {
        this->activationtype = activationtype;
        this->alpha = alpha;
    }

    std::string getActivationType() { return this->activationtype; }
    float getAlpha() { return this->alpha; }

    void forwardPass() {}
    void backwardPass() {}
};

/************************************************************************************************
* We use ModelDropout class as a meta model only for use  as entry point for python API.
* The actual model is the Dropout  class.
*************************************************************************************************/
class ModelDropout : public BaseOperator {
private:
    float probability = 0.05;
public: 
    ModelDropout(const float probability = 0.05) {
        this->probability = probability;
    }

    float getProbability() { return this->probability; }

    void forwardPass() {}
    void backwardPass() {}
};

/************************************************************************************************
* We use ModelFlatten class as a meta model only for use  as entry point for python API.
* The actual model is the Flatten  class.
*************************************************************************************************/
class ModelFlatten : public BaseOperator {
private:
public: 
    ModelFlatten() {}

    void forwardPass() {}
    void backwardPass() {}
};

/************************************************************************************************
* We use ModelConvolution class as a meta model only for use  as entry point for python API.
* The actual model is the Convolution  class.
*************************************************************************************************/
class ModelConvolution : public BaseOperator {
private:
    int kernel_size = 2;
    int stride      = 1;
    int padding     = 1;
    int dilation    = 1;

    bool bias = true; // Use bias by default.
public: 
    ModelConvolution(int kernel_size = 2, int stride = 1, int padding = 1, int dilation = 1, bool bias = true)   {
        this->kernel_size = kernel_size;
        this->stride      = stride;
        this->padding     = padding;
        this->dilation    = dilation;   
        this->bias        = bias;  
    }

    int getKernelSize() { return this->kernel_size; } 
    int getStride() { return this->stride; } 
    int getPadding() { return this->padding; } 
    int getDilation() { return this->dilation; } 
    int getBias() { return this->bias; } 

    void forwardPass() {}
    void backwardPass() {}
};



/************************************************************************************************
* We use ModelReduction class as a meta model only for use  as entry point for python API.
* The actual model is the Reduction  class.
*************************************************************************************************/
class ModelReduction : public BaseOperator {
private:
    std::string reducttype = "add";
public: 
    ModelReduction(const std::string& reducttype = "add") {
        this->reducttype = reducttype;
    }
    void forwardPass() {}
    void backwardPass() {}
};

/************************************************************************************************
* We use ModelAttention class as a meta model only for use  as entry point for python API.
* The actual model is the Attention  class.
*************************************************************************************************/
class ModelAttention : public BaseOperator {
private:
    int W = 0;  // number of weights (or number of features)
    int H = 1;  // number of heads
    bool bias = false;
    bool masked = false;
public: 
    ModelAttention(int size = 3, bool bias = false, bool masked = false)  {
        this->H = 1;
        this->W = size;
        this->bias = bias;
        this->masked = masked;
    }

    int getSize() { return this->W; }
    bool getBias() { return this->bias; }
    bool getMasked() { return this->masked; }

    void forwardPass() {}
    void backwardPass() {}
};

/************************************************************************************************
* We use ModelFeedForward class as a meta model only for use  as entry point for python API.
* The actual model is the FeedForward  class.
*************************************************************************************************/
class ModelFeedForward : public BaseOperator {
private:
    int W = 0;
    bool bias = true;
    std::string activationtype = "leakyrelu";
    float alpha = 0.01;
public: 
    ModelFeedForward(int size = 3, bool bias = true, const std::string& activationtype = "leakyrelu") {
        this->activationtype = activationtype;
        this->W = size;
        this->bias = bias;
    }

    ModelFeedForward(int size = 3, bool bias = true, const std::string& activationtype = "leakyrelu", const float alpha=0.01) {
        this->activationtype = activationtype;
        this->alpha = alpha;
        this->W = size;
        this->bias = bias;
    }

    int getSize() { return this->W; }
    bool getBias() { return this->bias; }
    std::string getActivationType() { return this->activationtype; }
    float getAlpha() { return this->alpha; }

    void forwardPass() {}
    void backwardPass() {}
};

/************************************************************************************************
* We use ModelMultiHeadAttention class as a meta model only for use  as entry point for python API.
* The actual model is the MultiHeadAttention  class.
*************************************************************************************************/
class ModelMultiHeadAttention : public BaseOperator {
private:
    int W = 0;  // number of weights (or number of features)
    int H = 1;  // number of heads
    bool bias = true;
    bool masked = false;
public: 
    ModelMultiHeadAttention(int heads = 3, int size = 3, bool bias = false, bool masked = false)  {
        this->W = size;
        this->H = heads;
        this->bias = bias;
        this->masked = masked;
        log_info( "**** MultiHeadAttention instance created ****" );
    }

    int getHead() { return this->H; }
    int getAttentionSize() { return this->W; }
    bool getBias() { return this->bias; }
    bool getMasked() { return this->masked; }

    void forwardPass() {}
    void backwardPass() {}
};

/************************************************************************************************
* We use ModelEncoder class as a meta model only for use  as entry point for python API.
* The actual model is the Encoder  class.
*************************************************************************************************/
class ModelEncoder : public BaseOperator {
private:
    std::string activationtype = "leakyrelu";
    int W       = 0;  // number of weights (or number of features) for K, V, Q (attention layer)
    int F       = 0;  // number of weights (or number of features) for feedforward layer
    int H = 1;  // number of heads
    int L = 1;  // number of layers
    bool bias = true;
    float alpha = 0.01;
public: 
    ModelEncoder(int heads = 1, int attention_size = 4, int feed_size = 4,  
                 int layers = 1, bool bias = true, const std::string& activationtype = "leakyrelu") {
        this->activationtype = activationtype;
        this->W    = attention_size;
        this->F    = feed_size;
        this->L    = layers;
        this->bias = bias;
        this->H    = heads;
        if (attention_size % heads != 0) throw AIException("heads is not multiple of attention_size ...");
    }

    ModelEncoder(int heads = 1, int attention_size = 4, int feed_size = 4,  
                 int layers = 1, bool bias = true, const std::string& activationtype = "leakyrelu", const float alpha=0.01) {       
        this->activationtype = activationtype;
        this->alpha = alpha;
        this->W     = attention_size;
        this->F     = feed_size;
        this->L     = layers;
        this->bias  = bias;
        this->H     = heads;
        if (attention_size % heads != 0) throw AIException("heads is not multiple of attention_size ...");
    }

    int getHead() { return this->H; }
    int getAttentionSize() { return this->W; }
    int getFeedSize() { return this->F; }
    int getLayers() { return this->L; }
    bool getBias() { return this->bias; }
    std::string getActivationType() { return this->activationtype; }
    float getAlpha() { return this->alpha; }

    void forwardPass() {}
    void backwardPass() {}
};

/************************************************************************************************
* We use ModelDecoder class as a meta model only for use  as entry point for python API.
* The actual model is the Encoder  class.
*************************************************************************************************/
class ModelDecoder : public BaseOperator {
private:
    std::string activationtype = "leakyrelu";
    int W       = 0;  // number of weights (or number of features) for K, V, Q (attention layer)
    int F       = 0;  // number of weights (or number of features) for feedforward layer
    int H       = 1;  // number of heads
    int L       = 1;  // number of layers

    bool  bias  = true;
    float alpha = 0.01;
public: 
    ModelDecoder(int heads = 1, int attention_size = 4, int feed_size = 4, 
                 int layers = 1, bool bias = true, const std::string& activationtype = "leakyrelu") {
        this->activationtype = activationtype;
        this->W    = attention_size;
        this->F    = feed_size;
        this->L    = layers;
        this->bias = bias;
        this->H    = heads;
        if (attention_size % heads != 0) throw AIException("heads is not multiple of attention_size ...");
    }

    ModelDecoder(int heads = 1, int attention_size = 4, int feed_size = 4, 
                 int layers = 1, bool bias = true, const std::string& activationtype = "leakyrelu", const float alpha=0.01) {
        this->activationtype = activationtype;
        this->alpha = alpha;
        this->W     = attention_size;
        this->F     = feed_size;
        this->L     = layers;
        this->bias  = bias;
        this->H     = heads;
        if (attention_size % heads != 0) throw AIException("heads is not multiple of attention_size ...");
    }

    int getHead() { return this->H; }
    int getAttentionSize() { return this->W; }
    int getFeedSize() { return this->F; }
    int getLayers() { return this->L; }
    bool getBias() { return this->bias; }
    std::string getActivationType() { return this->activationtype; }
    float getAlpha() { return this->alpha; }

    void forwardPass() {}
    void backwardPass() {}
};

/************************************************************************************************
* We use ModelRNN class as a meta model only for use  as entry point for python API.
* The actual model is the RNN  class.
*************************************************************************************************/
class ModelRNN : public BaseOperator {
private:
   int hidden_size;
   int output_size;
   int output_sequence_length;
   int num_layers;
   bool bidirectional;
   RNNType rnntype;
public: 
    ModelRNN(int hidden_size = 1, int output_size = 3, int output_sequence_length = 1, 
             int num_layers = 1, bool bidirectional = true, RNNType rnntype = RNNType::MANY_TO_MANY) {
        this->hidden_size = hidden_size;
        this->output_size = output_size;
        this->output_sequence_length = output_sequence_length;
        this->num_layers = num_layers;
        this->bidirectional = bidirectional;
        this->rnntype = rnntype;
    }

    int getHiddenSize() { return this->hidden_size; }
    int getOuputSize() { return this->output_size; }
    int getOutputSequenceLength() { return this->output_sequence_length; }
    int getNumLayers() { return this->num_layers; }
    bool getBiDirection() { return this->bidirectional; }
    RNNType getRNNType() { return this->rnntype; }

    void forwardPass() {}
    void backwardPass() {}
};

/************************************************************************************************
* We use ModelLSTM class as a meta model only for use  as entry point for python API.
* The actual model is the RNN  class.
*************************************************************************************************/
class ModelLSTM : public BaseOperator {
private:
   int hidden_size;
   int output_size;
   int output_sequence_length;
   int num_layers;
   bool bidirectional;
   RNNType rnntype;
public: 
    ModelLSTM(int hidden_size = 1, int output_size = 3, int output_sequence_length = 1, 
              int num_layers = 1, bool bidirectional = true, RNNType rnntype = RNNType::MANY_TO_MANY) {
        this->hidden_size = hidden_size;
        this->output_size = output_size;
        this->output_sequence_length = output_sequence_length;
        this->num_layers = num_layers;
        this->bidirectional = bidirectional;
        this->rnntype = rnntype;
    }

    int getHiddenSize() { return this->hidden_size; }
    int getOuputSize() { return this->output_size; }
    int getOutputSequenceLength() { return this->output_sequence_length; }
    int getNumLayers() { return this->num_layers; }
    bool getBiDirection() { return this->bidirectional; }
    RNNType getRNNType() { return this->rnntype; }

    void forwardPass() {}
    void backwardPass() {}
};

/************************************************************************************************
* We use ModelGRU class as a meta model only for use  as entry point for python API.
* The actual model is the RNN  class.
*************************************************************************************************/
class ModelGRU : public BaseOperator {
private:
   int hidden_size;
   int output_size;
   int output_sequence_length;
   int num_layers;
   bool bidirectional;
   RNNType rnntype;
public: 
    ModelGRU(int hidden_size = 1, int output_size = 3, int output_sequence_length = 1, 
             int num_layers = 1, bool bidirectional = true, RNNType rnntype = RNNType::MANY_TO_MANY) {
        this->hidden_size = hidden_size;
        this->output_size = output_size;
        this->output_sequence_length = output_sequence_length;
        this->num_layers = num_layers;
        this->bidirectional = bidirectional;
        this->rnntype = rnntype;
    }

    int getHiddenSize() { return this->hidden_size; }
    int getOuputSize() { return this->output_size; }
    int getOutputSequenceLength() { return this->output_sequence_length; }
    int getNumLayers() { return this->num_layers; }
    bool getBiDirection() { return this->bidirectional; }
    RNNType getRNNType() { return this->rnntype; }

    void forwardPass() {}
    void backwardPass() {}
};



#endif
