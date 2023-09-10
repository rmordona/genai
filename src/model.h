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

#ifndef BASEMODEL_H
#define BASEMODEL_H


template <class T>
class BaseModel {
private:
    Graph<T>* graph;
    aitensor<T> target;
    aitensor<T> predicted;
    aitensor<T> gradients;
    aiscalar<T> loss;
    std::string losstype = "mse";
    std::string optimizertype = "adam";
    T learningRate = 0.01;
    int itermax = 1;
public:
    BaseModel(const std::string& losstype = "mse", const std::string& optimizertype = "adam", 
          const T learningRate = 0.01, const int itermax = 1) {
        this->losstype = losstype;
        this->optimizertype = optimizertype;
        this->learningRate = learningRate;
        this->itermax = itermax;
    }

    void setGraph(Graph<T>* graph);

    Graph<T>* getGraph();

    void setLoss(std::string& losstype);

    // The input is assumed to have NxM where N=number of samples, M=embedding vector size
    // This allows to compute for the output size,  MxW where W is the number of weights (features) to use.
    void setTarget(const py::array_t<T>& target);

    aitensor<T> getTarget();

    void useCrossEntropy();

    void train(std::string& losstype, std::string& optimizertype, T& learningRate = 0.01, int& itermax = 1);

};

class ModelNode {
private:
    std::string name;
    NodeType ntype;
    std::vector<std::shared_ptr<BaseOperator>> operations;
    float* input_fdata;
    double* input_ddata;
public: 
    ModelNode(std::string name, NodeType ntype) { 
        this->name = name; 
        this->ntype = ntype;
    }

    std::string getName() { return this->name; }

    NodeType getNodeType() { return this->ntype; }

    void setDataFloat(const py::array_t<float>& input_data);
    void setDataDouble(const py::array_t<double>& input_data);

    void setOperations(std::vector<std::shared_ptr<BaseOperator>>& operations) {
        this->operations = operations;
    }

    std::vector<std::shared_ptr<BaseOperator>> getOperations() { return this->operations; }

};

class Model {
private:
    std::string losstype = "mse";
    std::string optimizertype = "adam";
    double learningRate = 0.01;
    int itermax = 1;
    std::string datatype = "float";
    Graph<float> graphXf;
    Graph<double> graphXd;
    BaseModel<float> modelXf;
    BaseModel<double> modelXd;
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
    Model(const std::string& losstype = "mse", const std::string& optimizertype = "adam", 
          const double learningRate = 0.01, const int itermax = 1, const std::string& datatype = "float");

    void createGraph();

    // void addNode(const std::string& name, NodeType ntype, std::vector<std::shared_ptr<BaseOperator>>& operations);
    std::shared_ptr<ModelNode> addNode(const std::string& name, NodeType ntype);

    void setData(const std::string& name, NodeType ntype, const py::array_t<float>& input_dataf, const py::array_t<double>& input_datad);

    void connect(std::shared_ptr<ModelNode> from, std::shared_ptr<ModelNode> to);

    void connect(std::vector<std::shared_ptr<ModelNode>> from_nodes, std::shared_ptr<ModelNode> to);

};

/************ Base Model initialize templates ************/

template class BaseModel<float>;  // Instantiate with float
template class BaseModel<double>;  // Instantiate with double

/************ Wrapper Objects for Python API */

class ModelLinear : public BaseOperator {
private:
    int W = 0; // number of weights 
    bool bias = true; // Use bias by default.
public: 
    ModelLinear(int size, bool bias = true)  {
        this->W = size;
        this->bias = bias;
    }
    void forwardPass() {}
    void backwardPass() {}
};

class ModelBatchNorm : public BaseOperator {
private:
public: 
    ModelBatchNorm() {}
    void forwardPass() {}
    void backwardPass() {}
};

class ModelLayerNorm : public BaseOperator {
private:
public: 
    ModelLayerNorm() {}
    void forwardPass() {}
    void backwardPass() {}
};

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
    void forwardPass() {}
    void backwardPass() {}
};

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

class ModelAttention : public BaseOperator {
private:
    int W = 0;  // number of weights (or number of features)
    int H = 1;  // number of heads
    bool bias = true;
public: 
    ModelAttention(int size = 3, bool bias = false)  {
        this->H = 1;
        this->W = size;
        this->bias = bias;
    }
    void forwardPass() {}
    void backwardPass() {}
};

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

    void forwardPass() {}
    void backwardPass() {}
};

class ModelMultiHeadAttention : public BaseOperator {
private:
    int W = 0;  // number of weights (or number of features)
    int H = 1;  // number of heads
    bool bias = true;
public: 
    ModelMultiHeadAttention(int heads = 3, int size = 3, bool bias = false)  {
        this->W = size;
        this->H = heads;
        this->bias = bias;
        // M1.setZero();
        log_info( "**** MultiHeadAttention instance created ****" );
    }
    void forwardPass() {}
    void backwardPass() {}
};

class ModelEncoder : public BaseOperator {
private:
    std::string activationtype = "leakyrelu";
    int W = 0;  // number of weights (or number of features)
    int H = 1;  // number of heads
    bool bias = true;
    float alpha = 0.01;
public: 
    ModelEncoder(int heads = 1, int size = 3, bool bias = true, const std::string& activationtype = "leakyrelu") {
        this->activationtype = activationtype;
        this->W = size;
        this->bias = bias;
        this->H = heads;
    }

    ModelEncoder(int heads = 1, int size = 3, bool bias = true, const std::string& activationtype = "leakyrelu", const float alpha=0.01) {
        this->activationtype = activationtype;
        this->alpha = alpha;
        this->W = size;
        this->bias = bias;
        this->H = heads;
    }

    void forwardPass() {}
    void backwardPass() {}
};


#endif
