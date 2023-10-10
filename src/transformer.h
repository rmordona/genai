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

#pragma once

#include "operators.h"

#ifndef TRANSFORMER_H
#define TRANSFORMER_H

/**************************************************************************************************
  Tensor Helper Functions
**************************************************************************************************/

// This function takes a sub-tensor slice of a tensor. 
template <class T>
static std::vector<aitensor<T>> head_split(const aitensor<T>& tensor, int splits) {
    int dim0 = tensor.size();
    int dim1 = tensor.at(0).rows();
    int dim2 = tensor.at(0).cols();

    std::vector<aitensor<T>> heads;
    int splitSize =  dim2 / splits;

    log_info("Splitting Features for Multi-Head Attention ...");
    log_detail("Tensor: {0} x {1} x {2}", dim0, dim1, dim2);

    for (int i = 0; i < splits; i++) {
        aitensor<T> head;
        int start = i * splitSize;
        for (int j = 0; j < dim0; j++) {
            aimatrix<T> input = tensor.at(j).block(0, start, dim1, splitSize);
            head.push_back(input);
            log_detail("Split matrix i={0} j={1}: {2} x {3}", i, j, input.rows(), input.cols());
        }
        heads.push_back(head);
    }

    return heads;
}


template <class T>
class Attention : public BaseOperator {
private:
    aitensor<T> input_data;   // Input Dimention: BxNxM
    aitensor<T> encoder_data;  // If we are passing an encoded input to a decoder
    aitensor<T> encoder_gradient;  // If we are passing an encoded input to a decoder

    Linear<T>* Q  = nullptr;  // BxNxW
    Linear<T>* K  = nullptr;  // BxNxW
    Linear<T>* V  = nullptr;  // BxNxW
    Linear<T>* Wo = nullptr; // this extra weight matrix will align the output dimension the same as the input. // BxNxM

    aitensor<T> Qout;
    aitensor<T> Kout;
    aitensor<T> Vout;

    aitensor<T> QKweight;
    aitensor<T> QKweightV;

    int B = 0;  // batch size
    int N = 0;  // input size
    int M = 0;  // number of features (embedding vector size)
    int W = 0;  // number of weights (or number of features)
    int H = 1;  // number of heads
    int Dk = 0; // number of dimensions per head (M/H)

    bool bias = false;

public:
    Attention(int size = 3, bool bias = false)  {
        this->W = size;
        this->bias = bias;
        log_info( "**** Attention instance created ****" );
    }

    // While the parameter weight has dimension BxMxW,  the resulting transformation has dimension of BxNxW.
    // We only need the M dimension from an BxNxM input to generate parameter matrix.
    // where weights is BxNxW and bias is W.
    const aitensor<T> forward(const aitensor<T>& input_data);

    // Leave the gradients as is. They are cached in the Node. 
    // They will be used to update the parameters in next parallel operations.
    // the dInput is the gradient we propagate to source Nodes in the graph;
    // while the parameter gradients get cached to be used to update the parameters later.
    const aitensor<T> backward(const aitensor<T>& gradients);

    void updateParameters(std::string& optimizertype, T& learningRate, int& iter);

    // if passing encoded input to decoder, we take care of the K and V input.
    void setEncoderData(const aitensor<T>& encoder_data) {
        this->encoder_data = encoder_data;
    }

    void setEncoderGradient(const aitensor<T>& encoder_gradient) {
        this->encoder_gradient = encoder_gradient;
    }

    const aitensor<T> getEncoderGradient() {
        return this->encoder_gradient;
    }

    void forwardPass() {}
    void backwardPass() {}
    std::string generateDotFormat();

};

/*****************************************************************************************************
* Base Multi-Head Attention Layer
*****************************************************************************************************/
template <class T>
class MultiHeadAttention : public BaseOperator {
private:
    aitensor<T> input_data; // NxW samples, by using & 
    aitensor<T> encoder_data;  // If we are passing an encoded input to a decoder
    aitensor<T> encoder_gradient;  // If we are passing an encoded input to a decoder

    std::vector<Attention<T>*> M1;
    bool bias = true;

    int B = 0;  // batch size
    int N = 0;  // number of samples
    int M = 0;  // number of features (embedding vector size)
    int W = 0;  // number of weights (or number of features)
    int H = 1;  // number of heads
    int Dk = 0; // number of dimensions per head (M/H)
    int split = 0; // number of values in an array to jump.

    std::string activationtype = "leakyrelu";
    float alpha = 0.01; // for leakyReLU

public:
    MultiHeadAttention(int heads = 3, int size = 3, bool bias = false)  {
        this->W = size;
        this->H = heads;
        this->bias = bias;
        // M1.setZero();
        log_info( "**** MultiHeadAttention instance created ****" );
    }

    // While the parameter weight has dimension BxMxW,  the resulting transformation has dimension of BxNxW.
    // We only need the M dimension from an BxNxM input to generate parameter matrix.
    // where weights is BxNxW and bias is W.
    const aitensor<T> forward(const aitensor<T>& input_data);

    // Leave the gradients as is. They are cached in the Node. 
    // They will be used to update the parameters in next parallel operations.
    // the dInput is the gradient we propagate to source Nodes in the graph;
    // while the parameter gradients get cached to be used to update the parameters later.
    const aitensor<T> backward(const aitensor<T>& gradients);

    void updateParameters(std::string& optimizertype, T& learningRate, int& iter);

    // if passing encoded input to decoder, we take care of the K and V input.
    void setEncoderData(const aitensor<T>& encoder_data) {
        this->encoder_data = encoder_data;
    }

    void setEncoderGradient(const aitensor<T>& encoder_gradient) {
        this->encoder_gradient = encoder_gradient;
    }

    void forwardPass() {}
    void backwardPass() {}
    std::string generateDotFormat();
};

/*****************************************************************************************************
* Base FeedForward  Layer
*****************************************************************************************************/
template <class T>
class FeedForward : public BaseOperator {
private:
    aitensor<T> input_data; // NxW samples, by using & 
    Linear<T>* L1 = nullptr;
    Linear<T>* L2 = nullptr; // required to align the dimension similar to the input
    Activation<T>* A1 = nullptr;

    aitensor<T> L1out; // Cache output for use by activation backprop

    bool bias = true;
    int B = 0;
    int N = 0;
    int M = 0;
    int W = 0;

    std::string activationtype = "leakyrelu";
    float alpha = 0.01; // for leakyReLU

public:

    FeedForward(int size = 3, bool bias = true, const std::string& activationtype = "leakyrelu") {
        this->activationtype = activationtype;
        this->W = size;
        this->bias = bias;
        log_info( "**** FeedForward instance created ****" );
    }

    FeedForward(int size = 3, bool bias = true, const std::string& activationtype = "leakyrelu", const float alpha=0.01) {
        this->activationtype = activationtype;
        this->alpha = alpha;
        this->W = size;
        this->bias = bias;
        log_info( "**** FeedForward instance created ****" );
    }

    // While the parameter weight has dimension BxMxW,  the resulting transformation has dimension of BxNxW.
    // We only need the M dimension from an BxNxM input to generate parameter matrix.
    // where weights is BxNxW and bias is W.
    const aitensor<T> forward(const aitensor<T>& input_data);

    // Leave the gradients as is. They are cached in the Node. 
    // They will be used to update the parameters in next parallel operations.
    // the dInput is the gradient we propagate to source Nodes in the graph;
    // while the parameter gradients get cached to be used to update the parameters later.
    const aitensor<T> backward(const aitensor<T>& gradients);

    void updateParameters(std::string& optimizertype, T& learningRate, int& iter);

    void forwardPass() {}
    void backwardPass() {}
    std::string generateDotFormat();
};

/*****************************************************************************************************
* Base Encoder  Layer
*****************************************************************************************************/
template <class T>
class Encoder : public BaseOperator {
private:
    aitensor<T> input_data; // NxW samples, by using & 
    MultiHeadAttention<T>* M1 = nullptr;
    LayerNorm<T>* LN1 = nullptr;
    FeedForward<T>* F1 = nullptr;
    LayerNorm<T>* LN2 = nullptr;

    aitensor<T> M1out; // Cache output for use by attention backprop
    aitensor<T> F1out; // Cache output for use by feedforward backprop
    aitensor<T> LN1out; // Cache output for use by feedforward backprop

    bool bias = true;
    int B = 0;
    int N = 0;
    int M = 0;
    int H = 0;
    int W = 0;

    std::string activationtype = "leakyrelu";
    float alpha = 0.01; // for leakyReLU

public:

    Encoder(int heads = 1, int size = 3, bool bias = true, const std::string& activationtype = "leakyrelu") {
        this->activationtype = activationtype;
        this->W = size;
        this->bias = bias;
        this->H = heads;
        log_info( "**** Encoder instance created ****" );
    }

    Encoder(int heads = 1, int size = 3, bool bias = true, const std::string& activationtype = "leakyrelu", const float alpha=0.01) {
        this->activationtype = activationtype;
        this->alpha = alpha;
        this->W = size;
        this->bias = bias;
        this->H = heads;
        log_info( "**** Encoder instance created ****" );
    }

    // While the parameter weight has dimension BxMxW,  the resulting transformation has dimension of BxNxW.
    // We only need the M dimension from an BxNxM input to generate parameter matrix.
    // where weights is BxNxW and bias is W.
    const aitensor<T> forward(const aitensor<T>& input_data);

    // Leave the gradients as is. They are cached in the Node. 
    // They will be used to update the parameters in next parallel operations.
    // the dInput is the gradient we propagate to source Nodes in the graph;
    // while the parameter gradients get cached to be used to update the parameters later.
    const aitensor<T> backward(const aitensor<T>&  gradients);

    void updateParameters(std::string& optimizertype, T& learningRate, int& iter);

    void forwardPass() {}
    void backwardPass() {}
    std::string generateDotFormat();
};


/*****************************************************************************************************
* Base Encoder  Layer
*****************************************************************************************************/
template <class T>
class Decoder : public BaseOperator {
private:
    aitensor<T> input_data; // NxW samples, by using & 
    MultiHeadAttention<T>* M1 = nullptr;
    LayerNorm<T>* LN1 = nullptr;
    MultiHeadAttention<T>* M2 = nullptr;
    LayerNorm<T>* LN2 = nullptr;
    FeedForward<T>* F1 = nullptr;
    LayerNorm<T>* LN3 = nullptr;

    aitensor<T> M1out; // Cache output for use by attention backprop
    aitensor<T> M2out; // Cache output for use by attention backprop
    aitensor<T> LN1out; // Cache output for use by feedforward backprop
    aitensor<T> LN2out; // Cache output for use by feedforward backprop
    aitensor<T> F1out; // Cache output for use by feedforward backprop

    bool bias = true;
    int B = 0;
    int N = 0;
    int M = 0;
    int H = 0;
    int W = 0;

    std::string activationtype = "leakyrelu";
    float alpha = 0.01; // for leakyReLU

public:

    Decoder(int heads = 1, int size = 3, bool bias = true, const std::string& activationtype = "leakyrelu") {
        this->activationtype = activationtype;
        this->W = size;
        this->bias = bias;
        this->H = heads;
        log_info( "**** Encoder instance created ****" );
    }

    Decoder(int heads = 1, int size = 3, bool bias = true, const std::string& activationtype = "leakyrelu", const float alpha=0.01) {
        this->activationtype = activationtype;
        this->alpha = alpha;
        this->W = size;
        this->bias = bias;
        this->H = heads;
        log_info( "**** Encoder instance created ****" );
    }

    // While the parameter weight has dimension BxMxW,  the resulting transformation has dimension of BxNxW.
    // We only need the M dimension from an BxNxM input to generate parameter matrix.
    // where weights is BxNxW and bias is W.
    const aitensor<T> forward(const aitensor<T>& input_data);

    // Leave the gradients as is. They are cached in the Node. 
    // They will be used to update the parameters in next parallel operations.
    // the dInput is the gradient we propagate to source Nodes in the graph;
    // while the parameter gradients get cached to be used to update the parameters later.
    const aitensor<T> backward(const aitensor<T>&  gradients);

    void updateParameters(std::string& optimizertype, T& learningRate, int& iter);

    void forwardPass() {}
    void backwardPass() {}
    std::string generateDotFormat();
};


#endif

