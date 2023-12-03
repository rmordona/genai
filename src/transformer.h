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

#include <limits>
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
            log_matrix(input);
        }
        heads.push_back(head);
    }

    return heads;
}

/**********************************************************************************************************************
* Unlike the RNN, note that the design of the Attention is based on using a matrix with dimension BxSxP
* where B is the number of samples, S is the length of the sequence, and P is the embedding size (the dimension).
* Here is a better illustration for an input matrix.
* 
* Input Data (5xP):
* token     sample 1    sample 2    sample 3        sample B
* token 1   word1_embed word1_embed word1_embed ... word1_embed  
* token 2   word2_embed word2_embed word3_embed ... word2_embed  
* ...
* token N   wordN_embed wordN_embed wordN_embed ... wordN_embed  
*
* Sequences should be padded for consistent length.
**********************************************************************************************************************/
template <class T>
class Attention : public BaseOperator {
private:
    aitensor<T> input_data = {};   // Input Dimention: BxNxM
    aitensor<T> encoder_data = {};      // If we are passing an encoded input
    aitensor<T> encoder_gradient = {};  // If we are passing an encoded input 

    Linear<T>* Wk  = nullptr;  // BxNxW  where W = dmodel (typically), though in practise, W can be explicit
    Linear<T>* Wv  = nullptr;  // BxNxW  where W = dmodel (typically), though in practise, W can be explicit
    Linear<T>* Wq  = nullptr;  // BxNxW  where W = dmodel (typically), though in practise, W can be explicit
    Linear<T>* Wo = nullptr; // this extra weight matrix will align the output dimension the same as the input. // BxNxM

    aimatrix<T> Mask; // Mask Kernel for Decoder in Transformer

    aitensor<T> Q;  // Q projection
    aitensor<T> K;  // K projection
    aitensor<T> V;  // V projection

    aitensor<T> QKweight;
    // aitensor<T> QKweightV;

    int B = 0;  // batch size
    int N = 0;  // input size
    int M = 0;  // number of features (embedding vector size)
    int W = 0;  // number of weights (or number of features)
    int H = 1;  // number of heads
    int Dk = 0; // number of dimensions per head (M/H)

    bool bias = true;
    bool masked = false;
    bool output_projection = true; // By setting it to true, we perform the output projection (Wo)
                                   // otherwise, we defer that step as last operation in multihead
                                   // so that when we split attention unit into multi heads, we only
                                   // project K, V, and Q and calculate all the way to the softmax(KQ/sqrt(dk))V
public:
    Attention(int size = 3, bool bias = true, bool masked = false, bool output_projection = true)  {
        this->W                 = size;
        this->bias              = bias;
        this->masked            = masked;
        this->output_projection = output_projection;
        log_info( "**** Attention instance created ****" );
    }

    // While the parameter weight has dimension BxMxW,  the resulting transformation has dimension of BxNxW.
    // We only need the M dimension from an BxNxM input to generate parameter matrix.
    // where weights is BxNxW and bias is W.
    const aitensor<T> forward(const aitensor<T>& input_data, const aitensor<T>& encoder_data = {});

    // Leave the gradients as is. They are cached in the Node. 
    // They will be used to update the parameters in next parallel operations.
    // the dInput is the gradient we propagate to source Nodes in the graph;
    // while the parameter gradients get cached to be used to update the parameters later.
    const aitensor<T> backward(const aitensor<T>& gradients);

    void updateParameters(std::string& optimizertype, T& learningRate, int& iter);

    // If an encoder gradient is preserved ...
    const aitensor<T> getEncoderGradient() { return this->encoder_gradient; }

    std::string generateDotFormat(const std::string& name = "generic", bool operators = false, bool weights = false);

    void forwardPass() {}
    void backwardPass() {}

};

/*****************************************************************************************************
* Base Multi-Head Attention Layer
*****************************************************************************************************/
template <class T>
class MultiHeadAttention : public BaseOperator {
private:
    aitensor<T> input_data = {};   // Input Dimention: BxNxM
    aitensor<T> encoder_data = {};      // If we are passing an encoded input
    aitensor<T> encoder_gradient = {};  // If we are passing an encoded input 

    std::vector<Attention<T>*> M1;

    Linear<T>* Wo = nullptr; // this extra weight matrix will align the output dimension the same as the input. // BxNxM

    int B = 0;  // batch size
    int N = 0;  // number of samples
    int M = 0;  // number of features (embedding vector size) for Q, K, V layer
    int W = 0;  // number of weights (or number of features)
    int F = 0;  // number of weights for First FeedForward Layer
    int H = 1;  // number of heads
    int Dk = 0; // number of dimensions per head (M/H)
    int split = 0; // number of values in an array to jump.

    bool bias = true;
    bool masked = false; // for Decoder Transformer

    std::string activationtype = "leakyrelu";
    float alpha = 0.01; // for leakyReLU

public:
    MultiHeadAttention(int heads = 3, int attention_size = 3, int feed_size = 3, bool bias = true, bool masked = false)  {
        this->W = attention_size;
        this->F = feed_size;
        this->H = heads;
        this->bias = bias;
        this->masked = masked;

        // M1.setZero();
        log_info( "**** MultiHeadAttention instance created ****" );
    }

    // While the parameter weight has dimension BxMxW,  the resulting transformation has dimension of BxNxW.
    // We only need the M dimension from an BxNxM input to generate parameter matrix.
    // where weights is BxNxW and bias is W.
    const aitensor<T> forward(const aitensor<T>& input_data = {}, const aitensor<T>& encoder_data = {});

    // Leave the gradients as is. They are cached in the Node. 
    // They will be used to update the parameters in next parallel operations.
    // the dInput is the gradient we propagate to source Nodes in the graph;
    // while the parameter gradients get cached to be used to update the parameters later.
    const aitensor<T> backward(const aitensor<T>& gradients);

    void updateParameters(std::string& optimizertype, T& learningRate, int& iter);

    // If an encoder gradient is preserved ...
    const aitensor<T> getEncoderGradient() { return this->encoder_gradient; }

    std::string generateDotFormat(const std::string& name = "generic", bool operators = false, bool weights = false);

    void forwardPass() {}
    void backwardPass() {} 
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
    int W = 0;
    int M = 0;

    std::string activationtype = "leakyrelu";
    float alpha = 0.01; // for leakyReLU

public:

    FeedForward(int feed_size = 3, bool bias = true, const std::string& activationtype = "leakyrelu") {
        this->activationtype = activationtype;
        this->W = feed_size;
        this->bias = bias;
        log_info( "**** FeedForward instance created ****" );
    }

    FeedForward(int feed_size = 3,  bool bias = true, const std::string& activationtype = "leakyrelu", const float alpha=0.01) {
        this->activationtype = activationtype;
        this->alpha = alpha;
        this->W = feed_size;
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

    std::string generateDotFormat(const std::string& name = "generic", bool operators = false, bool weights = false);

    void forwardPass() {}
    void backwardPass() {}
};

/*****************************************************************************************************
* Base Encoder Block
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
    int F = 0;

    std::string activationtype = "leakyrelu";
    float alpha = 0.01; // for leakyReLU

public:

    Encoder(int heads = 1, int attention_size = 3, int feed_size = 3, 
                    bool bias = true, const std::string& activationtype = "leakyrelu", const float alpha=0.01) {
        this->activationtype = activationtype;
        this->alpha = alpha;
        this->W = attention_size;
        this->F = feed_size;
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

    std::string generateDotFormat(const std::string& name = "generic", bool operators = false, bool weights = false);

    void forwardPass() {}
    void backwardPass() {}

};

/*****************************************************************************************************
* Base Encoder Layer
*
* The Transformer Encoder allows two types of training (in tandem with the Decoder Layer):
*
* - Inference (Without Teacher Forcing):
*
* 1. Encode the Input Sequence:
*    Pass the input sequence through the encoder to obtain the context vectors.
*
* 2. Initialize Decoder Input:
*    Set the input to the decoder as the start-of-sequence (SOS) token.
*
* 3. Decoding Loop:
*    Repeat until an end-of-sequence (EOS) token is generated or a maximum sequence length is reached:
*       Forward pass through the decoder to get the next token.
*       The next token is appended to the previously generated sequence.
*       The newly generated token becomes the input for the next decoding step.
*
*
* - Training (With Teacher Forcing):
*
* 1. Encode the Input Sequence:
*    Pass the input sequence through the encoder to obtain the context vectors.
*
* 2. Initialize Decoder Input:
*    Set the input to the decoder as the start-of-sequence (SOS) token.
*
* 3. Decoding Loop:
*    Repeat until an end-of-sequence (EOS) token is generated or a maximum sequence length is reached:
*       Forward pass through the decoder to get the next token.
*       The ground truth token from the target sequence is used as the input for the 
*          next decoding step instead of the model's own prediction.
*       The actual ground truth token is also used to calculate the loss during backpropagation.
*
* 4. Backpropagation:
*    Calculate the loss between the predicted sequence and the target sequence.
*    Perform backpropagation to update the model's parameters.
*
*****************************************************************************************************/
template <class T>
class EncoderLayer : public BaseOperator {
private:
    bool bias = true;
    int B = 0;
    int N = 0;
    int M = 0;
    int H = 0;
    int W = 0;
    int F = 0;
    int L = 1;

    std::string activationtype = "leakyrelu";
    float alpha = 0.01; // for leakyReLU

    std::vector<Encoder<T>> encoders;

public:

    EncoderLayer(int heads = 1, int attention_size = 3, int feed_size = 3,  int layers = 1, 
                    bool bias = true, const std::string& activationtype = "leakyrelu", const float alpha=0.01) {
        this->activationtype = activationtype;
        this->alpha = alpha;
        this->W = attention_size;
        this->F = feed_size;
        this->L = layers;
        this->bias = bias;
        this->H = heads;
        for (int i = 0; i < this->L; i++) {
            Encoder<T> encoder(heads, attention_size, feed_size, bias, activationtype, alpha);
            encoders.push_back(encoder);
        }
    }

    const aitensor<T> forward(const aitensor<T>& input_data);

    const aitensor<T> backward(const aitensor<T>&  gradients);

    void updateParameters(std::string& optimizertype, T& learningRate, int& iter);

    std::string generateDotFormat(const std::string& name , bool operators, bool weights);

    void forwardPass() {}
    void backwardPass() {}

};

/********************************************************************************************************
* Base Decoder  Block
*
* The Transformer Decoder allows two types of training:
*
* Without Teacher Forcing (Inference):
*
*    During inference or generation, the decoder's input at each step is its own previously generated 
*    token. This token is fed back into the decoder to predict the next token.
*
* With Teacher Forcing (Training):
*
*    During training, the true target sequence is known. Instead of using the decoder's own predictions 
*    as input for the next step, the actual ground truth (target) token is fed as input to the decoder 
*    at each step.
*
*    This means that, during training, the decoder is provided with the correct information at each step, 
*    allowing it to learn more effectively.
*
*********************************************************************************************************/
template <class T>
class Decoder : public BaseOperator {
private:
    aitensor<T> input_data; // NxW samples 
    aitensor<T> encoder_gradient = {};  // If we are passing an encoded input 

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
    int F = 0;

    std::string activationtype = "leakyrelu";
    float alpha = 0.01; // for leakyReLU

public:


    Decoder(int heads = 1, int attention_size = 3, int feed_size = 3, 
                    bool bias = true, const std::string& activationtype = "leakyrelu", const float alpha=0.01) {
        this->activationtype = activationtype;
        this->alpha = alpha;
        this->W = attention_size;
        this->F = feed_size;
        this->bias = bias;
        this->H = heads;
        log_info( "**** Encoder instance created ****" );
    }

    // While the parameter weight has dimension BxMxW,  the resulting transformation has dimension of BxNxW.
    // We only need the M dimension from an BxNxM input to generate parameter matrix.
    // where weights is BxNxW and bias is W.
    const aitensor<T> forward(const aitensor<T>& decoder_data, const aitensor<T>& encoder_data);

    // Leave the gradients as is. They are cached in the Node. 
    // They will be used to update the parameters in next parallel operations.
    // the dInput is the gradient we propagate to source Nodes in the graph;
    // while the parameter gradients get cached to be used to update the parameters later.
    const aitensor<T> backward(const aitensor<T>&  gradients);

    void updateParameters(std::string& optimizertype, T& learningRate, int& iter);

    // If an encoder gradient is preserved ...
    const aitensor<T> getEncoderGradient() { return this->encoder_gradient; }

    std::string generateDotFormat(const std::string& name = "generic", bool operators = false, bool weights = false);

    void forwardPass() {}
    void backwardPass() {}
};

/*****************************************************************************************************
* Base Decoder Layer
*****************************************************************************************************/
template <class T>
class DecoderLayer : public BaseOperator {
private:
    bool bias = true;
    int B = 0;
    int N = 0;
    int M = 0;
    int H = 0;
    int W = 0;
    int F = 0;
    int L = 1;

    std::string activationtype = "leakyrelu";
    float alpha = 0.01; // for leakyReLU

    std::vector<Decoder<T>> decoders;

public:

    DecoderLayer(int heads = 1, int attention_size = 3, int feed_size = 3, int layers = 1, 
                    bool bias = true, const std::string& activationtype = "leakyrelu", const float alpha=0.01) {
        this->activationtype = activationtype;
        this->alpha = alpha;
        this->W = attention_size;
        this->F = feed_size;
        this->L = layers;
        this->bias = bias;
        this->H = heads;
        for (int i = 0; i < this->L; i++) {
            Decoder<T> decoder(heads, attention_size, feed_size, bias, activationtype, alpha);
            decoders.push_back(decoder);
        }
    }

    const aitensor<T> forward(const aitensor<T>& decoder_data, const aitensor<T>& encoder_data);

    const aitensor<T> backward(const aitensor<T>&  gradients);

    void updateParameters(std::string& optimizertype, T& learningRate, int& iter);

    std::string generateDotFormat(const std::string& name , bool operators, bool weights);

    void forwardPass() {}
    void backwardPass() {}

};

/*****************************************************************************************************
* Base PositionalEncoder Layer
*****************************************************************************************************/
template <class T>
class PositionalEncoder {
private:
public:

    PositionalEncoder() {}
    
    // Get the positional encoding for a given position
    static airowvector<T> generate_positional_embedding(int pos,int N, int M)  {
        airowvector<T> encoding(M);
        for (int i = 0; i < M; i++) {
            T angle = pos / static_cast<T>(N); // the max length of tokens in a token matrix
            T exponent_term = i / static_cast<T>(M);
            encoding(i) = sin(angle * pow(10000, 2 * exponent_term))  + cos(angle * pow(10000, 2 * exponent_term));
        }
        return encoding;
    }


    static aitensor<T>  encode(const aitensor<T>& input_data) {

        log_info("=====================================================");
        log_info( "Entering Positional Encoding ..." );

        // aimatrix<T> input_matrix;
        std::vector<airowvector<T>> pos_encoding;  // Placeholder structure for the Encoding

        // dimension is BxNxW
        int B = input_data.size();
        int N = input_data.at(0).rows();
        int M = input_data.at(0).cols();  

        // initialize the Positional Embedding
        pos_encoding.resize(N, airowvector<T>(M));
        for (int pos = 0; pos < N; ++pos) {
            pos_encoding[pos] = generate_positional_embedding(pos, N, M);
        }

        aitensor<T> standard = input_data;

        // Adding positional embedding
        for (int i = 0; i < B; ++i) {
            for (int j = 0; j < N; ++j) {
                standard.at(i).row(j) += pos_encoding[j % N];
            }
        }
        return standard;
    }
};


#endif

