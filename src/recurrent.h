

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
 */

#pragma once
#ifndef RECURRENT_H
#define RECURRENT_H

#include "operators.h"

enum class RNNType {
    MANY_TO_ONE,
    ONE_TO_MANY,
    MANY_TO_MANY
};

enum class CellType {
    RNN_VANILLA,
    RNN_LSTM,
    RNN_GRU
};

template <class T>
class CellBase {
public:

    // Here we are splitting the concatenated XH where dimension is Nx(P+H)
    // such that we split XH into two matrices and get the original dimensions for X and H
    // so that X will have NxP and H will have NxH
    static std::tuple<aimatrix<T>, aimatrix<T>> split(const aimatrix<T>& XH, int param_size, int hidden_size) {
        aimatrix<T> A(param_size, hidden_size);
        aimatrix<T> B(hidden_size, hidden_size);
        B = XH.block(param_size, 0, hidden_size, hidden_size);
        A = XH.block(0, 0, param_size, hidden_size);
        return std::make_tuple(A, B);
    }

    virtual const aimatrix<T> forward(const aimatrix<T>& X) = 0; 

    // Used by vanilla RNN and GRU cells
    virtual const std::tuple<aimatrix<T>,aimatrix<T>> backward(int step, 
            const aimatrix<T>& dOut, const aimatrix<T>& dnext_h) = 0; 

    // Used by LSTM Cell
    virtual const std::tuple<aimatrix<T>, aimatrix<T>, aimatrix<T>> backward(int step, 
            const aimatrix<T>& dOut, const aimatrix<T>& dnext_h, const aimatrix<T>& dnext_c) = 0;  

    virtual void updateParameters(std::string& optimizertype, T& learningRate, int& iter) = 0;

};

/************************************************************************************************************
* Note that the design of the Recurrent Network Cells is based on using a matrix with dimension NxP
* where N is the number of words and P is the embedding size (the dimension).
* Each word in N does does not belong to one sequence, rather is a set of sequences.
* Here is a better illustration for an input matrix of 5xP:
* 
* Input Data (5xP):
* step t      step t+1    step t+2        step t+P-1
* word1_embed word2_embed word3_embed ... wordP_embed (sequence 1)
* word1_embed word2_embed word3_embed ... wordP_embed (sequence 2)
* ...
* word1_embed word2_embed word3_embed ... wordP_embed (sequence 5)
*
* This will allow us to process sequences in batches instead of just looping over one step at a time
* for a single sequence. Now, if that's the case, we pad each sequence to achieve uniform sequence lengths.
*************************************************************************************************************/
template <class T>
class RNNCell : public CellBase<T> {
private:
   // Parameters (weights and biases)
    aimatrix<T> W;  // Weight for the Input   (p x h)
    aimatrix<T> U;  // Weight for the Hidden State  (h x h)  (rows x columns)

    airowvector<T> bh; // Hidden bias

    std::vector<aimatrix<T>> H;  // Hidden state  (n x h) where n = number of words, h = hidden size
    std::vector<aimatrix<T>> X;
    std::vector<aimatrix<T>> dH; // Gradient with respect to Hidden state
    std::vector<aimatrix<T>> dX; // Gradient with respect to Input

   // Caching Gradients with respect to hidden-to-hidden weights  W, U
    aimatrix<T> dW;
    aimatrix<T> dU;

    // Caching Gradients with respect to hidden biases bh
    airowvector<T> dbh;

   // Caching optimizer parameters
    Optimizer<T>* opt_W  = nullptr; // for optimizer
    Optimizer<T>* opt_U  = nullptr; // for optimizer
    Optimizer<T>* opt_bh = nullptr; // for optimizer

    int input_size = 0;
    int param_size = 0;
    int hidden_size;
    bool last_layer = false;
    // int output_size;

public:
    RNNCell(int hidden_size, bool last_layer) : hidden_size(hidden_size), last_layer(last_layer) {}

    void setInitialWeights(int N, int P);
    const aimatrix<T> forward(const aimatrix<T>& X);
    const std::tuple<aimatrix<T>,aimatrix<T>> backward(int step, const aimatrix<T>& dOut, const aimatrix<T>& dnext_h);

    // Not used by RNNCell
    const std::tuple<aimatrix<T>,aimatrix<T>,aimatrix<T>> backward(int step, 
            const aimatrix<T>& dOut, const aimatrix<T>& dnext_h, const aimatrix<T>& dnext_c) {
            return std::make_tuple(aimatrix<T>(), aimatrix<T>(), aimatrix<T>());
    } 

    void updateParameters(std::string& optimizertype, T& learningRate, int& iter);
};

template <class T>
class LSTMCell : public CellBase<T> {
private:
    // Parameters (weights and biases)
    aimatrix<T> Wf;      // Weight matrix for input gate from input x         (p + h) x h
    aimatrix<T> Wi;      // Weight matrix for input gate from input x         (p + h) x h
    aimatrix<T> Wo;      // Weight matrix for output gate from input x        (p + h) x h
    aimatrix<T> Wg;      // Weight matrix for candidate state from input x    (p + h) x h

    airowvector<T> bf;   // Bias vector for input gate        (1xh)
    airowvector<T> bi;   // Bias vector for input gate        (1xh)
    airowvector<T> bg;   // Bias vector for candidate state   (1xh)
    airowvector<T> bo;   // Bias vector for output gate       (1xh)

    aimatrix<T> Ft;      // Forget Gate       (nxh)
    aimatrix<T> It;      // Input Gate        (nxh)
    aimatrix<T> Ot;      // Output Gate       (nxh)
    aimatrix<T> Gt;      // Candidate State   (nxh)

    std::vector<aimatrix<T>> H;  // Hidden state  (n x h) where n = number of words, h = hidden size
    std::vector<aimatrix<T>> C;  // Cell state  (n x h) where n = number of words, h = hidden size
    std::vector<aimatrix<T>> X;
    std::vector<aimatrix<T>> dH; // Gradient with respect to Hidden state
    std::vector<aimatrix<T>> dC;
    std::vector<aimatrix<T>> dX; // Gradient with respect to Input

    aimatrix<T> XH;

    // Caching Gradients with respect to hidden-to-hidden weights  Ft, It, Gt, Ot
    aimatrix<T> dWf;
    aimatrix<T> dWi;
    aimatrix<T> dWg;
    aimatrix<T> dWo;

    // Caching Gradients with respect to hidden biases bf, bi, bg, bo
    airowvector<T> dbf;
    airowvector<T> dbi;
    airowvector<T> dbg;
    airowvector<T> dbo;

    // Caching optimizer parameters
    Optimizer<T>* opt_Wf = nullptr; // for optimizer
    Optimizer<T>* opt_Wi = nullptr; // for optimizer
    Optimizer<T>* opt_Wg = nullptr; // for optimizer
    Optimizer<T>* opt_Wo = nullptr; // for optimizer
    Optimizer<T>* opt_bf = nullptr; // for optimizer
    Optimizer<T>* opt_bi = nullptr; // for optimizer
    Optimizer<T>* opt_bg = nullptr; // for optimizer
    Optimizer<T>* opt_bo = nullptr; // for optimizer

    int input_size;
    int param_size;
    int hidden_size;
    bool last_layer = false;
    // int output_size;

public:
    LSTMCell(int hidden_size, bool last_layer) : hidden_size(hidden_size), last_layer(last_layer) {}

    void setInitialWeights(int N, int P);
    const aimatrix<T> forward(const aimatrix<T>& X);

    // Not used by LSTMCell
    const std::tuple<aimatrix<T>,aimatrix<T>> backward(int step, const aimatrix<T>& dOut, const aimatrix<T>& dnext_h)  {
            return std::make_tuple(aimatrix<T>(), aimatrix<T>());
    } 
    const std::tuple<aimatrix<T>,aimatrix<T>,aimatrix<T>> backward(int step, 
            const aimatrix<T>& dOut, const aimatrix<T>& dnext_h, const aimatrix<T>& dnext_c);

    void updateParameters(std::string& optimizertype, T& learningRate, int& iter);
};

template <class T>
class GRUCell : public CellBase<T> {
private:
    // Weights for the input-to-hidden connections
    aimatrix<T> Wz;      // Weight matrix for the update gate               (p + h) x h
    aimatrix<T> Wr;      // Weight matrix for the reset gate                (p + h) x h
    aimatrix<T> Wg;      // Weight matrix for the candidate hidden state    (p + h) x h

    // Biases for the hidden units
    airowvector<T> bz;   // Bias vector for the update gate              (1xh)
    airowvector<T> br;   // Bias vector for the reset gate               (1xh)
    airowvector<T> bg;   // Bias vector for the candidate hidden state   (1xh)

    aimatrix<T> Zt;      // Forget Gate       (nxh)
    aimatrix<T> Rt;      // Input Gate        (nxh)
    aimatrix<T> Gt;      // Candidate State   (nxh)

    std::vector<aimatrix<T>> H;  // Hidden state  (n x h) where n = number of words, h = hidden size
    std::vector<aimatrix<T>> X;  // (n x p)
    std::vector<aimatrix<T>> dH; // Gradient with respect to Hidden state
    std::vector<aimatrix<T>> dX; // Gradient with respect to Input

    aimatrix<T> XH;      // Concatenate X and H

    // Caching Gradients with respect to hidden-to-hidden weights Wf, Wi, Wo, Wc
    aimatrix<T> dWz;
    aimatrix<T> dWr;
    aimatrix<T> dWg;

    // Caching Gradients with respect to hidden biases bf, bi, bo, bc
    airowvector<T> dbz;
    airowvector<T> dbr;
    airowvector<T> dbg;

    // Caching optimizer parameters
    Optimizer<T>* opt_Wz = nullptr; // for optimizer
    Optimizer<T>* opt_Wr = nullptr; // for optimizer
    Optimizer<T>* opt_Wg = nullptr; // for optimizer
    Optimizer<T>* opt_bz = nullptr; // for optimizer
    Optimizer<T>* opt_br = nullptr; // for optimizer
    Optimizer<T>* opt_bg = nullptr; // for optimizer

    int input_size;
    int param_size;
    int hidden_size;
    bool last_layer = false;
    // int output_size;

public:
    GRUCell(int hidden_size, bool last_layer) : hidden_size(hidden_size), last_layer(last_layer) {}

    void setInitialWeights(int N, int P);
    const aimatrix<T> forward(const aimatrix<T>& X);
    const std::tuple<aimatrix<T>,aimatrix<T>> backward(int step, const aimatrix<T>& dOut, const aimatrix<T>& dnext_h);

    // Not used by GRUCell
    const std::tuple<aimatrix<T>,aimatrix<T>,aimatrix<T>> backward(int step, 
            const aimatrix<T>& dOut, const aimatrix<T>& dnext_h, const aimatrix<T>& dnext_c) {
            return std::make_tuple(aimatrix<T>(), aimatrix<T>(), aimatrix<T>());
    } 

    void updateParameters(std::string& optimizertype, T& learningRate, int& iter);
};

template <class T>
class RecurrentBase {
private:
    std::vector<CellBase<T>*> fcells;
    std::vector<CellBase<T>*> bcells;
    std::vector<CellBase<T>*> cells;

    aitensor<T> input_data;

    int sequence_length = 0;
    int input_size = 0;
    int embedding_size = 0;
    int num_directions = 2;

    ActivationType otype;
    ReductionType rtype;

    // Consider a vector of outputs specially in a MANY-TO-MANY scenario
    // where each time-step produces an output.
    aitensor<T> foutput, boutput, output;
    aitensor<T> V;      // Weight for the predicted output  (h x o) 
    std::vector<airowvector<T>> bo;  // Bias for the predicted output  

    aitensor<T> dV;      // Weight for the predicted output  (h x o)
    std::vector<airowvector<T>> dbo;  // Bias for the predicted output

    aitensor<T> gradients;
    aitensor<T> Yhat;

    // Caching optimizer parameters, vector of optimization parameters per time-step
    std::vector<Optimizer<T>*> opt_V; // for optimizer
    std::vector<Optimizer<T>*> opt_bo; // for optimizer

    bool initialized = false; // used to ensure V and bo are not initialized multiple times during training.

    int hidden_size;
    int output_size;
    int num_layers;
    bool bidirectional;
    RNNType rnntype;

    int direction = 0;
    CellType celltype;

        // Factory function to create cell objects based on an option
    CellBase<T>* createCell(CellType celltype, int hidden_size, bool last_layer) {
        if (celltype == CellType::RNN_VANILLA) {
            return new RNNCell<T>(hidden_size, last_layer);
        } else
        if (celltype == CellType::RNN_LSTM) {
            return new LSTMCell<T>(hidden_size, last_layer);
        } else
        if (celltype == CellType::RNN_GRU) {
            return new GRUCell<T>(hidden_size, last_layer);
        } else {
            return nullptr; // Invalid option
        }
    }

public:
    RecurrentBase(int hidden_size, int output_size, int num_layers, bool bidirectional, RNNType rnntype, CellType celltype) 
        : hidden_size(hidden_size), output_size(output_size), num_layers(num_layers), bidirectional(bidirectional), rnntype(rnntype), celltype(celltype) {

        if (bidirectional != true) {
            this->num_directions = 1;
        }

        for (int layer = 0; layer < num_layers; ++layer) {
            CellBase<T>* rnn_ptr1 = createCell(celltype, hidden_size, layer + 1 == num_layers);
            CellBase<T>* rnn_ptr2 = createCell(celltype, hidden_size, layer + 1 == num_layers);
            this->fcells.push_back(std::move(rnn_ptr1)); 
            this->bcells.push_back(std::move(rnn_ptr2));
        }

    }

    std::tuple<aimatrix<T>, airowvector<T>> getWeights(int step, aimatrix<T> out);

    const aitensor<T> processOutputs();

    void processGradients(aitensor<T>& gradients);

    const aitensor<T> forwardpass(const aitensor<T>& input_data);

    const aitensor<T> backwardpass(const aitensor<T>& gradients);

    void updatingParameters(std::string& optimizertype, T& learningRate, int& iter);

    RNNType getRNNType() { return this->rnntype; }
    ActivationType getOType() { return this->otype; }
    ReductionType getRType() { return this->rtype; }

    int getNumDirections() { return this->num_directions;}
    int getNumLayers() { return this->num_layers; }

    void setOutput(const aitensor<T>& output) { this->output = output; }
    void setPrediction(const aitensor<T>& Yhat) { this->Yhat = Yhat; }
    const aitensor<T>& getPrediction() { return this->Yhat; }

    void setGradients(const aitensor<T>& gradients) { this->gradients = gradients; }
    const aitensor<T>& getGradients() { return this->gradients; }

    void setRNNType(RNNType rnntype) {  this->rnntype = rnntype; }
    void setOType(ActivationType otype) {  this->otype = otype; }
    void setRType(ReductionType rtype) {  this->rtype = rtype; }

    void setSequenceLength(int sequence_length) { this->sequence_length = sequence_length; }
    void setInputSize(int input_size) { this->input_size = input_size; }
    void setEmbeddingSize(int embedding_size) { this->embedding_size = embedding_size; }

    int getSequenceLength() { return this->sequence_length; }
    int getInputSize() { return this->input_size; }
    int getEmbeddingSize() { return this->embedding_size; }
    int getHiddenSize() { return this->hidden_size; }

    void setOutputSize(int output_size) { this->output_size = output_size; }
    int getOutputSize() { return this->output_size;  }

    void setInitialized() { this->initialized = true; }
    bool isInitialized() { return this->initialized; }

    CellType getCellType() { return this->celltype; }

    std::vector<CellBase<T>*> getCells(int direction)  {
        this->direction = direction;
        if (direction == 0) {
            return this->fcells;
        } else {
            return this->bcells;
        }
    }

};

template <class T>
class RNN : public BaseOperator {
private:
    RecurrentBase<T>* rnnbase;
public:
    RNN(int hidden_size, int output_size, int num_layers, bool bidirectional, RNNType rnntype) {
            this->rnnbase = new RecurrentBase<T>(hidden_size, output_size, num_layers, bidirectional, rnntype, CellType::RNN_VANILLA);
    }

    const aitensor<T> forward(const aitensor<T>& input_data);
    const aitensor<T> backward(const aitensor<T>& gradients);
    void updateParameters(std::string& optimizertype, T& learningRate, int& iter);

    void forwardPass() {} // virtual function of BaseOperator (different from those of the RecurrentBase)
    void backwardPass() {} // virtual function of BaseOperator (different from those of the RecurrentBase)

};

template <class T>
class LSTM : public BaseOperator {
private:
    RecurrentBase<T>* rnnbase;
public:
    LSTM(int hidden_size, int output_size, int num_layers, bool bidirectional, RNNType rnntype)  {
        this->rnnbase = new RecurrentBase<T>(hidden_size, output_size, num_layers, bidirectional, rnntype, CellType::RNN_LSTM);
    }

    const aitensor<T> forward(const aitensor<T>& input_data);
    const aitensor<T> backward(const aitensor<T>& gradients);
    void updateParameters(std::string& optimizertype, T& learningRate, int& iter);

    void forwardPass() {} // virtual function of BaseOperator (different from those of the RecurrentBase)
    void backwardPass() {} // virtual function of BaseOperator (different from those of the RecurrentBase)
};

template <class T>
class GRU : public BaseOperator {
private:
    RecurrentBase<T>* rnnbase;
public:
    GRU(int hidden_size, int output_size,  int num_layers, bool bidirectional, RNNType rnntype) {
        this->rnnbase = new RecurrentBase<T>(hidden_size, output_size, num_layers, bidirectional, rnntype, CellType::RNN_GRU);
    }
    const aitensor<T> forward(const aitensor<T>& input_data);
    const aitensor<T> backward(const aitensor<T>& gradients);
    void updateParameters(std::string& optimizertype, T& learningRate, int& iter);

    void forwardPass() {} // virtual function of BaseOperator (different from those of the RecurrentBase)
    void backwardPass() {} // virtual function of BaseOperator (different from those of the RecurrentBase)
};


#endif