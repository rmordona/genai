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
 
#include "genai.h"
#include "recurrent.h"

/***************************************************************************************************************************
*********** IMPLEMENTING RNNCell
****************************************************************************************************************************/
template <class T>
RNNCell<T>::RNNCell(int hidden_size, T learning_rate) 
    : hidden_size(hidden_size), learning_rate(learning_rate) {
    learning_rate = 0.01;
}

template <class T>
const aimatrix<T>& RNNCell<T>::getHiddenState() {
    return H;
}

template <class T>
void RNNCell<T>::setInitialWeights(int N, int P) {
    // Initialize parameters, gates, states, etc.
    if (this->input_size !=0 || this->param_size != 0) return;

    this->input_size = N;  
    this->param_size = P;  

    W.resize(param_size, hidden_size);
    U.resize(hidden_size, hidden_size);

    BaseOperator::heInitMatrix(W);
    BaseOperator::heInitMatrix(U);

    bh = airowvector<T>::Zero(hidden_size);

    H    = aimatrix<T>::Zero(input_size, hidden_size);
}

template <class T>
const aimatrix<T>& RNNCell<T>::forward(const aimatrix<T>& input_data) {

    // Store input for backward pass.
    // this can be the prev_hidden_state
    // this->X = input_data;
    const aimatrix<T>& X = input_data;

    setInitialWeights(input_data.rows(), input_data.cols());

    // Compute hidden state.
    //     (nxh) =  (nxp) * (pxh) + (nxh) * (hxh) + h
    H = BaseOperator::tanh((aimatrix<T>) (X * W + H * U + bh) );

    // Compute Yhat.
    //     (nxo) =  (nxh) * (hxo) + h
    // Yhat = BaseOperator::softmax(H * V + bo);

    // We return the Hidden State as output;
    return H; 
}

template <class T>
const aimatrix<T>& RNNCell<T>::backward(const aimatrix<T>& input_data, const aimatrix<T>& dnext_h) {
    // Backpropagation logic for RNN

    const aimatrix<T>& X = input_data;

    // Compute gradient with respect to tanh
    aimatrix<T> dtanh = (1.0 - H.array().square()).array() * dnext_h.array();

    // Compute gradient with respect to hidden-to-hidden weights Whh.
    aimatrix<T> dU = H.transpose() * dtanh;

    // Compute gradient with respect to input-to-hidden weights Wxh.
    aimatrix<T> dW = X.transpose() * dtanh;

    // Compute gradient with respect to hidden-state.
    dH = dtanh * U.transpose();

    // Compute gradient with respect to input (dInput).
    dX = dtanh * W.transpose();

    // Compute gradient with respect to hidden bias bh.
    airowvector<T> dbh = dtanh.colwise().sum(); 

    // Compute gradient with respect to output bias bo.
    //airowvector<T> dbo = Y.colwise().sum();

    // Update weights and biases.
    W -= learning_rate * dW;
    U -= learning_rate * dU;
    bh -= learning_rate * dbh;
    // bo -= learning_rate * dbo;

    // Return gradient with respect to input.
    // We return the gradient wrt X for each layer, not wrt H for each time step.
    // for the gradient wrt H, we just cache it for time step propagation, but layer propagation.
    return dX;
}
 
/***************************************************************************************************************************
*********** IMPLEMENTING LSTMCell
****************************************************************************************************************************/
template <class T>
LSTMCell<T>::LSTMCell(int hidden_size, T learning_rate) 
    : hidden_size(hidden_size), learning_rate(learning_rate) {
    learning_rate = 0.01;
}

template <class T>
const aimatrix<T>& LSTMCell<T>::getHiddenState() {
    return H;
}

template <class T>
const aimatrix<T>& LSTMCell<T>::getCellState() {
    return C;
}

template <class T>
void LSTMCell<T>::setInitialWeights(int N, int P) {
    // Initialize parameters, gates, states, etc.
    if (this->input_size !=0 || this->param_size != 0) return;

    this->input_size = N;  
    this->param_size = P;  

    // Initialize parameters (weights and biases)
    // Note: Initialize these parameters according to your initialization strategy
    Wf.resize(param_size + hidden_size, hidden_size);
    Wi.resize(param_size + hidden_size, hidden_size);
    Wo.resize(param_size + hidden_size, hidden_size);
    Wg.resize(param_size + hidden_size, hidden_size);

    BaseOperator::heInitMatrix(Wf);
    BaseOperator::heInitMatrix(Wi);
    BaseOperator::heInitMatrix(Wo);
    BaseOperator::heInitMatrix(Wg);

    bf = airowvector<T>::Zero(hidden_size);
    bi = airowvector<T>::Zero(hidden_size);
    bo = airowvector<T>::Zero(hidden_size);
    bg = airowvector<T>::Zero(hidden_size);

    H    = aimatrix<T>::Zero(input_size, hidden_size);
    C    = aimatrix<T>::Zero(input_size, hidden_size);

    XH   = aimatrix<T>::Zero(input_size, param_size + hidden_size); // concatenate X and H

}

template <class T>
const aimatrix<T>& LSTMCell<T>::forward(const aimatrix<T>& input_data) {

    // Store input for backward pass.
    // this->X = input_data;
    const aimatrix<T>& X = input_data;

    setInitialWeights(input_data.rows(), input_data.cols());

    // Concatenate input and hidden state. 
    XH << X, H;  // In terms of dimension, we have: (NxP) + (NxH) = Nx(P+H)

    // Calculate the forget gate values
    Ft = BaseOperator::sigmoid((aimatrix<T>) (XH * Wf + bf));

    // Calculate the input gate values
    It = BaseOperator::sigmoid((aimatrix<T>) (XH * Wi + bi));

    // Calculate the output gate values
    Ot = BaseOperator::sigmoid((aimatrix<T>) (XH * Wo + bo));

    // Calculate the candidate state values
    Gt = BaseOperator::tanh((aimatrix<T>) (XH * Wg + bg));

    // Calculate the new cell state by updating based on the gates
    C = Ft.array() * C.array() + It.array() * Gt.array();

    // Calculate the new hidden state by applying the output gate and tanh activation
    H = BaseOperator::tanh(C).array() * Ot.array();

    // Compute Yhat.
    //     (nxo) =  (nxh) * (hxo) + h
    // Yhat = BaseOperator::softmax(H * V + by);

    // We return the Hidden State as output;
    return H; 

}

template <class T>
const aimatrix<T>& LSTMCell<T>::backward(const aimatrix<T>& input_data, const aimatrix<T>& dnext_h) {
    // Backpropagation logic for LSTM

    const aimatrix<T>& X = input_data;

    // Compute gradients with respect to the gates
    dC = (Ot.array() * (1 - BaseOperator::tanh((aimatrix<T>) (C.array())).array().square()) * dnext_h.array()) + dC.array();
    aimatrix<T> dFt = dC.array() * C.array() * Ft.array() * (1 - Ft.array());
    aimatrix<T> dIt = dC.array() * Gt.array() * It.array() * (1 - It.array());
    aimatrix<T> dGt = dC.array() * It.array().array() * (1 - Gt.array().square().array());
    aimatrix<T> dOt = dnext_h.array() * BaseOperator::tanh(C).array() * Ot.array() * (1 - Ot.array());

    // Concatenate input and hidden state.
    XH << X, H;

    // Compute gradients with respect to hidden-to-hidden weights Wf, Wi, Wo, Wc
    aimatrix<T> dWf = XH.transpose() * dFt;
    aimatrix<T> dWi = XH.transpose() * dIt;
    aimatrix<T> dWg = XH.transpose() * dGt;
    aimatrix<T> dWo = XH.transpose() * dOt;


    // Compute gradients with respect to hidden biases bf, bi, bo, bc

    airowvector<T> dbf = dFt.colwise().sum();
    airowvector<T> dbi = dIt.colwise().sum();
    airowvector<T> dbo = dOt.colwise().sum();
    airowvector<T> dbg = dGt.colwise().sum();

    // Compute gradient with respect to the cell state C (dC)
    dC = dC.array() * Ft.array();

    // Compute gradient with respect to hidden state H (dH)
    int ps = param_size, hs = hidden_size;

    aimatrix<T> Wfx, Wfh, Wix, Wih, Wox, Woh, Wgx, Wgh;

    std::tie(Wfx, Wfh) = CellBase<T>::split(this->Wf, ps, hs);
    std::tie(Wix, Wih) = CellBase<T>::split(this->Wi, ps, hs);
    std::tie(Wox, Woh) = CellBase<T>::split(this->Wo, ps, hs);
    std::tie(Wgx, Wgh) = CellBase<T>::split(this->Wg, ps, hs);

    // Compute gradient with respect to hidden state.
    dH  = dFt * Wfx + dIt * Wix + dOt * Wox + dGt * Wgx;

    // Compute gradient with respect to input (dInput).
    dX  = dFt * Wfh + dIt * Wih + dOt * Woh + dGt * Wgh;

    // Compute gradient with respect to output bias bo.
    // airowvector<T> dby = Y.colwise().sum();

    // Update parameters and stored states
    this->Wf -= learning_rate * dWf;
    this->Wi -= learning_rate * dWi;
    this->Wo -= learning_rate * dWo;
    this->Wg -= learning_rate * dWg;
    this->bf -= learning_rate * dbf;
    this->bi -= learning_rate * dbi;
    this->bo -= learning_rate * dbo;
    this->bg -= learning_rate * dbg;
    // by -= learning_rate * dby;

    // Update hidden state  and cell state
    H -= learning_rate * dH;
    C -= learning_rate * dC;

    // Update stored states and gates
    Ft -= learning_rate * dFt;
    It -= learning_rate * dIt;
    Ot -= learning_rate * dOt;
    Gt -= learning_rate * dGt;

    // Return gradient with respect to input.
    // We return the gradient wrt X for each layer, not wrt H & C for each time step.
    // for the gradient wrt H & C, we just cache it for time step propagation, but layer propagation.
    return dX;
}

/***************************************************************************************************************************
 *********** IMPLEMENTING GRUCell
****************************************************************************************************************************/
template <class T>
GRUCell<T>::GRUCell(int hidden_size, T learning_rate) 
    : hidden_size(hidden_size), learning_rate(learning_rate) {
    learning_rate = 0.01;
}

template <class T>
const aimatrix<T>& GRUCell<T>::getHiddenState() {
    return H;
}

template <class T>
void GRUCell<T>::setInitialWeights(int N, int P) {
    // Initialize parameters, gates, states, etc.
    if (this->input_size !=0 || this->param_size != 0) return;

    this->input_size = N;  
    this->param_size = P;  

    // Initialize parameters (weights and biases)
    // Note: Initialize these parameters according to your initialization strategy
    Wz.resize(param_size + hidden_size, hidden_size);
    Wr.resize(param_size + hidden_size, hidden_size);
    Wg.resize(param_size + hidden_size, hidden_size);

    BaseOperator::heInitMatrix(Wz);
    BaseOperator::heInitMatrix(Wr);
    BaseOperator::heInitMatrix(Wg);

    bz = airowvector<T>::Zero(hidden_size);
    br = airowvector<T>::Zero(hidden_size);
    bg = airowvector<T>::Zero(hidden_size);

    H  = aimatrix<T>::Zero(input_size, hidden_size);

    XH = aimatrix<T>::Zero(input_size, param_size + hidden_size); // concatenate X and H

}

template <class T>
const aimatrix<T>& GRUCell<T>::forward(const aimatrix<T>& input_data) {

    // Store input for backward pass.
    // this can be the prev_hidden_state
    // X = input_data;
    const aimatrix<T>& X = input_data;

    setInitialWeights(input_data.rows(), input_data.cols());

    // Concatenate input and hidden state.
    XH << X, H;

    // Calculate the update gate using input, hidden state, and biases
    Zt =  BaseOperator::sigmoid((aimatrix<T>)(XH * Wz + bz));

    // Calculate the reset gate using input, hidden state, and biases
    Rt =  BaseOperator::sigmoid((aimatrix<T>)(XH * Wr + br));

    // Calculate the candidate hidden state using input, reset gate, hidden state, and biases
    aimatrix<T> RH = Rt.array() * H.array();
    aimatrix<T> rXH(input_size, hidden_size + hidden_size);

    rXH << X, RH;

    Rt = BaseOperator::tanh((aimatrix<T>)(XH * Wg + bg));

    // Calculate the new hidden state using update gate, previous hidden state, and candidate hidden state
    H = (1 - Zt.array()).array() * Gt.array() + Zt.array() * H.array(); 

    // Compute Yhat.
    //     (nxo) =  (nxh) * (hxo) + h
    // Yhat = BaseOperator::softmax(H * V + bo);

    // We return the Hidden State as output;
    return H;
}

template <class T>
const aimatrix<T>& GRUCell<T>::backward(const aimatrix<T>& input_data, const aimatrix<T>& dnext_h) {
    // Backpropagation logic for GRU

    const aimatrix<T>& X = input_data;

    // Compute gradients with respect to hidden-to-hidden weights Wz, Wr, Wg

    int ps = param_size, hs = hidden_size;

    aimatrix<T> Wzx, Wzh, Wrx, Wrh, Wgx, Wgh;

    std::tie(Wzx, Wzh) = CellBase<T>::split(this->Wz, ps, hs);
    std::tie(Wrx, Wrh) = CellBase<T>::split(this->Wr, ps, hs);
    std::tie(Wgx, Wgh) = CellBase<T>::split(this->Wg, ps, hs);

    // Compute gradients with respect to the gates
    aimatrix<T> dZt = dnext_h.array() * ( H.array() - Gt.array()) * Zt.array() * ( 1 - Zt.array());
    aimatrix<T> dGt = dnext_h.array() * ( 1 - Zt.array()) * ( 1 - Gt.array().square());
    aimatrix<T> dRt = ( dGt.array() * (H * Wrh).array()  * ( 1 - Rt.array().square())) * Rt.array() * ( 1 - Rt.array());

    // Concatenate input and hidden state.
    XH << X, H;

    // Compute gradients with respect to hidden-to-hidden weights Wz, Wr, Wg
    aimatrix<T> dWz = XH.transpose() * dZt;
    aimatrix<T> dWr = XH.transpose() * dRt;
    aimatrix<T> dWg = XH.transpose() * dGt;

    // Compute gradients with respect to hidden biases bz, br, bg
    airowvector<T> dbz = dZt.colwise().sum();
    airowvector<T> dbr = dGt.colwise().sum();
    airowvector<T> dbg = dRt.colwise().sum();

    // Compute gradient with respect to hidden state H (dH)
    dH = dZt * Wzx + dRt * Wrx + dGt * Wgx;

    // Compute gradient with respect to input (dInput)
    dH = dZt * Wzh + dRt * Wrh + dGt * Wgh;

    // Compute gradient with respect to output bias bo.
    // airowvector<T> dbo = Y.colwise().sum();

    // Update parameters
    Wz -= learning_rate * dWz;
    Wr -= learning_rate * dWr;
    Wg -= learning_rate * dWg;
    bz -= learning_rate * dbz;
    br -= learning_rate * dbr;
    bg -= learning_rate * dbg;
    // bo -= learning_rate * dbo;

    // Update hidden state  
    H -= learning_rate * dH;

    // Update stored states and gates
    Zt -= learning_rate * dZt;
    Rt -= learning_rate * dRt;
    Gt -= learning_rate * dGt;

    // Return gradient with respect to input.
    // We return the gradient wrt X for each layer, not wrt H & C for each time step.
    // for the gradient wrt H & C, we just cache it for time step propagation, but layer propagation.
    return dX;
}


/*****************************************************************************************************************
* Implement Recurrent Network
*****************************************************************************************************************/

/*****************************************************************************************************************
* This forward function will capture the final output of each step in each layer. Which therefore means that
* it can support many-to-one, one-to-many, and many-to-many scenario. By default, it already gives support to 
* the one-to-many and many-to-many-scenario. For many-to-one, just extract the last step in each layer. However, 
* for bidrectional, user should choose how to merge the last steps (one for the forward direction and the other 
* for the backward direction).
*
* This implementation allows to flexibly use different merge strategies (sum, concat, avg) and supports 
* different scenarios by capturing outputs at each step in each layer. The merging of outputs can be useful 
* for bidirectional RNNs, where one needs to merge the outputs from both directions in some way to obtain 
* a final output.
*****************************************************************************************************************/

template <class T>
const aitensor<T> RNN<T>::forward(const aitensor<T>& input_data) {
    const aitensor<T> Yhat = this->forwarding(input_data);
    this->setPrediction(Yhat);
    return Yhat;
}

template <class T>
const aitensor<T>  RNN<T>::backward(const aitensor<T>& gradients) {
    const aitensor<T> dInput = this->backprop(gradients);
    this->setGradients(dInput);
    return dInput;
}

template <class T>
const aitensor<T> LSTM<T>::forward(const aitensor<T>& input_data) {
    const aitensor<T> Yhat = this->forwarding(input_data);
    this->setPrediction(Yhat);
    return Yhat;
}

template <class T>
const aitensor<T> LSTM<T>::backward(const aitensor<T>& gradients) {
    const aitensor<T> dInput = this->backprop(gradients);
    this->setGradients(dInput);
    return dInput;
}

template <class T>
const aitensor<T> GRU<T>::forward(const aitensor<T>& input_data) {
    const aitensor<T> Yhat = this->forwarding(input_data);
    this->setPrediction(Yhat);
    return Yhat;
}

template <class T>
const aitensor<T>  GRU<T>::backward(const aitensor<T>& gradients) {
    const aitensor<T> dInput = this->backprop(gradients);
    this->setGradients(dInput);
    return dInput;
}
 
template <class T>
void RNN<T>::updateParameters(std::string& optimizertype, T& learningRate, int& iter) {
    // Learnable parameters already learnt inside the RNN cells
}

template <class T>
void LSTM<T>::updateParameters(std::string& optimizertype, T& learningRate, int& iter) {
    // Learnable parameters already learnt inside the LSTM cells
}

template <class T>
void GRU<T>::updateParameters(std::string& optimizertype, T& learningRate, int& iter) {
    // Learnable parameters already learnt inside the GRU cells
}


template <class T>
std::tuple<aimatrix<T>, airowvector<T>> RecurrentBase<T>::setInitialWeights(int step, aimatrix<T> out) {

    if (this->isInitialized()) {
        return std::make_tuple(this->get_V()[step], this->get_bo()[step]);
    }

    aimatrix<T> v(out.rows(), this->getOutputSize());
    airowvector<T> bo(out.cols());
    BaseOperator::heInitMatrix(v);
    bo.Zero(out.cols());
    return std::make_tuple(v, bo);
}

template <class T>
const aitensor<T> RecurrentBase<T>::processOutputs() {

    aimatrix<T> out, v, yhat;
    airowvector<T> bo;

    int sequence_length           = this->getSequenceLength();
    int input_size                = this->getInputSize();
    int embedding_size            = this->getEmbeddingSize();

    ReductionType rnntype         = this->getRType();
    aitensor<T> prediction;

    if (rnntype == ReductionType::CONCAT) {
        prediction(sequence_length, input_size, embedding_size * 2);
    } else {
        prediction(sequence_length, input_size, embedding_size);
    }

    for (int step = 0; step < this->getSequenceLength(); ++step) {

        // merge bidirectional outputs
        if (rnntype == ReductionType::SUM) {
            out = this->getFoutput()[step].array() + this->getBoutput()[step].array();
        } else 
        if (rnntype == ReductionType::AVG) {
            out = ( this->getFoutput()[step].array() + this->getBoutput()[step].array() ) / 2;
        } else 
        if (rnntype == ReductionType::CONCAT) {
            out << this->getFoutput()[step], this->getBoutput()[step];
        }

        // Initialize the weights for the output;
        std::tie(v, bo) = setInitialWeights(step, out);

        if (this->getOType() == ActivationType::SOFTMAX) {
            yhat = BaseOperator::softmax((aimatrix<T>) (out * v + bo));
        } else 
        if (this->getOType() == ActivationType::TANH) {
            yhat = BaseOperator::tanh((aimatrix<T>) (out * v + bo));
        } else
        if (this->getOType() == ActivationType::SIGMOID) {
            yhat = BaseOperator::sigmoid((aimatrix<T>) (out * v + bo));
        } else {
            this->setOutputSize( this->getHiddenSize());
            yhat = out; // no activation, pure Hidden State output
        }

        // rnn->outputs.chip(step, 0) = tensor_view(out);
        prediction.chip(step, 0) = tensor_view(yhat);

        if (!this->isInitialized()) {
            this->addV( v );
            this->addbo( bo );
        }
    }

    // this->setOutput(prediction);
    this->setPrediction(prediction);
    return prediction; // Cache output for backward pass while also pass it to next operations.
}

template <class T>
void RecurrentBase<T>::processGradients(aitensor<T>& gradients) {

    aimatrix<T> dOut, yhat;

    // Gradient will have dimension of: sequence_size, batch_size, output_size (instead of hidden_size)

    ActivationType       otype        = this->getOType();
    const aitensor<T>&   prediction   = this->getPrediction();

    for (int step = 0; step < this->getSequenceLength(); ++step) {

        // Process gradient for the activation operation
        dOut = matrix_view(chip(gradients, step, 0));  
        // out  = this->getOutput()[step];
        yhat = matrix_view(chip(prediction, step, 0));
        if (otype == ActivationType::SOFTMAX) {
            dOut = BaseOperator::softmaxGradient(dOut, yhat); // dsoftmaxV
            this->set_V(step, dOut * yhat.transpose());
            this->set_bo(step, dOut.colwise().sum());

        } else 
        if (otype == ActivationType::TANH) {
            dOut = BaseOperator::tanhGradient(dOut, yhat);      // dtanV
            this->set_V(step, dOut * yhat.transpose());
            this->set_bo(step, dOut.colwise().sum());;
        } else
        if (otype == ActivationType::SIGMOID) {
            dOut = BaseOperator::sigmoidGradient(dOut, yhat);  // dsigmoidV
            this->set_V(step, dOut * yhat.transpose());
            this->set_bo(step, dOut.colwise().sum());
        }

        gradients.chip(step, 0) = tensor_view((aimatrix<T>) (dOut * V[step].transpose())); // get dInput

    }
}

template <class T>
const aitensor<T> RecurrentBase<T>::forwarding(const aitensor<T>& input_data) {

    this->setData(input_data);
    this->setOType(ActivationType::SOFTMAX);
    this->setRType(ReductionType::AVG);

    int batch_size     = input_data.dimension(0);  // sequence
    int input_size     = input_data.dimension(1);
    int embedding_size = input_data.dimension(2);

    this->setSequenceLength(batch_size);
    this->setInputSize(input_size);
    this->setEmbeddingSize(embedding_size);

    // CellType celltype = this->getCellType();

    for (int direction = 0; direction < this->getNumDirections(); ++direction) {
        this->setCells(direction);

        // Forward pass: Run from first to last time step
        for (int step = 0; step < this->getSequenceLength(); ++step) {
            aimatrix<T> input_batch = matrix_view(chip(input_data, step, 0));


            // Forward pass through each layer of the RNN
            for (int layer = 0; layer < this->getNumLayers(); ++layer) {
                auto cell = this->getCells(); // std::dynamic_pointer_cast<RNNCell<T>>(this->getCells());
                input_batch = cell[layer]->forward(input_batch); // This is the Hidden State
            }

            if (direction == 0) {
                (this->getFoutput()).push_back(input_batch);
            } else {
                (this->getBoutput()).push_back(input_batch);
            }
        }
    }

    aitensor<T> Yhat = processOutputs();

    // marker to indicate the weights are all initialized, as they depend on these sizes.
    this->setInitialized();

    return Yhat;
}

template <class T>
const aitensor<T> RecurrentBase<T>::backprop(const aitensor<T>& gradients) {

    const aitensor<T>& input_data = this->getData();

    int batch_size      = input_data.dimension(0);  // sequence
    int input_size      = input_data.dimension(1);
    int embedding_size  = input_data.dimension(2);
    int sequence_length = batch_size;

    aitensor<T> dInput = gradients;

    processGradients(dInput);

    std::vector<aimatrix<T>> dOutf, dOutb;
    aimatrix<T> dOuts;

    ReductionType rtype = this->getRType();

    // Now, let us see if we need to split;
    // Process gradient for the reduction operation
    for (int step = sequence_length - 1; step >= 0; --step) {
        aimatrix<T> seq_f, seq_b, seq_m;
        if (rtype == ReductionType::SUM) {
            seq_f = matrix_view(chip(dInput, step, 0)); // gradients(step,0);
            seq_b = matrix_view(chip(dInput, step, 0)); // gradients(step,0);
        } else 
        if (rtype == ReductionType::AVG) {
            seq_f = matrix_view(chip(dInput, step, 0)) / 2; // gradients(step,0) / 2;
            seq_b = matrix_view(chip(dInput, step, 0)) / 2; // gradients(step,0) / 2;
        } else 
        if (rtype == ReductionType::CONCAT) {
            seq_m = matrix_view(chip(dInput, step, 0));
            seq_f = seq_m.block(0, 0, batch_size, embedding_size);
            seq_b = seq_m.block(0, embedding_size, batch_size, embedding_size);
        }
        dOutf.push_back(seq_f);
        dOutb.push_back(seq_b);
    }

    // We need to send back the same structure of input_gradients as the input to the forward pass
    aitensor<T> input_gradients(batch_size, input_size, embedding_size);

    for (int direction = 0; direction < this->getNumDirections(); ++direction) {
        this->setCells(direction);

        // Backward pass: Run from last to first time step
        for (int step = this->getSequenceLength() - 1; step >= 0; --step) {
            aimatrix<T> input_batch = matrix_view(chip(input_data, step, 0));  

            if (direction == 0) {
                dOuts = dOutf[step];
            } else {
                dOuts = dOutb[step];
            }

            // Backward pass through each layer of the RNN
            for (int layer = this->getNumLayers() - 1; layer >= 0; --layer) {
                auto cells = this->getCells();
                dOuts = cells[layer]->backward(input_batch, dOuts);
                input_batch = cells[layer]->getHiddenState();
            }

            // Store the gradients for input data
            input_gradients.chip(step, 0) = tensor_view(dOuts);
        }
    }
    return input_gradients;
}

/**********  Recurrent Network initialize templates *****************/

template class CellBase<float>;  // Instantiate with float
template class CellBase<double>;  // Instantiate with double

template class RNNCell<float>;  // Instantiate with float
template class RNNCell<double>;  // Instantiate with double

template class LSTMCell<float>;  // Instantiate with float
template class LSTMCell<double>;  // Instantiate with double

template class GRUCell<float>;  // Instantiate with float
template class GRUCell<double>;  // Instantiate with double

template class RecurrentBase<float>;  // Instantiate with float
template class RecurrentBase<double>;  // Instantiate with double

template class RNN<float>;  // Instantiate with float
template class RNN<double>;  // Instantiate with double

template class LSTM<float>;  // Instantiate with float
template class LSTM<double>;  // Instantiate with double

template class GRU<float>;  // Instantiate with float
template class GRU<double>;  // Instantiate with double


