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
RNNCell::RNNCell(int hidden_size, double learning_rate) 
    : hidden_size(hidden_size), learning_rate(learning_rate) {
    learning_rate = 0.01;
}

const aimatrix& RNNCell::getHiddenState() {
    return H;
}

void RNNCell::setInitialWeights(int N, int P) {
    // Initialize parameters, gates, states, etc.
    if (this->input_size !=0 || this->param_size != 0) return;

    this->input_size = N;  
    this->param_size = P;  

    W.resize(param_size, hidden_size);
    U.resize(hidden_size, hidden_size);

    BaseOperator::heInitialization(W);
    BaseOperator::heInitialization(U);

    bh = airowvector::Zero(hidden_size);

    H    = aimatrix::Zero(input_size, hidden_size);
}

const aimatrix& RNNCell::forward(const aimatrix& input_data) {

    // Store input for backward pass.
    // this can be the prev_hidden_state
    // this->X = input_data;
    const aimatrix& X = input_data;

    setInitialWeights(input_data.rows(), input_data.cols());

    // Compute hidden state.
    //     (nxh) =  (nxp) * (pxh) + (nxh) * (hxh) + h
    H = BaseOperator::tanh(X * W + H * U + bh);

    // Compute Yhat.
    //     (nxo) =  (nxh) * (hxo) + h
    // Yhat = BaseOperator::softmax(H * V + bo);

    // We return the Hidden State as output;
    return H; 
}

const aimatrix& RNNCell::backward(const aimatrix& input_data, const aimatrix& dnext_h) {
    // Backpropagation logic for RNN

    const aimatrix& X = input_data;

    // Compute gradient with respect to tanh
    aimatrix dtanh = (1.0 - H.array().square()).array() * dnext_h.array();

    // Compute gradient with respect to hidden-to-hidden weights Whh.
    aimatrix dU = H.transpose() * dtanh;

    // Compute gradient with respect to input-to-hidden weights Wxh.
    aimatrix dW = X.transpose() * dtanh;

    // Compute gradient with respect to hidden-state.
    dH = dtanh * U.transpose();

    // Compute gradient with respect to input (dInput).
    dX = dtanh * W.transpose();

    // Compute gradient with respect to hidden bias bh.
    airowvector dbh = dtanh.colwise().sum(); 

    // Compute gradient with respect to output bias bo.
    //airowvector dbo = Y.colwise().sum();

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
LSTMCell::LSTMCell(int hidden_size, double learning_rate) 
    : hidden_size(hidden_size), learning_rate(learning_rate) {
    learning_rate = 0.01;
}

const aimatrix& LSTMCell::getHiddenState() {
    return H;
}

const aimatrix& LSTMCell::getCellState() {
    return C;
}

void LSTMCell::setInitialWeights(int N, int P) {
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

    BaseOperator::heInitialization(Wf);
    BaseOperator::heInitialization(Wi);
    BaseOperator::heInitialization(Wo);
    BaseOperator::heInitialization(Wg);

    bf = airowvector::Zero(hidden_size);
    bi = airowvector::Zero(hidden_size);
    bo = airowvector::Zero(hidden_size);
    bg = airowvector::Zero(hidden_size);

    H    = aimatrix::Zero(input_size, hidden_size);
    C    = aimatrix::Zero(input_size, hidden_size);

    XH   = aimatrix::Zero(input_size, param_size + hidden_size); // concatenate X and H

}

const aimatrix& LSTMCell::forward(const aimatrix& input_data) {

    // Store input for backward pass.
    // this->X = input_data;
    const aimatrix& X = input_data;

    setInitialWeights(input_data.rows(), input_data.cols());

    // Concatenate input and hidden state.
    XH << X, H;

    // Calculate the forget gate values
    Ft = BaseOperator::sigmoid(XH * Wf + bf);

    // Calculate the input gate values
    It = BaseOperator::sigmoid(XH * Wi + bi);

    // Calculate the output gate values
    Ot = BaseOperator::sigmoid(XH * Wo + bo);

    // Calculate the candidate state values
    Gt = BaseOperator::tanh(XH * Wg + bg);

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

const aimatrix& LSTMCell::backward(const aimatrix& input_data, const aimatrix& dnext_h) {
    // Backpropagation logic for LSTM

    const aimatrix& X = input_data;

    // Compute gradients with respect to the gates
    dC = (Ot.array() * (1 - BaseOperator::tanh(C.array()).array().square()) * dnext_h.array()) + dC.array();
    aimatrix dFt = dC.array() * C.array() * Ft.array() * (1 - Ft.array());
    aimatrix dIt = dC.array() * Gt.array() * It.array() * (1 - It.array());
    aimatrix dGt = dC.array() * It.array().array() * (1 - Gt.array().square().array());
    aimatrix dOt = dnext_h.array() * BaseOperator::tanh(C).array() * Ot.array() * (1 - Ot.array());

    // Concatenate input and hidden state.
    XH << X, H;

    // Compute gradients with respect to hidden-to-hidden weights Wf, Wi, Wo, Wc
    aimatrix dWf = XH.transpose() * dFt;
    aimatrix dWi = XH.transpose() * dIt;
    aimatrix dWg = XH.transpose() * dGt;
    aimatrix dWo = XH.transpose() * dOt;


    // Compute gradients with respect to hidden biases bf, bi, bo, bc

    airowvector dbf = dFt.colwise().sum();
    airowvector dbi = dIt.colwise().sum();
    airowvector dbo = dOt.colwise().sum();
    airowvector dbg = dGt.colwise().sum();

    // Compute gradient with respect to the cell state C (dC)
    dC = dC.array() * Ft.array();

    // Compute gradient with respect to hidden state H (dH)
    int ps = param_size, hs = hidden_size;
    dH  = dFt * Split(Wf, ps, hs).transposedX() 
        + dIt * Split(Wi, ps, hs).transposedX()  
        + dOt * Split(Wo, ps, hs).transposedX() 
        + dGt * Split(Wg, ps, hs).transposedX();

    // Compute gradient with respect to input (dInput)
    dX  = dFt * Split(Wf, ps, hs).transposedH() 
        + dIt * Split(Wi, ps, hs).transposedH()  
        + dOt * Split(Wo, ps, hs).transposedH()
        + dGt * Split(Wg, ps, hs).transposedH();

    // Compute gradient with respect to output bias bo.
    // airowvector dby = Y.colwise().sum();

    // Update parameters and stored states
    Wf -= learning_rate * dWf;
    Wi -= learning_rate * dWi;
    Wo -= learning_rate * dWo;
    Wg -= learning_rate * dWg;
    bf -= learning_rate * dbf;
    bi -= learning_rate * dbi;
    bo -= learning_rate * dbo;
    bg -= learning_rate * dbg;
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
GRUCell::GRUCell(int hidden_size, double learning_rate) 
    : hidden_size(hidden_size), learning_rate(learning_rate) {
    learning_rate = 0.01;
}

const aimatrix& GRUCell::getHiddenState() {
    return H;
}

void GRUCell::setInitialWeights(int N, int P) {
    // Initialize parameters, gates, states, etc.
    if (this->input_size !=0 || this->param_size != 0) return;

    this->input_size = N;  
    this->param_size = P;  

    // Initialize parameters (weights and biases)
    // Note: Initialize these parameters according to your initialization strategy
    Wz.resize(param_size + hidden_size, hidden_size);
    Wr.resize(param_size + hidden_size, hidden_size);
    Wg.resize(param_size + hidden_size, hidden_size);

    BaseOperator::heInitialization(Wz);
    BaseOperator::heInitialization(Wr);
    BaseOperator::heInitialization(Wg);

    bz = airowvector::Zero(hidden_size);
    br = airowvector::Zero(hidden_size);
    bg = airowvector::Zero(hidden_size);

    H    = aimatrix::Zero(input_size, hidden_size);

    XH   = aimatrix::Zero(input_size, param_size + hidden_size); // concatenate X and H

}

const aimatrix& GRUCell::forward(const aimatrix& input_data) {

    // Store input for backward pass.
    // this can be the prev_hidden_state
    // X = input_data;
    const aimatrix& X = input_data;

    setInitialWeights(input_data.rows(), input_data.cols());

    // Concatenate input and hidden state.
    XH << X, H;

    // Calculate the update gate using input, hidden state, and biases
    Zt =  BaseOperator::sigmoid(XH * Wz + bz);

    // Calculate the reset gate using input, hidden state, and biases
    Rt =  BaseOperator::sigmoid(XH * Wr + br);

    // Calculate the candidate hidden state using input, reset gate, hidden state, and biases
    aimatrix RH = Rt.array() * H.array();
    aimatrix rXH(input_size, hidden_size + hidden_size);
    rXH << X, RH;
    Rt = BaseOperator::tanh(XH * Wg + bg);

    // Calculate Ghat


    // Calculate the new hidden state using update gate, previous hidden state, and candidate hidden state
    H = (1 - Zt.array()).array() * Gt.array() + Zt.array() * H.array(); 

    // Compute Yhat.
    //     (nxo) =  (nxh) * (hxo) + h
    // Yhat = BaseOperator::softmax(H * V + bo);

    // We return the Hidden State as output;
    return H;
}

const aimatrix& GRUCell::backward(const aimatrix& input_data, const aimatrix& dnext_h) {
    // Backpropagation logic for GRU

    const aimatrix& X = input_data;

    // Compute gradients with respect to hidden-to-hidden weights Wz, Wr, Wg

    int ps = param_size, hs = hidden_size;

    // Compute gradients with respect to the gates
    aimatrix dZt = dnext_h.array() * ( H.array() - Gt.array()) * Zt.array() * ( 1 - Zt.array());
    aimatrix dGt = dnext_h.array() * ( 1 - Zt.array()) * ( 1 - Gt.array().square());
    aimatrix dRt = ( dGt.array() * (H * Split(Wr, ps, hs).transposedH()).array() 
                          * ( 1 - Rt.array().square()) )
                          * Rt.array() * ( 1 - Rt.array());

    // Concatenate input and hidden state.
    XH << X, H;

    // Compute gradients with respect to hidden-to-hidden weights Wz, Wr, Wg
    aimatrix dWz = XH.transpose() * dZt;
    aimatrix dWr = XH.transpose() * dRt;
    aimatrix dWg = XH.transpose() * dGt;

    // Compute gradients with respect to hidden biases bz, br, bg
    airowvector dbz = dZt.colwise().sum();
    airowvector dbr = dGt.colwise().sum();
    airowvector dbg = dRt.colwise().sum();

    // Compute gradient with respect to hidden state H (dH)

    dH  = dZt * Split(Wz, ps, hs).transposedX() 
        + dRt * Split(Wr, ps, hs).transposedX()  
        + dGt * Split(Wg, ps, hs).transposedX();

    // Compute gradient with respect to input (dInput)
    dX  = dZt * Split(Wz, ps, hs).transposedH() 
        + dRt * Split(Wr, ps, hs).transposedH()  
        + dGt * Split(Wg, ps, hs).transposedH();

    // Compute gradient with respect to output bias bo.
    // airowvector dbo = Y.colwise().sum();

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

const aitensor&  RNN::forward(const aitensor& input_data) {
    this->Yhat = forwarding(input_data, this);
    return this->Yhat;
}

const aitensor&  RNN::backward(const aitensor& gradients) {
    this->gradients = backprop(gradients, this);
    return this->gradients;
}

const aitensor&  LSTM::forward(const aitensor& input_data) {
    this->Yhat = forwarding(input_data, this);
    return this->Yhat;
}

const aitensor&  LSTM::backward(const aitensor& gradients) {
    this->gradients = backprop(gradients, this);
    return this->gradients;
}

const aitensor&  GRU::forward(const aitensor& input_data) {
    this->Yhat = forwarding(input_data, this);
    return this->Yhat;
}

const aitensor&  GRU::backward(const aitensor& gradients) {
    this->gradients = backprop(gradients, this);
    return this->gradients;
}

void RNN::updateParameters(std::string& optimizertype, double& learningRate, int& iter) {
    // Learnable parameters already learnt inside the RNN cells
}

void LSTM::updateParameters(std::string& optimizertype, double& learningRate, int& iter) {
    // Learnable parameters already learnt inside the LSTM cells
}

void GRU::updateParameters(std::string& optimizertype, double& learningRate, int& iter) {
    // Learnable parameters already learnt inside the GRU cells
}


template <typename CellType>
std::tuple<aimatrix, airowvector> setInitialWeights(int step, aimatrix out, CellType& rnn) {

    if (rnn->iniitalized) {
        return std::make_tuple(rnn->V[step], rnn->bo[step]);
    }

    parameters.weights.resize(M, this->W); // allocates memory
    parameters.biases.resize(this->W); // allocates memory

    aimatrix v(out.rows(), rnn->output_size);
    airowvector bo(out.cols());
    BaseOperator::heInitialization(v);
    bo.Zero(out.cols());
    return std::make_tuple(v, bo);
}

template <typename CellType>
const std::vector<aimatrix>& processOutputs(CellType& rnn) {

    const aimatrix& out, v, yhat;
    const airowvector bo;

    for (int step = 0; step < rnn->sequence_length; ++step) {

        // merge bidirectional outputs
        if (rnn->rtype == ReductionType::SUM) {
            out = rnn->foutput[step].array() + rnn->boutput[step].array();
        } else 
        if (rnn->rtype == ReductionType::AVG) {
            out = ( rnn->foutput[step].array() + rnn->boutput[step].array() ) / 2;
        } else 
        if (rnn->rtype == ReductionType::CONCAT) {
            out << rnn->foutput[step], rnn->boutput[step];
        }

        // Initialize the weights for the output;
        std::tie(v, bo) = setInitialWeights(step, out, rnn);

        if (rnn->otype == ActivationType::SOFTMAX) {
            yhat = BaseOperator::softmax(out * v + bo);
        } else 
        if (rnn->otype == ActivationType::TANH) {
            yhat = BaseOperator::tanh(out * v + bo);
        } else
        if (rnn->otype == ActivationType::SIGMOID) {
            yhat = BaseOperator::sigmoid(out * v + bo);
        } else {
            rnn->output_size = rnn->hidden_size;
            yhat = out; // no activation, pure Hidden State output
        }

        rnn->outputs(step, 0) = out;
        rnn->Yhat(step, 0) = yhat;

        if (!rnn->initialized) {
            rnn->V.push_back( v );
            rnn->bo.push_back( bo );
        }
    }

    return rnn->Yhat; // Cache output for backward pass while also pass it to next operations.
}

template <typename CellType>
void processGradients(const aitensor& gradients, CellType& rnn) {

    const aimatrix& dOut, out, yhat;

    // Gradient will have dimension of: sequence_size, batch_size, output_size (instead of hidden_size)

    for (int step = 0; step < rnn->sequence_length; ++step) {

        // Process gradient for the activation operation
        dOut = gradients[step];
        out  = rnn->outputs[step];
        yhat = rnn->Yhat(step, 0);
        if (rnn->otype == ActivationType::SOFTMAX) {
            dOut = BaseOperator::softmaxGradient(dOut, yhat); // dsoftmaxV
            rnn->dV[step] = dOut * out.transpose();
            rnn->dbo[step] = dOut.colwise().sum;

        } else 
        if (rnn->otype == ActivationType::TANH) {
            dOut = BaseOperator::tanhGradient(dOut, yhat);      // dtanV
            rnn->dV[step] = dOut * out.transpose();
            rnn->dbo[step] = dOut.colwise().sum;
        } else
        if (rnn->otype == ActivationType::SIGMOID) {
            dOut = BaseOperator::sigmoidGradient(dOut, yhat);  // dsigmoidV
            rnn->dV[step] = dOut * out.transpose();
            rnn->dbo[step] = dOut.colwise().sum;
        }

        gradients[step] = dOut * rnn->V[step].transpose(); // get dInput

    }
}

template <typename CellType>
const aitensor& forwarding(const aitensor& input_data, CellType& rnn) {

    rnn->input_data = input_data;
    rnn->otype = ActivationType::SOFTMAX;
    rnn->rtype = ReductionType::AVG;

    // int batch_size = input_data.dimension(0);  
    // int embedding_size = input_data.dimension(1);
    // rnn->sequence_length = input_data.dimension(2);

    int batch_size     = input_data.dimension(0);  // sequence
    int input_size     = input_data.dimension(1);
    int embedding_size = input_data.dimension(2);
    rnn->sequence_length = batch_size;

    for (int direction = 0; direction < rnn->num_directions; ++direction) {
        rnn->cells  = (direction == 0) ? rnn->fcells : rnn->bcells;

        // Forward pass: Run from first to last time step
        for (int step = 0; step < rnn->sequence_length; ++step) {
            aimatrix input_batch = input_data.chip(step, 0);

            // Forward pass through each layer of the RNN
            for (int layer = 0; layer < rnn->num_layers; ++layer) {
                input_batch = rnn->cells[layer].forward(input_batch); // This is the Hidden State
            }

            if (direction == 0) {
                rnn->foutput.push_back(input_batch);
            } else {
                rnn->boutput.push_back(input_batch);
            }
        }
    }

    const aitensor& Yhat = processOutputs(rnn);

    // marker to indicate the weights are all initialized, as they depend on these sizes.
    rnn->initialized = true;

    return Yhat;
}

template <typename CellType>
const aitensor& backprop(const aitensor& gradients, CellType& rnn) {

    const aitensor& input_data = rnn->input_data;

    // int batch_size = input_data.dimension(0);
    // int embedding_size = input_data.dimension(1);
    // int sequence_length = input_data.dimension(2);

    int batch_size      = input_data.dimension(0);  // sequence
    int input_size      = input_data.dimension(1);
    int embedding_size  = input_data.dimension(2);
    int sequence_length = batch_size;

    processGradients(gradients, rnn);

    std::vector<aimatrix> dOutf, dOutb;
    aimatrix dOuts;

    // Now, let us see if we need to split;
    // Process gradient for the reduction operation
    for (int step = sequence_length - 1; step >= 0; --step) {
        aimatrix seq_f, seq_b;
        if (rnn->rtype == ReductionType::SUM) {
            seq_f = gradients(step,0);
            seq_b = gradients(step,0);
        } else 
        if (rnn->rtype == ReductionType::AVG) {
            seq_f = gradients(step,0) / 2;
            seq_b = gradients(step,0) / 2;
        } else 
        if (rnn->rtype == ReductionType::CONCAT) {
            seq_f = gradients(step,0).block(0, 0, batch_size, embedding_size);
            seq_b = gradients(step,0).block(0, embedding_size, batch_size, embedding_size);
        }
        dOutf.push_back(seq_f);
        dOutb.push_back(seq_b);
    }

    // We need to send back the same structure of input_gradients as the input to the forward pass
    aitensor input_gradients(batch_size, input_size, embedding_size);

    for (int direction = 0; direction < rnn->num_directions; ++direction) {
        rnn->cells  = (direction == 0) ? rnn->fcells : rnn->bcells;

        // Backward pass: Run from last to first time step
        for (int step = sequence_length - 1; step >= 0; --step) {
            aimatrix input_batch = input_data.chip(step, 0);

            if (direction == 0) {
                dOuts = dOutf[step];
            } else {
                dOuts = dOutb[step];
            }

            // Backward pass through each layer of the RNN
            for (int layer = rnn->num_layers - 1; layer >= 0; --layer) {
                dOuts = rnn->cells[layer].backward(input_batch, dOuts);
                input_batch = rnn->cells[layer].getHiddenState();
            }

            // Store the gradients for input data
            input_gradients.chip(step, 0) = dOuts;
        }
    }
    return input_gradients;
}



