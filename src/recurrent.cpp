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
void RNNCell<T>::setInitialWeights(int N, int P) {
    // Initialize parameters, gates, states, etc.
    if (this->input_size !=0 || this->param_size != 0) return;

    this->input_size = N;  
    this->param_size = P;  

    W.resize(this->param_size, this->hidden_size);
    U.resize(this->hidden_size, this->hidden_size);
    bh.resize(this->hidden_size);

    BaseOperator::heInitMatrix(W);
    BaseOperator::heInitMatrix(U);

    bh.setConstant(T(0.00));

    aimatrix<T> H  = aimatrix<T>::Zero(this->input_size, this->hidden_size);
    this->H.push_back(H);
}

template <class T>
const aimatrix<T> RNNCell<T>::forward(const aimatrix<T>& X) {

    log_detail("===============================================");
    log_detail("RNNCell Forward Pass ...");

    log_detail("Size: {0}", X.size());

    this->X.push_back(X);

    setInitialWeights(X.rows(), X.cols());

    log_detail("Dimension of X: {0}x{1}", X.rows(), X.cols());
    log_matrix(X);
    log_detail("Dimension of U: {0}x{1}", U.rows(), U.cols());
    log_matrix(U);
    log_detail("Dimension of W: {0}x{1}", W.rows(), W.cols());
    log_matrix(W);

    // Compute hidden state.
    //     (nxh) =  (nxp) * (pxh) + (nxh) * (hxh) + h
    // H = BaseOperator::tanh((aimatrix<T>) (X * W + H * U + bh) );
    aimatrix<T> H = this->H.back();  // previous time
    aimatrix<T> new_H = (aimatrix<T>)  BaseOperator::tanh((aimatrix<T>) ((BaseOperator::matmul(X, W) + BaseOperator::matmul(H, U)).rowwise() + bh ));

    log_detail("BaseOperator::matmul(X, W)");
    log_matrix(BaseOperator::matmul(X, W));

    log_detail("BaseOperator::matmul(H, U)");
    log_matrix(BaseOperator::matmul(H, U));

    log_detail("(BaseOperator::matmul(X, W) + BaseOperator::matmul(H, U)).rowwise() ");
    log_matrix((aimatrix<T>) (BaseOperator::matmul(X, W) + BaseOperator::matmul(H, U)) );

    log_detail("bh");
    log_rowvector(bh);  

    this->H.push_back(new_H);

    log_detail("New H (tanh of X * W + H * U)");
    log_matrix(new_H);

    log_info("RNNCell Forward Pass end ...");

    // We return Output next layer;
    return new_H;
}

template <class T>
const aimatrix<T> RNNCell<T>::backward(int step, const aimatrix<T>& dnext_h) {
    // Backpropagation logic for RNN
    log_detail("===============================================");
    log_detail("RNNCell Backward Pass ...");

    log_detail("Dimension of dnext_h: {0}x{1}", dnext_h.rows(), dnext_h.cols());
    log_matrix(dnext_h);

    aimatrix<T> H = this->H.at(step); // Hidden state at t-1
    aimatrix<T> X = this->X.at(step); // Input at t

    log_detail("Dimension of X: {0}x{1}", X.rows(), X.cols());
    log_matrix(X);
    log_detail("Dimension of H: {0}x{1}", H.rows(), H.cols());
    log_matrix(H);
    log_detail("Dimension of U: {0}x{1}", U.rows(), U.cols());
    log_matrix(U);
    log_detail("Dimension of W: {0}x{1}", W.rows(), W.cols());
    log_matrix(W);

    // Compute gradient with respect to tanh output
    aimatrix<T> dtanh = BaseOperator::tanhGradient(dnext_h, H); // (1.0 - H.array().square()).array() * dnext_h.array();
    log_detail("Dimension of dtanh: {0}x{1}", dtanh.rows(), dtanh.cols());
    log_matrix(dtanh);

    // Next operations is based on tanh( X * W + H * U + bh)

    // Compute gradient with respect to input (X)(dInput) from tanh perspective
    log_detail("Calculating gradient with respect to X (dX) ...");
    aimatrix<T> dX = BaseOperator::matmul(dtanh, W.transpose());
    log_matrix(this->dX);

    // Compute gradient with respect to W (dW) from tanh perspective
    log_detail("Calculating gradient with respect to W (dW) ...");
    this->dW = BaseOperator::matmul(X.transpose(), dtanh);
    log_matrix(this->dW);

    // Compute gradient with respect to H (dH) from tanh perspective
    log_detail("Calculating gradient with respect to H (dH) ...");
    aimatrix<T> dH = BaseOperator::matmul(dtanh, U.transpose());
    log_matrix(dH);

    // Compute gradient with respect to H (dH) from tanh perspective
    log_detail("Calculating gradient with respect to X (dX) ...");
    this->dU = BaseOperator::matmul(H.transpose(), dtanh);
    log_matrix(this->dU);

    // Compute gradient with respect to hidden bias bh.
    log_detail("Calculating dbh ...");
    this->dbh = dtanh.colwise().sum(); 
    log_rowvector(this->dbh);

    this->dH.push_back(dH);
    this->dX.push_back(dX);

    log_info("RNNCell Backward Pass end ...");

    // Return gradient with respect to input.
    // We return the gradient wrt X for each layer, not wrt H for each time step.
    // for the gradient wrt H, we just cache it for time step propagation, but layer propagation.
    return dH;
}

template <class T>
void RNNCell<T>::updateParameters(std::string& optimizertype, T& learningRate, int& iter) {
    // Update parameters
    if (opt_W == nullptr) {
        opt_W  = new Optimizer<T>(optimizertype, learningRate);
        opt_U  = new Optimizer<T>(optimizertype, learningRate);
        opt_bh = new Optimizer<T>(optimizertype, learningRate);
    }
    log_detail("Updating W in RNN Cell ...");
    opt_W->update(optimizertype, this->W, this->dW, iter);
    log_detail("Updating U in RNN Cell ...");
    opt_U->update(optimizertype, this->U, this->dU, iter);
    log_detail("Updating bh in RNN Cell ...");
    opt_bh->update(optimizertype, this->bh, this->dbh, iter);
    this->H.clear();
    this->X.clear();
    this->dH.clear();
    this->dX.clear();
    this->input_size = 0;
    this->param_size = 0;
}
 
/***************************************************************************************************************************
*********** IMPLEMENTING LSTMCell
****************************************************************************************************************************/
template <class T>
void LSTMCell<T>::setInitialWeights(int N, int P) {
    // Initialize parameters, gates, states, etc.
    if (this->input_size !=0 || this->param_size != 0) return;

    this->input_size = N;  
    this->param_size = P;  

    // Initialize parameters (weights and biases)
    // Note: Initialize these parameters according to your initialization strategy
    Wf.resize(this->param_size + this->hidden_size, this->hidden_size);
    Wi.resize(this->param_size + this->hidden_size, this->hidden_size);
    Wg.resize(this->param_size + this->hidden_size, this->hidden_size);
    Wo.resize(this->param_size + this->hidden_size, this->hidden_size);
    bf.resize(this->hidden_size);
    bi.resize(this->hidden_size);
    bg.resize(this->hidden_size);
    bo.resize(this->hidden_size);

    BaseOperator::heInitMatrix(Wf);
    BaseOperator::heInitMatrix(Wi);
    BaseOperator::heInitMatrix(Wg);
    BaseOperator::heInitMatrix(Wo);

    bf.setConstant(T(0.01));
    bi.setConstant(T(0.01));
    bg.setConstant(T(0.01));
    bo.setConstant(T(0.01));


    aimatrix<T> H  = aimatrix<T>::Zero(this->input_size, this->hidden_size);
    aimatrix<T> C  = aimatrix<T>::Zero(this->input_size, this->hidden_size);
    this->H.push_back(H);
    this->C.push_back(C);

    XH   = aimatrix<T>::Zero(this->input_size, this->param_size + this->hidden_size); // concatenate X and H

}

template <class T>
const aimatrix<T> LSTMCell<T>::forward(const aimatrix<T>& X) {

    log_detail("===============================================");
    log_detail("LSTMCell Forward Pass ...");

    this->X.push_back(X);

    setInitialWeights(X.rows(), X.cols());

    aimatrix<T> H = this->H.back();
    aimatrix<T> C = this->C.back();

    // Concatenate input and hidden state. 
    XH << X, H;  // In terms of dimension, we have: (NxP) + (NxH) = Nx(P+H)

    // Calculate the forget gate values (nx[p+h] * [p+h]xh = nxh)
    this->Ft = BaseOperator::sigmoid((aimatrix<T>) (BaseOperator::matmul(XH, Wf).rowwise() + bf));

    // Calculate the input gate values (nx[p+h] * [p+h]xh = nxh)
    this->It = BaseOperator::sigmoid((aimatrix<T>) (BaseOperator::matmul(XH, Wi).rowwise() + bi));

    // Calculate the output gate values (nx[p+h] * [p+h]xh = nxh)
    this->Ot = BaseOperator::sigmoid((aimatrix<T>) (BaseOperator::matmul(XH, Wo).rowwise() + bo));

    // Calculate the candidate state values (nx[p+h] * [p+h]xh = nxh)
    this->Gt = BaseOperator::tanh((aimatrix<T>) (BaseOperator::matmul(XH, Wg).rowwise() + bg));

    // Calculate the new cell state by updating based on the gates (result: nxh)
    aimatrix<T> new_C = Ft.array() * C.array() + It.array() * Gt.array();
    this->C.push_back(new_C);

    // Calculate the new hidden state by applying the output gate and tanh activation
    aimatrix<T> new_H = BaseOperator::tanh(new_C).array() * Ot.array();
    this->H.push_back(new_H);

    log_info("LSTMCell Forward Pass end ...");

    // We return Output next layer;
    return new_H; 

}

template <class T>
const std::tuple<aimatrix<T>, aimatrix<T>> LSTMCell<T>::backward(int step, const aimatrix<T>& dnext_h, const aimatrix<T>& dnext_c) {
    // Backpropagation logic for LSTM

    log_detail("===============================================");
    log_detail("LSTMCell Backward Pass ...");

    log_detail("Dimension of dnext_h: {0}x{1}", dnext_h.rows(), dnext_h.cols());
    log_matrix(dnext_h);

    log_detail("Dimension of dnext_c: {0}x{1}", dnext_c.rows(), dnext_c.cols());
    log_matrix(dnext_c);

    log_detail("Capture Hidden state at step {0}", step);
    aimatrix<T> H = this->H.at(step); // Hidden state at t-1
    log_detail("Capture Cell state at step {0}", step);
    aimatrix<T> C = this->C.at(step); // Cell state at t-1
    log_detail("Capture Input at step {0}", step);
    aimatrix<T> X = this->X.at(step); // Input at t


    log_detail("Dimension of H: {0}x{1}", H.rows(), H.cols());
    log_matrix(H);
    log_detail("Dimension of C: {0}x{1}", C.rows(), C.cols());
    log_matrix(C);
    log_detail("Dimension of Ft: {0}x{1}", Ft.rows(), Ft.cols());
    log_matrix(Ft); 
    log_detail("Dimension of Ft: {0}x{1}", It.rows(), It.cols());
    log_matrix(It); 
    log_detail("Dimension of Ft: {0}x{1}", Gt.rows(), Gt.cols());
    log_matrix(Gt); 
    log_detail("Dimension of Ft: {0}x{1}", Ot.rows(), Ot.cols());
    log_matrix(Ot); 

    // Gradient with respect to C
    aimatrix<T> tanh  = BaseOperator::tanh(C);
    aimatrix<T> dtanh = BaseOperator::tanhGradient(dnext_h, tanh);
    aimatrix<T> dC    = Ot.array() * dtanh.array() + dnext_c.array();

    // Gradient with respect to O
    aimatrix<T> dOt    = dnext_h.array() * tanh.array() * Ot.array() * ( 1 - Ot.array());

    // Gradient with respect to F
    aimatrix<T> dFt = dC.array() * C.array() * Ft.array() * (1 - Ft.array());

    // Gradient with respect to I
    aimatrix<T> dIt = dC.array() * Gt.array() * It.array() * (1 - It.array());

    // Gradient with respect to G
    aimatrix<T> dGt = dC.array() * It.array() * (1 - Gt.array().square());

    // Concatenate input and hidden state.
    XH << X, H;

    // Compute gradients with respect to hidden-to-hidden weights Wf, Wi, Wo, Wc
    this->dWf = BaseOperator::matmul(XH.transpose(), dFt);
    this->dWi = BaseOperator::matmul(XH.transpose(), dIt);
    this->dWg = BaseOperator::matmul(XH.transpose(), dGt);
    this->dWo = BaseOperator::matmul(XH.transpose(), dOt);

    // Compute gradients with respect to hidden biases bf, bi, bo, bc

    this->dbf = dFt.colwise().sum();
    this->dbi = dIt.colwise().sum();
    this->dbg = dGt.colwise().sum();
    this->dbo = dOt.colwise().sum();

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
    aimatrix<T> dH  = BaseOperator::matmul(dFt, Wfx) + BaseOperator::matmul(dIt, Wix) + 
                      BaseOperator::matmul(dOt, Wox) + BaseOperator::matmul(dGt, Wgx);

    // Compute gradient with respect to input (dInput).
    aimatrix<T> dX  = BaseOperator::matmul(dFt, Wfh) + BaseOperator::matmul(dIt, Wih) + 
                      BaseOperator::matmul(dOt, Woh) + BaseOperator::matmul(dGt, Wgh);

    this->dH.push_back(dH);
    this->dX.push_back(dX);

    log_info("LSTMCell Backward Pass end ...");

    // Return gradient with respect to input.
    // We return the gradient wrt X for each layer, not wrt H & C for each time step.
    // for the gradient wrt H & C, we just cache it for time step propagation, but layer propagation.
    return std::make_tuple(dX, dH);
}

template <class T>
void LSTMCell<T>::updateParameters(std::string& optimizertype, T& learningRate, int& iter) {
    // Update parameters
    if (opt_Ft == nullptr) {
        opt_Ft = new Optimizer<T>(optimizertype, learningRate);
        opt_It = new Optimizer<T>(optimizertype, learningRate);
        opt_Gt = new Optimizer<T>(optimizertype, learningRate);
        opt_Ot = new Optimizer<T>(optimizertype, learningRate);
        opt_bf = new Optimizer<T>(optimizertype, learningRate);
        opt_bi = new Optimizer<T>(optimizertype, learningRate);
        opt_bg = new Optimizer<T>(optimizertype, learningRate);
        opt_bo = new Optimizer<T>(optimizertype, learningRate);
    }
    log_detail("Updating Ft in LSTM Cell ...");
    opt_Ft->update(optimizertype, this->Ft, this->dWf, iter);
    log_detail("Updating It in LSTM Cell ...");
    opt_It->update(optimizertype, this->It, this->dWi, iter);
    log_detail("Updating Gt in LSTM Cell ...");
    opt_Gt->update(optimizertype, this->Gt, this->dWg, iter);
    log_detail("Updating Ot in LSTM Cell ...");
    opt_Ot->update(optimizertype, this->Ot, this->dWo, iter);
    log_detail("Updating bf in LSTM Cell ...");
    opt_bf->update(optimizertype, this->bf, this->dbf, iter);
    log_detail("Updating bi in LSTM Cell ...");
    opt_bi->update(optimizertype, this->bi, this->dbi, iter);
    log_detail("Updating bg in LSTM Cell ...");
    opt_bg->update(optimizertype, this->bg, this->dbg, iter);
    log_detail("Updating bo in LSTM Cell ...");
    opt_bo->update(optimizertype, this->bo, this->dbo, iter);
    this->H.clear();
    this->C.clear();
    this->X.clear();
    this->dH.clear();
    this->dC.clear();
    this->dX.clear();
    this->input_size = 0;
    this->param_size = 0;
}
 
/***************************************************************************************************************************
 *********** IMPLEMENTING GRUCell
****************************************************************************************************************************/
template <class T>
void GRUCell<T>::setInitialWeights(int N, int P) {
    // Initialize parameters, gates, states, etc.
    if (this->input_size !=0 || this->param_size != 0) return;
 
    this->input_size = N;  
    this->param_size = P;  

    // Initialize parameters (weights and biases)
    // Note: Initialize these parameters according to your initialization strategy
    Wz.resize(this->param_size + this->hidden_size, this->hidden_size);
    Wr.resize(this->param_size + this->hidden_size, this->hidden_size);
    Wg.resize(this->param_size + this->hidden_size, this->hidden_size);
    bz.resize(this->hidden_size);
    br.resize(this->hidden_size);
    bg.resize(this->hidden_size);
    BaseOperator::heInitMatrix(Wz);
    BaseOperator::heInitMatrix(Wr);
    BaseOperator::heInitMatrix(Wg);

    bz.setConstant(T(0.01));
    br.setConstant(T(0.01));
    bg.setConstant(T(0.01));

    aimatrix<T> H  = aimatrix<T>::Zero(this->input_size, this->hidden_size);
    this->H.push_back(H);

    XH = aimatrix<T>::Zero(this->input_size, this->param_size + this->hidden_size); // concatenate X and H

}

template <class T>
const aimatrix<T> GRUCell<T>::forward(const aimatrix<T>& X) {

    log_detail("===============================================");
    log_detail("GRUCell Forward Pass ...");

    this->X.push_back(X);

    setInitialWeights(X.rows(), X.cols());

    aimatrix<T> H = this->H.back(); // Hidden state at t-1

    XH << X, H;

    // Calculate the update gate using input, hidden state, and biases (nx[p+h] * [p+h]xh = nxh)
    Zt =  BaseOperator::sigmoid((aimatrix<T>)(BaseOperator::matmul(XH, Wz).rowwise() + bz));

    // Calculate the reset gate using input, hidden state, and biases (nx[p+h] * [p+h]xh = nxh)
    Rt =  BaseOperator::sigmoid((aimatrix<T>)(BaseOperator::matmul(XH, Wr).rowwise() + br));

    // Calculate the candidate hidden state using input, reset gate, hidden state, and biases
    aimatrix<T> RH = Rt.array() * H.array();
    aimatrix<T> rXH(input_size, hidden_size + hidden_size);

    rXH << X, RH;

    Gt = BaseOperator::tanh((aimatrix<T>)(BaseOperator::matmul(rXH, Wg).rowwise() + bg));

    log_detail("Forward Pass Zt ...");
    log_matrix(Zt);
    log_detail("Forward Pass RXH ...");
    log_matrix(rXH);
    log_detail("Forward Pass Rt ...");
    log_matrix(Rt);
    log_detail("Forward Pass Rt ...");
    log_matrix(Gt);

    // Calculate the new hidden state using update gate, previous hidden state, and candidate hidden state
    aimatrix<T> new_H = (1 - Zt.array()).array() * Gt.array() + Zt.array() * H.array(); 

    log_detail("Forward Pass new_H ...");
    log_matrix(new_H);

    this->H.push_back(new_H);

    log_info("GRUCell Forward Pass end ...");

    // We return Output next layer;
    return new_H; 
}

template <class T>
const aimatrix<T> GRUCell<T>:: backward(int step, const aimatrix<T>& dnext_h) {
    // Backpropagation logic for GRU

    log_detail("===============================================");
    log_detail("GRUCell Backward Pass ...");

    aimatrix<T> H = this->H.at(step);
    aimatrix<T> X = this->X.at(step);

    int ps = param_size, hs = hidden_size;

    aimatrix<T> Wzx, Wzh, Wrx, Wrh, Wgx, Wgh;

    std::tie(Wzx, Wzh) = CellBase<T>::split(this->Wz, ps, hs);
    std::tie(Wrx, Wrh) = CellBase<T>::split(this->Wr, ps, hs);
    std::tie(Wgx, Wgh) = CellBase<T>::split(this->Wg, ps, hs);

    // Compute gradients with respect to the gates
    aimatrix<T> dZt =  dnext_h.array() * ( H.array() - Gt.array()) * Zt.array() * ( 1 - Zt.array());
    aimatrix<T> dGt =  dnext_h.array() * ( 1 - Zt.array()) * ( 1 - Gt.array().square());
    aimatrix<T> dRt = ( dGt.array() * (H * Wrh).array()  * ( 1 - Rt.array().square())) * Rt.array() * ( 1 - Rt.array());

    // Concatenate input and hidden state.
    XH << X, H;

    // Compute gradients with respect to hidden-to-hidden weights Wz, Wr, Wg
    this->dWz = BaseOperator::matmul(XH.transpose(), dZt);
    this->dWr = BaseOperator::matmul(XH.transpose(), dRt);
    this->dWg = BaseOperator::matmul(XH.transpose(), dGt);

    // Compute gradients with respect to hidden biases bz, br, bg
    this->dbz = dZt.colwise().sum();
    this->dbr = dGt.colwise().sum();
    this->dbg = dRt.colwise().sum();

    // Compute gradient with respect to hidden state H (dH)
    aimatrix<T> dH = BaseOperator::matmul(dZt, Wzx) + BaseOperator::matmul(dRt, Wrx) + BaseOperator::matmul(dGt, Wgx);
    this->dH.push_back(dH);

    // Compute gradient with respect to input (dInput)
    aimatrix<T>  dX = BaseOperator::matmul(dZt, Wzh) + BaseOperator::matmul(dRt, Wrh) + BaseOperator::matmul(dGt, Wgh);
    this->dX.push_back(dX);

    log_info("GRUCell Backward Pass end ...");

    // Return gradient with respect to input.
    // We return the gradient wrt X for each layer, not wrt H & C for each time step.
    // for the gradient wrt H & C, we just cache it for time step propagation, but layer propagation.
    return dH;
}

template <class T>
void GRUCell<T>::updateParameters(std::string& optimizertype, T& learningRate, int& iter) {
    // Update parameters
    if (opt_Wz == nullptr) {
        opt_Wz = new Optimizer<T>(optimizertype, learningRate);
        opt_Wr = new Optimizer<T>(optimizertype, learningRate);
        opt_Wg = new Optimizer<T>(optimizertype, learningRate);
        opt_bz = new Optimizer<T>(optimizertype, learningRate);
        opt_br = new Optimizer<T>(optimizertype, learningRate);
        opt_bg = new Optimizer<T>(optimizertype, learningRate);
    }
    log_detail("Updating Wz in GRU Cell ...");
    opt_Wz->update(optimizertype, this->Wz, this->dWz, iter);
    log_detail("Updating Wr in GRU Cell ...");
    opt_Wr->update(optimizertype, this->Wr, this->dWr, iter);
    log_detail("Updating Wg in GRU Cell ...");
    opt_Wg->update(optimizertype, this->Wg, this->dWg, iter);
    log_detail("Updating bz in GRU Cell ...");
    opt_bz->update(optimizertype, this->bz, this->dbz, iter);
    log_detail("Updating br in GRU Cell ...");
    opt_br->update(optimizertype, this->br, this->dbr, iter);
    log_detail("Updating bg in GRU Cell ...");
    opt_bg->update(optimizertype, this->bg, this->dbg, iter);
    this->H.clear();
    this->X.clear();
    this->dH.clear();
    this->dX.clear();
    this->input_size = 0;
    this->param_size = 0;
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
    log_info("===============================================");
    log_info("RNN Forward Pass ...");
    const aitensor<T> Yhat = this->rnnbase->forwardpass(input_data);
    log_detail("End RNN Forward ... ");
    this->rnnbase->setPrediction(Yhat);
    return Yhat;
}

template <class T>
const aitensor<T>  RNN<T>::backward(const aitensor<T>& gradients) {
    const aitensor<T> dInput = this->rnnbase->backwardpass(gradients);
    this->rnnbase->setGradients(dInput);
    return dInput;
}

template <class T>
void RNN<T>::updateParameters(std::string& optimizertype, T& learningRate, int& iter) {
    // Learnable parameters already learnt inside the RNN cells
    this->rnnbase->updatingParameters(optimizertype, learningRate, iter);
}

template <class T>
const aitensor<T> LSTM<T>::forward(const aitensor<T>& input_data) {
    log_info("===============================================");
    log_info("LSTM Forward Pass ...");
    const aitensor<T> Yhat = this->rnnbase->forwardpass(input_data);
    log_detail("End LSTM Forward ... ");
    this->rnnbase->setPrediction(Yhat);
    return Yhat;
}

template <class T>
const aitensor<T> LSTM<T>::backward(const aitensor<T>& gradients) {
    const aitensor<T> dInput = this->rnnbase->backwardpass(gradients);
    this->rnnbase->setGradients(dInput);
    return dInput;
}

template <class T>
void LSTM<T>::updateParameters(std::string& optimizertype, T& learningRate, int& iter) {
    // Learnable parameters already learnt inside the LSTM cells
    this->rnnbase->updatingParameters(optimizertype, learningRate, iter);
}


template <class T>
const aitensor<T> GRU<T>::forward(const aitensor<T>& input_data) {
    log_info("===============================================");
    log_info("GRU Forward Pass ...");
    const aitensor<T> Yhat = this->rnnbase->forwardpass(input_data);
    log_detail("End GRU Forward ... ");
    this->rnnbase->setPrediction(Yhat);
    return Yhat;
}

template <class T>
const aitensor<T>  GRU<T>::backward(const aitensor<T>& gradients) {
    const aitensor<T> dInput = this->rnnbase->backwardpass(gradients);
    this->rnnbase->setGradients(dInput);
    return dInput;
}

template <class T>
void GRU<T>::updateParameters(std::string& optimizertype, T& learningRate, int& iter) {
    // Learnable parameters already learnt inside the GRU cells
    this->rnnbase->updatingParameters(optimizertype, learningRate, iter);
}


template <class T>
std::tuple<aimatrix<T>, airowvector<T>> RecurrentBase<T>::getWeights(int step, aimatrix<T> out) {

    if (this->isInitialized()) {
        RNNType       rnntype  = this->getRNNType();
        if (rnntype == RNNType::MANY_TO_ONE) {
            return std::make_tuple(this->V.at(0), this->bo.at(0));
        } else {
            return std::make_tuple(this->V.at(step), this->bo.at(step));
        }
    }

    aimatrix<T> v(out.rows(), out.cols());
    airowvector<T> bo(out.cols());
    BaseOperator::heInitMatrix(v);
    bo.Zero(out.cols());

    this->V.push_back(v);
    this->bo.push_back(bo);

    return std::make_tuple(v, bo);
}

template <class T>
const aitensor<T> RecurrentBase<T>::processOutputs() {
    log_info("===============================================");
    log_info("Recurrent Base Processing Output ...");
    aimatrix<T> out, v, yhat;
    airowvector<T> bo;

    int sequence_length    = this->getSequenceLength();
    ReductionType rtype    = this->getRType();
    RNNType       rnntype  = this->getRNNType();
    aitensor<T> prediction;

    log_detail("Entering Processing of Output");

    int start_step = 0; // Default for MANY_TO_MANY or ONE_TO_MANY scenario

    if (rnntype == RNNType::MANY_TO_ONE) {
        start_step = sequence_length - 1; // consider only the last sequence for MANY_TO_ONE
    }

    for (int step = start_step; step < sequence_length; ++step) {

        log_detail("Process Direction Step {0} with output size {1}", step, this->foutput.size());

        // merge bidirectional outputs
        if (this->bidirectional == true) {
            if (rtype == ReductionType::SUM) {
                out = this->foutput.at(step).array() + this->boutput.at(step).array();
            } else 
            if (rtype == ReductionType::AVG) {
                out = ( this->foutput.at(step).array() + this->boutput.at(step).array() ) / 2;
            } else 
            if (rtype == ReductionType::CONCAT) {
                out << this->foutput.at(step), this->boutput.at(step);
            }
        } else {
            out = this->foutput.at(step);
        }

        log_detail("Reduction Result at step {0}:", step);
        log_matrix(out);

        //  Get weights for the output;
        std::tie(v, bo) = getWeights(step, out);

        if (this->getOType() == ActivationType::SOFTMAX) {
            yhat = BaseOperator::softmax((aimatrix<T>) (BaseOperator::matmul(out, v).rowwise() + bo));
        } else 
        if (this->getOType() == ActivationType::TANH) {
            yhat = BaseOperator::tanh((aimatrix<T>) (BaseOperator::matmul(out, v).rowwise() + bo));
        } else
        if (this->getOType() == ActivationType::SIGMOID) {
            yhat = BaseOperator::sigmoid((aimatrix<T>) (BaseOperator::matmul(out, v).rowwise() + bo));
        } else {
            this->setOutputSize( this->getHiddenSize());
            yhat = out; // no non-linearity operations to logit, raw Hidden State output
        }

        log_detail("Non-Linearity result at step {0}:", step);
        log_matrix(yhat);

        prediction.push_back(yhat);

    }

    // marker to indicate the weights are all initialized, as they depend on these sizes.
    this->setInitialized();

    log_info("Recurrent Base Processing Output End with output size {0} ...", prediction.size());

    // this->setOutput(prediction);
    this->setPrediction(prediction);
    return prediction; // Cache output for backward pass while also pass it to next operations.
}

template <class T>
const aitensor<T> RecurrentBase<T>::forwardpass(const aitensor<T>& input_data) {

    log_info("===============================================");
    log_info("RecurrentBase Forward Pass ...");

    this->input_data = input_data;

    this->setOType(ActivationType::SOFTMAX);
    this->setRType(ReductionType::AVG);

    int sequence_length = this->input_data.size(); // sequence
    int input_size      = this->input_data.at(0).rows();
    int embedding_size  = this->input_data.at(0).cols();
    int num_directions  = this->getNumDirections();

    this->setSequenceLength(sequence_length);
    this->setInputSize(input_size);
    this->setEmbeddingSize(embedding_size);

    log_detail( "Batch Size: {0}, Row: {1}, Col: {2}", sequence_length, input_size, embedding_size );
    log_detail( "Number of Directions: {0}", num_directions );

    for (int direction = 0; direction < num_directions; ++direction) {

        log_detail("-------------------------------");
        log_detail("Next Direction: {0}", direction);

        std::vector<CellBase<T>*> cells = this->getCells(direction);

        // Forward pass: Run from first to last time step
        for (int step = 0; step < sequence_length; ++step) {

            log_detail("Direction {0} Step {1}: ", direction, step);

            // Reverse the order of data if backward direction (e.g. direction == 1);
            int idx_ = (direction == 0) ? step : ( sequence_length - step - 1);

            aimatrix<T> input_batch = input_data.at(idx_);

            log_detail("Input Batch:");
            log_detail("Input Batch Dim: {0}x{1}", input_size, embedding_size);
 
            // Forward pass through each layer of the RNN
            for (int layer = 0; layer < this->getNumLayers(); ++layer) {
                log_detail("Entering Cell Forward pass at Layer {0} ...", layer);
                if (celltype == CellType::RNN_VANILLA) {
                    log_detail("Casting to RNN Cell at layer {0} step {1}...", layer, step);
                    RNNCell<T>* cell = dynamic_cast<RNNCell<T>*>(cells[layer]);
                    input_batch = cell->forward(input_batch); // This is the Hidden State
                } else
                if (celltype == CellType::RNN_LSTM) {
                    LSTMCell<T>* cell = dynamic_cast<LSTMCell<T>*>(cells[layer]);
                    log_detail("Casting to LSTM Cell at layer {0} step {1}...", layer, step);
                    input_batch = cell->forward(input_batch); // This is the Hidden State
                } else
                if (celltype == CellType::RNN_GRU) {
                    GRUCell<T>* cell = dynamic_cast<GRUCell<T>*>(cells[layer]);
                    log_detail("Casting to GRU Cell at layer {0} step {1}...", layer, step);
                    input_batch = cell->forward(input_batch); // This is the Hidden State
                } 
                log_detail("Cell Forward pass output ...");
                log_matrix(input_batch);
                log_detail("Cell Forward pass exited ...");
            }

            log_detail("Layer forward done ...");

            if (direction == 0) {
                log_detail("Add Forward Direction output ...");
                this->foutput.push_back(input_batch);
                log_detail("Forward Direction output size: {0}", this->foutput.size());
            } else {
                log_detail("Add Backward Direction output ...");
                this->boutput.push_back(input_batch);
                log_detail("Backward Direction output size: {0}", this->boutput.size());
            }
        }
    }

    log_detail("Processing Output ...");

    aitensor<T> Yhat = processOutputs();

    log_detail("RecurrentBase Forward Pass end ...");

    return Yhat;
}

template <class T>
void RecurrentBase<T>::processGradients(aitensor<T>& gradients) {

    log_info("===============================================");
    log_info("RecurrentBase processing Gradients ...");

    aimatrix<T> dOut, doutput, yhat;

    // Gradient will have dimension of: sequence_size, batch_size, output_size (instead of hidden_size)

    ActivationType       otype        = this->getOType();
    const aitensor<T>&   prediction   = this->getPrediction();

    int last_sequence = this->getSequenceLength(); // Default for MANY_TO_MANY or ONE_TO_MANY scenario

    if (rnntype == RNNType::MANY_TO_ONE) {
        last_sequence = 1; // consider only the first sequence, given we only have 1 matrix in the vector
    }

    for (int step = 0; step < last_sequence; ++step) {

        log_detail("Sequence step {0} for gradient processing", step);

        // Process gradient for the activation operation
        dOut = gradients.at(step); 
        yhat = prediction.at(step); 
        if (otype == ActivationType::SOFTMAX) {
            dOut = BaseOperator::softmaxGradient(dOut, yhat); // dsoftmaxV
            doutput = BaseOperator::matmul(dOut, yhat.transpose());
            this->dV.push_back(doutput);
            this->dbo.push_back(dOut.colwise().sum());
        } else 
        if (otype == ActivationType::TANH) {
            dOut = BaseOperator::tanhGradient(dOut, yhat);      // dtanV
            doutput = BaseOperator::matmul(dOut, yhat.transpose());
            this->dV.push_back(doutput);
            this->dbo.push_back(dOut.colwise().sum());
        } else
        if (otype == ActivationType::SIGMOID) {
            dOut = BaseOperator::sigmoidGradient(dOut, yhat);  // dsigmoidV
            doutput = BaseOperator::matmul(dOut, yhat.transpose());
            this->dV.push_back(doutput);
            this->dbo.push_back(dOut.colwise().sum());
        }

        log_detail("Updating gradient at step {0} ...", step);
        log_detail("Before image of gradient matrix:");
        log_matrix(gradients.at(step));
        log_detail("Dimension of dOut: {0}x{1}", doutput.rows(), doutput.cols());
        log_matrix(doutput);
        log_detail("Dimension of V: {0}x{1}", this->V.at(step).rows(), this->V.at(step).cols());
        log_matrix(this->V.at(step));
        gradients.at(step) = BaseOperator::matmul(this->V.at(step).transpose(), doutput);
        log_detail("After image of gradient matrix:");
        log_matrix(gradients.at(step));
    }
    log_info("RecurrentBase Backward Pass end ...");
}

template <class T>
const aitensor<T> RecurrentBase<T>::backwardpass(const aitensor<T>& gradients) {

    log_info("===============================================");
    log_info("RecurrentBase Backward Pass ...");

    aitensor<T> dOutput = gradients;

    int sequence_length = this->input_data.size(); // sequence

    int input_size      = this->input_data.at(0).rows();

    int embedding_size  = this->input_data.at(0).cols();

    int hidden_size = this->getHiddenSize();

    int num_directions  = this->getNumDirections();

    log_detail( "Batch Size: {0}, Row: {1}, Col: {2}", sequence_length, input_size, embedding_size );
    log_detail( "Directions: {0}", num_directions );

    this->processGradients(dOutput);

    ReductionType rtype = this->getRType();
    RNNType     rnntype = this->getRNNType();

    std::vector<aimatrix<T>> dOutf, dOutb;

    int gradient_size = dOutput.size();  
 
    // Now, let us see if we need to split;
    // Process gradient for the reduction operation
    for (int step = gradient_size - 1; step >= 0; --step) {
        aimatrix<T> seq_f, seq_b, seq_m;
        if (this->bidirectional == true) {
            if (rtype == ReductionType::SUM) {
                seq_f = dOutput.at(step); 
                seq_b = dOutput.at(step); 
            } else 
            if (rtype == ReductionType::AVG) {
                seq_f = dOutput.at(step) / 2; 
                seq_b = dOutput.at(step) / 2; 
            } else 
            if (rtype == ReductionType::CONCAT) {
                seq_m = dOutput.at(step); 
                seq_f = seq_m.block(0, 0, sequence_length, embedding_size);
                seq_b = seq_m.block(0, embedding_size, sequence_length, embedding_size);
            }
            dOutf.push_back(seq_f);
            dOutb.push_back(seq_b);
        } else {
            seq_f = dOutput.at(step); 
            dOutf.push_back(seq_f);
        }
        if (rnntype == RNNType::MANY_TO_ONE) {
            break; // Since output is only one, then we take only the gradient.
        }
    }

    log_detail("Start Backward bidirectional loop ...");

    // We need to send back the same structure of input_gradients as the input to the forward pass
    aitensor<T> dInput; // (batch_size, input_size, embedding_size);
    aimatrix<T> dnextH, dnextC;

    dnextH = aimatrix<T>::Zero(input_size, hidden_size);
    dnextC = aimatrix<T>::Zero(input_size, hidden_size);

    for (int direction = 0; direction < this->getNumDirections(); ++direction) {

        std::vector<CellBase<T>*> cells = this->getCells(direction);

        // Backward pass: Run from last to first time step
        for (int step = sequence_length - 1; step >= 0; --step) {

            log_detail("Sequence forward: Direction {0} Step {1}: ", direction, step);
            log_detail("Input Batch Dim: {0}x{1}", input_size, embedding_size);

            if (rnntype == RNNType::MANY_TO_ONE) {
                // consider only the first step, afterwhich, 
                // gradient with respect to hidden states per step is propagated along
                // the gradient of the output is propagated only once at the beginning of the step.
                if (step == sequence_length - 1) { 
                    if (direction == 0) {
                        dnextH = dOutf.at(0) + dnextH;
                    } else {
                        dnextH = dOutb.at(0) + dnextH;
                    }
                }
            } else {
                if (direction == 0) {
                    dnextH = dOutf.at(step) + dnextH;
                } else {
                    dnextH = dOutb.at(step) + dnextH;
                }
            }
 
            log_detail("dOuts input:");
            log_matrix(dnextH);

            // Backward pass through each layer of the RNN
            for (int layer = this->getNumLayers() - 1; layer >= 0; --layer) {
                log_detail("Entering Cell Backward pass ... layer {0}", layer);

                if (celltype == CellType::RNN_VANILLA) {
                    RNNCell<T>* cell = dynamic_cast<RNNCell<T>*>(cells[layer]);
                    dnextH = cell->backward(step, dnextH);
                } else
                if (celltype == CellType::RNN_LSTM) {
                    LSTMCell<T>* cell = dynamic_cast<LSTMCell<T>*>(cells[layer]);
                    std::tie(dnextH, dnextC) = cell->backward(step, dnextH, dnextC);
                } else
                if (celltype == CellType::RNN_GRU) {
                    GRUCell<T>* cell = dynamic_cast<GRUCell<T>*>(cells[layer]);
                    dnextH = cell->backward(step, dnextH);
                } 

                log_detail("Cell Backward pass output");
                log_detail("dOuts output:");
                log_matrix(dnextH);
            }

            log_detail("Layer forward done ...");

            // Store the gradients for input data
            dInput.push_back(dnextH);  

            log_detail("Sequence add dInput  ...");
            log_matrix(dnextH);
        }
        log_detail("Sequence forward done: Direction {0}  ...", direction );
    }
    log_detail("RecurrentBase Backward Pass end ...");

    return dInput;
}

template <class T>
void RecurrentBase<T>::updatingParameters(std::string& optimizertype, T& learningRate, int& iter) {

    log_info("===============================================");
    log_info("RecurrentBase Updating Parameters ...");

    RNNType rnntype = this->getRNNType();

    int sequence_length = this->getSequenceLength();

    if (rnntype == RNNType::MANY_TO_ONE) {
        sequence_length = 1;
    }

    // initialize optimization parameters (V and bo) for output
    if (opt_V.size() == 0) {

        for (int step = 0; step < sequence_length; ++step) {
            Optimizer<T>* outV  = new Optimizer<T>(optimizertype, learningRate);
            Optimizer<T>* outbo = new Optimizer<T>(optimizertype, learningRate);
            this->opt_V.push_back(outV);
            this->opt_bo.push_back(outbo);
        }
    }

    // updating optimization parameters (V and bo) for output
    for (int step = 0; step < sequence_length; ++step) {
        log_detail("Updating V at output step {0} ...", step);
        this->opt_V.at(step)->update(optimizertype, this->V.at(step), this->dV.at(step), iter);
        log_detail("Updating bo at output step {0} ...", step);
        this->opt_bo.at(step)->update(optimizertype, this->bo.at(step), this->dbo.at(step), iter);
    }

    // clear for the next training/epoch
    this->dV.clear();
    this->dbo.clear();

    // Retrieve sequence again since  MANY_TO_ONE scenario does not matter.
    sequence_length = this->getSequenceLength(); 

    // if bidirectional is false, then we only have 1 direction (forward-direction)
    for (int direction = 0; direction < this->getNumDirections(); ++direction) {

        std::vector<CellBase<T>*> cells = this->getCells(direction);

        // Updating cells
        // for (int step = this->getSequenceLength() - 1; step >= 0; --step) {
        for (int step = 0; step < sequence_length; ++step) {
            for (int layer = this->getNumLayers() - 1; layer >= 0; --layer) {
                if (celltype == CellType::RNN_VANILLA) {
                    RNNCell<T>* cell = dynamic_cast<RNNCell<T>*>(cells[layer]);
                    log_detail("Starting RNN update (direction: {0}) (step: {1}) ...", direction, step);
                    cell->updateParameters(optimizertype, learningRate, iter);
                } else
                if (celltype == CellType::RNN_LSTM) {
                    LSTMCell<T>* cell = dynamic_cast<LSTMCell<T>*>(cells[layer]);
                    log_detail("Starting LSTM update (direction: {0}) (step: {1}) ...", direction, step);
                    cell->updateParameters(optimizertype, learningRate, iter);
                } else
                if (celltype == CellType::RNN_GRU) {
                    GRUCell<T>* cell = dynamic_cast<GRUCell<T>*>(cells[layer]);
                    log_detail("Starting GRU update (direction: {0}) (step: {1}) ...", direction, step);
                    cell->updateParameters(optimizertype, learningRate, iter);
                }
            }
        }
    }

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


