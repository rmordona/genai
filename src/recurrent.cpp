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

    // Cell Shared Weights
    this->W.resize(this->param_size, this->hidden_size);
    this->U.resize(this->hidden_size, this->hidden_size);
    this->bh.resize(this->hidden_size);

    this->dW = aimatrix<T>::Zero(this->param_size, this->hidden_size);
    this->dU = aimatrix<T>::Zero(this->hidden_size, this->hidden_size);
    this->dbh = airowvector<T>::Zero(this->hidden_size);

    BaseOperator::heInitMatrix(this->W);
    BaseOperator::heInitMatrix(this->U);
    this->bh.setConstant(T(0.01));

    if (rnntype == RNNType::ONE_TO_MANY) {
        this->O.resize(this->output_size, this->hidden_size); // This weight is required for subsequent ONE_TO_MANY time_steps.
        this->dO = aimatrix<T>::Zero(this->output_size, this->hidden_size);
        BaseOperator::heInitMatrix(this->O);
    }

    aimatrix<T> H  = aimatrix<T>::Zero(this->input_size, this->hidden_size);
    this->H.push_back(H);
}
 
template <class T>
const aimatrix<T> RNNCell<T>::forward(const aimatrix<T>& X) {

    log_detail("===============================================");
    log_detail("RNNCell Forward Pass ...");

    log_detail("Size: {0}", X.size());

    int step = this->X.size();

    this->X.push_back(X);
    setInitialWeights(X.rows(), X.cols());

    aimatrix<T> H = this->H.back();  // Hidden state at t-1

    aimatrix<T> W =  (this->rnntype != RNNType::ONE_TO_MANY || (this->rnntype == RNNType::ONE_TO_MANY && step == 0)) ?  this->W : this->O;

    log_detail("Dimension of X: {0}x{1}", X.rows(), X.cols());
    log_matrix(X);
    log_detail("Dimension of W: {0}x{1}", W.rows(), W.cols());
    log_matrix(W);
    log_detail("Dimension of O (One-to-Many only): {0}x{1}", O.rows(), O.cols());
    log_matrix(O);
    log_detail("Dimension of H: {0}x{1}", H.rows(), H.cols());
    log_matrix(H);
    log_detail("Dimension of U: {0}x{1}", U.rows(), U.cols());
    log_matrix(U);

    // Compute hidden state.
    //     (nxh) =  (nxp) * (pxh) + (nxh) * (hxh) + h
    // H = BaseOperator::tanh((aimatrix<T>) (X * W + H * U + bh) );
    aimatrix<T> new_H = (aimatrix<T>)  BaseOperator::tanh((aimatrix<T>) ((BaseOperator::matmul(X, W) + BaseOperator::matmul(H, U)).rowwise() + bh));

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
const std::tuple<aimatrix<T>,aimatrix<T>> RNNCell<T>::backward(int step, const aimatrix<T>& dOut, const aimatrix<T>& dnext_h) {
    // Backpropagation logic for RNN
    log_detail("===============================================");
    log_detail("RNNCell Backward Pass ...");

    log_detail("Dimension of dnext_h: {0}x{1}", dnext_h.rows(), dnext_h.cols());
    log_matrix(dnext_h);

    bool is_W_seq = (this->rnntype != RNNType::ONE_TO_MANY || (this->rnntype == RNNType::ONE_TO_MANY && step == 0));

    const aimatrix<T>& H = this->H.at(step); // Hidden state at t-1
    const aimatrix<T>& X = this->X.at(step); // Input at t
    const aimatrix<T>& W =  (is_W_seq) ?  this->W : this->O;

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


    // Compute gradient with respect to W (dW) from tanh perspective
    log_detail("Calculating gradient with respect to W (dW) ...");
    aimatrix<T> dW_ = BaseOperator::matmul(X.transpose(), dtanh);

    if (is_W_seq) { 
        if (this->rnntype == RNNType::ONE_TO_MANY) {
            this->dW += dW_;       // current step
            this->dW += this->dO;  // previous steps
        } else {
            this->dW += dW_;
        }
    } else {  
        this->dO += dW_; 
    }

    log_detail("Gradient of dW_:");
    log_matrix(dW_);
    log_detail("Gradient of this->dW:");
    log_matrix(this->dW);
    log_detail("Gradient of this->dO:");
    log_matrix(this->dO);

    // Compute gradient with respect to H (dH) from tanh perspective
    log_detail("Calculating gradient with respect to U (dU) ...");
    aimatrix<T> dU_ = BaseOperator::matmul(H.transpose(), dtanh); 
    this->dU += dU_;

    log_detail("Gradient of dU_:");
    log_matrix(dU_);
    log_detail("Gradient of dU:");
    log_matrix(this->dU);

    // Compute gradient with respect to hidden bias bh.
    log_detail("Calculating dbh ...");
    airowvector<T> dbh_ = dtanh.colwise().sum();  this->dbh += dbh_;
    log_rowvector(this->dbh);


    // Compute gradient with respect to H (dH) from tanh perspective
    log_detail("Calculating gradient with respect to H (dH) ...");
    aimatrix<T> dH = BaseOperator::matmul(dtanh, U.transpose());
    log_matrix(dH);


    // Compute gradient with respect to input (X)(dInput) from tanh perspective
    log_detail("Calculating gradient with respect to X (dX) ...");
    aimatrix<T> dX = BaseOperator::matmul(dtanh, W.transpose());
    log_matrix(dX);

    this->dH.push_back(dH);
    this->dX.push_back(dX);

    log_info("RNNCell Backward Pass end ...");

    // Return gradient with respect to input.
    // We return the gradient wrt X for each layer, not wrt H for each time step.
    // for the gradient wrt H, we just cache it for time step propagation, but layer propagation.
    return std::make_tuple(dX, dH);
}

template <class T>
void RNNCell<T>::updateParameters(std::string& optimizertype, T& learningRate, int& iter) {
    // Update parameters
    if (opt_W == nullptr) {
        opt_W  = new Optimizer<T>(optimizertype, learningRate);
        opt_U  = new Optimizer<T>(optimizertype, learningRate);
        opt_bh = new Optimizer<T>(optimizertype, learningRate);
        if (rnntype == RNNType::ONE_TO_MANY) {
            opt_O  = new Optimizer<T>(optimizertype, learningRate);
        }
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
    this->dW.setZero();
    this->dU.setZero();
    this->dbh.setZero();

    if (rnntype == RNNType::ONE_TO_MANY) {
        log_detail("Updating O in RNN Cell ...");
        opt_O->update(optimizertype, this->O, this->dO, iter);
        this->dO.setZero();
    }
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

    // Cell Shared Weights
    Wf.resize(this->param_size, this->hidden_size);
    Wi.resize(this->param_size, this->hidden_size);
    Wg.resize(this->param_size, this->hidden_size);
    Wo.resize(this->param_size, this->hidden_size);

    Uf.resize(this->hidden_size, this->hidden_size);
    Ui.resize(this->hidden_size, this->hidden_size);
    Ug.resize(this->hidden_size, this->hidden_size);
    Uo.resize(this->hidden_size, this->hidden_size);

    bf.resize(this->hidden_size);
    bi.resize(this->hidden_size);
    bg.resize(this->hidden_size);
    bo.resize(this->hidden_size);

    BaseOperator::heInitMatrix(Wf);
    BaseOperator::heInitMatrix(Wi);
    BaseOperator::heInitMatrix(Wg);
    BaseOperator::heInitMatrix(Wo);

    BaseOperator::heInitMatrix(Uf);
    BaseOperator::heInitMatrix(Ui);
    BaseOperator::heInitMatrix(Ug);
    BaseOperator::heInitMatrix(Uo);

    bf.setConstant(T(0.01));
    bi.setConstant(T(0.01));
    bg.setConstant(T(0.01));
    bo.setConstant(T(0.01));

    dWf = aimatrix<T>::Zero(this->param_size, this->hidden_size);
    dWi = aimatrix<T>::Zero(this->param_size, this->hidden_size);
    dWg = aimatrix<T>::Zero(this->param_size, this->hidden_size);
    dWo = aimatrix<T>::Zero(this->param_size, this->hidden_size);

    dUf = aimatrix<T>::Zero(this->hidden_size, this->hidden_size);
    dUi = aimatrix<T>::Zero(this->hidden_size, this->hidden_size);
    dUg = aimatrix<T>::Zero(this->hidden_size, this->hidden_size);
    dUo = aimatrix<T>::Zero(this->hidden_size, this->hidden_size);

    dbf = airowvector<T>::Zero(this->hidden_size);
    dbi = airowvector<T>::Zero(this->hidden_size);
    dbg = airowvector<T>::Zero(this->hidden_size);
    dbo = airowvector<T>::Zero(this->hidden_size);

    // This weight is required for subsequent ONE_TO_MANY time_steps.
    if (rnntype == RNNType::ONE_TO_MANY) {
        Of.resize(this->output_size, this->hidden_size); 
        Oi.resize(this->output_size, this->hidden_size); 
        Og.resize(this->output_size, this->hidden_size); 
        Oo.resize(this->output_size, this->hidden_size); 
        BaseOperator::heInitMatrix(Of);
        BaseOperator::heInitMatrix(Oi);
        BaseOperator::heInitMatrix(Og);
        BaseOperator::heInitMatrix(Oo);
        dOf = aimatrix<T>::Zero(this->output_size, this->hidden_size); 
        dOi = aimatrix<T>::Zero(this->output_size, this->hidden_size); 
        dOg = aimatrix<T>::Zero(this->output_size, this->hidden_size); 
        dOo = aimatrix<T>::Zero(this->output_size, this->hidden_size); 
    }
    aimatrix<T> H  = aimatrix<T>::Zero(this->input_size, this->hidden_size);
    aimatrix<T> C  = aimatrix<T>::Zero(this->input_size, this->hidden_size);
    H.setConstant(T(1e-8));
    C.setConstant(T(1e-8));
    this->H.push_back(H);
    this->C.push_back(C);

}

template <class T>
const aimatrix<T> LSTMCell<T>::forward(const aimatrix<T>& X) {

    log_detail("===============================================");
    log_detail("LSTMCell Forward Pass ...");

    int step = this->X.size();

    this->X.push_back(X);

    setInitialWeights(X.rows(), X.cols());

    aimatrix<T> H = this->H.back(); // Hidden state at t-1
    aimatrix<T> C = this->C.back(); // Cell state at t-1

    aimatrix<T> Wf =  (this->rnntype != RNNType::ONE_TO_MANY || (this->rnntype == RNNType::ONE_TO_MANY && step == 0)) ?  this->Wf : this->Of;
    aimatrix<T> Wi =  (this->rnntype != RNNType::ONE_TO_MANY || (this->rnntype == RNNType::ONE_TO_MANY && step == 0)) ?  this->Wi : this->Oi;
    aimatrix<T> Wg =  (this->rnntype != RNNType::ONE_TO_MANY || (this->rnntype == RNNType::ONE_TO_MANY && step == 0)) ?  this->Wg : this->Og;
    aimatrix<T> Wo =  (this->rnntype != RNNType::ONE_TO_MANY || (this->rnntype == RNNType::ONE_TO_MANY && step == 0)) ?  this->Wo : this->Oo;

    log_detail("Dimension of Wf: {0}x{1}", Wf.rows(), Wf.cols());
    log_matrix(Wf);

    log_detail("Dimension of Wi: {0}x{1}", Wi.rows(), Wi.cols());
    log_matrix(Wi);

    log_detail("Dimension of Wg: {0}x{1}", Wg.rows(), Wg.cols());
    log_matrix(Wg);

    log_detail("Dimension of Wo: {0}x{1}", Wo.rows(), Wo.cols());
    log_matrix(Wo);

    // Calculate the forget gate values (nxp * pxh + nxh * hxh  = nxh)
    aimatrix<T> Ft = BaseOperator::sigmoid((aimatrix<T>) ((BaseOperator::matmul(X, Wf) + BaseOperator::matmul(H, Uf)).rowwise() + bf));
    log_detail("Dimension of Ft: {0}x{1}", Ft.rows(), Ft.cols());
    log_matrix(Ft);

    // Calculate the input gate values (nxp * pxh + nxh * hxh  = nxh)
    aimatrix<T> It = BaseOperator::sigmoid((aimatrix<T>) ((BaseOperator::matmul(X, Wi) + BaseOperator::matmul(H, Ui)).rowwise() + bi));
    log_detail("Dimension of It: {0}x{1}", It.rows(), It.cols());
    log_matrix(It);

    // Calculate the output gate values (nxp * pxh + nxh * hxh  = nxh)
    aimatrix<T> Ot = BaseOperator::sigmoid((aimatrix<T>)  ((BaseOperator::matmul(X, Wo) + BaseOperator::matmul(H, Uo)).rowwise() + bo));
    log_detail("Dimension of Ot: {0}x{1}", Ot.rows(), Ot.cols());
    log_matrix(Ot);

    // Calculate the candidate state values (nxp * pxh + nxh * hxh  = nxh)
    aimatrix<T> Gt = BaseOperator::tanh((aimatrix<T>)  ((BaseOperator::matmul(X, Wg) + BaseOperator::matmul(H, Ug)).rowwise() + bg));
    log_detail("Dimension of Gt: {0}x{1}", Gt.rows(), Gt.cols());
    log_matrix(Gt);

    // Calculate the new cell state by updating based on the gates (result: nxh)
    aimatrix<T> new_C = Ft.array() * C.array() + It.array() * Gt.array();
    log_detail("Dimension of new_C: {0}x{1}", new_C.rows(), new_C.cols());
    log_matrix(new_C);

    // Calculate the new hidden state by applying the output gate and tanh activation
    aimatrix<T> new_H = BaseOperator::tanh(new_C).array() * Ot.array();
    log_detail("Dimension of new_H: {0}x{1}", new_H.rows(), new_H.cols());
    log_matrix(new_H);

    this->Ft.push_back(Ft);
    this->It.push_back(It);
    this->Gt.push_back(Gt);
    this->Ot.push_back(Ot);
    this->C.push_back(new_C);
    this->H.push_back(new_H);

    log_info("LSTMCell Forward Pass end ...");

    // We return Output next layer;
    return new_H; 

}

template <class T>
const std::tuple<aimatrix<T>,aimatrix<T>,aimatrix<T>> LSTMCell<T>::backward(int step, 
        const aimatrix<T>& dOut, const aimatrix<T>& dnext_h, const aimatrix<T>& dnext_c) {
    // Backpropagation logic for LSTM
    log_detail("===============================================");
    log_detail("LSTMCell Backward Pass ...");

    // Compute gradient with respect to hidden state H (dH)
    //int ps = param_size, hs = hidden_size;

    //aimatrix<T> Wfx, Wfh, Wix, Wih, Wox, Woh, Wgx, Wgh;

    log_detail("Dimension of dnext_h: {0}x{1}", dnext_h.rows(), dnext_h.cols());
    log_matrix(dnext_h);

    log_detail("Dimension of dnext_c: {0}x{1}", dnext_c.rows(), dnext_c.cols());
    log_matrix(dnext_c);

    aimatrix<T> Ft = this->Ft.at(step);
    aimatrix<T> It = this->It.at(step);
    aimatrix<T> Gt = this->Gt.at(step);
    aimatrix<T> Ot = this->Ot.at(step);

    aimatrix<T> nC = this->C.at(step+1); // Cell state at t
    aimatrix<T> nH = this->H.at(step+1); // Hidden state at t
    aimatrix<T> H = this->H.at(step); // Hidden state at t-1
    aimatrix<T> C = this->C.at(step); // Cell state at t-1
    aimatrix<T> X = this->X.at(step); // Input at t

    log_detail("Dimension of X: {0}x{1}", X.rows(), X.cols());
    log_matrix(X);
    log_detail("Dimension of H: {0}x{1}", H.rows(), H.cols());
    log_matrix(H);
    log_detail("Dimension of C: {0}x{1}", C.rows(), C.cols());
    log_matrix(C);
    log_detail("Dimension of Ft: {0}x{1}", Ft.rows(), Ft.cols());
    log_matrix(Ft); 
    log_detail("Dimension of It: {0}x{1}", It.rows(), It.cols());
    log_matrix(It); 
    log_detail("Dimension of Gt: {0}x{1}", Gt.rows(), Gt.cols());
    log_matrix(Gt); 
    log_detail("Dimension of Ot: {0}x{1}", Ot.rows(), Ot.cols());
    log_matrix(Ot); 

    // Concatenate input and hidden state.
    // XH << X, H;

    // log_detail("Dimension of XH: {0}x{1}", XH.rows(), XH.cols());
    //log_matrix(XH);

    aimatrix<T> tanhC  = BaseOperator::tanh(nC);
    aimatrix<T> dtanhC = BaseOperator::tanhGradient(dnext_h, tanhC);
    aimatrix<T> dC     = Ot.array() * dtanhC.array() + dnext_c.array();

    log_detail("Dimension of dC: {0}x{1}", dC.rows(), dC.cols());
    log_matrix(dC);

    // Gradient with respect to O
    aimatrix<T> dOt = nC.array() * dnext_h.array() * Ot.array() * ( 1 - Ot.array()); // BaseOperator::sigmoidGradient(dnext_h, Ot).array(); 
    log_detail("Dimension of dOt: {0}x{1}", dOt.rows(), dOt.cols());
    log_matrix(dOt);

    // Gradient with respect to F
    aimatrix<T> dFt = C.array() *  dtanhC.array() * Ft.array() * (1 - Ft.array()); // BaseOperator::sigmoidGradient(dC, Ft).array();
    log_detail("Dimension of dFt: {0}x{1}", dFt.rows(), dFt.cols());
    log_matrix(dFt);

    // Gradient with respect to I
    aimatrix<T> dIt = Gt.array() *  dtanhC.array() * It.array() * (1 - It.array()); //  BaseOperator::sigmoidGradient(dC, It).array(); 
    log_detail("Dimension of dIt: {0}x{1}", dIt.rows(), dIt.cols());
    log_matrix(dIt);

    // Gradient with respect to G
    aimatrix<T> dGt = It.array() * dtanhC.array() * (1 - Gt.array().square()); // BaseOperator::tanhGradient(dC, Gt).array();
    log_detail("Dimension of dGt: {0}x{1}", dGt.rows(), dGt.cols());
    log_matrix(dGt);

    // Compute gradients with respect to hidden-to-hidden weights Wf, Wi, Wg, Wo, Uf, Ui, Ug, Uo
    aimatrix<T> dWf_ = BaseOperator::matmul(X.transpose(), dFt);  this->dWf += dWf_;
    aimatrix<T> dWi_ = BaseOperator::matmul(X.transpose(), dIt);  this->dWi += dWi_;
    aimatrix<T> dWg_ = BaseOperator::matmul(X.transpose(), dGt);  this->dWg += dWg_;
    aimatrix<T> dWo_ = BaseOperator::matmul(X.transpose(), dOt);  this->dWo += dWo_;

    aimatrix<T> dUf_ = BaseOperator::matmul(H.transpose(), dFt);  this->dUf += dUf_;
    aimatrix<T> dUi_ = BaseOperator::matmul(H.transpose(), dIt);  this->dUi += dUi_;
    aimatrix<T> dUg_ = BaseOperator::matmul(H.transpose(), dGt);  this->dUg += dUg_;
    aimatrix<T> dUo_ = BaseOperator::matmul(H.transpose(), dOt);  this->dUo += dUo_;

    log_detail("Dimension of dWf: {0}x{1}", this->dWf.rows(), this->dWf.cols());
    log_matrix(this->dWf);

    log_detail("Dimension of dWi: {0}x{1}", this->dWi.rows(), this->dWi.cols());
    log_matrix(this->dWi);

    // Compute gradients with respect to hidden biases bf, bi, bg, bo
    airowvector<T> dbf_ = dFt.colwise().sum(); this->dbf += dbf_;
    airowvector<T> dbi_ = dIt.colwise().sum(); this->dbi += dbi_;
    airowvector<T> dbg_ = dGt.colwise().sum(); this->dbg += dbg_;
    airowvector<T> dbo_ = dOt.colwise().sum(); this->dbo += dbo_;


    log_detail("Dimension of this->Wf: {0}x{1}", this->Wf.rows(), this->Wf.cols());
    log_matrix(this->Wf);

    log_detail("Dimension of this->Uf: {0}x{1}", this->Uf.rows(), this->Uf.cols());
    log_matrix(this->Uf);


    log_detail("Dimension of this->Wi: {0}x{1}", this->Wi.rows(), this->Wi.cols());
    log_matrix(this->Wi);

    log_detail("Dimension of this->Ui: {0}x{1}", this->Ui.rows(), this->Ui.cols());
    log_matrix(this->Ui);

    log_detail("Dimension of this->Wg: {0}x{1}", this->Wg.rows(), this->Wg.cols());
    log_matrix(this->Wg);

    log_detail("Dimension of this->Ug: {0}x{1}", this->Ug.rows(), this->Ug.cols());
    log_matrix(this->Ug);

    log_detail("Dimension of this->Wo: {0}x{1}", this->Wo.rows(), this->Wo.cols());
    log_matrix(this->Wo);

    log_detail("Dimension of this->Uo: {0}x{1}", this->Uo.rows(), this->Uo.cols());
    log_matrix(this->Uo);

    // Compute gradient with respect to input (dInput).
    aimatrix<T> dX  = BaseOperator::matmul(dFt, Wf.transpose()) + BaseOperator::matmul(dIt, Wi.transpose()) + 
                      BaseOperator::matmul(dOt, Wo.transpose()) + BaseOperator::matmul(dGt, Wg.transpose());

    // Compute gradient with respect to hidden state.
    aimatrix<T> dH  = BaseOperator::matmul(dFt, Uf.transpose()) + BaseOperator::matmul(dIt, Ui.transpose()) + 
                      BaseOperator::matmul(dOt, Uo.transpose()) + BaseOperator::matmul(dGt, Ug.transpose());

    // Compute gradient with respect to the cell state C (dC)
    dC = dtanhC.array() * Ft.array();

    log_detail("Dimension of dC (dC * Ft): {0}x{1}", dC.rows(), dC.cols());
    log_matrix(dC);

    log_detail("Dimension of dH: {0}x{1}", dH.rows(), dH.cols());
    log_matrix(dH);

    log_detail("Dimension of dX: {0}x{1}", dX.rows(), dX.cols());
    log_matrix(dX);

    log_info("LSTMCell Backward Pass end ...");

    // Return gradient with respect to input.
    // We return the gradient wrt X for each layer, not wrt H & C for each time step.
    // for the gradient wrt H & C, we just cache it for time step propagation, but layer propagation.
    return std::make_tuple(dX, dH, dC);
}

template <class T>
void LSTMCell<T>::updateParameters(std::string& optimizertype, T& learningRate, int& iter) {
    // Update parameters
    if (opt_Wf == nullptr) {
        opt_Wf = new Optimizer<T>(optimizertype, learningRate);
        opt_Wi = new Optimizer<T>(optimizertype, learningRate);
        opt_Wg = new Optimizer<T>(optimizertype, learningRate);
        opt_Wo = new Optimizer<T>(optimizertype, learningRate);

        opt_Uf = new Optimizer<T>(optimizertype, learningRate);
        opt_Ui = new Optimizer<T>(optimizertype, learningRate);
        opt_Ug = new Optimizer<T>(optimizertype, learningRate);
        opt_Uo = new Optimizer<T>(optimizertype, learningRate);

        opt_bf = new Optimizer<T>(optimizertype, learningRate);
        opt_bi = new Optimizer<T>(optimizertype, learningRate);
        opt_bg = new Optimizer<T>(optimizertype, learningRate);
        opt_bo = new Optimizer<T>(optimizertype, learningRate);

        if (rnntype == RNNType::ONE_TO_MANY) {
            opt_Of = new Optimizer<T>(optimizertype, learningRate);
            opt_Oi = new Optimizer<T>(optimizertype, learningRate);
            opt_Og = new Optimizer<T>(optimizertype, learningRate);
            opt_Oo = new Optimizer<T>(optimizertype, learningRate);
        }
    }

    log_detail("Updating Wf in LSTM Cell ...");
    opt_Wf->update(optimizertype, this->Wf, this->dWf, iter);
    log_detail("Updating Wi in LSTM Cell ...");
    opt_Wi->update(optimizertype, this->Wi, this->dWi, iter);
    log_detail("Updating Wg in LSTM Cell ...");
    opt_Wg->update(optimizertype, this->Wg, this->dWg, iter);
    log_detail("Updating Wo in LSTM Cell ...");
    opt_Wo->update(optimizertype, this->Wo, this->dWo, iter);

    log_detail("Updating Uf in LSTM Cell ...");
    opt_Uf->update(optimizertype, this->Uf, this->dUf, iter);
    log_detail("Updating Ui in LSTM Cell ...");
    opt_Ui->update(optimizertype, this->Ui, this->dUi, iter);
    log_detail("Updating Ui in LSTM Cell ...");
    opt_Ug->update(optimizertype, this->Ui, this->dUg, iter);
    log_detail("Updating Ui in LSTM Cell ...");
    opt_Uo->update(optimizertype, this->Ui, this->dUo, iter);

    log_detail("Updating bf in LSTM Cell ...");
    opt_bf->update(optimizertype, this->bf, this->dbf, iter);
    log_detail("Updating bi in LSTM Cell ...");
    opt_bi->update(optimizertype, this->bi, this->dbi, iter);
    log_detail("Updating bg in LSTM Cell ...");
    opt_bg->update(optimizertype, this->bg, this->dbg, iter);
    log_detail("Updating bo in LSTM Cell ...");
    opt_bo->update(optimizertype, this->bo, this->dbo, iter);

    this->Ft.clear();
    this->It.clear();
    this->Gt.clear();
    this->Ot.clear();
    this->H.clear();
    this->C.clear();
    this->X.clear();

    this->dWf.setZero();
    this->dWi.setZero();
    this->dWg.setZero();
    this->dWo.setZero();

    this->dUf.setZero();
    this->dUi.setZero();
    this->dUg.setZero();
    this->dUo.setZero();

    this->dbf.setZero();
    this->dbi.setZero();
    this->dbg.setZero();
    this->dbo.setZero();

    if (rnntype == RNNType::ONE_TO_MANY) {
        log_detail("Updating Of in LSTM Cell ...");
        opt_Of->update(optimizertype, this->Of, this->dOf, iter);
        log_detail("Updating Oi in LSTM Cell ...");
        opt_Oi->update(optimizertype, this->Oi, this->dOi, iter);
        log_detail("Updating Og in LSTM Cell ...");
        opt_Og->update(optimizertype, this->Og, this->dOg, iter);
        log_detail("Updating Og in LSTM Cell ...");
        opt_Oo->update(optimizertype, this->Oo, this->dOo, iter);
        this->dOf.setZero();
        this->dOi.setZero();
        this->dOg.setZero();
        this->dOo.setZero();
    }
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
    Wz.resize(this->param_size, this->hidden_size);
    Wr.resize(this->param_size, this->hidden_size);
    Wg.resize(this->param_size, this->hidden_size);

    Uz.resize(this->hidden_size, this->hidden_size);
    Ur.resize(this->hidden_size, this->hidden_size);
    Ug.resize(this->hidden_size, this->hidden_size);

    bz.resize(this->hidden_size);
    br.resize(this->hidden_size);
    bg.resize(this->hidden_size);

    BaseOperator::heInitMatrix(Wz);
    BaseOperator::heInitMatrix(Wr);
    BaseOperator::heInitMatrix(Wg);

    bz.setConstant(T(0.01));
    br.setConstant(T(0.01));
    bg.setConstant(T(0.01));


    dWz = aimatrix<T>::Zero(this->param_size, this->hidden_size);
    dWr = aimatrix<T>::Zero(this->param_size, this->hidden_size);
    dWg = aimatrix<T>::Zero(this->param_size, this->hidden_size);

    dUz = aimatrix<T>::Zero(this->hidden_size, this->hidden_size);
    dUr = aimatrix<T>::Zero(this->hidden_size, this->hidden_size);
    dUg = aimatrix<T>::Zero(this->hidden_size, this->hidden_size);

    dbz = airowvector<T>::Zero(this->hidden_size);
    dbr = airowvector<T>::Zero(this->hidden_size);
    dbg = airowvector<T>::Zero(this->hidden_size);

    // This weight is required for subsequent ONE_TO_MANY time_steps.
    if (rnntype == RNNType::ONE_TO_MANY) {
        Oz.resize(this->output_size, this->hidden_size); 
        Or.resize(this->output_size, this->hidden_size); 
        Og.resize(this->output_size, this->hidden_size); 
        BaseOperator::heInitMatrix(Oz);
        BaseOperator::heInitMatrix(Or);
        BaseOperator::heInitMatrix(Og);
        dOz.aimatrix<T>::Zero(this->output_size, this->hidden_size); 
        dOr.aimatrix<T>::Zero(this->output_size, this->hidden_size); 
        dOg.aimatrix<T>::Zero(this->output_size, this->hidden_size); 
    }
    aimatrix<T> H  = aimatrix<T>::Zero(this->input_size, this->hidden_size);
    this->H.push_back(H);

}

template <class T>
const aimatrix<T> GRUCell<T>::forward(const aimatrix<T>& X) {

    log_detail("===============================================");
    log_detail("GRUCell Forward Pass ...");

    int step = this->X.size();

    this->X.push_back(X);

    setInitialWeights(X.rows(), X.cols());

    aimatrix<T> H = this->H.back(); // Hidden state at t-1

    aimatrix<T> Wz =  (this->rnntype != RNNType::ONE_TO_MANY || (this->rnntype == RNNType::ONE_TO_MANY && step == 0)) ?  this->Wz : this->Oz;
    aimatrix<T> Wr =  (this->rnntype != RNNType::ONE_TO_MANY || (this->rnntype == RNNType::ONE_TO_MANY && step == 0)) ?  this->Wr : this->Or;
    aimatrix<T> Wg =  (this->rnntype != RNNType::ONE_TO_MANY || (this->rnntype == RNNType::ONE_TO_MANY && step == 0)) ?  this->Wg : this->Og;

    // Calculate the update gate using input, hidden state, and biases (nxp * pxh + nxh * hxh  = nxh)
    aimatrix<T> Zt =  BaseOperator::sigmoid((aimatrix<T>)((BaseOperator::matmul(X, Wz) + BaseOperator::matmul(H, Uz)).rowwise() + bz));

    // Calculate the reset gate using input, hidden state, and biases (nxp * pxh + nxh * hxh  = nxh)
    aimatrix<T> Rt =  BaseOperator::sigmoid((aimatrix<T>)((BaseOperator::matmul(X, Wr) + BaseOperator::matmul(H, Ur)).rowwise() + br));

    // Calculate the candidate hidden state using input, reset gate, hidden state, and biases
    aimatrix<T> RH = Rt.array() * H.array();
    aimatrix<T> rXH(input_size, hidden_size + hidden_size);

    aimatrix<T> Gt = BaseOperator::tanh((aimatrix<T>)((BaseOperator::matmul(X, Wg) + BaseOperator::matmul(RH, Ug)).rowwise() + bg));

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

    this->Zt.push_back(Zt);
    this->Rt.push_back(Rt);
    this->Gt.push_back(Gt);
    this->H.push_back(new_H);

    log_info("GRUCell Forward Pass end ...");

    // We return Output next layer;
    return new_H; 
}

template <class T>
const std::tuple<aimatrix<T>,aimatrix<T>> GRUCell<T>::backward(int step, const aimatrix<T>& dOut, const aimatrix<T>& dnext_h) {
    // Backpropagation logic for GRU

    log_detail("===============================================");
    log_detail("GRUCell Backward Pass ...");

    //int ps = param_size, hs = hidden_size;

    //aimatrix<T> Wzx, Wzh, Wrx, Wrh, Wgx, Wgh;

    log_detail("Retrievint H and X at step {0} ...", step);
    aimatrix<T> Zt = this->Zt.at(step);
    aimatrix<T> Rt = this->Rt.at(step);
    aimatrix<T> Gt = this->Gt.at(step);
    aimatrix<T> H  = this->H.at(step);
    aimatrix<T> X  = this->X.at(step);

    log_detail("Dimension of H: {0}x{1}", H.rows(), H.cols());
    log_matrix(H);

    log_detail("Dimension of X: {0}x{1}", X.rows(), X.cols());
    log_matrix(X);

    // Compute gradients with respect to the gates
    aimatrix<T> dZt = dnext_h.array() * (H.array() - Gt.array()) * Zt.array() * (1 - Zt.array());
    aimatrix<T> dGt = dnext_h.array() * (1 - Zt.array()) * (1 - Gt.array().square());
    aimatrix<T> dRt = (dGt.array() * (H * Ur).array()  * (1 - Rt.array().square())) * Rt.array() * (1 - Rt.array());

    aimatrix<T> RH = Rt.array() * H.array();
 
    // Compute gradients with respect to hidden-to-hidden weights Wz, Wr, Wg
    aimatrix<T> dWz_ = BaseOperator::matmul(X.transpose(), dZt); this->dWz += dWz_;
    aimatrix<T> dWr_ = BaseOperator::matmul(X.transpose(), dRt); this->dWr += dWr_;
    aimatrix<T> dWg_ = BaseOperator::matmul(X.transpose(), dGt); this->dWg += dWg_;

    aimatrix<T> dUz_ = BaseOperator::matmul(H.transpose(), dZt); this->dUz += dUz_;
    aimatrix<T> dUr_ = BaseOperator::matmul(H.transpose(), dRt); this->dUr += dUr_;
    aimatrix<T> dUg_ = BaseOperator::matmul(RH.transpose(), dGt); this->dUg += dUg_;

    log_detail("Dimension of this->dWz: {0}x{1}", this->dWz.rows(), this->dWz.cols());
    log_matrix(this->dWz);

    log_detail("Dimension of this->dWr: {0}x{1}", this->dWr.rows(), this->dWr.cols());
    log_matrix(this->dWr);

    log_detail("Dimension of this->dWg: {0}x{1}", this->dWg.rows(), this->dWg.cols());
    log_matrix(this->dWg);

    // Compute gradients with respect to hidden biases bz, br, bg
    airowvector<T> dbz_ = dZt.colwise().sum(); this->dbz += dbz_;
    airowvector<T> dbr_ = dRt.colwise().sum(); this->dbr += dbr_;
    airowvector<T> dbg_ = dGt.colwise().sum(); this->dbg += dbg_;

    log_detail("Dimension of this->Wz: {0}x{1}", this->Wz.rows(), this->Wz.cols());
    log_matrix(this->Wz);


    // Compute gradient with respect to hidden state H (dH)
    aimatrix<T> dX = BaseOperator::matmul(dZt, Wz.transpose()) + BaseOperator::matmul(dRt, Wr.transpose()) + BaseOperator::matmul(dGt, Wg.transpose());

    // Compute gradient with respect to input (dInput)
    aimatrix<T> dH = BaseOperator::matmul(dZt, Uz.transpose()) + BaseOperator::matmul(dRt, Ur.transpose()) + BaseOperator::matmul(dGt, Ug.transpose());

    log_info("GRUCell Backward Pass end ...");

    // Return gradient with respect to input.
    // We return the gradient wrt X for each layer, not wrt H & C for each time step.
    // for the gradient wrt H & C, we just cache it for time step propagation, but layer propagation.
    return std::make_tuple(dX, dH);
}

template <class T>
void GRUCell<T>::updateParameters(std::string& optimizertype, T& learningRate, int& iter) {
    // Update parameters
    if (opt_Wz == nullptr) {
        opt_Wz = new Optimizer<T>(optimizertype, learningRate);
        opt_Wr = new Optimizer<T>(optimizertype, learningRate);
        opt_Wg = new Optimizer<T>(optimizertype, learningRate);

        opt_Uz = new Optimizer<T>(optimizertype, learningRate);
        opt_Ur = new Optimizer<T>(optimizertype, learningRate);
        opt_Ug = new Optimizer<T>(optimizertype, learningRate);

        opt_bz = new Optimizer<T>(optimizertype, learningRate);
        opt_br = new Optimizer<T>(optimizertype, learningRate);
        opt_bg = new Optimizer<T>(optimizertype, learningRate);

        if (rnntype == RNNType::ONE_TO_MANY) {
            opt_Oz = new Optimizer<T>(optimizertype, learningRate);
            opt_Or = new Optimizer<T>(optimizertype, learningRate);
            opt_Og = new Optimizer<T>(optimizertype, learningRate);
        }
    }
    log_detail("Updating Wz in GRU Cell ...");
    opt_Wz->update(optimizertype, this->Wz, this->dWz, iter);
    log_detail("Updating Wr in GRU Cell ...");
    opt_Wr->update(optimizertype, this->Wr, this->dWr, iter);
    log_detail("Updating Wg in GRU Cell ...");
    opt_Wg->update(optimizertype, this->Wg, this->dWg, iter);

    log_detail("Updating Uz in GRU Cell ...");
    opt_Uz->update(optimizertype, this->Uz, this->dUz, iter);
    log_detail("Updating Ur in GRU Cell ...");
    opt_Ur->update(optimizertype, this->Ur, this->dUr, iter);
    log_detail("Updating Ug in GRU Cell ...");
    opt_Ug->update(optimizertype, this->Ug, this->dUg, iter);

    log_detail("Updating bz in GRU Cell ...");
    opt_bz->update(optimizertype, this->bz, this->dbz, iter);
    log_detail("Updating br in GRU Cell ...");
    opt_br->update(optimizertype, this->br, this->dbr, iter);
    log_detail("Updating bg in GRU Cell ...");
    opt_bg->update(optimizertype, this->bg, this->dbg, iter);

    this->Zt.clear();
    this->Rt.clear();
    this->Gt.clear();
    this->H.clear();
    this->X.clear();
    this->dH.clear();
    this->dX.clear();

    this->dWz.setZero();
    this->dWr.setZero();
    this->dWg.setZero();

    this->dUz.setZero();
    this->dUr.setZero();
    this->dUg.setZero();

    this->dbz.setZero();
    this->dbr.setZero();
    this->dbg.setZero();

    if (rnntype == RNNType::ONE_TO_MANY) {
        log_detail("Updating Oz in GRU Cell ...");
        opt_Oz->update(optimizertype, this->Oz, this->dOz, iter);
        log_detail("Updating Or in GRU Cell ...");
        opt_Or->update(optimizertype, this->Or, this->dOr, iter);
        log_detail("Updating Og in GRU Cell ...");
        opt_Og->update(optimizertype, this->Og, this->dOg, iter);
        this->dOz.setZero();
        this->dOr.setZero();
        this->dOg.setZero();
    }
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

    RNNType       rnntype  = this->getRNNType();
    if (this->isInitialized()) {
        // there is only one element in a vector of predicted outputs 
        if (rnntype == RNNType::MANY_TO_ONE) {
            return std::make_tuple(this->V.at(0), this->by.at(0));
        } else {
            log_detail("Getting weight {0}x{1} at step {2} ...", this->input_size, this->output_size, step);
            return std::make_tuple(this->V.at(step), this->by.at(step));
        }
    }
    log_detail("Initializing  weight {0}x{1} at step {2} ...", out.cols(), this->output_size, step);
    aimatrix<T> V;
   // if (rnntype == RNNType::ONE_TO_MANY) {
   //    V.resize(this->input_size, this->output_size);
   // } else {
       V.resize(out.cols(), this->output_size);
   // } 
    airowvector<T> by(this->output_size);
    BaseOperator::heInitMatrix(V);
    by.setConstant(T(0.01));

    this->V.push_back(V);
    this->by.push_back(by);

    log_detail("End Initializing  weight {0}x{1} at step {2} ...", out.cols(), this->output_size, step);

    return std::make_tuple(V, by);
}

// Handle output for ONE_TO_MANY scenarios
template <class T>
const aimatrix<T> RecurrentBase<T>::processPrediction(int step, const aimatrix<T>& H) {
    log_info("===============================================");
    log_info("Recurrent Base Processing Output ...");
    aimatrix<T> V, Yhat; // H being the final output.
    airowvector<T> by;

    log_detail("Entering Processing of Cell Output");
    log_detail("Reduction Result at step {0}:", step);

    //  Get weights for the output;
    std::tie(V, by) = getWeights(step, H);

    log_detail("Get (H) new_H");
    log_matrix(H);

    log_detail("Get (V) weight");
    log_matrix(V);

    log_detail("Get (by) weight");
    log_rowvector(by);

    if (this->getOType() == ActivationType::SOFTMAX) {
        log_detail("Non-Linearity (Softmax) ...");
        log_matrix((aimatrix<T>) (BaseOperator::matmul(H, V).rowwise() + by));
        Yhat = BaseOperator::softmax((aimatrix<T>) (BaseOperator::matmul(H, V).rowwise() + by));
    } else 
    if (this->getOType() == ActivationType::TANH) {
        log_detail("Non-Linearity (TANH) ...");
        Yhat = BaseOperator::tanh((aimatrix<T>) (BaseOperator::matmul(H, V).rowwise() + by));
    } else
    if (this->getOType() == ActivationType::SIGMOID) {
        log_detail("Non-Linearity (SIGMOID) ...");
        Yhat = BaseOperator::sigmoid((aimatrix<T>) (BaseOperator::matmul(H, V).rowwise() + by));
    } else {
        this->setOutputSize( this->getHiddenSize());
        Yhat = H; // no non-linearity operations to logit, raw Hidden State output
    }

    log_detail("Non-Linearity result (yhat) at step {0}:", step);
    log_matrix(Yhat);

    this->output.push_back(H);
    this->Yhat.push_back(Yhat);

    log_info("Recurrent Base Processing Output End with output size {0} ...", this->Yhat.size());

    return Yhat; 
}

// Handle output for MANY_TO_ONE or MANY_TO_MANY scenarios
template <class T>
const aitensor<T> RecurrentBase<T>::processPredictions() {
    log_info("===============================================");
    log_info("Recurrent Base Processing Output ...");
    aimatrix<T> H, V, Yhat; // H being the final output.
    airowvector<T> by;

    int sequence_length    = this->getSequenceLength();
    ReductionType rtype    = this->getRType();
    RNNType       rnntype  = this->getRNNType();
    aitensor<T> prediction;
    aitensor<T> output;

    log_detail("Entering Processing of Output");

    int start_step = 0; // Default for MANY_TO_MANY or ONE_TO_MANY scenario

    if (rnntype == RNNType::MANY_TO_ONE) {
        start_step = sequence_length - 1; // consider only the last sequence for MANY_TO_ONE
    }

    for (int step = start_step; step < sequence_length; ++step) {

        log_detail("Process Output Step {0} with output size {1}", step, this->foutput.size());

        // merge bidirectional outputs
        if (this->bidirectional == true) {
            if (rtype == ReductionType::SUM) {
                log_detail("Reduction type: SUM");
                H = this->foutput.at(step).array() + this->boutput.at(step).array(); // H being the final output
            } else 
            if (rtype == ReductionType::AVG) {
                log_detail("Reduction type: AVG");
                H = ( this->foutput.at(step).array() + this->boutput.at(step).array() ) / 2; // H being the final output
            } else 
            if (rtype == ReductionType::CONCAT) {
                log_detail("Reduction type: CONCAT");
                H << this->foutput.at(step), this->boutput.at(step); // H being the final output
            }
        } else {
            H = this->foutput.at(step);
        }

        log_detail("Reduction Result at step {0}:", step);

        //  Get weights for the output;
        std::tie(V, by) = getWeights(step, H);

        log_detail("H output ...");
        log_matrix(H);

        log_detail("V output ...");
        log_matrix(V);

        log_detail("bo output ...");
        log_rowvector(by);

        log_detail("Matmul (H * v) ...");
        log_matrix(BaseOperator::matmul(H, V));

        if (this->getOType() == ActivationType::SOFTMAX) {
            log_detail("Non-Linearity (Softmax) ...");
            Yhat = BaseOperator::softmax((aimatrix<T>) (BaseOperator::matmul(H, V).rowwise() + by));
        } else 
        if (this->getOType() == ActivationType::TANH) {
            log_detail("Non-Linearity (TANH) ...");
            Yhat = BaseOperator::tanh((aimatrix<T>) (BaseOperator::matmul(H, V).rowwise() + by));
        } else
        if (this->getOType() == ActivationType::SIGMOID) {
            log_detail("Non-Linearity (SIGMOID) ...");
            Yhat = BaseOperator::sigmoid((aimatrix<T>) (BaseOperator::matmul(H, V).rowwise() + by));
        } else {
            this->setOutputSize( this->getHiddenSize());
            Yhat = H; // no non-linearity operations to logit, raw Hidden State output
        }

        log_detail("Non-Linearity result (yhat) at step {0}:", step);
        log_matrix(Yhat);

        output.push_back(H);
        prediction.push_back(Yhat);

    }

    log_info("Recurrent Base Processing Output End with output size {0} ...", prediction.size());

    this->setOutput(output);
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

    RNNType     rnntype = this->getRNNType();

    int sequence_length = this->input_data.size(); // sequence
    int input_size      = this->input_data.at(0).rows();
    int embedding_size  = this->input_data.at(0).cols();
    int num_directions  = this->getNumDirections();

    if (rnntype == RNNType::ONE_TO_MANY) {
        sequence_length = this->getOutputSequenceLength();
        if (sequence_length == 0) {
            throw AIException("Sequence Length for One-To-Many Scenario not specified ...");
        }
    } 

    this->setSequenceLength(sequence_length);
    this->setInputSize(input_size);
    this->setEmbeddingSize(embedding_size);

    log_detail( "Batch Size: {0}, Row: {1}, Col: {2}", sequence_length, input_size, embedding_size );
    log_detail( "Number of Directions: {0}", num_directions );

    if (this->bidirectional == true &&  rnntype == RNNType::ONE_TO_MANY) {
        throw AIException("Bidirectional RNN for One-To-Many Scenario is not allowed ...");
    }

    aimatrix<T> input_batch; // absorbs input from dataset or output from cell (in ONE_TO_MANY systems)

    for (int direction = 0; direction < num_directions; ++direction) {

        log_detail("-------------------------------");
        log_detail("Next Direction: {0}", direction);

        std::vector<CellBase<T>*> cells = this->getCells(direction);

        // Forward pass: Run from first to last time step
        for (int step = 0; step < sequence_length; ++step) {

            log_detail("Direction {0} Step {1}: ", direction, step);

            // Reverse the order of data if backward direction (e.g. direction == 1);
            int idx_ = (direction == 0) ? step : ( sequence_length - step - 1);

            //  handle only first input if ONE_TO_MANY or all input if not ONE_TO_MANY
            if ((step == 0 && rnntype == RNNType::ONE_TO_MANY) || rnntype != RNNType::ONE_TO_MANY) {
                input_batch = input_data.at(idx_);
            } 

            log_detail("Input Batch:");
            log_detail("Input Batch Dim: {0}x{1}", input_size, embedding_size);
 
            // Forward pass through each layer of the RNN
            for (int layer = 0; layer < this->getNumLayers(); ++layer) {
                log_detail("Entering Cell Forward pass at Layer {0} ...", layer);
                if (celltype == CellType::RNN_VANILLA) {
                    log_detail("Casting to RNN Cell at layer {0} step {1}...", layer, step);
                    RNNCell<T>* cell = dynamic_cast<RNNCell<T>*>(cells[layer]);
                    input_batch = cell->forward(input_batch); // Output the Hidden State
                } else
                if (celltype == CellType::RNN_LSTM) {
                    LSTMCell<T>* cell = dynamic_cast<LSTMCell<T>*>(cells[layer]);
                    log_detail("Casting to LSTM Cell at layer {0} step {1}...", layer, step);
                    input_batch = cell->forward(input_batch); // Output the Hidden State
                } else
                if (celltype == CellType::RNN_GRU) {
                    GRUCell<T>* cell = dynamic_cast<GRUCell<T>*>(cells[layer]);
                    log_detail("Casting to GRU Cell at layer {0} step {1}...", layer, step);
                    input_batch = cell->forward(input_batch); // Output the Hidden State
                } 
                log_detail("Cell Forward pass output ...");
                log_matrix(input_batch);
                log_detail("Cell Forward pass exited ...");
            }

            log_detail("Layer forward done ...");

            if (direction == 0) {
                log_detail("Add Forward Direction output ...");

                // pass predicted output of hidden state to next time step as input in ONE_TO_MANY scenario.
                if (rnntype == RNNType::ONE_TO_MANY) {
                    aimatrix<T> Yhat = processPrediction(step, input_batch);
                    input_batch = Yhat;
                }

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

    // otherwise, if scenario is not ONE_TO_MANY, then generate output as is without passing to next time step.
    if (rnntype != RNNType::ONE_TO_MANY) {
        Yhat = processPredictions();
    }

    // marker to indicate the weights are all initialized, as they depend on these sizes.
    this->setInitialized();

    log_detail("RecurrentBase Forward Pass end ...");

    return Yhat;
}

// For Many-To-Many or MANY-To-One Scenario
template <class T>
void RecurrentBase<T>::processGradients(aitensor<T>& gradients) {

    log_info("===============================================");
    log_info("RecurrentBase processing Gradients ...");

    aimatrix<T> dOut, dV, yhat, H, V; // H being the final output

    // Gradient will have dimension of: sequence_size, batch_size, output_size (instead of hidden_size)

    ActivationType       otype        = this->getOType();
    const aitensor<T>&   prediction   = this->getPrediction();

    // Default for MANY_TO_MANY scenario
    int last_sequence   = this->getSequenceLength(); 

    // for ONE_TO_MANY scenario
    if (this->rnntype == RNNType::ONE_TO_MANY) {
        last_sequence = this->getOutputSequenceLength();
    } else // For MANY_TO_ONE scenario
    if (this->rnntype == RNNType::MANY_TO_ONE) {
        last_sequence = 1; // consider only the first sequence, given we only have 1 matrix in the vector
    }

    for (int step = 0; step < last_sequence; ++step) {

        log_detail("-----------------------------------------");
        log_detail("Sequence step {0} for gradient processing", step);

        // Process gradient for the activation operation
        dOut = gradients.at(step); 
        yhat = prediction.at(step); 
        V = this->V.at(step);
        H = this->output.at(step);
        log_detail("Gradients (dOut) at step {0}", step);
        log_matrix(dOut);

        log_detail("Dimension of H: {0}x{1}", H.rows(), H.cols());
        log_matrix(H);

        log_detail("Dimension of V: {0}x{1}", V.rows(), V.cols());
        log_matrix(V);

        log_detail("Yhat output (prediction at step {})", step);
        log_matrix(yhat);

        if (otype == ActivationType::SOFTMAX) {
            log_detail("Non-Linearity Softmax Gardient ...");
            dOut = BaseOperator::softmaxGradient(dOut, yhat); // dsoftmaxV
            dV = BaseOperator::matmul(H.transpose(), dOut);
            this->dV.push_back(dV);
            this->dby.push_back(dOut.colwise().sum());
        } else 
        if (otype == ActivationType::TANH) {
            log_detail("Non-Linearity TANH Gardient ...");
            dOut = BaseOperator::tanhGradient(dOut, yhat);      // dtanV
            dV = BaseOperator::matmul(H.transpose(), dOut);
            this->dV.push_back(dV);
            this->dby.push_back(dOut.colwise().sum());
        } else
        if (otype == ActivationType::SIGMOID) {
            log_detail("Non-Linearity SIGMOID Gardient ...");
            dOut = BaseOperator::sigmoidGradient(dOut, yhat);  // dsigmoidV
            dV = BaseOperator::matmul(H.transpose(), dOut);
            this->dV.push_back(dV);
            this->dby.push_back(dOut.colwise().sum());
        } // else just use dOut

        log_detail("Dimension of dOut: {0}x{1}", dOut.rows(), dOut.cols());
        log_matrix(dOut);

        log_detail("Dimension of dV: {0}x{1}", dV.rows(), dV.cols());
        log_matrix(dV);

        log_detail("dby");
        log_rowvector((airowvector<T>) dOut.colwise().sum());

        log_detail("Setting gradient at step {0} ...", step);
        log_detail("Before image of gradient matrix:");
        log_matrix(gradients.at(step));

        // Calculate dInput (dnext_H)
        gradients.at(step) = BaseOperator::matmul(dOut, V.transpose());
        log_detail("After image of gradient (dnextH: BaseOperator::matmul(dOut, V.transpose())");
        log_matrix(gradients.at(step));

        aimatrix<T> dInput = BaseOperator::matmul(H.transpose(), dOut);
        log_detail("After image of gradient (new dV: BaseOperator::matmul(H.transpose(), dOut)");
        log_matrix(dInput);

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
                seq_f.resize(sequence_length, embedding_size);
                seq_b.resize(sequence_length, embedding_size);
                seq_b = seq_m.block(0, embedding_size, sequence_length, embedding_size);
                seq_f = seq_m.block(0, 0, sequence_length, embedding_size);
            }
            dOutf.push_back(seq_f);
            dOutb.push_back(seq_b);
        } else {
            seq_f = dOutput.at(step); 
            dOutf.push_back(seq_f);
        }
        if (rnntype == RNNType::MANY_TO_ONE) {
            break; // Since output is only one, then we take only the last gradient.
        }
    }

    log_detail("Start Backward bidirectional loop ...");

    // We need to send back the same structure of input_gradients as the input to the forward pass
    aitensor<T> dInput; // (batch_size, input_size, embedding_size);

    aimatrix<T> dOut, dnextH, dnextC;

    for (int direction = 0; direction < this->getNumDirections(); ++direction) {

        dnextH = aimatrix<T>::Zero(input_size, hidden_size);
        dnextC = aimatrix<T>::Zero(input_size, hidden_size);

        std::vector<CellBase<T>*> cells = this->getCells(direction);

        // Backward pass: Run from last to first time step
        for (int step = sequence_length - 1; step >= 0; --step) {

            log_detail("*** Sequence forward: Direction {0} Step {1}: ", direction, step);
            log_detail("*** Input Batch Dim: {0}x{1}", input_size, embedding_size);

            if (rnntype == RNNType::MANY_TO_ONE) {
                // consider only the first step, afterwhich, 
                // gradient with respect to hidden states per step is propagated along
                // the gradient of the output is propagated only once at the beginning of the step.
                if (step == sequence_length - 1) { 
                    if (direction == 0) {
                        dOut = dOutf.at(0);
                        dnextH = dOutf.at(0) + dnextH;

                    } else {
                        dOut = dOutb.at(0);
                        dnextH = dOutb.at(0) + dnextH;
                    }
                } 
            } else {
                if (direction == 0) {
                    dOut = dOutf.at(step);
                    dnextH = dOut + dnextH;
                } else {
                    dOut = dOutb.at(step);
                    dnextH = dOut + dnextH;
                }
            }

            log_detail("dOut output:");
            log_matrix(dOut);
 
            log_detail("dnextH input:");
            log_matrix(dnextH);

            // Backward pass through each layer of the RNN
            for (int layer = this->getNumLayers() - 1; layer >= 0; --layer) {
                log_detail("Entering Cell Backward pass ... layer {0}", layer);

                if (celltype == CellType::RNN_VANILLA) {
                    RNNCell<T>* cell = dynamic_cast<RNNCell<T>*>(cells[layer]);
                    std::tie(dOut, dnextH) = cell->backward(step, dOut, dnextH);
                } else
                if (celltype == CellType::RNN_LSTM) {
                    LSTMCell<T>* cell = dynamic_cast<LSTMCell<T>*>(cells[layer]);
                    std::tie(dOut, dnextH, dnextC) = cell->backward(step, dOut, dnextH, dnextC);
                } else
                if (celltype == CellType::RNN_GRU) {
                    GRUCell<T>* cell = dynamic_cast<GRUCell<T>*>(cells[layer]);
                    std::tie(dOut, dnextH) = cell->backward(step, dOut, dnextH);
                } 
                log_detail("Cell Backward pass output");
                log_detail("dOuts output:");
                log_matrix(dnextH);
                dnextH = dOut + dnextH;
            }


            log_detail("Layer forward done ...");

            // Store the gradients for input data
            dInput.push_back(dOut);  

            log_detail("Sequence add dInput  ...");
            log_matrix(dOut);
        }
        log_detail("Sequence forward done: Direction {0}  ...", direction );
    }
    log_detail("RecurrentBase Backward Pass end ...");

    return dInput;
}

template<class T>
void RecurrentBase<T>::clearCache() {
    this->foutput.clear();
    this->boutput.clear();
    this->output.clear();
    this->dV.clear();
    this->dby.clear();
    this->gradients.clear();
    this->Yhat.clear();
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
            Optimizer<T>* outby = new Optimizer<T>(optimizertype, learningRate);
            this->opt_V.push_back(outV);
            this->opt_by.push_back(outby);
        }
    }

    // updating optimization parameters (V and bo) for output
    for (int step = 0; step < sequence_length; ++step) {
        log_detail("Updating V at output step {0} ...", step);
        this->opt_V.at(step)->update(optimizertype, this->V.at(step), this->dV.at(step), iter);
        log_detail("Updating bo at output step {0} ...", step);
        this->opt_by.at(step)->update(optimizertype, this->by.at(step), this->dby.at(step), iter);
    }

    // clear Cache for the next training/epoch
    this->clearCache();

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


