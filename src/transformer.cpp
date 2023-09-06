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
 *
*/

#include "genai.h"
#include "operators.h"
#include "transformer.h"

namespace py = pybind11;
using namespace py::literals;


/*****************************************************************************************************
* Base Attention Head Layer
*****************************************************************************************************/

// While the parameter weight has dimension BxMxW,  the resulting transformation has dimension of BxNxW.
// We only need the M dimension from an BxNxM input to generate parameter matrix.
// where weights is BxNxW and bias is W.
template <class T>
const aitensor<T> Attention<T>::forward(const aitensor<T>& input_data) { 

    log_info("=====================================");
    log_info( "Entering Attention Forward Pass ..." );

    // Cache for later back propagation.
    this->input_data = input_data;

    this->B  = input_data.dimension(0);
    this->N  = input_data.dimension(1);
    this->M  = input_data.dimension(2);

    this->Dk = this->M / this->H;

    log_detail( "Size of input:", input_data.size() );

    if (Q == nullptr || K == nullptr || V == nullptr || Wo == nullptr) {
        Q  = new Linear<T>(this->W, this->bias);
        K  = new Linear<T>(this->W, this->bias);
        V  = new Linear<T>(this->W, this->bias);
        Wo = new Linear<T>(this->M, this->bias);
    }

    // Perform Linear Transformation.
    this->Qout = Q->forward(input_data);
    this->Kout = K->forward(input_data);
    this->Vout = V->forward(input_data);

    log_detail( "Q Linear output" );
    log_matrix( this->Qout );

    log_detail( "K Linear output" );
    log_matrix( this->Kout );

    log_detail( "V Linear output" );
    log_matrix( this->Vout );

    aitensor<T> voutputs(this->B, this->N, this->W); // Dimension: BxNxW
    QKsoft.resize(this->B, this->N, this->N); // Dimension: BxNxN

    aimatrix<T> mQout, mKout, mVout, mQKsoft, mQKsoftV, QK;

    for (int i = 0; i < this->B; ++i) {

        mQout = matrix_view(chip(this->Qout, i, 0)); // NxW
        mKout = matrix_view(chip(this->Kout, i, 0)); // NxW
        mVout = matrix_view(chip(this->Vout, i, 0)); // NxW

        // MatMul (QK^T)
        QK = BaseOperator::matmul(mQout, mKout.transpose()); // NxN

        log_detail( "QK matmul" );
        log_matrix( QK );

        // Include some Masking (still to be implemented)

        // Scale sqrt(Dk)
        QK = QK.array() / sqrt(this->Dk);

        log_detail( "Scaling -> QK / sqrt(Dk)" );
        log_matrix( QK );

        // Mask if required (Decoder)
        // ...

        // Perform Softmax
        mQKsoft = BaseOperator::softmax(QK);

        log_detail( "Softmax QKsoft output" );
        log_matrix(  mQKsoft );

        log_detail( "Vout output" );
        log_matrix(  mVout  );

        // Cache to be used by Back propagation
        QKsoft.chip(i, 0) = tensor_view(mQKsoft);

        // Include dropout (still to be implemented)

        // Perform matmul with V
        mQKsoftV = BaseOperator::matmul(mQKsoft, mVout);

        voutputs.chip(i, 0) = tensor_view(mQKsoftV);
    
    }

    // Perform another transform to align dimension.
    aitensor<T> output = Wo->forward(voutputs);

    log_detail( "Attention forward pass output ..." );
    log_matrix( output );

    return output; // this becomes input to the next Node or next Layer.
}


// Leave the gradients as is. They are cached in the Node. 
// They will be used to update the parameters in next parallel operations.
// the dInput is the gradient we propagate to source Nodes in the graph;
// while the parameter gradients get cached to be used to update the parameters later.
template <class T>
const aitensor<T> Attention<T>::backward(const aitensor<T>& gradients) { 

    log_info("=====================================");
    log_info( "Entering Attention Backward Pass ..." );

    // Gradient with Respect to W linear operations.
    aitensor<T> WodInput = Wo->backward(gradients); 

    aimatrix<T> mWodInput, mKout, mQout, mVout, mQKsoft;

    aitensor<T>  QdInput(this->B, this->N, this->M);
    aitensor<T>  KdInput(this->B, this->N, this->M);
    aitensor<T>  VdInput(this->B, this->N, this->M);
    aitensor<T> dInput(this->B, this->N, this->M);

    for (int i = 0; i < this->B; ++i) {

        mWodInput = matrix_view( chip(WodInput, i, 0));
        mKout     = matrix_view( chip(Kout, i, 0));
        mQout     = matrix_view( chip(Qout, i, 0));
        mVout     = matrix_view( chip(Vout, i, 0));
        mQKsoft   = matrix_view( chip(QKsoft, i, 0));

        // Gradient with Respect to QKsoft (matmul operation)
        aimatrix<T> dQKsoft = BaseOperator::matmul(mWodInput, mVout.transpose());

        log_detail( "dQKsoft gradient" );
        log_matrix( dQKsoft );

        // Gradient with Respect to Vout (matmul operation)
        aimatrix<T> VmInput = BaseOperator::matmul(mQKsoft.transpose(), mWodInput);

        log_detail( "VmInput gradient" );
        log_matrix( VmInput );

        // Propagate Gradient to softmax operation.
        aimatrix<T> dInput = BaseOperator::softmaxGradient(dQKsoft, mQKsoft);

        log_detail( "dInput gradient" );
        log_matrix( dInput );

        // Propagate Gradient to scale operation.
        aimatrix<T> dScale = -0.5 * dInput.array() * (1.0 / std::sqrt(this->Dk));

        log_detail( "dScale gradient" );
        log_matrix( dInput );

        // Gradient with Respect to Q (matmul operation)
        aimatrix<T> QdQK = BaseOperator::matmul(dScale, mKout); // Kout was already tranposed during forward.

        log_detail( "QdQK gradient" );
        log_matrix( QdQK );

        // Gradient with Respect to V (matmul operation)
        aimatrix<T> KdQK = BaseOperator::matmul(mQout.transpose(), dScale);

        log_detail( "KdQK gradient" );
        log_matrix( KdQK );

        QdInput.chip(i, 0) = tensor_view(QdQK);
        KdInput.chip(i, 0) = tensor_view(KdQK);
        VdInput.chip(i, 0) = tensor_view(VmInput);

    }

    log_detail( " Done Backprop to Q, K, V ..." );

    // Propagate Gradient to the Q,K,V linear operations.
    QdInput = Q->backward(QdInput);  // NxM
    KdInput = K->backward(KdInput);  // NxM
    VdInput = V->backward(VdInput);  // NxM

    log_detail( " Done Backprop to Q, K, V ..." );

    log_detail( "VdInput gradient" );
    log_matrix( VdInput );

    log_detail( "QdInput gradient" );
    log_matrix( QdInput );

    log_detail( "KdInput gradient" );
    log_matrix( KdInput );

    // dInput = QdInput.data() + KdInput.data() + VdInput.data(); // BxNxM

    log_detail( "dInput gradient" );
    log_matrix( dInput );

    return dInput;  
}

template <class T>
void Attention<T>::updateParameters(std::string& optimizertype, T& learningRate, int& iter) {

    log_info("=========================================");
    log_info( "Entering Attention Update Parameter ..." );

    Q->updateParameters(optimizertype, learningRate, iter);
    K->updateParameters(optimizertype, learningRate, iter);
    V->updateParameters(optimizertype, learningRate, iter);
    Wo->updateParameters(optimizertype, learningRate, iter);
}

template <class T>
std::string Attention<T>::generateDotFormat() {   
    std::string dot = "{* Attention *}|";  
    if (Q == nullptr || K == nullptr || V == nullptr || Wo == nullptr) {
        dot += "{- No training done yet -}";
    } else {
        dot += Q->generateDotFormat("Q") + "|";
        dot += K->generateDotFormat("K") + "|";
        dot += V->generateDotFormat("V") + "|";
        dot += Wo->generateDotFormat("Wo");
    }
    return dot; 
}

/*****************************************************************************************************
* Base Multi-Head Attention Layer:
*****************************************************************************************************/
// While the parameter weight has dimension BxMxW,  the resulting transformation has dimension of BxNxW.
// We only need the M dimension from an NxM input to generate parameter matrix.
// where weights is BxNxW and bias is W.
template <class T>
const aitensor<T> MultiHeadAttention<T>::forward(const aitensor<T>& input_data) { 


    log_info("===============================================");
    log_info( "Entering Multi-Head Attention Forward Pass ..." );

    // Cache for later back propagation.
    this->input_data = input_data;

    this->B  = input_data.dimension(0);
    this->N  = input_data.dimension(1);
    this->M  = input_data.dimension(2);
    this->Dk = this->M / this->H;

    log_detail( "Size of input: {:d}" , input_data.size() );
    log_detail( "Size of Head: {:d}", this->H );

    if (M1.empty()) {
        for (int i = 0; i < this->H; i++) {
            Attention<T>* A1  = new Attention<T>(this->H, this->W);
            M1.push_back(A1);
        }
    }

    log_detail( "Size of DK ...{:d}" , Dk );

    aitensor<T> output(this->B, this->N, this->M);

    std::vector<aitensor<T>> heads = feature_slice(input_data, this->H);

    Eigen::array<Eigen::Index, 3> starting;
    Eigen::array<Eigen::Index, 3> ending;

    for (int i = 0; i < this->H; i++) {
        aitensor<T> output = M1[i]->forward(heads[i]);

        int start = i * this->H;
        starting  = {0, 0, start};
        ending    = {this->B, this->N, this->H};
        output.slice(starting, ending) = output;
    }

    log_detail( "MultiAttention Forward pass done ..." );

    return output;
}

// Leave the gradients as is. They are cached in the Node. 
// They will be used to update the parameters in next parallel operations.
// the dInput is the gradient we propagate to source Nodes in the graph;
// while the parameter gradients get cached to be used to update the parameters later.
template <class T>
const aitensor<T> MultiHeadAttention<T>::backward(const aitensor<T>& gradients) { 

    log_info("================================================");
    log_info( "Entering Multi-Head Attention Backward Pass ..." );

    this->Dk = this->M / this->H;

    log_detail( "Size of DK ...{:d}" , Dk );

    aitensor<T> dInput(this->B, this->N, this->M);

    std::vector<aitensor<T>> heads = feature_slice(gradients, this->H);

    Eigen::array<Eigen::Index, 3> starting;
    Eigen::array<Eigen::Index, 3> ending;

    for (int i = 0; i < this->H; i++) {
        aitensor<T> input = M1[i]->backward(heads[i]);

        int start = i * this->H;
        starting  = {0, 0, start};
        ending    = {this->B, this->N, this->H};
        dInput.slice(starting, ending) = input;
    }

    return dInput;
}

template <class T>
void MultiHeadAttention<T>::updateParameters(std::string& optimizertype, T& learningRate, int& iter) {

    log_info("=====================================================");
    log_info( "Entering Multi-Head Attention Update Parameters ..." );

    for (int i = 1; i < this->H; i++) {
        M1[i]->updateParameters(optimizertype, learningRate, iter);
    }
}

template <class T>
std::string MultiHeadAttention<T>::generateDotFormat() {   
    std::string dot = "{* MultiHeadAttention *}|";  
    int cnt = 1;
    for (int i = 1; i < this->H; i++) {
        dot +=  M1[i]->generateDotFormat();
        if (++cnt < (int) this->H) { dot += "|"; }
    }
    return dot; 
}

/*****************************************************************************************************
* Base FeedForward  Layer
*****************************************************************************************************/

// While the parameter weight has dimension MxW,  the resulting transformation has dimension of NxW.
// We only need the M dimension from an NxM input to generate parameter matrix.
// where weights is NxW and bias is W.
template <class T>
const aitensor<T> FeedForward<T>::forward(const aitensor<T>& input_data) { 

    log_info("=====================================================");
    log_info( "Entering Feedforward Forward Pass ..." );

    // Cache for later back propagation.
    this->input_data = input_data;

    log_detail( "Input Result" );
    log_matrix( input_data );

    this->B  = input_data.dimension(0);
    this->N  = input_data.dimension(1);
    this->M  = input_data.dimension(2);

    if (L1 == nullptr || L2 == nullptr || A1 == nullptr) {
        L1 = new Linear<T>(this->W, bias);
        L2 = new Linear<T>(this->M, bias); // requires to have dimension as the feedforward input
        A1 = new Activation<T>(this->activationtype, this->alpha);
    }

    // Perform Linear Transformation.
    L1out = L1->forward(input_data);  // Cache output for use by backward activation later
    aitensor<T> A1out = A1->forward(L1out);
    aitensor<T> output = L2->forward(A1out);

    log_detail( "Output Result" );
    log_matrix( output );

    return output;
}

// Leave the gradients as is. They are cached in the Node. 
// They will be used to update the parameters in next parallel operations.
// the dInput is the gradient we propagate to source Nodes in the graph;
// while the parameter gradients get cached to be used to update the parameters later.
template <class T>
const aitensor<T> FeedForward<T>::backward(const aitensor<T>& gradients) { 
    // Propagate Gradient to feedforward operations.

    log_info("=======================================");
    log_info( "Entering Feedforward Backward Pass ..." );

    aitensor<T> L2gradients = L2->backward(gradients); 
    aitensor<T> A1gradients = A1->backward(L2gradients);
    aitensor<T> dInput = L1->backward(A1gradients);

    log_detail( "dInput" );
    log_matrix( dInput );

    return dInput;
}

template <class T>
void FeedForward<T>::updateParameters(std::string& optimizertype, T& learningRate, int& iter) {

    log_info("==========================================");
    log_info( "Entering Feedforward Update Paramter ..." );

    L1->updateParameters(optimizertype, learningRate, iter);
    L2->updateParameters(optimizertype, learningRate, iter);

}

template <class T>
std::string FeedForward<T>::generateDotFormat() {
    std::string dot = "{* FeedForward *}|";  
    if (L1 == nullptr || L2 == nullptr || A1 == nullptr) {
        dot += "{- No training done yet -}";
    } else {
        dot +=  L1->generateDotFormat("L1") + "|";
        dot +=  A1->generateDotFormat() + "|"; 
        dot +=  L2->generateDotFormat("L2"); 
    }
    return dot; 
}

/*****************************************************************************************************
* Base Encoder  Layer
*****************************************************************************************************/

// While the parameter weight has dimension BxMxW,  the resulting transformation has dimension of BxNxW.
// We only need the M dimension from an NxM input to generate parameter matrix.
// where weights is BxNxW and bias is W.
template <class T>
const aitensor<T> Encoder<T>::forward(const aitensor<T>& input_data) { 

    log_info("=====================================================");
    log_info( "Entering Encoder Forward Pass ..." );

    // Cache for later back propagation.
    this->input_data = input_data;

    this->B  = input_data.dimension(0);
    this->N  = input_data.dimension(1);
    this->M  = input_data.dimension(2);

    log_detail( "Size of input: {:d}", this->input_data.size() );

    if (M1 == nullptr || LN1 == nullptr || F1 == nullptr || LN2 == nullptr) {
        LN2 = new LayerNorm<T>(this->W);
        F1  = new FeedForward<T>(this->W, this->bias, this->activationtype,  this->alpha);
        LN1 = new LayerNorm<T>(this->W); 
        M1  = new MultiHeadAttention<T>(this->H, this->W);
    }

    // Perform Linear Transformation.
    M1out = M1->forward(input_data);

    log_detail( "Input Data ...." );
    log_matrix( input_data  );

    log_detail( "Encoder Attention forward output ..." );
    log_matrix( M1out );

    aitensor<T> InputM1out = M1out + input_data;

    log_detail( "Encoder Add 1 forward output ..." );
    log_matrix( InputM1out  );

    LN1out = LN1->forward(InputM1out);

    log_detail( "Encoder LN1 forward output ..." );
    log_matrix( LN1out  );

    F1out = F1->forward(LN1out);

    log_detail( "Encoder FeedForward forward output ..." );
    log_matrix( F1out  );

    aitensor<T> LN1F1out = F1out + LN1out;

    log_detail( "Encoder Add 2 forward output ..." );
    log_matrix( LN1F1out  );

    aitensor<T> output = LN2->forward(LN1F1out);

    log_detail( "Encoder LN2 forward output ..." );
    log_matrix( output  );

    return output;
}

// Leave the gradients as is. They are cached in the Node. 
// They will be used to update the parameters in next parallel operations.
// the dInput is the gradient we propagate to source Nodes in the graph;
// while the parameter gradients get cached to be used to update the parameters later.
template <class T>
const aitensor<T> Encoder<T>::backward(const aitensor<T>& gradients) { 

    log_info("=====================================================");
    log_info( "Entering Encoder Backward Pass ..." );

    log_detail( "Entering Encoder backpropagation ..." );
    // Propagate Gradient to Encoder.
    aitensor<T>  LN2gradients = LN2->backward(gradients); 

    log_detail( "Encoder LN2 backprop output ..." );
    log_matrix( LN2gradients );

    aitensor<T> F1LN2gradients = LN2gradients; // F1out.array() * LN2gradients.array();

    aitensor<T> F1gradients = F1->backward(F1LN2gradients);

    log_detail( "Encoder F1 backprop output ..." );
    log_matrix( F1gradients );

    aitensor<T> InputLN2gradients = LN2gradients; // LN1out.array() * LN2gradients.array();

    log_detail( "Encoder input * F1 backprop output ..." );
    log_matrix( InputLN2gradients );

    F1gradients = InputLN2gradients + F1gradients;

    log_detail( "Encoder (InputLN1gradients + F1gradients) backprop output ..." );
    log_matrix( F1gradients );

    aitensor<T>  LN1gradients = LN1->backward(F1gradients);

    log_detail( "Encoder LN1 backprop output ..." );
    log_matrix( LN1gradients );

    aitensor<T> M1LN1gradients = LN1gradients; // A1out.array() * LN1gradients.array();

    aitensor<T>  M1gradients =  M1->backward(M1LN1gradients);

    log_detail( "Encoder A1 backprop output ..." );
    log_matrix( M1gradients );

    aitensor<T> LN1outLN1gradients = LN1gradients; // input_data.array() * LN1gradients.array();

    aitensor<T> dInput = LN1outLN1gradients + M1gradients;      

    log_detail( "Encoder dInput ..." );
    log_matrix( dInput );

    return dInput;
}  

template <class T>
void Encoder<T>::updateParameters(std::string& optimizertype, T& learningRate, int& iter) {

    log_info("=====================================================");
    log_info( "Entering Encoder Update Parameter ..." );

    LN1->updateParameters(optimizertype, learningRate, iter);
    LN2->updateParameters(optimizertype, learningRate, iter);
    F1->updateParameters(optimizertype, learningRate, iter);
    M1->updateParameters(optimizertype, learningRate, iter);

}

template <class T>
std::string Encoder<T>::generateDotFormat() {
    std::string dot = "{* Encoder *}|";

    if (M1 == nullptr || LN1 == nullptr || F1 == nullptr || LN2 == nullptr)  {
        dot += "{- No training done yet -}";
    } else {
        dot +=  M1->generateDotFormat() + "|";
        dot +=  LN1->generateDotFormat("add_norm1") + "|";
        dot +=  F1->generateDotFormat() + "|";
        dot +=  LN2->generateDotFormat("add_norm2");
    }
    return dot; 
}

/************ Attention & Transformers initialize template ************/

template class Attention<float>;  // Instantiate with float
template class Attention<double>;  // Instantiate with double

template class FeedForward<float>;  // Instantiate with float
template class FeedForward<double>;  // Instantiate with double

template class MultiHeadAttention<float>;  // Instantiate with float
template class MultiHeadAttention<double>;  // Instantiate with double

template class Encoder<float>;  // Instantiate with float
template class Encoder<double>;  // Instantiate with double
