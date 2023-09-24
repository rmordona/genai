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
#include "logger.h"
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

    // dimension is BxNxW
    this->B = input_data.size();
    this->N = input_data.at(0).rows();
    this->M = input_data.at(0).cols(); // M dimension from input will be the same dimension as output (Wo)

    this->Dk = this->M / this->H;

    log_detail( "Size of input:", input_data.size() );

    if (Q == nullptr || K == nullptr || V == nullptr || Wo == nullptr) {
        Q  = new Linear<T>(this->W, this->bias);
        K  = new Linear<T>(this->W, this->bias);
        V  = new Linear<T>(this->W, this->bias);
        Wo = new Linear<T>(this->M, this->bias);
    }

    // Perform Linear Transformation.
    log_info( "Q Linear forward pass ..." );
    this->Qout = Q->forward(input_data);
    log_detail( "Q Linear output" );
    log_matrix( this->Qout );

    log_info( "K Linear forward pass ..." );
    this->Kout = K->forward(input_data);
    log_detail( "K Linear output" );
    log_matrix( this->Kout );

    log_info( "V Linear forward pass ..." );
    this->Vout = V->forward(input_data);
    log_detail( "V Linear output" );
    log_matrix( this->Vout );

    aitensor<T> voutputs; // (this->B, this->N, this->W); // Dimension: BxNxW

    aimatrix<T> mQout, mKout, mVout, QKscore, QKscale, mQKweight, mQKweightV, QKweightV;

    for (int i = 0; i < this->B; ++i) {

        mQout = this->Qout.at(i); // NxW
        mKout = this->Kout.at(i); // NxW
        mVout = this->Vout.at(i); // NxW

        log_detail("dimension of mQout {0}x{1}", mQout.rows(), mQout.cols());

        // Computing the Attention Score - MatMul (QK^T)
        QKscore = BaseOperator::matmul(mQout, mKout.transpose()); // NxN

        log_detail( "QK matmul" );
        log_matrix( QKscore );

        // Include some Masking (still to be implemented)

        // Scale sqrt(Dk)
        QKscale = QKscore.array() / sqrt(this->Dk);

        log_detail( "Scaling -> QKscore / sqrt(Dk)" );
        log_matrix( QKscale );

        // Mask if required (for Decoder)
        // ...

        // Computing the Attention WeightPerform via Softmax
        mQKweight = BaseOperator::softmax(QKscale);

        log_detail( "Attention Weight mQKweight output" );
        log_matrix(  mQKweight );

        // Cache to be used by Back propagation
        this->QKweight.push_back( mQKweight );

        // Include dropout (still to be implemented)

        // Perform matmul with V
        QKweightV = BaseOperator::matmul(mQKweight, mVout);

        log_detail( "Matmul of weight * V (QKweightV)" );
        log_matrix(  QKweightV );

        voutputs.push_back(QKweightV);
    
    }

    // Perform another transform to align dimension.
    log_info( "Wo Linear forward pass ..." );
    aitensor<T> output = Wo->forward(voutputs);
    log_detail( "Wo Linear output" );
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

    aimatrix<T>  mdWo, mKout, mQout, mVout, mQKweight;
    aitensor<T>  QdInput, KdInput, VdInput, dInput;

    // Gradient with Respect to W linear operations.
    aitensor<T>  dWo = Wo->backward(gradients); 

    log_detail( "Wo gradient" );
    log_matrix(dWo);

    for (int i = 0; i < this->B; ++i) {

        mdWo      = dWo.at(i);
        mKout     = Kout.at(i); 
        mQout     = Qout.at(i); 
        mVout     = Vout.at(i); 
        mQKweight = QKweight.at(i); 

        // Gradient with Respect to Result of softmax  
        log_detail("Gradient with respect to QKweight");
        aimatrix<T> dQKweight = BaseOperator::matmul(mdWo, mVout.transpose());

        log_detail( "dQKweight" );
        log_matrix( dQKweight );

        // Gradient with Respect to V (matmul operation)
        log_detail("Gradient with respect to vOut");
        aimatrix<T> d_V = BaseOperator::matmul(mQKweight.transpose(), mdWo);

        log_detail( "d_V gradient" );
        log_matrix( d_V );

        // Propagate Gradient to softmax operation.
        aimatrix<T> dQKscale = BaseOperator::softmaxGradient(dQKweight, mQKweight);

        log_detail( "dQKscale gradient" );
        log_matrix( dQKscale );

        // Calculate the scale.
        aimatrix<T> dScale = -0.5 * dQKscale.array() * (1.0 / std::sqrt(this->Dk));

        log_detail( "dScale gradient" );
        log_matrix( dScale );

        // Gradient with Respect to Q (matmul operation)
        aimatrix<T> d_Q = BaseOperator::matmul(dScale.transpose(), mKout);
        log_detail( "d_Q gradient" );
        log_matrix( d_Q );

        // Gradient with Respect to K (matmul operation)
        aimatrix<T> d_K = BaseOperator::matmul(mQout.transpose(), dScale).transpose();

        log_detail( "d_K gradient" );
        log_matrix( d_K );

        QdInput.push_back(d_Q); 
        KdInput.push_back(d_K); 
        VdInput.push_back(d_V); 

    }

    log_detail( " Perform Backprop for Q, K, V ..." );

    // Propagate Gradient to the Q,K,V linear operations.
    log_detail( "Q Linear backward pass ..." );
    QdInput = Q->backward(QdInput);  // NxM
    log_detail( "K Linear backward pass ..." );
    KdInput = K->backward(KdInput);  // NxM
    log_detail( "V Linear backward pass ..." );
    VdInput = V->backward(VdInput);  // NxM

    log_detail( "VdInput gradient" );
    log_matrix( VdInput );

    log_detail( "QdInput gradient" );
    log_matrix( QdInput );

    log_detail( "KdInput gradient" );
    log_matrix( KdInput );

    // dInput = QdInput.data() + KdInput.data() + VdInput.data(); // BxNxM

    for (int i = 0; i < this->B; ++i) {
        dInput.push_back(QdInput.at(i) +  KdInput.at(i) + VdInput.at(i));
    }

    return dInput;  
}

template <class T>
void Attention<T>::updateParameters(std::string& optimizertype, T& learningRate, int& iter) {

    log_info("=========================================");
    log_info( "Entering Attention Update Parameter ..." );

    log_detail("Q Linear parameter update");
    Q->updateParameters(optimizertype, learningRate, iter);
    log_detail("K Linear parameter update");
    K->updateParameters(optimizertype, learningRate, iter);
    log_detail("V Linear parameter update");
    V->updateParameters(optimizertype, learningRate, iter);
    log_detail("Wo Linear parameter update");
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

    // dimension is BxNxW
    this->B = input_data.size();
    this->N = input_data.at(0).rows();
    this->M = input_data.at(0).cols();

    this->Dk = this->M / this->H;

    log_detail( "Size of input: {:d}" , input_data.size() );
    log_detail( "Size of Head: {:d}", this->H );

    if (M1.empty()) {
        for (int i = 0; i < this->H; i++) {
            Attention<T>* A1  = new Attention<T>(this->W); // specify weight size.
            M1.push_back(A1);
        }
    }

    log_detail( "Size of DK ...{:d}" , Dk );

    aitensor<T> output; 

    std::vector<aitensor<T>> heads = head_split(input_data, this->H);

    for (int i = 0; i < this->H; i++) {
        log_detail( "Multi Attention Forward at split ({0}) ...", i );
        aitensor<T> head = M1[i]->forward(heads[i]);
        if (i == 0) {
            for (int j = 0; j < this->B; j++) {
                output.push_back(head.at(j));
            }
        } else {
            for (int j = 0; j < this->B; j++) {
                aimatrix<T> C(output.at(j).rows(), output.at(j).cols() + head.at(j).cols());
                C << output.at(j), head.at(j);
                output.at(j) = C;
            }
        }
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

    aitensor<T> dInput;

    std::vector<aitensor<T>> heads = head_split(gradients, this->H);

    for (int i = 0; i < this->H; i++) {
        log_detail( "Multi Attention Backward at split ({0}) ...", i );
        aitensor<T> head = M1[i]->backward(heads[i]);
        if (i == 0) {
            for (int j = 0; j < this->B; j++) {
                dInput.push_back(head.at(j));
            }
        } else {
            for (int j = 0; j < this->B; j++) {
                aimatrix<T> C(dInput.at(j).rows(), dInput.at(j).cols() + head.at(j).cols());
                C << dInput.at(j), head.at(j);
                dInput.at(j) = C;
            }
        }
    }

    return dInput;
}

template <class T>
void MultiHeadAttention<T>::updateParameters(std::string& optimizertype, T& learningRate, int& iter) {

    log_info("=====================================================");
    log_info( "Entering Multi-Head Attention Update Parameters ..." );

    for (int i = 0; i < this->H; i++) {
        M1[i]->updateParameters(optimizertype, learningRate, iter);
    }
}

template <class T>
std::string MultiHeadAttention<T>::generateDotFormat() {   
    std::string dot = "{* MultiHeadAttention *}|";  
    int cnt = 1;
    for (int i = 0; i < this->H; i++) {
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

    // dimension is BxNxW
    this->B = input_data.size();
    this->N = input_data.at(0).rows();
    this->M = input_data.at(0).cols();

    if (L1 == nullptr || L2 == nullptr || A1 == nullptr) {
        L1 = new Linear<T>(this->W, bias);
        L2 = new Linear<T>(this->M, bias); // requires to have dimension as the feedforward input
        A1 = new Activation<T>(this->activationtype, this->alpha);
    }

    // Perform Linear Transformation.
    log_detail( "L1 Linear forward pass ..." );
    L1out = L1->forward(input_data);  // Cache output for use by backward activation later
    log_detail( "L1 Linear output" );
    log_matrix( L1out );

    log_detail( "A1 Activation forward pass ..." );
    aitensor<T> A1out = A1->forward(L1out);
    log_detail( "A1 Activation output" );
    log_matrix( A1out );

    log_detail( "L2 Linear forward pass ..." );
    aitensor<T> output = L2->forward(A1out);
    log_detail( "L2 Linear output" );
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

    log_detail( "L2 Linear backward pass ..." );
    aitensor<T> L2gradients = L2->backward(gradients); 
    log_detail( "A1 Activation backward pass ..." );
    aitensor<T> A1gradients = A1->backward(L2gradients);
    log_detail( "L1 Linear backward pass ..." );
    aitensor<T> dInput = L1->backward(A1gradients);

    return dInput;
}

template <class T>
void FeedForward<T>::updateParameters(std::string& optimizertype, T& learningRate, int& iter) {

    log_info("==========================================");
    log_info( "Entering Feedforward Update Paramter ..." );

    log_detail("L1 Linear parameter update");
    L1->updateParameters(optimizertype, learningRate, iter);
    log_detail("L2 Linear parameter update");
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

    // dimension is BxNxW
    this->B = input_data.size();
    this->N = input_data.at(0).rows();
    this->M = input_data.at(0).cols();

    log_detail( "Size of input: {:d}", this->input_data.size() );

    if (M1 == nullptr || LN1 == nullptr || F1 == nullptr || LN2 == nullptr) {
        LN2 = new LayerNorm<T>();
        F1  = new FeedForward<T>(this->W, this->bias, this->activationtype,  this->alpha);
        LN1 = new LayerNorm<T>(); 
        M1  = new MultiHeadAttention<T>(this->H, this->W);
    }

    // Perform Linear Transformation.
    M1out = M1->forward(input_data);

    log_detail( "Input Data ...." );
    log_matrix( input_data  );

    log_detail( "Encoder Attention forward output (M1out)..." );
    log_matrix( M1out );

    aitensor<T> InputM1out;
    
    for (int i = 0; i < (int) this->B; i++) {
        InputM1out.push_back( M1out.at(i) + input_data.at(i) );
    }

    log_detail( "Encoder M1 forward output ..." );
    log_matrix( InputM1out  );

    log_detail( "Encoder LN1 forward pass ..." );
    LN1out = LN1->forward(InputM1out);

    log_detail( "Encoder LN1 forward output ..." );
    log_matrix( LN1out  );

    log_detail( "Encoder F1 forward pass ..." );
    F1out = F1->forward(LN1out);

    log_detail( "Encoder FeedForward forward output ..." );
    log_matrix( F1out  );

    aitensor<T> LN1F1out;

    for (int i = 0; i < (int) this->B; i++) {
        LN1F1out.push_back( F1out.at(i) + LN1out.at(i) );
    }

    log_detail( "Encoder F1 forward output ..." );
    log_matrix( LN1F1out  );

    log_detail( "Encoder LN2 forward pass ..." );
    aitensor<T> output = LN2->forward(LN1F1out);

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
    aitensor<T>  dLN2 = LN2->backward(gradients);  // dLN2

    log_detail( "Encoder LN2 backprop output (dLN2) ..." );
    log_matrix( dLN2 );

    // aitensor<T> F1LN2gradients = LN2gradients; // F1out.array() * LN2gradients.array();

    aitensor<T> dF1 = F1->backward(dLN2);  // (F1LN2gradients);  // dF1

    log_detail( "Encoder F1 backprop output (dF1) ..." );
    log_matrix( dF1 );

    // aitensor<T> InputLN2gradients = LN2gradients; // LN1out.array() * LN2gradients.array();

    log_detail( "Encoder input * F1 backprop output ..." );
    // log_matrix( InputLN2gradients );

    for (int i = 0; i < (int) gradients.size(); i++) {

        dF1.at(i) = dLN2.at(i) + dF1.at(i);

    }

    log_detail( "Encoder (dLN2 + dF1) backprop output ..." );
    log_matrix( dF1 );

    aitensor<T>  dLN1 = LN1->backward(dF1);

    log_detail( "Encoder LN1 backprop output (dLN1) ..." );
    log_matrix( dLN1 );

    // aitensor<T> M1LN1gradients = LN1gradients; // A1out.array() * LN1gradients.array();

    aitensor<T>  dM1 =  M1->backward(dLN1);

    log_detail( "Encoder M1 backprop output (dM1) ..." );
    log_matrix( dM1 );

    // aitensor<T> LN1outLN1gradients = LN1gradients; // input_data.array() * LN1gradients.array();

    aitensor<T> dInput;
    
    for (int i = 0; i < (int) gradients.size(); i++) {

        dInput.push_back( dLN1.at(i) + dM1.at(i) );

    }
       
    return dInput;
}  
 
template <class T>
void Encoder<T>::updateParameters(std::string& optimizertype, T& learningRate, int& iter) {

    log_info("=====================================================");
    log_info( "Entering Encoder Update Parameter ..." );

    log_detail("LN1 Linear parameter update");
    LN1->updateParameters(optimizertype, learningRate, iter);
    log_detail("LN2 Linear parameter update");
    LN2->updateParameters(optimizertype, learningRate, iter);
    log_detail("F1 Linear parameter update");
    F1->updateParameters(optimizertype, learningRate, iter);
    log_detail("M1 Linear parameter update");
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
