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
const aitensor<T> Attention<T>::forward(const aitensor<T>& input_data, const aitensor<T>& encoder_data) { 
 
    log_info("=====================================");
    log_info( "Entering Attention Forward Pass ..." );

    // Cache for later back propagation.
    this->input_data = input_data;

    if (encoder_data.size() != 0) {
        this->encoder_data = encoder_data;
    }

    // dimension is BxNxW
    this->B = input_data.size();
    this->N = input_data.at(0).rows();
    this->M = input_data.at(0).cols(); // M dimension from input will be the same dimension as output (Wo)

    this->Dk = this->M / this->H;

    log_detail( "Size of input:", input_data.size() );

    if (Q == nullptr || K == nullptr || V == nullptr || Wo == nullptr) {
        Q  = new Linear<T>(this->W, false);  
        K  = new Linear<T>(this->W, this->bias);  
        V  = new Linear<T>(this->W, this->bias);  
        Wo = new Linear<T>(this->M, this->bias);
        if (this->masked) { // Apply Mask only for first multihead of decoder in transformer.
            T negative_infinity = -std::numeric_limits<T>::infinity();
            this->Mask = aimatrix<T>::Zero(this->N, this->N);    // Mask Kernel (NxN)
            for (int i = 0; i < this->N; i++) {
                for (int j = i+1; j < this->N; j++) {
                    this->Mask(i,j) = negative_infinity;
                }
            }
        }
        log_detail("Mask Kernel:");
        log_matrix(this->Mask);
    }

    // Perform Linear Transformation.
    log_info( "Q Linear forward pass ..." );

    // if we expect an encoder_data, then the data will be transformed to K and V
    // Otherwise, the decoder_data will assume K and V.
    this->Qout = Q->forward(input_data); 

    if (encoder_data.size() != 0) { 
        this->Kout = K->forward(encoder_data);
        this->Vout = V->forward(encoder_data);
    } else {
        this->Kout = K->forward(input_data);
        this->Vout = V->forward(input_data);
    }

    log_detail( "Q Linear output" );
    log_matrix( this->Qout );

    log_detail( "K Linear output" );
    log_matrix( this->Kout );

    log_detail( "V Linear output" );
    log_matrix( this->Vout );

    aitensor<T> voutputs; // (this->B, this->N, this->W); // Dimension: BxNxW

    aimatrix<T> mQout, mKout, mVout, QKscore, QKscale, mQKweight, mQKweightV, QKweightV;

    log_detail( "Proceeding with main Attention steps (matmul, scale, softmax) ..." );

    for (int i = 0; i < this->B; ++i) {

        mQout = this->Qout.at(i); // NxW
        mKout = this->Kout.at(i); // NxW
        mVout = this->Vout.at(i); // NxW

        log_detail( "dimension of mQout {0}x{1} at sequence {2}", mQout.rows(), mQout.cols(), i);

        // Computing the Attention Score - MatMul (QK^T)
        QKscore = BaseOperator::matmul(mQout, mKout.transpose()); // NxN

        log_detail( "QK matmul" );
        log_matrix( QKscore );

        // Scale sqrt(Dk)
        QKscale = QKscore.array() / sqrt(this->Dk);

        log_detail( "Scaling -> QKscore / sqrt(Dk)" );
        log_matrix( QKscale );

        // Apply Mask Kernel if required (for Decoder)
        if (this->masked) {
            QKscale += this->Mask;
        }

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
    aitensor<T>  QdInput, KdInput, VdInput, dInput, encoder_gradient;

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

    for (int i = 0; i < this->B; ++i) {
        if (this->encoder_data.size() != 0) {
            encoder_gradient.push_back(KdInput.at(i) + VdInput.at(i));
            dInput.push_back(QdInput.at(i));
        } else {
            dInput.push_back(QdInput.at(i) +  KdInput.at(i) + VdInput.at(i));
        }
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

    encoder_gradient.clear();
}

template <class T>
std::string Attention<T>::generateDotFormat(const std::string& name , bool operators, bool weights) { 
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
const aitensor<T> MultiHeadAttention<T>::forward(const aitensor<T>& input_data, const aitensor<T>& encoder_data) { 

    log_info("===============================================");
    log_info( "Entering Multi-Head Attention Forward Pass ..." );

    // Cache for later back propagation.
    this->input_data = input_data;

    if (encoder_data.size() != 0) {
        this->encoder_data = encoder_data;
    }


    // dimension is BxNxW
    this->B = input_data.size();
    this->N = input_data.at(0).rows();
    this->M = input_data.at(0).cols();

    this->Dk = this->M / this->H;

    log_detail( "Size of input: {:d}" , input_data.size() );
    log_detail( "Size of Head: {:d}", this->H );

    if (M1.empty()) {
        for (int i = 0; i < this->H; i++) {
            Attention<T>* A1  = new Attention<T>(this->W, this->bias, this->masked); // specify weight size.
            M1.push_back(A1);
        }
    }

    log_detail( "Size of DK ...{:d}" , Dk );

    aitensor<T> output; 

    std::vector<aitensor<T>> decoder_heads = head_split(input_data, this->H);

    std::vector<aitensor<T>> encoder_heads;

    if (this->encoder_data.size() != 0) {
        encoder_heads = head_split(encoder_data, this->H);
    }

    aitensor<T> decoder_head;
    for (int i = 0; i < this->H; i++) {
        log_detail( "Multi Attention Forward at split ({0}) ...", i );

        if (encoder_data.size() != 0) {
            decoder_head = M1[i]->forward(decoder_heads[i], encoder_heads[i]);
        } else {
            decoder_head = M1[i]->forward(decoder_heads[i]);
        }
        if (i == 0) {
            for (int j = 0; j < this->B; j++) {
                output.push_back(decoder_head.at(j));
            }
        } else {
            for (int j = 0; j < this->B; j++) {
                aimatrix<T> C(output.at(j).rows(), output.at(j).cols() + decoder_head.at(j).cols());
                C << output.at(j), decoder_head.at(j);
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

    aitensor<T> dInput, encoder_gradient;

    aitensor<T> head;
    aitensor<T> encoder_head;
    std::vector<aitensor<T>> heads = head_split(gradients, this->H);

    for (int i = 0; i < this->H; i++) {
        log_detail( "Multi Attention Backward at split ({0}) ...", i );
        head = M1[i]->backward(heads[i]);
        // if we have an encoder input, then take care of the gradient also.
        if (this->encoder_data.size() != 0) {
            encoder_head = M1[i]->getEncoderGradient();
        }
        // Let us assemble  / join back all the heads.
        if (i == 0) {
            for (int j = 0; j < this->B; j++) {
                dInput.push_back(head.at(j));
                if (this->encoder_data.size() != 0) {
                    encoder_gradient.push_back(encoder_head.at(j));
                }
            }
        } else {
            for (int j = 0; j < this->B; j++) {
                aimatrix<T> C(dInput.at(j).rows(), dInput.at(j).cols() + head.at(j).cols());
                C << dInput.at(j), head.at(j);
                dInput.at(j) = C;

                if (this->encoder_data.size() != 0) {
                    aimatrix<T> C(encoder_gradient.at(j).rows(),encoder_gradient.at(j).cols() + encoder_head.at(j).cols());
                    C << encoder_gradient.at(j), encoder_head.at(j);
                    encoder_gradient.at(j) = C;
                }

            }
        }
    }

    this->encoder_gradient = encoder_gradient;

    return dInput;
}

template <class T>
void MultiHeadAttention<T>::updateParameters(std::string& optimizertype, T& learningRate, int& iter) {

    log_info("=====================================================");
    log_info( "Entering Multi-Head Attention Update Parameters ..." );

    for (int i = 0; i < this->H; i++) {
        M1[i]->updateParameters(optimizertype, learningRate, iter);
    }
    encoder_gradient.clear();
}

template <class T>
std::string MultiHeadAttention<T>::generateDotFormat(const std::string& name , bool operators, bool weights) {
    std::string dot = "{* MultiHeadAttention (" + name + ") *}|";  
    for (int i = 0; i < this->H; i++) {
        dot += "{* Head " + std::to_string(i) + " *}|";  
        dot +=  M1[i]->generateDotFormat() + "|";
    }
    dot.pop_back(); // Remove the last comma
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

    if (this->W == 0) {
        this->W = input_data.at(0).cols();
    }

    if (L1 == nullptr || L2 == nullptr || A1 == nullptr) {
        L1 = new Linear<T>(this->W, true);
        A1 = new Activation<T>(this->activationtype, this->alpha);
        L2 = new Linear<T>(this->M, true); // requires to have dimension as the feedforward input
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
std::string FeedForward<T>::generateDotFormat(const std::string& name , bool operators, bool weights) {
    std::string dot = "{* FeedForward (" + name + ") *}|";  
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
* Base Encoder Layer
*****************************************************************************************************/

// While the parameter weight has dimension BxMxW,  the resulting transformation has dimension of BxNxW.
// We only need the M dimension from an NxM input to generate parameter matrix.
// where weights is BxNxW and bias is W.
template <class T>
const aitensor<T> Encoder<T>::forward(const aitensor<T>& input_data) { 

    log_info("=====================================================");
    log_info( "Entering Encoding  ..." );

    // Cache for later back propagation.
    this->input_data = input_data;

    // dimension is BxNxW
    this->B = input_data.size();
    this->N = input_data.at(0).rows();
    // this->M = input_data.at(0).cols();

    log_detail( "Size of input: {:d}", this->input_data.size() );

    if (M1 == nullptr || LN1 == nullptr || F1 == nullptr || LN2 == nullptr) {
        M1  = new MultiHeadAttention<T>(this->H, this->W, this->F, this->bias, false);
        LN1 = new LayerNorm<T>(); 
        F1  = new FeedForward<T>(this->F, this->bias, this->activationtype,  this->alpha);
        LN2 = new LayerNorm<T>();
    }

    // Perform MultiHead Operation.
    M1out = M1->forward(input_data);

    log_detail( "Input Data ...." );
    log_matrix( input_data  );

    log_detail( "Encoder Attention forward output (M1out)..." );
    log_matrix( M1out );

    aitensor<T> InputM1out;
    
    // Add the residual from previous input
    for (int i = 0; i < (int) this->B; i++) {
        InputM1out.push_back( M1out.at(i) + input_data.at(i) );
    }

    log_detail( "Encoder M1 forward output ..." );
    log_matrix( InputM1out  );

    log_detail( "Encoder LN1 forward pass ..." );
    LN1out = LN1->forward(InputM1out);

    log_detail( "Encoder LN1 forward output ..." );
    log_matrix( LN1out  );


    // Perform FeedForward Operation.
    log_detail( "Encoder F1 forward pass ..." );
    F1out = F1->forward(LN1out);

    log_detail( "Encoder FeedForward forward output ..." );
    log_matrix( F1out  );

    aitensor<T> LN1F1out;

    // Add the residual from previous input
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

    aitensor<T> dF1 = F1->backward(dLN2);  // (F1LN2gradients);  // dF1

    log_detail( "Encoder F1 backprop output (dF1) ..." );
    log_matrix( dF1 );

    log_detail( "Encoder input * F1 backprop output ..." );

    // Add the gradients from F1 and gradient from L1
    for (int i = 0; i < (int) gradients.size(); i++) {
        dF1.at(i) = dLN2.at(i) + dF1.at(i);
    }

    log_detail( "Encoder (dLN2 + dF1) backprop output ..." );
    log_matrix( dF1 );

    aitensor<T>  dLN1 = LN1->backward(dF1);

    log_detail( "Encoder LN1 backprop output (dLN1) ..." );
    log_matrix( dLN1 );

    aitensor<T>  dM1 =  M1->backward(dLN1);

    log_detail( "Encoder M1 backprop output (dM1) ..." );
    log_matrix( dM1 );

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
std::string Encoder<T>::generateDotFormat(const std::string& name , bool operators, bool weights) {
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

// where weights is BxNxW and bias is W.
template <class T>
const aitensor<T> EncoderLayer<T>::forward(const aitensor<T>& input_data) { 

    log_info("=====================================================");
    log_info( "Entering EncoderLayer Forward Pass ..." );

    aitensor<T> output = input_data;

    // Perform Forward Pass for each layer
    for (int i = 0; i < this->L; i++) {
        Encoder<T> encoder = this->encoders.at(i);
        output = encoder.forward(output);
    }
    return output;
}

// where weights is BxNxW and bias is W.
template <class T>
const aitensor<T> EncoderLayer<T>::backward(const aitensor<T>& gradients) { 

    log_info("=====================================================");
    log_info( "Entering EncoderLayer Backward Pass ..." );

    aitensor<T> dInput = gradients;

    // Perform Backward Prop for each layer
    for (int i = 0; i < this->L; i++) {
        Encoder<T> encoder = this->encoders.at(i);
        dInput = encoder.backward(dInput);
    }
    return dInput;
}

// where weights is BxNxW and bias is W.
template <class T>
void EncoderLayer<T>::updateParameters(std::string& optimizertype, T& learningRate, int& iter) {

    log_info("=====================================================");
    log_info( "Entering EncoderLayer Update Parameter ..." );

    // Perform Parameter Update for each layer
    for (int i = 0; i < this->L; i++) {
        Encoder<T> encoder = this->encoders.at(i);
        encoder.updateParameters(optimizertype, learningRate, iter);
    }
}

template <class T>
std::string EncoderLayer<T>::generateDotFormat(const std::string& name , bool operators, bool weights) {
    std::string dot = "{* Encoder Layer *}|";

    for (int i = 0; i < this->L; i++) {
        Encoder<T> encoder = this->encoders.at(i);
        dot += encoder.generateDotFormat("encoder " + i);
    }

    return dot; 
}

/*****************************************************************************************************
* Base Decoder  Block
*****************************************************************************************************/

// While the parameter weight has dimension BxMxW,  the resulting transformation has dimension of BxNxW.
// We only need the M dimension from an NxM input to generate parameter matrix.
// where weights is BxNxW and bias is W.
template <class T>
const aitensor<T> Decoder<T>::forward(const aitensor<T>& decoder_data, const aitensor<T>& encoder_data) { 

    log_info("=====================================================");
    log_info( "Entering Decoder Forward Pass ..." );

    // Cache for later back propagation.
    this->input_data = decoder_data;

    // dimension is BxNxW
    this->B = this->input_data.size();
    this->N = this->input_data.at(0).rows();
    // this->M = this->input_data.at(0).cols();

    log_detail( "Size of input: {:d}", this->input_data.size() );

    if (M1 == nullptr || M2 == nullptr || LN1 == nullptr || F1 == nullptr || LN2 == nullptr) {
        M1  = new MultiHeadAttention<T>(this->H, this->W, this->F,  this->bias, true); // masked is set.
        LN1 = new LayerNorm<T>(); 
        M2  = new MultiHeadAttention<T>(this->H, this->W, this->F, this->bias, false);
        LN2 = new LayerNorm<T>(); 
        F1  = new FeedForward<T>(this->F, this->bias, this->activationtype,  this->alpha);
        LN3 = new LayerNorm<T>();
    }

    // Perform First MultiHead Operation.
    M1out = M1->forward(this->input_data);

    log_detail( "Input Data ...." );
    log_matrix( this->input_data  );

    log_detail( "Encoder Attention forward output (M1out)..." );
    log_matrix( M1out );

    aitensor<T> InputM1out;
    
    // Add the residual from previous input
    for (int i = 0; i < (int) this->B; i++) {
        InputM1out.push_back( M1out.at(i) + input_data.at(i) );
    }

    log_detail( "Decoder M1 forward output ..." );
    log_matrix( InputM1out  );

    log_detail( "Decoder LN1 forward pass ..." );
    LN1out = LN1->forward(InputM1out);

    log_detail( "Decoder LN1 forward output ..." );
    log_matrix( LN1out  );


    // Perform Second MultiHead Operation.
    M2out = M2->forward(input_data, encoder_data);

    log_detail( "Input Data ...." );
    log_matrix( input_data  );

    log_detail( "Decoder Attention forward output (M1out)..." );
    log_matrix( M1out );

    aitensor<T> InputM2out;
    
    // Add the residual from previous input
    for (int i = 0; i < (int) this->B; i++) {
        InputM2out.push_back( M2out.at(i) + LN1out.at(i) );
    }

    log_detail( "Decoder M2 forward output ..." );
    log_matrix( InputM2out  );

    log_detail( "Decoder LN2 forward pass ..." );
    LN2out = LN2->forward(InputM2out);

    log_detail( "Decoder LN2 forward output ..." );
    log_matrix( LN2out  );

    // Perform FeedForward Operation.
    log_detail( "Decoder F1 forward pass ..." );
    F1out = F1->forward(LN2out);

    log_detail( "Decoder FeedForward forward output ..." );
    log_matrix( F1out  );

    aitensor<T> LN2F1out;

    for (int i = 0; i < (int) this->B; i++) {
        LN2F1out.push_back( F1out.at(i) + LN2out.at(i) );
    }

    log_detail( "Decoder F1 forward output ..." );
    log_matrix( LN2F1out  );

    log_detail( "Decoder LN2 forward pass ..." );
    aitensor<T> output = LN3->forward(LN2F1out);

    return output;
}

// Leave the gradients as is. They are cached in the Node. 
// They will be used to update the parameters in next parallel operations.
// the dInput is the gradient we propagate to source Nodes in the graph;
// while the parameter gradients get cached to be used to update the parameters later.
template <class T>
const aitensor<T> Decoder<T>::backward(const aitensor<T>& gradients) { 

    log_info("=====================================================");
    log_info( "Entering Decoder Backward Pass ..." );

    log_detail( "Entering Decoder backpropagation ..." );
    // Propagate Gradient to Encoder.
    aitensor<T>  dLN3 = LN3->backward(gradients);  // dLN2

    log_detail( "Decoder LN3 backprop output (dLN3) ..." );
    log_matrix( dLN3 );

    aitensor<T> dF1 = F1->backward(dLN3); 

    log_detail( "Decoder F1 backprop output (dF1) ..." );
    log_matrix( dF1 );

    log_detail( "Decoder input * F1 backprop output ..." );


    for (int i = 0; i < (int) gradients.size(); i++) {
        dF1.at(i) = dLN3.at(i) + dF1.at(i);
    }

    log_detail( "Decoder (dLN2 + dF1) backprop output ..." );
    log_matrix( dF1 );

    aitensor<T>  dLN2 = LN2->backward(dF1);

    log_detail( "Decoder LN1 backprop output (dLN2) ..." );
    log_matrix( dLN2 );

    aitensor<T>  dM2 =  M2->backward(dLN2);

    // Retrieve the gradient for the encoder output.
    this->encoder_gradient = M2->getEncoderGradient();

    log_detail( "Encoder M1 backprop output (dM2) ..." );
    log_matrix( dM2 );

    for (int i = 0; i < (int) gradients.size(); i++) {
        dM2.push_back( dLN2.at(i) + dM2.at(i) );
    }

    aitensor<T>  dLN1 = LN1->backward(dM2);

    log_detail( "Decoder LN1 backprop output (dLN1) ..." );
    log_matrix( dLN1 );

    aitensor<T>  dM1 =  M1->backward(dLN1);

    log_detail( "Encoder M1 backprop output (dM1) ..." );
    log_matrix( dM1 );

    aitensor<T> dInput;
    
    for (int i = 0; i < (int) gradients.size(); i++) {
        dInput.push_back( dLN1.at(i) + dM1.at(i) );
    }
       
    return dInput;
}  
  
template <class T>
void Decoder<T>::updateParameters(std::string& optimizertype, T& learningRate, int& iter) {

    log_info("=====================================================");
    log_info( "Entering Decoder Update Parameter ..." );

    log_detail("LN3 Linear parameter update");
    LN3->updateParameters(optimizertype, learningRate, iter);

    log_detail("F1 Linear parameter update");
    F1->updateParameters(optimizertype, learningRate, iter);

    log_detail("LN2 Linear parameter update");
    LN2->updateParameters(optimizertype, learningRate, iter);

    log_detail("M1 Linear parameter update");
    M2->updateParameters(optimizertype, learningRate, iter);

    log_detail("LN1 Linear parameter update");
    LN1->updateParameters(optimizertype, learningRate, iter);

    log_detail("M1 Linear parameter update");
    M1->updateParameters(optimizertype, learningRate, iter);

}

template <class T>
std::string Decoder<T>::generateDotFormat(const std::string& name , bool operators, bool weights) {
    std::string dot = "{* Decoder *}|";

    if (M1 == nullptr || LN1 == nullptr || F1 == nullptr || LN2 == nullptr)  {
        dot += "{- No training done yet -}";
    } else {
        dot +=  M1->generateDotFormat("Masked") + "|";
        dot +=  LN1->generateDotFormat("add_norm1") + "|";
        dot +=  M2->generateDotFormat() + "|";
        dot +=  LN2->generateDotFormat("add_norm2") + "|";
        dot +=  F1->generateDotFormat() + "|";
        dot +=  LN3->generateDotFormat("add_norm3");
    }
    return dot; 
}

// where weights is BxNxW and bias is W.
template <class T>
const aitensor<T> DecoderLayer<T>::forward(const aitensor<T>& decoder_data, const aitensor<T>& encoder_data) { 

    log_info("=====================================================");
    log_info( "Entering DecoderLayer Forward Pass ..." );

    aitensor<T> output = decoder_data;

    // Perform Forward Pass for each layer
    for (int i = 0; i < this->L; i++) {
        Decoder<T> decoder = this->decoders.at(i);
        output = decoder.forward(output, encoder_data);
    }
    return output;
}

// where weights is BxNxW and bias is W.
template <class T>
const aitensor<T> DecoderLayer<T>::backward(const aitensor<T>& gradients) { 

    log_info("=====================================================");
    log_info( "Entering DecoderLayer Backward Pass ..." );

    aitensor<T> decoder_gradient = gradients;
    aitensor<T> encoder_gradient, dInput = {};

    // Perform Backward Prop for each layer
    for (int i = 0; i < this->L; i++) {
        Decoder<T> decoder = this->decoders.at(i);
        decoder_gradient = decoder.backward(decoder_gradient);
        encoder_gradient = decoder.getEncoderGradient();
        if (dInput.size() == 0) {
            dInput = encoder_gradient;
        } else {
            int enc_gradient_size = encoder_gradient.size();
            for (int j = 0; j < enc_gradient_size; j++) {
                dInput.at(j).array() += encoder_gradient.at(j).array();
            }
        }
    }
    return dInput;
}

// where weights is BxNxW and bias is W.
template <class T>
void DecoderLayer<T>::updateParameters(std::string& optimizertype, T& learningRate, int& iter) {

    log_info("=====================================================");
    log_info( "Entering DecoderLayer Update Parameter ..." );

    // Perform Parameter Update for each layer
    for (int i = 0; i < this->L; i++) {
        Decoder<T> decoder = this->decoders.at(i);
        decoder.updateParameters(optimizertype, learningRate, iter);
    }
}

template <class T>
std::string DecoderLayer<T>::generateDotFormat(const std::string& name , bool operators, bool weights) {
    std::string dot = "{* Decoder Layer *}|";

    for (int i = 0; i < this->L; i++) {
        Decoder<T> decoder = this->decoders.at(i);
        dot += decoder.generateDotFormat("decoder " + i);
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

template class Decoder<float>;  // Instantiate with float
template class Decoder<double>;  // Instantiate with double

template class EncoderLayer<float>;  // Instantiate with float
template class EncoderLayer<double>;  // Instantiate with double

template class DecoderLayer<float>;  // Instantiate with float
template class DecoderLayer<double>;  // Instantiate with double
 
template class PositionalEncoder<float>;  // Instantiate with float
template class PositionalEncoder<double>;  // Instantiate with double

