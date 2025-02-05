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
 
    this->Dk = this->W; // Each input_data matrix is already treated as one head from multiheader attention.
 
    log_detail( "Size of input:", input_data.size() );

    if (this->Wq == nullptr || this->Wk == nullptr || this->Wv == nullptr) {
        this->Wq  = new Linear<T>(this->W, false);  
        this->Wk  = new Linear<T>(this->W, false);  
        this->Wv  = new Linear<T>(this->W, false);  

        if (this->output_projection) { // if stand-alone self-attention, no multi-head involved.
            this->Wo  = new Linear<T>(this->M, this->bias);
        }

        if (this->masked) { // Apply Mask only for first multihead of decoder in transformer.
            T negative_infinity = -std::numeric_limits<T>::infinity();
            this->Mask = aimatrix<T>::Zero(this->N, this->N);   // Mask Kernel (NxN)
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
    this->Q = Wq->forward(input_data); 

    if (encoder_data.size() != 0) { 
        log_info( "K Linear forward pass (encoder data) ..." );
        this->K = Wk->forward(encoder_data);
        log_info( "V Linear forward pass (encoder data) ..." );
        this->V = Wv->forward(encoder_data);
    } else {
        log_info( "K Linear forward pass (input data) ..." );
        this->K = Wk->forward(input_data);
        log_info( "V Linear forward pass (input data) ..." );
        this->V = Wv->forward(input_data);
    }

    log_detail( "Q Linear output: {0}", this->Q.size() );  // B x N x W
    log_matrix( this->Q[0] );

    log_detail( "K Linear output: {0}", this->K.size()  );  // B x N x W
    log_matrix( this->K[0] );

    log_detail( "V Linear output: {0}" , this->V.size() );  // B x N x W
    log_matrix( this->V[0] );

    aitensor<T> voutputs; // (this->B, this->N, this->W); // Dimension: BxNxW

    aimatrix<T> mQout, mKout, mVout, QKscore, QKscale, mQKweight, mQKweightV, QKweightV;

    log_detail( "Proceeding with main Attention steps (matmul, scale, softmax) ..." );

    for (int i = 0; i < this->B; ++i) {

        mQout = this->Q.at(i); // N x W
        mKout = this->K.at(i); // N x W
        mVout = this->V.at(i); // N x W

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

        // Computing the Attention Weight. Perform via Softmax
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

    if (this->output_projection) {
        // Perform another transform to align dimension.
        log_info( "Wo Linear forward pass ..." );
        aitensor<T> output = Wo->forward(voutputs);
        log_detail( "Wo Linear output" );
        log_matrix( output );
        return output;
    }

    return voutputs; // this becomes input to the next Node or next Layer.
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

    aitensor<T>  dWo;
    if (this->output_projection) {
        // Gradient with Respect to W linear operations.
        dWo = Wo->backward(gradients); 
        log_detail( "Wo gradient" );
        log_matrix(dWo);
    } else {
        dWo = gradients;
    }

    for (int i = 0; i < this->B; ++i) {

        mdWo      = dWo.at(i);
        mKout     = this->K.at(i); 
        mQout     = this->Q.at(i); 
        mVout     = this->V.at(i); 
        mQKweight = this->QKweight.at(i); 

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
        // aimatrix<T> dScale = -0.5 * dQKscale.array() * (1.0 / std::sqrt(this->Dk));
        aimatrix<T> dScale = dQKscale.array() * ( 1.0 / std::sqrt(this->Dk));

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
    QdInput = Wq->backward(QdInput);  // NxM
    log_detail( "K Linear backward pass ..." );
    KdInput = Wk->backward(KdInput);  // NxM
    log_detail( "V Linear backward pass ..." );
    VdInput = Wv->backward(VdInput);  // NxM

    log_detail( "VdInput gradient" );
    log_matrix( VdInput );

    log_detail( "QdInput gradient" );
    log_matrix( QdInput );

    log_detail( "KdInput gradient" );
    log_matrix( KdInput );

    for (int i = 0; i < this->B; ++i) {
        if (this->encoder_data.size() != 0) {
            this->encoder_gradient.push_back(KdInput.at(i) + VdInput.at(i));
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
    Wq->updateParameters(optimizertype, learningRate, iter);
    log_detail("K Linear parameter update");
    Wk->updateParameters(optimizertype, learningRate, iter);
    log_detail("V Linear parameter update");
    Wv->updateParameters(optimizertype, learningRate, iter);

    if (this->output_projection) {
        log_detail("Wo Linear parameter update");
        Wo->updateParameters(optimizertype, learningRate, iter);
    }

    this->encoder_gradient.clear();
    this->QKweight.clear();

}

template <class T>
Topology Attention<T>::generateDotFormat(const std::string& name , bool operators, bool weights) { 
    Topology topology ;
    topology.dot = "{* Attention *}|";  
    topology.parameters = 0;
    if (Wq == nullptr || Wk == nullptr || Wv == nullptr) {
        topology.dot += "{- No training done yet -}";
    } else {
        Topology top1 = Wq->generateDotFormat("Q") ;
        Topology top2 = Wk->generateDotFormat("K") ;
        Topology top3 = Wv->generateDotFormat("V") ;
        topology.dot += top1.dot + "|" + top2.dot + "|" + top3.dot + "|";
        topology.parameters += top1.parameters + top2.parameters + top3.parameters;
        if (this->output_projection) {
            Topology top4 = Wo->generateDotFormat("Wo");
            topology.dot += top4.dot;
            topology.parameters += top4.parameters;
        }
    }
    return topology; 
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

    aitensor<T> attention_score, voutput; 

    // Cache for later back propagation.
    this->input_data = input_data;

    if (encoder_data.size() != 0) {
        this->encoder_data = encoder_data;
    }

    // dimension is BxNxW
    this->B = input_data.size();
    this->N = input_data.at(0).rows();
    this->M = input_data.at(0).cols();

    // Split base on the feature size of the Attention unit, not based on the embedding size of the input.
    // However, typically we keep this->W and this->M the same.
    this->Dk = this->W / this->H;

    log_detail( "Size of input: {0}, Dimension: {1}x{2}" , input_data.size(), this->N, this->M );
    log_detail( "Size of W = {0}, M = {1}, Head: {2}", this->W, this->M, this->H );
    log_detail( "Size of DK ...{:d}" , Dk );


    if (M1.empty() || this->Wo == nullptr) {

        // Split Weights at this stage.
        for (int i = 0; i < this->H; i++) {
            Attention<T>* A1  = new Attention<T>(this->Dk, this->bias, this->masked, false); // specify weight size.
            M1.push_back(A1);
        }

        // Given that we deal with Residuals, it helps for the size of the attention and feedforward to follow
        // the size of the embedding which becomes constant throughout the encoder / decoder.
        // We achieve this using an additional transformer layer that follows the embedding size.
        this->Wo  = new Linear<T>(this->M, this->bias);

    }


    for (int i = 0; i < this->H; i++) {
        log_detail( "Multi Attention Forward at split ({0}) ...", i );

        if (encoder_data.size() != 0) {
            attention_score = M1[i]->forward(this->input_data, encoder_data); // B x N x W
        } else {
            attention_score = M1[i]->forward(this->input_data);               // B x N x W
        }
        if (i == 0) {
            for (int j = 0; j < this->B; j++) {
                voutput.push_back(attention_score.at(j));
            }
        } else {
            for (int j = 0; j < this->B; j++) {
                aimatrix<T> C(voutput.at(j).rows(), voutput.at(j).cols() + attention_score.at(j).cols());
                C << voutput.at(j), attention_score.at(j);
                voutput.at(j) = C;
            }
        }
    }

    // Perform  transform to align dimension.
    log_info( "Wo Linear forward pass ..." );
    aitensor<T> output = Wo->forward(voutput);
    log_detail( "Wo Linear output" );
    log_matrix( output[0] );

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

    aitensor<T> dWo;                // Calculate gradient with respect to Wo
    aitensor<T> decoder_gradient;   // Calculate gradient with respect to Q (for decoder)
    aitensor<T> encoder_gradient;   // Calculate gradient with respect to K & V (for encoder)
    aitensor<T> dInput;             // Calculate gradient with respect to Input

    aitensor<T> encoder_head, attention_head;

    // Gradient with Respect to M linear operations.
    dWo = Wo->backward(gradients);  // BxNxM
    log_detail( "Wo gradient" );
    log_matrix(dWo);

    std::vector<aitensor<T>> hgradients = head_split(dWo, this->H);  // H x B x N x dK

    for (int i = 0; i < this->H; i++) {
        log_detail( "Multi Attention Backward at split ({0}) ...", i );
        attention_head = M1[i]->backward(hgradients[i]);  // B x N x W
        // if we have an encoder input, then take care of the gradient also.
        if (this->encoder_data.size() != 0) {
            encoder_head = M1[i]->getEncoderGradient();
        }
        // Let us assemble  / join back all the heads.
        if (i == 0) {
            for (int j = 0; j < this->B; j++) {
                decoder_gradient.push_back(attention_head.at(j));
                if (this->encoder_data.size() != 0) {
                    encoder_gradient.push_back(encoder_head.at(j));
                }
            }
        } else {
            for (int j = 0; j < this->B; j++) {
                decoder_gradient.at(j) += attention_head.at(j);
                if (this->encoder_data.size() != 0) {
                    encoder_gradient.at(j) += encoder_head.at(j);
                }

            }
        }
    }

    this->encoder_gradient = encoder_gradient;

    return decoder_gradient;
}

template <class T>
void MultiHeadAttention<T>::updateParameters(std::string& optimizertype, T& learningRate, int& iter) {

    log_info("=====================================================");
    log_info( "Entering Multi-Head Attention Update Parameters ..." );

    for (int i = 0; i < this->H; i++) {
        M1[i]->updateParameters(optimizertype, learningRate, iter);
    }

    log_detail("Wo Linear parameter update");
    Wo->updateParameters(optimizertype, learningRate, iter);

}

template <class T>
Topology MultiHeadAttention<T>::generateDotFormat(const std::string& name , bool operators, bool weights) {
    Topology topology;
    topology.dot = "{* MultiHeadAttention (" + name + ") *}|";
    topology.parameters = 0;
    for (int i = 0; i < this->H; i++) {
        topology.dot += "{* Head " + std::to_string(i) + " *}|";  
        Topology top =  M1[i]->generateDotFormat() ;
        topology.dot += top.dot + "|";
        topology.parameters += top.parameters;
    }
    topology.dot.pop_back(); // Remove the last comma
    return topology; 
}

/*****************************************************************************************************
* Base FeedForward  Layer
*****************************************************************************************************/
// We expect the input to be BxNxM (where M is embedding size). This layer goes through the
// first linear transformation which has a weight of MxW such that the output will be BxNxW.
// The second linear transformation has weight of WxW by default which also results in BxNxW.
// If the output size is provided, then the second linear transformation will have WxO weights
// and thus will result in BxNxO. 
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
        this->W = this->M;
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
    log_detail( "L2 Linear output {0} {1} {2}", output.size(), output.at(0).rows(), output.at(0).cols() );
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

    // Note that there are no parameters to update in Activation functions.
}

template <class T>
Topology FeedForward<T>::generateDotFormat(const std::string& name , bool operators, bool weights) {
    Topology topology;
    topology.dot = "{* FeedForward (" + name + ") *}|";  
    topology.parameters = 0;
    if (L1 == nullptr || L2 == nullptr || A1 == nullptr) {
        topology.dot += "{- No training done yet -}";
    } else {
        Topology top1 =  L1->generateDotFormat("L1") ;
        Topology top2 =  A1->generateDotFormat() ; 
        Topology top3 =  L2->generateDotFormat("L2"); 
        topology.dot += top1.dot + "|" + top2.dot + "|" + top3.dot;
        topology.parameters += top1.parameters + top2.parameters + top3.parameters;
    }
    return topology; 
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
    this->M = input_data.at(0).cols();

    log_detail( "Size of input: {:d}", this->input_data.size() );

    if (M1 == nullptr || LN1 == nullptr || F1 == nullptr || LN2 == nullptr) {
        M1  = new MultiHeadAttention<T>(this->H, this->W, this->bias, false);
        LN1 = new LayerNorm<T>(); 
        F1  = new FeedForward<T>(this->F, this->bias, this->activationtype,  this->alpha);
        LN2 = new LayerNorm<T>();
    }

    log_detail( "Input Data ...." );
    log_matrix( input_data[0]  );

    // Perform MultiHead Operation.
    log_detail( "Encoder M1 forward ..." );
    M1out = M1->forward(input_data);

    log_detail( "Encoder M1 forward output (M1out)..." );
    log_matrix( M1out[0] );

    log_detail( "Encoder LN1 forward pass ..." );
    LN1out = LN1->forward(M1out);

    log_detail( "Encoder LN1 forward output ..." );
    log_matrix( LN1out  );

    // Add the residual from previous input
    for (int i = 0; i < (int) this->B; i++) {
        LN1out.at(i) = M1out.at(i) + input_data.at(i);
    }

    // Perform FeedForward Operation.
    log_detail( "Encoder F1 forward pass ..." );
    F1out = F1->forward(LN1out);

    log_detail( "Encoder FeedForward forward output ..." );
    log_matrix( F1out  );


    log_detail( "Encoder LN2 forward pass ..." );
    LN2out = LN2->forward(F1out);

    log_detail( "Encoder LN2 forward output ..." );
    log_matrix( LN2out  );

    // Add the residual from previous input
    aitensor<T> output;
    for (int i = 0; i < (int) this->B; i++) {
        output.push_back( LN2out.at(i) + LN1out.at(i) );
    }

    log_detail( "Encoder forward output ..." );
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

    log_detail(" Gradient: {0}", gradients.size());
    log_matrix(gradients[0]);

    // Propagate Gradient to Encoder.
    aitensor<T>  dLN2 = LN2->backward(gradients);  // dLN2

    log_detail( "Encoder LN2 backprop output (dLN2) ..." );
    log_matrix( dLN2 );

    aitensor<T> dF1 = F1->backward(dLN2);    // dF1

    log_detail( "Encoder F1 backprop output (dF1) ..." );
    log_matrix( dF1 );
 
    log_detail( "Encoder input * F1 backprop output ..." );

    // Add the gradients from F1 and gradient from L1
    for (int i = 0; i < (int) gradients.size(); i++) {
        dLN2.at(i) = dF1.at(i) + gradients.at(i);
    }

    log_detail( "Encoder (dLN2 + dF1) backprop output ..." );
    log_matrix( dF1 );

    aitensor<T>  dLN1 = LN1->backward(dLN2);

    log_detail( "Encoder LN1 backprop output (dLN1) ..." );
    log_matrix( dLN1 );

    aitensor<T>  dM1 =  M1->backward(dLN1);

    log_detail( "Encoder M1 backprop output (dM1) ..." );
    log_matrix( dM1 );

    aitensor<T> dInput;
    for (int i = 0; i < (int) gradients.size(); i++) {
        dInput.push_back( dM1.at(i) + dLN1.at(i) );
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
Topology Encoder<T>::generateDotFormat(const std::string& name , bool operators, bool weights) {
    Topology topology;
    topology.dot = "{* Encoder *}|";
    topology.parameters = 0;
    if (M1 == nullptr || LN1 == nullptr || F1 == nullptr || LN2 == nullptr)  {
        topology.dot += "{- No training done yet -}";
    } else {
        Topology top1 =  M1->generateDotFormat("M1") ;
        Topology top2 =  LN1->generateDotFormat("add_norm1") ;
        Topology top3 =  F1->generateDotFormat("F1") ;
        Topology top4 =  LN2->generateDotFormat("add_norm2");
        topology.dot += top1.dot +  "|" + top2.dot + + "|" + top3.dot + "|" + top4.dot;
        topology.parameters += top1.parameters + top2.parameters + top3.parameters + top4.parameters;
    }
    return topology; 
}

// where weights is BxNxW and bias is W.
template <class T>
const aitensor<T> EncoderLayer<T>::forward(const aitensor<T>& input_data) { 

    log_info("=====================================================");
    log_info( "Entering EncoderLayer Forward Pass ..." );

    aitensor<T> output = input_data;

    // Perform Forward Pass for each layer
    for (int i = 0; i < this->L; i++) {
        Encoder<T>* encoder = this->encoders.at(i);
        output = encoder->forward(output);
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
        Encoder<T>* encoder = this->encoders.at(i);
        dInput = encoder->backward(dInput);
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
        Encoder<T>* encoder = this->encoders.at(i);
        encoder->updateParameters(optimizertype, learningRate, iter);
    }
}

template <class T>
Topology EncoderLayer<T>::generateDotFormat(const std::string& name , bool operators, bool weights) {
    Topology topology;
    topology.dot = "{* Encoder Layer *}|";
    topology.parameters = 0;

    for (int i = 0; i < this->L; i++) {
        Encoder<T>* encoder = this->encoders.at(i);
        Topology top = encoder->generateDotFormat("encoder " + i);
        topology.dot += top.dot;
        topology.parameters += top.parameters;
    }

    return topology; 
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

    log_detail(" Size of input data: {0}", decoder_data.size());

    // Cache for later back propagation.
    this->input_data = decoder_data;

    // dimension is BxNxW
    this->B = this->input_data.size();
    this->N = this->input_data.at(0).rows();
    this->M = this->input_data.at(0).cols();

    log_detail( "Size of input: {:d}", this->input_data.size() );

    if (M1 == nullptr || M2 == nullptr || LN1 == nullptr || F1 == nullptr || LN2 == nullptr) {
        M1  = new MultiHeadAttention<T>(this->H, this->W,  this->bias, true); // masked is set.
        LN1 = new LayerNorm<T>(); 
        M2  = new MultiHeadAttention<T>(this->H, this->W, this->bias, false);
        LN2 = new LayerNorm<T>(); 
        F1  = new FeedForward<T>(this->F, this->bias, this->activationtype,  this->alpha);
        LN3 = new LayerNorm<T>();
    }

    log_detail( "Input Data ...." );
    log_matrix( this->input_data  );


    // Perform First MultiHead Operation.
    log_detail( "Decoder M1 forward output ..." );
    M1out = M1->forward(this->input_data);

    log_detail( "Decoder Attention forward output (M1out)..." );
    log_matrix( M1out );


    log_detail( "Decoder LN1 forward pass ..." );
    LN1out = LN1->forward(M1out);

    log_detail( "Decoder LN1 forward output ..." );
    log_matrix( LN1out  );

    // Add the residual from previous input
    for (int i = 0; i < (int) this->B; i++) {
        LN1out.at(i) = LN1out.at(i) + input_data.at(i)  ;
    }

    log_detail( "Decoder M1_LN1 forward output ..." );
    log_matrix( LN1out  );

    // Perform Second MultiHead Operation.
    M2out = M2->forward(LN1out, encoder_data);

    log_detail( "Decoder Attention forward output (M2out)..." );
    log_matrix( M2out );

    log_detail( "Decoder LN2 forward pass ..." );
    LN2out = LN2->forward(M2out);

    // Add the residual from previous input
    for (int i = 0; i < (int) this->B; i++) {
        LN2out.at(i) =  LN2out.at(i) + LN1out.at(i)  ;
    }

    log_detail( "Decoder M1_LN1 forward output ..." );
    log_matrix(LN2out);

    // Perform FeedForward Operation.
    log_detail( "Decoder F1 forward pass ..." );
    F1out = F1->forward(LN2out);

    log_detail( "Decoder F1out forward pass ..." );
    LN3out = LN3->forward(F1out);

    // Add the residual from previous input
    aitensor<T> output;
    for (int i = 0; i < (int) this->B; i++) {
        output.push_back( LN3out.at(i) + LN2out.at(i) );
    }
    
    log_detail( "Decoder output forward pass ..." );
    log_matrix( output  );

    log_detail( "Decoder output  ... {0} {1}x{2}", output.size(), output.at(0).rows(), output.at(0).cols() );

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


    log_detail( "Decoder LN3 backprop  ..." );
    aitensor<T>  dLN3 = LN3->backward(gradients);   

    log_detail( "Decoder LN3 backprop output (dLN3) ..." );
    log_matrix( dLN3 );

    log_detail( "Decoder F1 backprop  ..." );
    aitensor<T> dF1 = F1->backward(dLN3); 

    log_detail( "Decoder LN3 backprop output (dF1) ..." );
    log_matrix( dF1 );

    // Gradient is additive for residual
    for (int i = 0; i < (int) gradients.size(); i++) {
        dLN3.at(i) =  dF1.at(i) + gradients.at(i)  ;
    }

    log_detail( "Decoder LN2 backprop  ..." );
    aitensor<T>  dLN2 = LN2->backward(dLN3);  // dLN2

    log_detail( "Decoder LN2 backprop output (dLN2) ..." );
    log_matrix( dLN2 );

    log_detail( "Decoder M2 backprop  ..." );
    aitensor<T> dM2 = M2->backward(dLN2); 

    log_detail( "Decoder M2 backprop output (dM2) ..." );
    log_matrix( dM2 );

    // Retrieve the gradient for the encoder backpass.
    this->encoder_gradient = M2->getEncoderGradient();

    // Gradient is additive for residual
    for (int i = 0; i < (int) gradients.size(); i++) {
        dLN2.at(i) =  dM2.at(i) + dLN3.at(i) ;
    }

    log_detail( "Decoder LN1 backprop  ..." );
    aitensor<T>  dLN1 = LN1->backward(dLN2);  // dLN2

    log_detail( "Decoder LN1 backprop output (dLN1) ..." );
    log_matrix( dLN1 );

    log_detail( "Decoder M1 backprop  ..." );
    aitensor<T> dM1 = M1->backward(dLN1); 

    log_detail( "Decoder M1 backprop output (M1) ..." );
    log_matrix( dM1 );

    aitensor<T> dInput;
    // Gradient is additive for residual
    for (int i = 0; i < (int) gradients.size(); i++) {
        dInput.push_back( dM1.at(i) + dLN2.at(i) );
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
Topology Decoder<T>::generateDotFormat(const std::string& name , bool operators, bool weights) {
    Topology topology;
    topology.dot = "{* Decoder *}|";
    topology.parameters = 0;

    if (M1 == nullptr || LN1 == nullptr || F1 == nullptr || LN2 == nullptr)  {
        topology.dot += "{- No training done yet -}";
    } else {
        Topology top1 =  M1->generateDotFormat("Masked") ;
        Topology top2 =  LN1->generateDotFormat("add_norm1") ;
        Topology top3 =  M2->generateDotFormat("M2") ;
        Topology top4 =  LN2->generateDotFormat("add_norm2") ;
        Topology top5 =  F1->generateDotFormat("F1") ;
        Topology top6 =  LN3->generateDotFormat("add_norm3");
        topology.dot += top1.dot + "|" + top2.dot + "|" + top3.dot + "|" + top4.dot + "|" + top5.dot + "|" + top6.dot ;
        topology.parameters += top1.parameters + top2.parameters + top3.parameters +
                               top4.parameters + top5.parameters + top6.parameters;
    }
    return topology; 
}

// where weights is BxNxW and bias is W.
template <class T>
const aitensor<T> DecoderLayer<T>::forward(const aitensor<T>& decoder_data, const aitensor<T>& encoder_data) { 

    log_info("=====================================================");
    log_info( "Entering DecoderLayer Forward Pass ..." );

    aitensor<T> output = decoder_data;

    // Perform Forward Pass for each layer
    for (int i = 0; i < this->L; i++) {
        Decoder<T>* decoder = this->decoders.at(i);
        output = decoder->forward(output, encoder_data);
    }
    return output;
}

// where weights is BxNxW and bias is W.
template <class T>
const aitensor<T> DecoderLayer<T>::backward(const aitensor<T>& gradients) { 

    log_info("=====================================================");
    log_info( "Entering DecoderLayer Backward Pass ..." );

    aitensor<T> decoder_gradients = gradients;  // decoder_gradients
    aitensor<T> encoder_gradient = {};

    // Perform Backward Prop for each layer
    for (int i = 0; i < this->L; i++) {
        Decoder<T>* decoder = this->decoders.at(i);
        decoder_gradients = decoder->backward(decoder_gradients);
        encoder_gradient = decoder->getEncoderGradient(); // get the gradient for the encoder.
        if (this->encoder_gradients.size() == 0) {
            this->encoder_gradients = encoder_gradient;
        } else {
            int enc_gradient_size = this->encoder_gradients.size();
            for (int j = 0; j < enc_gradient_size; j++) {
                this->encoder_gradients.at(j).array() += encoder_gradient.at(j).array();
            }
        }
    }
    return decoder_gradients;
}
  
// where weights is BxNxW and bias is W.
template <class T>
void DecoderLayer<T>::updateParameters(std::string& optimizertype, T& learningRate, int& iter) {

    log_info("=====================================================");
    log_info( "Entering DecoderLayer Update Parameter ..." );

    // Perform Parameter Update for each layer
    for (int i = 0; i < this->L; i++) {
        Decoder<T>* decoder = this->decoders.at(i);
        decoder->updateParameters(optimizertype, learningRate, iter);
    }

    this->encoder_gradients.clear();
}

template <class T>
Topology DecoderLayer<T>::generateDotFormat(const std::string& name , bool operators, bool weights) {
    Topology topology;
    topology.dot = "{* Decoder Layer *}|";
    topology.parameters = 0;
    for (int i = 0; i < this->L; i++) {
        Decoder<T>* decoder = this->decoders.at(i);
        Topology top = decoder->generateDotFormat("decoder " + i);
        topology.dot += top.dot;
        topology.parameters += top.parameters;
    }
    return topology; 
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

