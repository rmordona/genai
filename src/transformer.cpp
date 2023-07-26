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

namespace py = pybind11;
using namespace py::literals;


/*****************************************************************************************************
* Base Attention Head Layer
*****************************************************************************************************/

// While the parameter weight has dimension MxW,  the resulting transformation has dimension of NxW.
// We only need the M dimension from an NxM input to generate parameter matrix.
// where weights is NxW and bias is W.
Eigen::MatrixXd Attention::forward(Eigen::MatrixXd& input_data) { 

    log_info("=====================================");
    log_info( "Entering Attention Forward Pass ..." );

    // Cache for later back propagation.
    this->input_data = input_data;

    this->N = input_data.rows();
    this->M = input_data.cols();
    this->Dk = this->M / this->H;

    log_detail( "Size of input:", this->input_data.size() );

    if (Q == nullptr || K == nullptr || V == nullptr || Wo == nullptr) {
        Q  = new Linear(this->W, bias);
        K  = new Linear(this->W, bias);
        V  = new Linear(this->W, bias);
        Wo = new Linear(this->M, bias);
    }

    // Perform Linear Transformation.
    Qout = Q->forward(input_data);
    Kout = K->forward(input_data);
    Vout = V->forward(input_data);

    log_detail( "Q Linear output" );
    log_matrix( Qout );

    log_detail( "K Linear output" );
    log_matrix( Kout );

    log_detail( "V Linear output" );
    log_matrix( Vout );

    // MatMul (QK^T)
    Eigen::MatrixXd QK = BaseOperator::matmul(Qout, Kout.transpose());

    log_detail( "QK matmul" );
    log_matrix( QK );

    // Include some Masking (still to be implemented)

    // Scale sqrt(Dk)
    QK = QK.array() / sqrt(Dk);

    log_detail( "Scaling -> QK / sqrt(Dk)" );
    log_matrix( QK );

    // Mask if required (Decoder)
    // ...

    // Perform Softmax
    QKsoft = BaseOperator::softmax(QK);

    log_detail( "Softmax QKsoft output" );
    log_matrix(  QKsoft );

    log_detail( "Vout output" );
    log_matrix(  Vout  );

    // Include dropout (still to be implemented)

    // Perform matmul with V
    QKsoftV = BaseOperator::matmul(QKsoft, Vout);

    // Perform another transform to align dimension.
    Eigen::MatrixXd output = Wo->forward(QKsoftV);

    log_detail( "Attention forward pass output ..." );
    log_matrix( output );

    return output; // this becomes input to the next Node or next Layer.
}


// Leave the gradients as is. They are cached in the Node. 
// They will be used to update the parameters in next parallel operations.
// the dInput is the gradient we propagate to source Nodes in the graph;
// while the parameter gradients get cached to be used to update the parameters later.
Eigen::MatrixXd Attention::backward(Eigen::MatrixXd& gradients) { 

    log_info("=====================================");
    log_info( "Entering Attention Backward Pass ..." );

    // Gradient with Respect to W linear operations.
    Eigen::MatrixXd WodInput = Wo->backward(gradients); 

    // Gradient with Respect to QKsoft (matmul operation)
    Eigen::MatrixXd dQKsoft = BaseOperator::matmul(WodInput, Vout.transpose());

    log_detail( "dQKsoft gradient" );
    log_matrix( dQKsoft );

    // Gradient with Respect to Vout (matmul operation)
    Eigen::MatrixXd VmInput = BaseOperator::matmul(QKsoft.transpose(), WodInput);

    log_detail( "VmInput gradient" );
    log_matrix( VmInput );



    // Propagate Gradient to softmax operation.
    Eigen::MatrixXd dInput = BaseOperator::softmaxGradient(dQKsoft, QKsoft);

    log_detail( "dInput gradient" );
    log_matrix( dInput );

    // Propagate Gradient to scale operation.
    Eigen::MatrixXd dScale = -0.5 * dInput.array() * (1.0 / std::sqrt(Dk));

    log_detail( "dScale gradient" );
    log_matrix( dInput );

    // Gradient with Respect to Q (matmul operation)
    Eigen::MatrixXd QdQK = BaseOperator::matmul(dScale, Kout); // Kout was already tranposed during forward.

    log_detail( "QdQK gradient" );
    log_matrix( QdQK );

    // Gradient with Respect to V (matmul operation)
    Eigen::MatrixXd KdQK = BaseOperator::matmul(Qout.transpose(), dScale);

    log_detail( "KdQK gradient" );
    log_matrix( KdQK );


    // Propagate Gradient to the Q,K,V linear operations.
    Eigen::MatrixXd  QdInput = Q->backward(QdQK);
    Eigen::MatrixXd  KdInput = V->backward(KdQK);
    Eigen::MatrixXd  VdInput = V->backward(VmInput); 

    log_detail( " Done Backprop to Q, K, V ..." );

    log_detail( "VdInput gradient" );
    log_matrix( VdInput );

    log_detail( "QdInput gradient" );
    log_matrix( QdInput );

    log_detail( "KdInput gradient" );
    log_matrix( KdInput );

    dInput = QdInput + KdInput + VdInput;

    log_detail( "dInput gradient" );
    log_matrix( dInput );

    return dInput;
}

void Attention::updateParameters(std::string& optimizertype, double& learningRate, int& iter) {

    log_info("=========================================");
    log_info( "Entering Attention Update Parameter ..." );

    Q->updateParameters(optimizertype, learningRate, iter);
    K->updateParameters(optimizertype, learningRate, iter);
    V->updateParameters(optimizertype, learningRate, iter);
    Wo->updateParameters(optimizertype, learningRate, iter);
}


/*****************************************************************************************************
* Base Multi-Head Attention Layer
*****************************************************************************************************/

// While the parameter weight has dimension MxW,  the resulting transformation has dimension of NxW.
// We only need the M dimension from an NxM input to generate parameter matrix.
// where weights is NxW and bias is W.
Eigen::MatrixXd MultiAttention::forward(Eigen::MatrixXd& input_data) { 


    log_info("===============================================");
    log_info( "Entering Multi-Head Attention Forward Pass ..." );

    // Cache for later back propagation.
    this->input_data = input_data;

    this->N = input_data.rows();
    this->M = input_data.cols();

    log_detail( "Size of input: {:d}" , this->input_data.size() );
    log_detail( "Size of Head: {:d}", H  );

    if (M1.empty()) {
        for (int i = 0; i < this->H; i++) {
            Attention* A1  = new Attention(1, this->W);
            M1.push_back(A1);
        }
    }


    this->Dk = this->M / this->H;
    int splits = 0; 

    log_detail( "Size of DK ...{:d}" , Dk );

    std::vector<Eigen::MatrixXd> outputs;

    for (int i = 0; i < this->H; i++) {
        splits = this->Dk * this->N * i; 
        Eigen::Map<Eigen::MatrixXd> O(this->input_data.data()+splits, N, Dk);
        outputs.push_back(O);
        
    }  

    // Perform Forward pass.
    for (int i = 0; i < this->H; i++) {
        outputs[i] = M1[i]->forward(outputs[i]);
    }

    int totalCols = 0;
    for (const auto& output : outputs) {
        totalCols += output.cols();
    }

    // now concatenate
    Eigen::MatrixXd concatenated_output = Eigen::MatrixXd::Zero(outputs[0].rows(), totalCols);
    int colOffset = 0;
    for (const auto& output : outputs) {
        concatenated_output.block(0, colOffset, output.rows(), output.cols()) = output;
        colOffset += output.cols();
    }

    log_detail( "MultiAttention Forward pass done ..." );

    return concatenated_output;
}

// Leave the gradients as is. They are cached in the Node. 
// They will be used to update the parameters in next parallel operations.
// the dInput is the gradient we propagate to source Nodes in the graph;
// while the parameter gradients get cached to be used to update the parameters later.
Eigen::MatrixXd MultiAttention::backward(Eigen::MatrixXd& gradients) { 

    log_info("================================================");
    log_info( "Entering Multi-Head Attention Backward Pass ..." );

    // Propagate Gradient to multihead operations.
    Eigen::MatrixXd  dInput = Eigen::MatrixXd::Zero(gradients.rows(), gradients.cols());

    // Perform MultiAttention backward pass.
    for (int i = 0; i < this->H; i++) {
        dInput += M1[i]->backward(gradients);
    }
    return dInput;
}

void MultiAttention::updateParameters(std::string& optimizertype, double& learningRate, int& iter) {

    log_info("=====================================================");
    log_info( "Entering Multi-Head Attention Update Parameters ..." );

    for (int i = 1; i < this->H; i++) {
        M1[i]->updateParameters(optimizertype, learningRate, iter);
    }
}


/*****************************************************************************************************
* Base FeedForward  Layer
*****************************************************************************************************/

// While the parameter weight has dimension MxW,  the resulting transformation has dimension of NxW.
// We only need the M dimension from an NxM input to generate parameter matrix.
// where weights is NxW and bias is W.
Eigen::MatrixXd FeedForward::forward(Eigen::MatrixXd& input_data) { 

    log_info("=====================================================");
    log_info( "Entering Feedforward Forward Pass ..." );

    // Cache for later back propagation.
    this->input_data = input_data;

    log_detail( "Input Result" );
    log_matrix( input_data );

    this->N = input_data.rows();
    this->M = input_data.cols();

    if (L1 == nullptr || L2 == nullptr || A1 == nullptr) {
        L1 = new Linear(this->W, bias);
        L2 = new Linear(this->M, bias); // requires to have dimension as the feedforward input
        A1 = new Activation(this->activationtype, this->alpha);
    }

    // Perform Linear Transformation.
    L1out = L1->forward(input_data);  // Cache output for use by backward activation later
    Eigen::MatrixXd A1out = A1->forward(L1out);
    Eigen::MatrixXd output = L2->forward(A1out);

    log_detail( "Output Result" );
    log_matrix( output );

    return output;
}

// Leave the gradients as is. They are cached in the Node. 
// They will be used to update the parameters in next parallel operations.
// the dInput is the gradient we propagate to source Nodes in the graph;
// while the parameter gradients get cached to be used to update the parameters later.
Eigen::MatrixXd FeedForward::backward(Eigen::MatrixXd& gradients) { 
    // Propagate Gradient to feedforward operations.

    log_info("=======================================");
    log_info( "Entering Feedforward Backward Pass ..." );

    Eigen::MatrixXd  L2gradients = L2->backward(gradients); 
    Eigen::MatrixXd  A1gradients = A1->backward(L2gradients, L1out);
    Eigen::MatrixXd  dInput = L1->backward(A1gradients);

    log_detail( "dInput" );
    log_matrix( dInput );

    return dInput;
}

void FeedForward::updateParameters(std::string& optimizertype, double& learningRate, int& iter) {

    log_info("==========================================");
    log_info( "Entering Feedforward Update Paramter ..." );

    L1->updateParameters(optimizertype, learningRate, iter);
    L2->updateParameters(optimizertype, learningRate, iter);

}


/*****************************************************************************************************
* Base Encoder  Layer
*****************************************************************************************************/

// While the parameter weight has dimension MxW,  the resulting transformation has dimension of NxW.
// We only need the M dimension from an NxM input to generate parameter matrix.
// where weights is NxW and bias is W.
Eigen::MatrixXd Encoder::forward(Eigen::MatrixXd& input_data) { 

    log_info("=====================================================");
    log_info( "Entering Encoder Forward Pass ..." );

    // Cache for later back propagation.
    this->input_data = input_data;

    this->N = input_data.rows();

    log_detail( "Size of input: {:d}", this->input_data.size() );

    if (M1 == nullptr || LN1 == nullptr || F1 == nullptr || LN2 == nullptr) {
        LN2 = new LayerNorm(this->W);
        F1  = new FeedForward(this->W, this->bias, this->activationtype,  this->alpha);
        LN1 = new LayerNorm(this->W); 
        M1  = new MultiAttention(this->H, this->W);
    }

    // Perform Linear Transformation.
    M1out = M1->forward(input_data);

    log_detail( "Input Data ...." );
    log_matrix( input_data  );

    log_detail( "Encoder Attention forward output ..." );
    log_matrix( M1out );

    Eigen::MatrixXd InputM1out = M1out.array() + input_data.array();

    log_detail( "Encoder Add 1 forward output ..." );
    log_matrix( InputM1out  );

    LN1out = LN1->forward(InputM1out);

    log_detail( "Encoder LN1 forward output ..." );
    log_matrix( LN1out  );

    F1out = F1->forward(LN1out);

    log_detail( "Encoder FeedForward forward output ..." );
    log_matrix( F1out  );

    Eigen::MatrixXd LN1F1out = F1out.array() + LN1out.array();

    log_detail( "Encoder Add 2 forward output ..." );
    log_matrix( LN1F1out  );

    Eigen::MatrixXd output = LN2->forward(LN1F1out);

    log_detail( "Encoder LN2 forward output ..." );
    log_matrix( output  );

    return output;
}

// Leave the gradients as is. They are cached in the Node. 
// They will be used to update the parameters in next parallel operations.
// the dInput is the gradient we propagate to source Nodes in the graph;
// while the parameter gradients get cached to be used to update the parameters later.
Eigen::MatrixXd Encoder::backward(Eigen::MatrixXd& gradients) { 

    log_info("=====================================================");
    log_info( "Entering Encoder Backward Pass ..." );

    log_detail( "Entering Encoder backpropagation ..." );
    // Propagate Gradient to Encoder.
    Eigen::MatrixXd  LN2gradients = LN2->backward(gradients); 

    log_detail( "Encoder LN2 backprop output ..." );
    log_matrix( LN2gradients );

    Eigen::MatrixXd F1LN2gradients = LN2gradients.array(); // F1out.array() * LN2gradients.array();

    Eigen::MatrixXd F1gradients = F1->backward(F1LN2gradients);

    log_detail( "Encoder F1 backprop output ..." );
    log_matrix( F1gradients );

    Eigen::MatrixXd InputLN2gradients = LN2gradients.array(); // LN1out.array() * LN2gradients.array();

    log_detail( "Encoder input * F1 backprop output ..." );
    log_matrix( InputLN2gradients );

    F1gradients = InputLN2gradients + F1gradients;

    log_detail( "Encoder (InputLN1gradients + F1gradients) backprop output ..." );
    log_matrix( F1gradients );

    Eigen::MatrixXd  LN1gradients = LN1->backward(F1gradients);

    log_detail( "Encoder LN1 backprop output ..." );
    log_matrix( LN1gradients );

    Eigen::MatrixXd M1LN1gradients = LN1gradients.array(); // A1out.array() * LN1gradients.array();

    Eigen::MatrixXd  M1gradients =  M1->backward(M1LN1gradients);

    log_detail( "Encoder A1 backprop output ..." );
    log_matrix( M1gradients );

    Eigen::MatrixXd LN1outLN1gradients = LN1gradients.array(); // input_data.array() * LN1gradients.array();

    Eigen::MatrixXd dInput = LN1outLN1gradients + M1gradients;      

    log_detail( "Encoder dInput ..." );
    log_matrix( dInput );

    return dInput;
}

void Encoder::updateParameters(std::string& optimizertype, double& learningRate, int& iter) {

    log_info("=====================================================");
    log_info( "Entering Encoder Update Parameter ..." );

    LN1->updateParameters(optimizertype, learningRate, iter);
    LN2->updateParameters(optimizertype, learningRate, iter);
    F1->updateParameters(optimizertype, learningRate, iter);
    M1->updateParameters(optimizertype, learningRate, iter);

}


