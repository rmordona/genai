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

    // Cache for later back propagation.
    this->input_data = input_data;

    this->N = input_data.rows();
    this->M = input_data.cols();
    this->Dk = this->M / this->H;

    std::cout << "Size of input:" << this->input_data.size() << "\n";

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

    std::cout << "Just Qout\n";
    std::cout << Qout << "\n";

    std::cout << "Just Kout\n";
    std::cout << Kout << "\n";

    std::cout << "Just Vout\n";
    std::cout << Vout << "\n";

    // MatMul (QK^T)
    Eigen::MatrixXd QK = BaseOperator::matmul(Qout, Kout.transpose());

    std::cout << "Just QK (matmul)\n";
    std::cout << QK << "\n";

    // Include some Masking (still to be implemented)

    // Scale sqrt(Dk)
    QK = QK.array() / sqrt(Dk);

    std::cout << "Just QK / sqrt(Dk\n";
    std::cout << QK << "\n";

    // Mask if required (Decoder)
    // ...

    // Perform Softmax
    QKsoft = BaseOperator::softmax(QK);

    std::cout << "Softmax QKsoft output\n";
    std::cout << QKsoft << "\n";

    std::cout << "Vout output\n";
    std::cout << Vout << "\n";

    // Include dropout (still to be implemented)

    // Perform matmul with V
    QKsoftV = BaseOperator::matmul(QKsoft, Vout);

    // Perform another transform to align dimension.
    Eigen::MatrixXd output = Wo->forward(QKsoftV);

    print_string("Attention forward pass output ...", true); 
    std::cout << output << "\n\n";

    return output; // this becomes input to the next Node or next Layer.
}


// Leave the gradients as is. They are cached in the Node. 
// They will be used to update the parameters in next parallel operations.
// the dInput is the gradient we propagate to source Nodes in the graph;
// while the parameter gradients get cached to be used to update the parameters later.
Eigen::MatrixXd Attention::backward(Eigen::MatrixXd& gradients) { 

    // Gradient with Respect to W linear operations.
    Eigen::MatrixXd WodInput = Wo->backward(gradients); 

    // Gradient with Respect to QKsoft (matmul operation)
    Eigen::MatrixXd dQKsoft = BaseOperator::matmul(WodInput, Vout.transpose());

    std::cout << "dQKsoft gradient\n";
    std::cout << dQKsoft << "\n";

    // Gradient with Respect to Vout (matmul operation)
    Eigen::MatrixXd VmInput = BaseOperator::matmul(QKsoft.transpose(), WodInput);

    std::cout << "VmInput gradient\n";
    std::cout << VmInput << "\n";



    // Propagate Gradient to softmax operation.
    Eigen::MatrixXd dInput = BaseOperator::softmaxGradient(dQKsoft, QKsoft);

    std::cout << "dInput gradient\n";
    std::cout << dInput << "\n";

    // Propagate Gradient to scale operation.
    Eigen::MatrixXd dScale = -0.5 * dInput.array() * (1.0 / std::sqrt(Dk));

    std::cout << "dScale gradient\n";
    std::cout << dInput << "\n";

    // Gradient with Respect to Q (matmul operation)
    Eigen::MatrixXd QdQK = BaseOperator::matmul(dScale, Kout); // Kout was already tranposed during forward.

    std::cout << "QdQK gradient\n";
    std::cout << QdQK << "\n";

    // Gradient with Respect to V (matmul operation)
    Eigen::MatrixXd KdQK = BaseOperator::matmul(Qout.transpose(), dScale);

    std::cout << "KdQK gradient\n";
    std::cout << KdQK << "\n";


    // Propagate Gradient to the Q,K,V linear operations.
    // std::cout << " Backprop to Q ...\n";
    Eigen::MatrixXd  QdInput = Q->backward(QdQK);
    // std::cout << " Backprop to K ...\n";
    Eigen::MatrixXd  KdInput = V->backward(KdQK);
    // std::cout << " Backprop to V ...\n";
    Eigen::MatrixXd  VdInput = V->backward(VmInput); 

    std::cout << " Done Backprop to Q, K, V ...\n";

    std::cout << "VdInput gradient\n";
    std::cout << VdInput << "\n";

    std::cout << "QdInput gradient\n";
    std::cout << QdInput << "\n";

    std::cout << "KdInput gradient\n";
    std::cout << KdInput << "\n";

    dInput = QdInput + KdInput + VdInput;

    std::cout << "dInput gradient\n";
    std::cout << dInput << "\n\n";

    return dInput;
}

void Attention::updateParameters(std::string& optimizertype, double& learningRate, int& iter) {
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

    // Cache for later back propagation.
    this->input_data = input_data;

    this->N = input_data.rows();
    this->M = input_data.cols();

    std::cout << "Size of input:" << this->input_data.size() << "\n";
    std::cout << "Size of Head:" << H << "\n";

    if (M1.empty()) {
        for (int i = 0; i < this->H; i++) {
            Attention* A1  = new Attention(1, this->W);
            M1.push_back(A1);
        }
    }


    this->Dk = this->M / this->H;
    int splits = 0; 

    std::cout << "Size of DK ..." << Dk << "\n";

    std::vector<Eigen::MatrixXd> outputs;

    for (int i = 0; i < this->H; i++) {
        splits = this->Dk * this->N * i; 
        std::cout << "Loop " << i << " ... split " << splits << "\n";
        Eigen::Map<Eigen::MatrixXd> O(this->input_data.data()+splits, N, Dk);
        outputs.push_back(O);
        
    }  

    std::cout << "Outputs" << "\n";
    for (int i = 0; i < this->H; i++) {
        std::cout << "FIrst head: " << i << "\n";
        std::cout << outputs[i] << "\n";
    }

    std::cout << "Now performing Forward pass \n";

    // Perform Forward pass.
    for (int i = 0; i < this->H; i++) {
        /// std::cout << "Attention forward entering ..." << i << "\n";
        outputs[i] = M1[i]->forward(outputs[i]);
        // std::cout << "Attention forward done ..." << i << "\n";
    }

    std::cout << "Now performing concatenation \n";

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

    std::cout << "MultiAttention forward pass done. \n";

    return concatenated_output;
}

// Leave the gradients as is. They are cached in the Node. 
// They will be used to update the parameters in next parallel operations.
// the dInput is the gradient we propagate to source Nodes in the graph;
// while the parameter gradients get cached to be used to update the parameters later.
Eigen::MatrixXd MultiAttention::backward(Eigen::MatrixXd& gradients) { 
    // Propagate Gradient to multihead operations.
    Eigen::MatrixXd  dInput = Eigen::MatrixXd::Zero(gradients.rows(), gradients.cols());

    std::cout << "Inside MultiAttention backward function ...\n";
    // Perform MultiAttention backward pass.
    for (int i = 0; i < this->H; i++) {
        std::cout << "calling attention backprop ...\n";
        std::cout << "attention gradient output \n";
        dInput += M1[i]->backward(gradients);
    }
    return dInput;
}

void MultiAttention::updateParameters(std::string& optimizertype, double& learningRate, int& iter) {
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

    // Cache for later back propagation.
    this->input_data = input_data;

    this->N = input_data.rows();
    this->M = input_data.cols();

    std::cout << "Size of input:" << this->input_data.size() << "\n";

    if (L1 == nullptr || L2 == nullptr || A1 == nullptr) {
        L1 = new Linear(this->W, bias);
        L2 = new Linear(this->M, bias); // requires to have dimension as the feedforward input
        A1 = new Activation(this->activationtype, this->alpha);
    }

    // Perform Linear Transformation.
    L1out = L1->forward(input_data);  // Cache output for use by backward activation later
    Eigen::MatrixXd A1out = A1->forward(L1out);
    Eigen::MatrixXd output = L2->forward(A1out);

    return output;
}

// Leave the gradients as is. They are cached in the Node. 
// They will be used to update the parameters in next parallel operations.
// the dInput is the gradient we propagate to source Nodes in the graph;
// while the parameter gradients get cached to be used to update the parameters later.
Eigen::MatrixXd FeedForward::backward(Eigen::MatrixXd& gradients) { 
    // Propagate Gradient to feedforward operations.
    Eigen::MatrixXd  L2gradients = L2->backward(gradients); 
    Eigen::MatrixXd  A1gradients = A1->backward(L2gradients, L1out);
    Eigen::MatrixXd  dInput = L1->backward(A1gradients);
    return dInput;
}

void FeedForward::updateParameters(std::string& optimizertype, double& learningRate, int& iter) {
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

    // Cache for later back propagation.
    this->input_data = input_data;

    this->N = input_data.rows();

    std::cout << "Size of input:" << this->input_data.size() << "\n";

    if (M1 == nullptr || LN1 == nullptr || F1 == nullptr || LN2 == nullptr) {
        LN2 = new LayerNorm(this->W);
        F1  = new FeedForward(this->W, this->bias, this->activationtype,  this->alpha);
        LN1 = new LayerNorm(this->W); 
        M1  = new MultiAttention(this->H, this->W);
    }

    // Perform Linear Transformation.
    M1out = M1->forward(input_data);

    std::cout << "Input Data ....\n";
    std::cout << input_data << "\n";

    std::cout << "\nEncoder Attention forward output ...\n";
    std::cout << M1out << "\n";

    Eigen::MatrixXd InputM1out = M1out.array() + input_data.array();

    std::cout << "\nEncoder Add 1 forward output ...\n";
    std::cout << InputM1out << "\n";

    LN1out = LN1->forward(InputM1out);

    std::cout << "\nEncoder LN1 forward output ...\n";
    std::cout << LN1out << "\n";

    F1out = F1->forward(LN1out);

    std::cout << "\nEncoder FeedForward forward output ...\n";
    std::cout << F1out << "\n";

    Eigen::MatrixXd LN1F1out = F1out.array() + LN1out.array();

    std::cout << "\nEncoder Add 2 forward output ...\n";
    std::cout << LN1F1out << "\n";

    Eigen::MatrixXd output = LN2->forward(LN1F1out);

    std::cout << "\nEncoder LN2 forward output ...\n";
    std::cout << output << "\n\n";

    return output;
}

// Leave the gradients as is. They are cached in the Node. 
// They will be used to update the parameters in next parallel operations.
// the dInput is the gradient we propagate to source Nodes in the graph;
// while the parameter gradients get cached to be used to update the parameters later.
Eigen::MatrixXd Encoder::backward(Eigen::MatrixXd& gradients) { 

    std::cout << "\nEntering Encoder backpropagation ...\n";
    // Propagate Gradient to Encoder.
    Eigen::MatrixXd  LN2gradients = LN2->backward(gradients); 

    std::cout << "\nEncoder LN2 backprop output ...\n";
    std::cout << LN2gradients << "\n\n";

    Eigen::MatrixXd F1LN2gradients = LN2gradients.array(); // F1out.array() * LN2gradients.array();

    Eigen::MatrixXd F1gradients = F1->backward(F1LN2gradients);

    std::cout << "\nEncoder F1 backprop output ...\n";
    std::cout << F1gradients << "\n\n";

    Eigen::MatrixXd InputLN2gradients = LN2gradients.array(); // LN1out.array() * LN2gradients.array();

    std::cout << "\nEncoder input * F1 backprop output ...\n";
    std::cout << InputLN2gradients << "\n\n";

    F1gradients = InputLN2gradients + F1gradients;

    std::cout << "\nEncoder (InputLN1gradients + F1gradients) backprop output ...\n";
    std::cout << F1gradients << "\n\n";

    Eigen::MatrixXd  LN1gradients = LN1->backward(F1gradients);

    std::cout << "\nEncoder LN1 backprop output ...\n";
    std::cout << LN1gradients << "\n\n";

    Eigen::MatrixXd M1LN1gradients = LN1gradients.array(); // A1out.array() * LN1gradients.array();

    Eigen::MatrixXd  M1gradients =  M1->backward(M1LN1gradients);

    std::cout << "\nEncoder A1 backprop output ...\n";
    std::cout << M1gradients << "\n\n";

    Eigen::MatrixXd LN1outLN1gradients = LN1gradients.array(); // input_data.array() * LN1gradients.array();

    Eigen::MatrixXd dInput = LN1outLN1gradients + M1gradients;      

    std::cout << "\nEncoder dInput ...\n";
    std::cout << dInput << "\n";

    return dInput;
}

void Encoder::updateParameters(std::string& optimizertype, double& learningRate, int& iter) {
    LN1->updateParameters(optimizertype, learningRate, iter);
    LN2->updateParameters(optimizertype, learningRate, iter);
    F1->updateParameters(optimizertype, learningRate, iter);
    M1->updateParameters(optimizertype, learningRate, iter);
}

//};


