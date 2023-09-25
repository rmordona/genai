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

namespace py = pybind11;
using namespace py::literals;

/*****************************************************************************************************
* Some common reduction operations used in machine learning and deep learning include:
*   Sum: Computes the sum of all the values.
*   Avg: Computes the average of all the values.
*   Max: Finds the maximum value among all the values.
*   Min: Finds the minimum value among all the values.
*   Argmax: Returns the index or position of the maximum value.
*   Argmin: Returns the index or position of the minimum value.
*   Matmul: Returns matrix multiplication result.
*   Mul: Returns element-wise multiplication result.
*****************************************************************************************************/
std::string Reduction::getType() {
    return this->reducttype;
}


/*****************************************************************************************************
* Base Optimizer Functions
*****************************************************************************************************/

/*****************************************************************************************************
* Stochastic Gradient Descent
*****************************************************************************************************/
template <class T>
void Optimizer<T>::sgd(aimatrix<T>& weights, const aimatrix<T>& gradients, int currentEpoch ,
                bool useStepDecay, T decayRateStep, int decayStep) {
    // Update weights
    weights.array() -= learningRate * gradients.array();

    if (useStepDecay) {
        stepDecay(learningRate, decayRateStep, currentEpoch, decayStep);
    }
}

template <class T>
void Optimizer<T>::sgd(aivector<T>& weights, const aivector<T>& gradients, int currentEpoch ,
                bool useStepDecay, T decayRateStep, int decayStep) {
        // Optimizer requires aimatrix<T>, so let's use that data structure with dimenion Rx1
        aimatrix<T> weights_(weights.rows(), 1);  // Column-Wise vector  Nx1
        aimatrix<T> gradients_(gradients.rows(), 1);  // Column-Wise vector  Nx1
        weights_.col(0) = weights;
        gradients_.col(0) = gradients;
        sgd(weights_, gradients_, currentEpoch);
        weights.col(0) = weights_.col(0);
}

template <class T>
void Optimizer<T>::sgd(airowvector<T>& weights, const airowvector<T>& gradients, int currentEpoch ,
                bool useStepDecay, T decayRateStep, int decayStep) {
        // Optimizer requires aimatrix<T>, so let's use that data structure with dimenion Rx1
        aimatrix<T> weights_(1, weights.cols());  // Row-Wise vector  1xM
        aimatrix<T> gradients_(1, gradients.cols());  // Row-Wise vector  1xM
        weights_.row(0) = weights;
        gradients_.row(0) = gradients;
        sgd(weights_, gradients_, currentEpoch);
        weights = weights_.row(0);
}

/*****************************************************************************************************
* Momentum optimizer with optional step decay
*****************************************************************************************************/
template <class T>
void Optimizer<T>::momentum(aimatrix<T>& weights, const aimatrix<T>& gradients, int currentEpoch ,
                T momentumRate, bool useStepDecay, T decayRateStep,  int decayStep) {
    // Initialize Momentum optimizer variables
    //static Eigen::MatrixXd velocity = Eigen::MatrixXd::Zero(weights.rows(), weights.cols());
    if (velocity.cols() == 0 && velocity.rows() == 0) {
        velocity = aimatrix<T>::Zero(weights.rows(), weights.cols());
    }

    // Update velocity
    velocity = momentumRate * velocity + learningRate * gradients;

    // Update weights
    weights.array() -= velocity.array();

    if (useStepDecay) {
        stepDecay(learningRate, decayRateStep, currentEpoch, decayStep);
    }
}

template <class T>
void Optimizer<T>::momentum(aivector<T>& weights, const aivector<T>& gradients, int currentEpoch,
                T momentumRate, bool useStepDecay, T decayRateStep,  int decayStep) {
        // Optimizer requires aimatrix<T>, so let's use that data structure with dimenion Rx1
        aimatrix<T> weights_(weights.rows(), 1);  // Column-Wise vector  Nx1
        aimatrix<T> gradients_(gradients.rows(), 1);  // Column-Wise vector  Nx1
        weights_.col(0) = weights;
        gradients_.col(0) = gradients;
        momentum(weights_, gradients_, currentEpoch);
        weights = weights_.col(0);
}

template <class T>
void Optimizer<T>::momentum(airowvector<T>& weights, const airowvector<T>& gradients, int currentEpoch ,
                T momentumRate, bool useStepDecay, T decayRateStep,  int decayStep) {
        // Optimizer requires aimatrix<T>, so let's use that data structure with dimenion Rx1
        aimatrix<T> weights_(1, weights.cols());  // Row-Wise vector  1xM
        aimatrix<T> gradients_(1, gradients.cols());  // Row-Wise vector  1xM
        weights_.row(0) = weights;
        gradients_.row(0) = gradients;
        momentum(weights_, gradients_, currentEpoch);
        weights = weights_.row(0);
}

/*****************************************************************************************************
*  Adam optimizer with optional step decay
*****************************************************************************************************/
template <class T>
void Optimizer<T>::adam(aimatrix<T>& weights, const aimatrix<T>& gradients, int currentEpoch ,
                T beta1, T beta2, T epsilon,  bool useStepDecay, T decayRateStep,  int decayStep) {

    log_info("============================");
    log_info("Entering Adam Optimation ...");

    T beta1_t = std::pow(beta1, currentEpoch + 1);
    T beta2_t = std::pow(beta2, currentEpoch + 1);

    log_detail( "Beta 1 Calculation: {:2.10f} based on beta1: {:2.10f}", beta1_t, beta1 );
    log_detail( "Beta 2 Calculation: {:2.10f} based on beta2: {:2.10f}", beta2_t, beta2 );

    if (moments.cols() == 0 && moments.rows() == 0) {
        log_info("Initializing moments to zeroes ...");
        moments = aimatrix<T>::Zero(weights.rows(), weights.cols());
    }
    if (velocity.cols() == 0 && velocity.rows() == 0) {
        log_info("Initializing velocity to zeroes ...");
        velocity = aimatrix<T>::Zero(weights.rows(), weights.cols());
    }

    log_info( "Calculating: beta1 * moments.array() [ note: all zeroes initially ]" );
    log_matrix( (aimatrix<T>) (beta1 * moments.array()) );

    log_info( "Calculating: beta2 * velocity.array() [ note: all zeroes initially ]" );
    log_matrix( (aimatrix<T>) (beta2 * velocity.array()) );


    // Update momentum and velocity
    moments = beta1 * moments.array() + (1 - beta1) * gradients.array();
    velocity = beta2 * velocity.array() + (1 - beta2) * (gradients.array() * gradients.array());

    log_detail( "Gradients" );
    log_matrix( (aimatrix<T>) gradients.array() );

    log_detail( "power of gradients" );
    log_matrix( (aimatrix<T>) (gradients.array() * gradients.array()) );


    log_detail( "momentum" );
    log_matrix( moments );
    log_detail( "velocity" );
    log_matrix( velocity );

    // Compute bias-corrected moment estimates
    aimatrix<T> m_hat = moments / (1 - beta1_t);
    aimatrix<T> v_hat = velocity / (1 - beta2_t);

    log_detail("momentum hat" );
    log_matrix( m_hat );
    log_detail( "velocity hat" );
    log_matrix( v_hat );

    // Update weights
    weights.array() -= learningRate * (m_hat.array() / (v_hat.array().sqrt() + epsilon));

    if (useStepDecay) {
        stepDecay(learningRate, decayRateStep, currentEpoch, decayStep);
    }
}

template <class T>
void Optimizer<T>::adam(aivector<T>& weights, const aivector<T>& gradients, int currentEpoch ,
                T beta1, T beta2, T epsilon,  bool useStepDecay, T decayRateStep,  int decayStep) {
        // Optimizer requires aimatrix<T>, so let's use that data structure with dimenion Rx1
        aimatrix<T> weights_(weights.rows(), 1);  // Column-Wise vector  Nx1
        aimatrix<T> gradients_(gradients.rows(), 1);  // Column-Wise vector  Nx1
        weights_.col(0) = weights;
        gradients_.col(0) = gradients;
        adam(weights_, gradients_, currentEpoch);
        weights = weights_.col(0);
}

template <class T>
void Optimizer<T>::adam(airowvector<T>& weights, const airowvector<T>& gradients, int currentEpoch ,
                T beta1, T beta2, T epsilon,  bool useStepDecay, T decayRateStep,  int decayStep) {
        // Optimizer requires aimatrix<T>, so let's use that data structure with dimenion Rx1
        aimatrix<T> weights_(1, weights.cols());  // Row-Wise vector  1xM
        aimatrix<T> gradients_(1, gradients.cols());  // Row-Wise vector  1xM
        weights_.row(0) = weights;
        gradients_.row(0) = gradients;
        adam(weights_, gradients_, currentEpoch);
        weights = weights_.row(0);
}

/*****************************************************************************************************
*  RMSprop optimizer with optional step decay
*****************************************************************************************************/
template <class T>
void Optimizer<T>::rmsprop(aimatrix<T>& weights, const aimatrix<T>& gradients, int currentEpoch ,
                T rho, T epsilon,  bool useStepDecay, T decayRateStep,  int decayStep) {
    // Initialize RMSprop optimizer variables
    if (rms.cols() == 0 && rms.rows() == 0) {
        rms = aimatrix<T>::Zero(weights.rows(), weights.cols());
    }

    // Update RMSprop cache
    rms = rho * rms + (1 - rho) * (gradients.array() * gradients.array()).matrix();

    // Update weights
    weights.array() -= learningRate * (gradients.array() / (rms.array().sqrt() + epsilon));

    if (useStepDecay) {
        stepDecay(learningRate, decayRateStep, currentEpoch, decayStep);
    }
}

template <class T>
void Optimizer<T>::rmsprop(aivector<T>& weights, const aivector<T>& gradients, int currentEpoch ,
                T rho, T epsilon,  bool useStepDecay, T decayRateStep,  int decayStep) {
        // Optimizer requires aimatrix<T>, so let's use that data structure with dimenion Rx1
        aimatrix<T> weights_(weights.rows(), 1);  // Column-Wise vector  Nx1
        aimatrix<T> gradients_(gradients.rows(), 1);  // Column-Wise vector  Nx1
        weights_.col(0) = weights;
        gradients_.col(0) = gradients;
        rmsprop(weights_, gradients_, currentEpoch);
        weights = weights_.col(0);
}

template <class T>
void Optimizer<T>::rmsprop(airowvector<T>& weights, const airowvector<T>& gradients, int currentEpoch ,
                T rho, T epsilon,  bool useStepDecay, T decayRateStep,  int decayStep) {
        // Optimizer requires aimatrix<T>, so let's use that data structure with dimenion Rx1
        aimatrix<T> weights_(1, weights.cols());  // Row-Wise vector  1xM
        aimatrix<T> gradients_(1, gradients.cols());  // Row-Wise vector  1xM
        weights_.row(0) = weights;
        gradients_.row(0) = gradients;
        rmsprop(weights_, gradients_, currentEpoch);
        weights = weights_.row(0);
}

/*****************************************************************************************************
*  Adagrad optimizer with optional step decay
*****************************************************************************************************/
template <class T>
void Optimizer<T>::adagrad(aimatrix<T>& weights, const aimatrix<T>& gradients, int currentEpoch ,
                T epsilon,
                bool useStepDecay, T decayRateStep,  int decayStep) {
    // Initialize Adagrad optimizer variables
    // static Eigen::MatrixXd accum = Eigen::MatrixXd::Zero(weights.rows(), weights.cols());
    if (accum.cols() == 0 && accum.rows() == 0) {
        accum = aimatrix<T>::Zero(weights.rows(), weights.cols());
    }

    // Update sum of squared gradients
    accum.array() += gradients.array() * gradients.array();

    // Update weights
    weights.array() -= learningRate * gradients.array() / (accum.array().sqrt() + epsilon);

    if (useStepDecay) {
        stepDecay(learningRate, decayRateStep, currentEpoch, decayStep);
    }
}

/*****************************************************************************************************
*  Adamax optimizer with optional step decay
*****************************************************************************************************/
template <class T>
void Optimizer<T>::adamax(aimatrix<T>& weights, const aimatrix<T>& gradients, int currentEpoch , 
                T beta1, T beta2, T epsilon,
                bool useStepDecay, T decayRateStep, int decayStep) {
    // Initialize Adamax optimizer variables
    if (moments.cols() == 0 && moments.rows() == 0) {
        moments = aimatrix<T>::Zero(weights.rows(), weights.cols());
    }
    if (nu.cols() == 0 && nu.rows() == 0) {
        nu = aimatrix<T>::Zero(weights.rows(), weights.cols());
    }

    // Update biased first moment estimate
    moments = beta1 * moments + (1 - beta1) * gradients;

    // Update the exponentially weighted infinity norm
    nu = nu.cwiseMax(beta2 * nu.cwiseAbs()) + (1 - beta2) * gradients.cwiseAbs();

    // Update weights
    weights.array() -= learningRate * (moments.array() / (nu.array() + epsilon));

    // Compute learning rate decay
    if (useStepDecay) {
        stepDecay(learningRate, decayRateStep, currentEpoch, decayStep);
    }
    
}

/*****************************************************************************************************
*  Nadam optimizer with optional step decay
*****************************************************************************************************/
template <class T>
void Optimizer<T>::nadam(aimatrix<T>& weights, const aimatrix<T>& gradients, int currentEpoch , 
                T beta1, T beta2, T epsilon,
                bool useStepDecay, T decayRateStep, int decayStep) {

    if (moments.cols() == 0 && moments.rows() == 0) {
        moments = aimatrix<T>::Zero(weights.rows(), weights.cols());
    }
    if (velocity.cols() == 0 && velocity.rows() == 0) {
        velocity = aimatrix<T>::Zero(weights.rows(), weights.cols());
    }


    T beta1_t = std::pow(beta1, currentEpoch + 1);
    T beta2_t = std::pow(beta2, currentEpoch + 1);

    // Update momentum and velocity
    moments = beta1 * moments + (1 - beta1) * gradients;
    velocity = beta2 * velocity + (1 - beta2) * (gradients.array() * gradients.array()).matrix();

    // Compute bias-corrected moment estimates
    aimatrix<T> m_hat = moments / (1 - beta1_t);
    aimatrix<T> v_hat = velocity / (1 - beta2_t);

    // Update weights
    weights.array() -= learningRate * (beta1 * m_hat + (1 - beta1) * gradients).array()
                / (v_hat.array().sqrt() + epsilon);

    if (useStepDecay) {
        stepDecay(learningRate, decayRateStep, currentEpoch, decayStep);
    }
}

/*****************************************************************************************************
*  Step decay for learning rate
*****************************************************************************************************/
template <class T>
void Optimizer<T>::stepDecay(T& learningRate, T decayRate, int currentEpoch, int decayStep) {
    if (currentEpoch != 0 && currentEpoch % decayStep == 0) {
        learningRate *= decayRate;
    }
}

/**************************************************************************************************************************
 * Linear Class:
***************************************************************************************************************************/
// This assumes that the input is defined with NxM dimensionality.
// Therefore the size of the parameters and thus gradients will be based on MxW where W is the number of weights to use;
// The output will have NxW dimension.
template <class T>
void Linear<T>::setInitialWeights(int M) {

    // if size is already set, 
    // it means weights have previously already been set by an initial forward pass
    if (this->M != 0) return;

    this->M = M;

    parameters.weights.resize(M, this->W); // allocates memory
    parameters.biases.resize(this->W); // allocates memory

        log_detail( "manual: input Dimension: {0}x{1}",  parameters.weights.rows(),  parameters.weights.cols());

    // Initialize Weights & Biases
    heInitMatrix(parameters.weights);
    // heInitialization(parameters.biases);
    parameters.biases.setConstant(T(0.01));

    log_detail( "manual: input Dimension: {0}x{1}",  parameters.weights.rows(),  parameters.weights.cols());


}

template <class T>
OperationParams<T> Linear<T>::getParameters() const {
    return parameters;
}

template <class T> 
std::vector<OperationParams<T>>  Linear<T>::getGradients() const {
    return vgradients;
}

// While the parameter weight has dimension MxW,  the resulting transformation has dimension of NxW.
// We only need the M dimension from an NxM input to generate parameter matrix.
// where weights is NxW and bias is W.
template <class T>
const aimatrix<T> Linear<T>::linearTransform(const aimatrix<T>& input_data) {

    log_info("==================================");
    log_info("Entering Linear Transformation ...");

    // Initialize the parameters.
    setInitialWeights(input_data.cols());

    log_detail( "Size of Input: {:d}", input_data.size() );
    log_detail( "Size of Weights: {:d}", parameters.weights.size() );
    log_detail( "Size of Biases: {:d}", parameters.biases.size() );
    log_detail( "---");
    log_detail( "Input" );
    log_matrix( input_data );
    log_detail( "Weights" );
    log_matrix( parameters.weights );
    log_detail( "Matrix Multiplication of Input and Weights (Standard)" );
    log_matrix( aimatrix<T> (input_data * parameters.weights) );
    log_detail( "Matrix Multiplication of Input and Weights (Using BaseOperator)" );
    log_matrix( aimatrix<T> (BaseOperator::matmul(input_data, parameters.weights)) );
    log_detail( "bias" );
    log_rowvector( parameters.biases );

    aimatrix<T> output1 = (input_data * parameters.weights);
        log_detail( "manual: input Dimension: {0}x{1}", input_data.rows(), input_data.cols());
        log_detail( "manual: weights Dimension: {0}x{1}", parameters.weights.rows(), parameters.weights.cols());
         log_detail( "manual: Prior Bias: Linear output Dimension: {0}x{1}", output1.rows(), output1.cols());
        log_matrix(output1);

    aimatrix<T> output = BaseOperator::matmul(input_data, parameters.weights);
        log_detail( "matmul: Prior Bias: Linear output Dimension: {0}x{1}", output.rows(), output.cols());
        log_matrix(output);
    if (bias == true) {
        // rowwise() means, slice one row at a time and add to bias (which is 1 row horizontally).
        output = output.rowwise() + parameters.biases;// Ax + b = Wx + b; NxW dimension.
        log_detail( "After Bias: Linear output Dimension: {0}x{1}", output.rows(), output.cols());
        log_matrix(output);
    } 
    return output;
}


template <class T>
const aitensor<T> Linear<T>::forward(const aitensor<T>& input_data) { 

    log_info("===============================================");
    log_info("Linear Transformation Forward Pass ...");

    // Cache for later back propagation.
    this->input_data = input_data;

    if (input_data.size() == 0) {
        return input_data;
    }

    this->batch_size = this->input_data.size();
    this->input_size = this->input_data.at(0).rows();
    this->embedding_size = this->input_data.at(0).cols();

    // std::unique_ptr<aitensor<T>> output_data = std::make_unique<aitensor<T>>(this->batch_size, this->input_size, this->W); // BxNxW
    // this->output_data.resize(this->batch_size, this->input_size, this->W);
    aimatrix<T> input(this->input_size, this->W);

    log_detail( "Batch Size: {0}, Row: {1}, Col: {2}", this->batch_size, this->input_size, this->W );

    aitensor<T> output_data;

    for (int i = 0; i < this->batch_size; ++i) {

        input = this->input_data.at(i); // matrix_view(chip(this->input_data,i, 0)); // input_size x embedding_size

        log_detail( "Size of input: {0}", input.size() );

        // Perform Linear Transformation.
        aimatrix<T> output = linearTransform(input);

        log_detail( "Linear output Dimension: {0}x{1}", output.rows(), output.cols());

        // output_data->chip(i, 0) = tensor_view(output);
        output_data.push_back(output);

    }

    log_info( "End Linear transformation ....\n" );

    return output_data; // this becomes input to the next Node or next Layer. It returns a copy of the output.
}

template <class T>
OperationParams<T> Linear<T>::gradient_Wrt_Weight_Bias(const aimatrix<T>& new_gradients, const aimatrix<T>& input_data) {
    int N = new_gradients.rows();

    // initialize gradients for next iteration.
    OperationParams<T> gradients;

    log_detail("Computing Gradient with respect to bias:");
    
    log_detail( "Size of gradient: {:d}" , new_gradients.size() );
    log_detail( "Size of input: {:d}" , input_data.size() );
    log_detail( "---");
    log_detail( "Before multiply of input and gradients");
    log_detail( "The input:" );
    log_matrix( aimatrix<T>  (input_data)  );
    log_detail( "The gradient:"  );
    log_matrix( new_gradients );

    log_info("Multiply input and gradient.");
    // Compute the gradient with respect to the weights (dW)
    // This is just only because, by standard we have Y = W * X, but in our forward, we have Y = X * W
    gradients.weights = BaseOperator::matmul(new_gradients.transpose(), input_data).transpose();  // dL/W = (dL/DC.T * x).T

    log_detail( "Computing Gradient with respect to weight:");
    log_matrix( aimatrix<T> (gradients.weights) );

    // Compute the gradient with respect to the bias (db)
    if (bias == true) {
        log_detail( "Add the bias." );
        gradients.biases = new_gradients.colwise().sum();
        log_detail(  "gradient biases rows: {:d}, cols: {:d}" , gradients.biases.rows(), gradients.biases.cols() );
        log_rowvector( gradients.biases );
    }

    // Normalize the gradients by dividing by the number of samples (N)
    gradients.weights /= N;  // dW - gradients of weights (MxW)
    log_detail( "Normalized Gradient weights:" );
    log_matrix( aimatrix<T>  (gradients.weights) );

    if (bias == true) {
        gradients.biases /= N;   // db - gradients of bias (1xW)
        log_detail( "Normalized Gradient biases:" );
        log_rowvector( gradients.biases );
    }

    // Return Gradients
    return gradients;
}

template <class T>
const aimatrix<T> Linear<T>::gradient_Wrt_Input(const aimatrix<T>& new_gradients) { 
    int N = new_gradients.rows();

    log_detail( "Computing Gradient with respect to Input:" );
    log_detail( "The weight:" );
    log_matrix( aimatrix<T> (parameters.weights) );
    log_detail( "The gradient:" ) ;
    log_matrix( aimatrix<T> (new_gradients) );
    // Compute the gradient with respect to the input (dInput)
    // This is just only because, by standard we have Y = W * X, but in our forward, we have Y = X * W
    aimatrix<T> dInput = BaseOperator::matmul(parameters.weights, new_gradients.transpose()).transpose();   // dL/x = (W * dL/DC.T)

    log_detail( "The computed gradient with respect to input:" );
    log_matrix( dInput );

    // Normalize the gradients by dividing by the number of samples (N)
    dInput /= N;  // dInput - gradients of input (NxM)

    log_detail( "normalized dInput" );
    log_matrix( aimatrix<T> (dInput) );
    return dInput;
}

// Leave the gradients as is. They are cached in the Node. 
// They will be used to update the parameters in next parallel operations.
// the dInput is the gradient we propagate to source Nodes in the graph;
// while the parameter gradients get cached to be used to update the parameters later.
template <class T>
const aitensor<T> Linear<T>::backward(const aitensor<T>& gradients) { 

    log_info("===========================================");
    log_info( "Entering Linear Gradient Backward pass ..." );

    aimatrix<T>& weights = parameters.weights;
    airowvector<T>& biases = parameters.biases;

    log_detail( "Size of gradients: {:d}" , gradients.size()  );
    log_detail( "Size of weights: {:d}"  , weights.size()  );
    log_detail( "Size of biases: {:d}" , biases.size()  );
    log_detail( "---" );
    log_detail( "Computing Gradient now ..." );

    // The dimension is based on that of the output_data.  See Linear<T>::forward()
    aitensor<T> dInput; // (this->batch_size, this->input_size, this->W); 

    aimatrix<T> gradient, input;

    for (int i = 0; i < this->batch_size; ++i) {

        gradient = gradients.at(i); // matrix_view(chip(gradients, i, 0)); 
        input    = this->input_data.at(i); // matrix_view(chip(this->input_data, i, 0));

        OperationParams<T> wbgradients = gradient_Wrt_Weight_Bias(gradient, input);

        log_detail( "Computing Delta Error now ..."  );
        log_matrix(wbgradients.weights);


        gradient = gradient_Wrt_Input(gradient);

        // dInput.chip(i, 0) = tensor_view(gradient);
        dInput.push_back(gradient);

        vgradients.push_back(wbgradients); // Cache gradients
    }

    log_detail( "Done with Gradients..." );
    return dInput;
}

template <class T>
void Linear<T>::updateParameters(std::string& optimizertype, T& learningRate, int& iter) {

    log_info("=======================================");
    log_info( "Entering Linear Parameter Updates ..." );
    log_detail( "size: {:d}", this->W );

    for (int i = 0; i < this->batch_size; ++i) {

        OperationParams<T>  gradients = vgradients[i];

        log_detail( "Assigning ..." );

        log_detail( "pbiases rows: {:d}, cols: {:d}" ,  parameters.biases.rows() , parameters.biases.cols() );
        log_detail( "gbiases rows: {:d}, cols: {:d}", gradients.biases.rows() , gradients.biases.cols() );

        log_detail( "Gradient weights:" );
        log_matrix( aimatrix<T> (gradients.weights) );
        log_detail( "Gradient biases:" );
        log_rowvector( gradients.biases );

        log_detail( "Before Updated weights:" );
        log_matrix( aimatrix<T> (parameters.weights) );
        log_detail( "Before Updated biases:" );
        log_rowvector( parameters.biases );

        if (opt_weights == nullptr) {
            opt_weights = new Optimizer<T>(optimizertype, learningRate);
        }

        if (opt_biases == nullptr && bias == true) {
            opt_biases = new Optimizer<T>(optimizertype, learningRate);
        }

        log_detail( "Updating Linear weights" );
        opt_weights->adam(parameters.weights, gradients.weights, iter);
        log_detail( "Updating Linear biases" );

        if (bias == true) {
            opt_biases->adam(parameters.biases, gradients.biases, iter);
        }

        log_detail( "Updated weights:" );
        log_matrix( aimatrix<T> (parameters.weights) );
        log_detail( "Updated biases:" );
        log_rowvector( parameters.biases  );

    }

    // initialize gradients for next iteration.
    vgradients.clear();
}

template <class T>
std::string Linear<T>::generateDotFormat(const std::string& name) {
    std::string dot = "{* Linear Transformation (" + name + ") *}|";  
    T min_weights = 0.0, max_weights = 0.0;
    T min_biases = 0.0, max_biases = 0.0;
    if (this->M != 0) // if weights are already initialized.
    try {
        min_weights = parameters.weights.minCoeff();
        max_weights = parameters.weights.maxCoeff();
        min_biases = parameters.biases.minCoeff();
        max_biases = parameters.biases.maxCoeff();
    } catch (...) {};
    dot += "{Weights|min=" + scalar_to_string(min_weights) + "|max=" + scalar_to_string(max_weights) + "}|";    
    dot += "{Biases|min=" + scalar_to_string(min_biases) + "|max=" + scalar_to_string(max_biases) + "}"; 
    return dot;
}

/*****************************************************************************************************
* Base Batch Normalization Function
* Suppose we have a sample input represented as a matrix of NxM where N=number of samples
* and M=embedding vector size (features).  Our Batch normalization takes mean and variance
* along the N dimension.
*****************************************************************************************************/

// This assumes that the input is defined with NxM dimensionality.
// Therefore the size of the parameters and thus gradients will be based on MxW where W is the number of weights to use.
template <class T>
void BatchNorm<T>::setInitialWeights(int M) {

    // if size is already set, 
    // it means weights have previously already been set by an initial forward pass
    if (this->M != 0) return;

    this->M = M;

    // Initialize scaling and shifting parameters   
    scale.resize(this->M); // allocates memory
    shift.resize(this->M); // allocates memory

    // Initialize Weights & Biases
    heInitVector(scale);
    shift.setConstant(0.01);

    // Initialize Gradients     
    //dScale.setZero();
    //dShift.setZero();
}
 
template <class T>
std::tuple<aimatrix<T>, aimatrix<T>, aimatrix<T>> BatchNorm<T>::normalize(const aimatrix<T>& input_data) {

    log_info("=================================");
    log_info( "Entering Batch Normalization ..." );

    setInitialWeights(input_data.rows());

    log_detail(" Input Data ...");
    log_matrix( input_data );

    // Calculate layer mean along the N dimension (tranposed).
    aivector<T> batchMean = (aivector<T>) (input_data.rowwise().mean());

    log_detail( "Mean ..." );
    log_vector( batchMean );

    // Calculate: X - mean
    aimatrix<T> Xmu = input_data.colwise() - batchMean;

    log_detail( "Xmu" );
    log_matrix( Xmu );

    // Calculate batch variance along the N dimension (tranposed).
    aivector<T> batchVariance = Xmu.array().square().rowwise().mean();

    log_detail( "Variance ..." );
    log_vector( batchVariance );

    // Add a small epsilon for numerical stability
    aivector<T> epsilonVector = aivector<T>::Constant(batchVariance.size(), epsilon);

    log_detail( "Epsilon ..." );
    log_vector( epsilonVector );

    // Calculate batch standard deviation across the N dimension (tranposed).
    this->batchStdDev = (batchVariance + epsilonVector).array().sqrt();

    log_detail( "stdDev ..." );
    log_vector( this->batchStdDev );

    // Normalize the inputs along the N dimension (tranposed).
    aimatrix<T> normalizedInput = Xmu.array().colwise()  / this->batchStdDev.array();

    log_detail( "normalizedInput along the N  ..." );
    log_matrix( normalizedInput );

    // Scale and shift the normalized inputs
    aimatrix<T> normalizedOutput = (normalizedInput.array().colwise() * scale.array()).array().colwise() + shift.array();

    log_detail( "scale ..." );
    log_vector( scale );

    log_detail( "shift ..." );
    log_vector( shift );

    log_detail( "normalizedOutput scaled ..." );
    log_matrix( normalizedOutput );
    
    return std::make_tuple(normalizedInput, normalizedOutput, Xmu);
}

template <class T>
const aitensor<T> BatchNorm<T>::forward(const aitensor<T>& input_data) { 

    // Cache for later back propagation.
    this->input_data = input_data;

    // if input_data is from liner transform, then dimension is BxNxW
    this->batch_size = this->input_data.size();
    this->input_size = this->input_data.at(0).rows();
    this->param_size = this->input_data.at(0).cols();

    aitensor<T> normalizedOutput; // (this->batch_size, this->input_size, this->param_size);

    aimatrix<T> input, normInput, normOutput, Xmu;

    for (int i = 0; i < this->batch_size; ++i) {

        input = input_data.at(i).transpose(); // batch_size x embedding_size

        log_detail( "Size of input at iteration {0}: {1}", i, input.size() );

        // Perform Linear Transformation.
        std::tie(normInput, normOutput, Xmu) = normalize(input);

        log_detail("normInput");
        log_matrix(normInput);

        log_detail("normOutput");
        log_matrix(normOutput);

        log_detail("Xmu");
        log_matrix(Xmu);

        this->normalizedInput.push_back(normInput.transpose()); 
        this->minusMean.push_back(Xmu.transpose()); 
        normalizedOutput.push_back(normOutput.transpose()); 

    }

    return normalizedOutput; // this becomes input to the next Node or next Layer.
}

// Leave the gradients as is for the scale and shift. They are cached in the Node. 
// They will be used to update the parameters in next parallel operations.
// the dInput is the gradient we propagate to source Nodes in the graph;
// while the parameter gradients get cached to be used to update the parameters later.
template <class T>
const aitensor<T> BatchNorm<T>::backward(const aitensor<T>& gradients) {

    int W = this->input_data.at(0).rows();  

    log_info("==============================================");
    log_info("Entering Batch Normalization Backward pass ...");

    aitensor<T> dInput; // (this->batch_size, this->input_size, this->param_size);

    aimatrix<T> gradient, normInput, Xmu;

    for (int i = 0; i < this->batch_size; ++i) {

        log_detail("Backward Iteration at {0} transpose prior", i);

        log_detail("gradient");
        log_matrix(gradients.at(i));

        log_detail("normInput");
        log_matrix(normalizedInput.at(i));

        log_detail("Xmu");
        log_matrix(minusMean.at(i));


        gradient = gradients.at(i).transpose(); // matrix_view(chip(gradients, i, 0));

        normInput = this->normalizedInput.at(i).transpose(); // matrix_view(chip(this->normalizedInput, i, 0));

        Xmu = this->minusMean.at(i).transpose(); // matrix_view(chip(this->minusMean, i, 0));

        log_detail("Backward Iteration at {0} transpose after", i);

        log_detail("gradient");
        log_matrix(gradient);

        log_detail("normInput");
        log_matrix(normInput);

        log_detail("Xmu");
        log_matrix(Xmu);

        // Compute the gradient with respect to the scale parameter (Gamma)
        aivector<T> dScale = (gradient.array() * normInput.array()).rowwise().sum();
        vgscale.push_back(dScale);

        // Compute the gradient with respect to the shift parameter (Beta)
        aivector<T> dShift = gradient.rowwise().sum();
        vgshift.push_back(dShift);

        log_detail( "scale and shift gradients" );
        log_vector( this->scale );
        log_vector( this->shift );

        log_detail( "dScale and dShift gradients" );
        log_vector( dScale );
        log_vector( dShift );

        // Compute the gradient with respect to the normalized input
        aimatrix<T> dNormalizedInput = gradient.array().colwise() * this->scale.array();

        log_detail( "dNormalizedInput" );
        log_matrix( (aimatrix<T>) (dNormalizedInput) );

        log_detail( "dNormalizedInput * Xmu" );
        log_matrix( (aimatrix<T>) (dNormalizedInput.array() * Xmu.array()) );

        log_detail( "dNormalizedInput * Xmu.rowwise.sum" );
        log_matrix( (aimatrix<T>) ((dNormalizedInput.array() * Xmu.array()).rowwise().sum()) );

        log_detail( "batchStdDev" );
        log_vector( (aivector<T>) (this->batchStdDev.array() ) );

        log_detail( "batchStdDev * batchStdDev" );
        log_vector( (aivector<T>) ( (this->batchStdDev.array() * this->batchStdDev.array()) ) );

        // Compute the gradient with respect to the layer standard deviation
        aivector<T> dbatchStdDev = -(dNormalizedInput.array() * Xmu.array()).rowwise().sum() / (this->batchStdDev.array() * this->batchStdDev.array());

        log_detail( "dLayerStdDev" );
        log_vector( dbatchStdDev );

        // Compute the gradient with respect to the layer variance
        aivector<T> dbatchVariance = 0.5 * (dbatchStdDev.array() / this->batchStdDev.array());

        log_detail( "dbatchVariance" );
        log_vector( dbatchVariance );

        aimatrix<T> dNormMinusMean1 = (dNormalizedInput.array() * (1.0 / (this->batchStdDev.array())).replicate(1,W)) + 
                                            (2.0 * Xmu.array() * (1.0/W * dbatchVariance.replicate(1,W)).array());

        log_detail( "dxmu1" );
        log_matrix( (aimatrix<T>) ((dNormalizedInput.array() * (1.0 / (this->batchStdDev.array())).replicate(1,W))) );
        log_detail( "dxmu2" );
        log_matrix( (aimatrix<T>) ( (2.0 * Xmu.array() * (1.0/W * dbatchVariance.replicate(1,W)).array()))  );

        log_detail( "dNormMinusMean1" );
        log_matrix( dNormMinusMean1  );

        log_detail( "istd" );
        log_matrix( (aimatrix<T>) ((1.0/(this->batchStdDev.array() * this->batchStdDev.array())).replicate(1,W))  );

        log_detail( "dsq" );
        log_matrix( (aimatrix<T>) (1.0/W * dbatchVariance.replicate(1, 2)) );

        log_detail( "xmu" );
        log_matrix( (aimatrix<T>) (Xmu) );

        // Compute the gradient with respect to the batch mean
        aivector<T> dLayerMean = -1.0 * dNormMinusMean1.array().rowwise().sum();

        log_detail( "dLayerMean" );
        log_vector( dLayerMean );

        aimatrix<T> dNormMinusMean2 = 1.0/W *  dLayerMean.replicate(1,W).array();

        log_detail("dNormMinusMean2" );
        log_matrix( (aimatrix<T>) (dNormMinusMean2) );

        // Compute the gradient with respect to the input
        gradient = dNormMinusMean1.array() + dNormMinusMean2.array();

        log_detail( "gradient" );
        log_matrix( (aimatrix<T>) (gradient) );

        dInput.push_back(gradient.transpose()); // dInput.chip(i, 0) = tensor_view(gradient);
    
    }

    return dInput;
}

template <class T>
void BatchNorm<T>::updateParameters(std::string& optimizertype, T& learningRate, int& iter) {

    log_info("==================================================");
    log_info("Entering Batch Normalization Upgrade Parameters...");

    for (int i = 0; i < this->batch_size; ++i) {

        if (opt_scale == nullptr) {
            opt_scale = new Optimizer<T>(optimizertype, learningRate);
            opt_shift = new Optimizer<T>(optimizertype, learningRate);
        }

        log_detail( "Updating Scale" );

        opt_scale->adam(this->scale, vgscale[i], iter);

        log_detail( "Updating Shift" );

        opt_shift->adam(this->shift, vgshift[i], iter);


    }
    
    // initialize gradients for next iteration.
    vgscale.clear();
    vgshift.clear();
}

template <class T>
std::string BatchNorm<T>::generateDotFormat(const std::string& name) {
    std::string dot = "{* Batch Normalization (" + name + ") *}|";  
    T min_scale = 0.0, max_scale = 0.0;
    T min_shift = 0.0, max_shift = 0.0;
    if (this->M != 0) // if weights are already initialized.
    try {
        max_scale = scale.maxCoeff();
        min_scale = scale.minCoeff();
        max_shift = shift.maxCoeff();
        min_shift = shift.minCoeff();
    } catch (...) {};
    dot += "{Shape|min=" + scalar_to_string(min_scale) + "|max=" + scalar_to_string(max_scale) + "}|";
    dot += "{Shift|min=" + scalar_to_string(min_shift) + "|max=" + scalar_to_string(max_shift) + "}";
    return dot;
}


/*****************************************************************************************************
* Base Layer Normalization Function:
*   Suppose we have a sample input represented as a matrix of NxM where N=number of samples
*   and M=embedding vector size (features).  Our Batch normalization takes mean and variance
*   along the M dimension.
*****************************************************************************************************/

// This assumes that the input is defined with NxM dimensionality.
// Therefore the size of the parameters and thus gradients will be based on MxW where W is the number of weights to use.
template <class T>
void LayerNorm<T>::setInitialWeights(int N) {

    // if size is already set, 
    // it means weights have previously already been set by an initial forward pass
    if (this->N != 0) return;

    this->N = N;

    // Initialize scaling and shifting parameters   
    scale.resize(this->N); // allocates memory
    shift.resize(this->N); // allocates memory

    // Initialize Weights & Biases
    heInitVector(scale);
    shift.setConstant(0.01);

    // Initialize Gradients     
    // dScale.setZero();
    // dShift.setZero();
}

template <class T>
std::tuple<aimatrix<T>, aimatrix<T>, aimatrix<T>> LayerNorm<T>::normalize(const aimatrix<T>& input_data) {
    log_info("=================================");
    log_info( "Entering Layer Normalization ..." );

    setInitialWeights(input_data.rows());

    log_detail(" Input Data ...");
    log_matrix( input_data );

    // Calculate layer mean along the M dimension.
    aivector<T> layerMean = (aivector<T>) (input_data.rowwise().mean());

    log_detail( "Mean ..." );
    log_vector( layerMean );

    // Calculate: X - mean
    aimatrix<T> Xmu = input_data.colwise() - layerMean;

    log_detail( "Xmu" );
    log_matrix( Xmu );

    // Calculate batch variance along the M dimension
    aivector<T> layerVariance = Xmu.array().square().rowwise().mean();

    log_detail( "Variance ..." );
    log_vector( layerVariance );

    // Add a small epsilon for numerical stability
    aivector<T> epsilonVector = aivector<T>::Constant(layerVariance.size(), epsilon);

    log_detail( "Epsilon ..." );
    log_vector( epsilonVector );

    // Calculate batch standard deviation across the M dimension
    this->layerStdDev = (layerVariance + epsilonVector).array().sqrt();

    log_detail( "stdDev ..." );
    log_vector( this->layerStdDev );

    // Normalize the inputs along the M dimension
    aimatrix<T> normalizedInput = Xmu.array().colwise()  / this->layerStdDev.array();

    log_detail( "normalizedInput along the N  ..." );
    log_matrix( normalizedInput );

    // Scale and shift the normalized inputs
    aimatrix<T> normalizedOutput = (normalizedInput.array().colwise() * scale.array()).array().colwise() + shift.array();

    log_detail( "scale ..." );
    log_vector( scale );

    log_detail( "shift ..." );
    log_vector( shift );

    log_detail( "normalizedOutput scaled ..." );
    log_matrix( normalizedOutput );

    return std::make_tuple(normalizedInput, normalizedOutput, Xmu);
}

template <class T>
const aitensor<T> LayerNorm<T>::forward(const aitensor<T>& input_data) { 

    // Cache for later back propagation.
    this->input_data = input_data;

    // if input_data is from liner transform, then dimension is BxNxW
    this->batch_size = this->input_data.size();
    this->input_size = this->input_data.at(0).rows();
    this->param_size = this->input_data.at(0).cols();

    aitensor<T> normalizedOutput; // (this->batch_size, this->input_size, this->param_size);

    aimatrix<T> input, normInput, normOutput, Xmu;

    for (int i = 0; i < this->batch_size; ++i) {

        input = input_data.at(i); // matrix_view(chip(input_data, i, 0)); // input_size x param_size

        log_detail( "Size of input:", input.size() );

        // Perform Linear Transformation.
        std::tie(normInput, normOutput, Xmu) = normalize(input);

        // Cache 
        this->normalizedInput.push_back(normInput);   // normalizedInput.chip(i, 0) = tensor_view(normInput);
        this->minusMean.push_back(Xmu);               // minusMean.chip(i, 0) = tensor_view(Xmu);

        normalizedOutput.push_back(normOutput); //   normalizedOutput.chip(i, 0) = tensor_view(normOutput);

    }

    return normalizedOutput; // this becomes input to the next Node or next Layer.
}

// Leave the gradients as is for the scale and shift. They are cached in the Node. 
// They will be used to update the parameters in next parallel operations.
// the dInput is the gradient we propagate to source Nodes in the graph;
// while the parameter gradients get cached to be used to update the parameters later.
template <class T>
const aitensor<T> LayerNorm<T>::backward(const aitensor<T>& gradients) {

    int W = this->input_data.at(0).cols();

    log_info("==============================================");
    log_info("Entering Layer Normalization Backward pass ...");


    aitensor<T> dInput; // (this->batch_size, this->input_size, this->param_size);

    aimatrix<T> gradient, normInput, Xmu;

    for (int i = 0; i < this->batch_size; ++i) {

        gradient = gradients.at(i); 

        normInput = this->normalizedInput.at(i); 

        Xmu = this->minusMean.at(i); 

        log_detail("Backward Iteration at {0}", i);

        log_detail("gradient");
        log_matrix(gradient);

        log_detail("normInput");
        log_matrix(normInput);

        log_detail("Xmu");
        log_matrix(Xmu);

        // Compute the gradient with respect to the scale parameter (Gamma)
        aivector<T> dScale = (gradient.array() * normInput.array()).rowwise().sum();
        vgscale.push_back(dScale);

        // Compute the gradient with respect to the shift parameter (Beta)
        aivector<T> dShift = gradient.rowwise().sum();
        vgshift.push_back(dShift);

        log_detail( "scale and shift gradients" );
        log_vector( this->scale );
        log_vector( this->shift );

        log_detail( "dScale and dShift gradients" );
        log_vector( dScale );
        log_vector( dShift );

        // Compute the gradient with respect to the normalized input
        aimatrix<T> dNormalizedInput = gradient.array().colwise() * this->scale.array();

        log_detail( "dNormalizedInput" );
        log_matrix( (aimatrix<T>) (dNormalizedInput) );

        log_detail( "dNormalizedInput * Xmu" );
        log_matrix( (aimatrix<T>) (dNormalizedInput.array() * Xmu.array()) );

        log_detail( "dNormalizedInput * Xmu.rowwise.sum" );
        log_matrix( (aimatrix<T>) ((dNormalizedInput.array() * Xmu.array()).rowwise().sum()) );

        log_detail( "layerStdDev" );
        log_vector( (aivector<T>) (this->layerStdDev.array() ) );

        log_detail( "layerStdDev * layerStdDev" );
        log_vector( (aivector<T>) ( (this->layerStdDev.array() * this->layerStdDev.array()) ) );

        // Compute the gradient with respect to the layer standard deviation
        aivector<T> dLayerStdDev = -(dNormalizedInput.array() * Xmu.array()).rowwise().sum() / (this->layerStdDev.array() * this->layerStdDev.array());

        log_detail( "dLayerStdDev" );
        log_vector( dLayerStdDev );

        // Compute the gradient with respect to the layer variance
        aivector<T> dLayerVariance = 0.5 * (dLayerStdDev.array() / this->layerStdDev.array());

        log_detail( "dLayerVariance" );
        log_vector( dLayerVariance );

        aimatrix<T> dNormMinusMean1 = (dNormalizedInput.array() * (1.0 / (this->layerStdDev.array())).replicate(1,W)) + 
                                            (2.0 * Xmu.array() * (1.0/W * dLayerVariance.replicate(1,W)).array());

        log_detail( "dxmu1" );
        log_matrix( (aimatrix<T>) ((dNormalizedInput.array() * (1.0 / (this->layerStdDev.array())).replicate(1,W))) );
        log_detail( "dxmu2" );
        log_matrix( (aimatrix<T>) ( (2.0 * Xmu.array() * (1.0/W * dLayerVariance.replicate(1,W)).array()))  );

        log_detail( "dNormMinusMean1" );
        log_matrix( dNormMinusMean1  );

        log_detail( "istd" );
        log_matrix( (aimatrix<T>) ((1.0/(this->layerStdDev.array() * this->layerStdDev.array())).replicate(1,W))  );

        log_detail( "dsq" );
        log_matrix( (aimatrix<T>) (1.0/W * dLayerVariance.replicate(1, 2)) );

        log_detail( "xmu" );
        log_matrix( (aimatrix<T>) (Xmu) );

        // Compute the gradient with respect to the batch mean
        aivector<T> dLayerMean = -1.0 * dNormMinusMean1.array().rowwise().sum();

        log_detail( "dLayerMean" );
        log_vector( dLayerMean );

        aimatrix<T> dNormMinusMean2 = 1.0/W *  dLayerMean.replicate(1,W).array();

        log_detail("dNormMinusMean2" );
        log_matrix( (aimatrix<T>) (dNormMinusMean2) );

        // Compute the gradient with respect to the input
        gradient = dNormMinusMean1.array() + dNormMinusMean2.array();

        log_detail( "gradient" );
        log_matrix( (aimatrix<T>) (gradient) );

        dInput.push_back(gradient); // dInput.chip(i, 0) = tensor_view(gradient);
    
    }

    return dInput;
}

template <class T>
void LayerNorm<T>::updateParameters(std::string& optimizertype, T& learningRate, int& iter) {

    log_info("==================================================");
    log_info("Entering Layer Normalization Upgrade Parameters...");

    for (int i = 0; i < this->batch_size; ++i) {

        if (opt_scale == nullptr) {
            opt_scale = new Optimizer<T>(optimizertype, learningRate);
            opt_shift = new Optimizer<T>(optimizertype, learningRate);
        }

        log_detail( "Updating Scale" );

        opt_scale->adam(this->scale, vgscale[i], iter);

        log_detail( "Updating Shift" );

        opt_shift->adam(this->shift, vgshift[i], iter);

    }
    
    // initialize gradients for next iteration.
    vgscale.clear();
    vgshift.clear();
}

template <class T>
std::string LayerNorm<T>::generateDotFormat(const std::string& name) {
    std::string dot = "{* Layer Normalization (" + name + ") *}|"; 
    T min_scale = 0.0, max_scale = 0.0;
    T min_shift = 0.0, max_shift = 0.0;
    if (this->N != 0) // if weights are already initialized.
    try {
        max_scale = scale.maxCoeff();
        min_scale = scale.minCoeff();
        max_shift = shift.maxCoeff();
        min_shift = shift.minCoeff();
    } catch (...) {};
    dot += "{Shape|min=" + scalar_to_string(min_scale) + "|max=" + scalar_to_string(max_scale) + "}|";
    dot += "{Shift|min=" + scalar_to_string(min_shift) + "|max=" + scalar_to_string(max_shift) + "}";
    return dot;   
}

/*****************************************************************************************************
* Base Activation Functions
*****************************************************************************************************/
template <class T>
const aimatrix<T> Activation<T>::relu(const aimatrix<T>& x) {
    return x.array().max(0.0);
}
 
/*****************************************************************************************************
 * So, the gradient of the ReLU function with respect to its input (z) can be expressed as follows:
 * dy/dz = propagated_gradient * 1 for z > 0
 * dy/dz = propagated_gradient * 0 for z <= 0
 *****************************************************************************************************/
template <class T>
const aimatrix<T> Activation<T>::reluGradient(const aimatrix<T>& gradients, const aimatrix<T>& input) {
    aimatrix<T> dInput = input.array().max(T(0.0)).template cast<bool>().template cast<T>() * gradients.array();
    return dInput;
}

template <class T>
const aimatrix<T> Activation<T>::leakyReLU(const aimatrix<T>& x, float alpha) {
    return x.array().max(alpha * x.array());
}

/*****************************************************************************************************
 * So, the gradient of the LeakyReLU function with respect to its input (z) can be expressed as follows:
 * dy/dz = propagated_gradient * 1 for z > 0
 * dy/dz = propagated_gradient * alpha for z <= 0
*****************************************************************************************************/
template <class T>
const aimatrix<T> Activation<T>::leakyReluGradient(const aimatrix<T>& gradients, const aimatrix<T>& input) {
    aimatrix<T> dInput = input.array().max(T(0.0)).template cast<bool>().template cast<T>().max(alpha) * gradients.array();
    return dInput;
}

template <class T>
const aimatrix<T> Activation<T>::gelu(const aimatrix<T>& x) {
    return 0.5 * x.array() * (1.0 + ((x.array() * std::sqrt(2.0 / M_PI)).tanh()));
}

/*****************************************************************************************************
 * Gelu Gradient ...
*****************************************************************************************************/
template <class T>
const aimatrix<T> Activation<T>::geluGradient(const aimatrix<T>& gradients, const aimatrix<T>& input) {
    // Calculate the coefficient used in the GELU formula
    T coefficient = sqrt(2.0 / M_PI);
    // Compute the cumulative distribution function (CDF) part of the GELU gradient
    aimatrix<T> cdf = 0.5 * (1.0 + (gradients.array() / coefficient).tanh());
    // Compute the probability density function (PDF) part of the GELU gradient
    aimatrix<T> pdf = exp(-0.5 * gradients.array().square()) / coefficient;
    // Combine the CDF and PDF components to obtain the final gradient values
    // Apply element-wise operations on arrays: add CDF, multiply x by PDF, add a term based on x^3
    return 0.5 * (1.0 + (cdf.array() + gradients.array() * pdf.array() + 0.044715 * gradients.array().cube())).matrix();
}

/*****************************************************************************************************
 * This assumes that the input is defined with BxNxM dimensionality.
 * Therefore the size of the parameters and thus gradients will be based on BxMxW 
 * where B is batch size, N is input size, and W is the number of weights to use.
*****************************************************************************************************/
template <class T>
void Activation<T>::setInitSize(const aimatrix<T>& input_data) {

    // if size is already set, 
    // it means weights have previously already been set by an initial forward pass
    if (this->N != 0 || this->M != 0) return;

    this->N = input_data.rows();
    this->M = input_data.cols();

    // Initialize scaling and shifting parameters   
    // this->dInput.resize(this->N, this->M); // allocates memory
}

template <class T>
const aimatrix<T> Activation<T>::computeActivation(const aimatrix<T>& input_data) { 
    aimatrix<T> output;
    setInitSize(input_data);

    if (activationtype == "sigmoid") {
        output = sigmoid(input_data);
    } else
    if (activationtype == "tanh") {
        output = tanh(input_data);
    } else
    if (activationtype == "relu") {
        output = relu(input_data);
    } else
    if (activationtype == "leakyrelu") {
        output = leakyReLU(input_data, alpha);
    } else
    if (activationtype == "gelu") {
        output = gelu(input_data);
    } else
    if (activationtype == "softmax") {
        output = softmax(input_data);
    }
    log_detail( "Activation output" );
    log_matrix( output );
    return output; // this becomes input to the next Node or next Layer.
}
 
template <class T>
const aimatrix<T> Activation<T>::computeGradient(const aimatrix<T>& gradients, const aimatrix<T>& output, const aimatrix<T>& input) {
    aimatrix<T> dInput;

    if (activationtype == "softmax") {
        dInput = softmaxGradient(gradients, output);
    } else
    if (activationtype == "sigmoid") {
        dInput = sigmoidGradient(gradients, output);
    } else
    if (activationtype == "tanh") {
        dInput = tanhGradient(gradients, output);
    } else
    if (activationtype == "relu") {
        dInput = reluGradient(gradients, input);
    } else
    if (activationtype == "leakyrelu") {
        dInput = leakyReluGradient(gradients, input);
    } else
    if (activationtype == "gelu") {
        dInput = geluGradient(gradients, input);
    }

    return dInput; // this becomes input to the next Node or next Layer backward.
}

template <class T>
const aitensor<T> Activation<T>::forward(const aitensor<T>& input_data) { 

    log_info( "=====================================" );
    log_info( "Entering Activation Forward Pass ..." );

    // Cache for later back propagation.
    this->input_data = input_data;

    this->output_data.clear();

    // if input_data is from liner transform, then dimension is BxNxW
    this->batch_size = this->input_data.size();
    this->input_size = this->input_data.at(0).rows();
    this->param_size = this->input_data.at(0).cols();

    std::cout << "Batch Size:" <<  this->batch_size  << " Row: " << this->input_size << " Col:" << this->param_size << std::endl;

    for (int i = 0; i < this->batch_size; ++i) {

        aimatrix<T> input = this->input_data.at(i); 

        // Perform Activation
        this->output_data.push_back(computeActivation(input)); 
    }

    log_detail("Activation Result" );
    log_matrix(this->output_data.at(0));

    return this->output_data; // this becomes input to the next Node or next Layer.
}

template <class T>
const aitensor<T> Activation<T>::backward(const aitensor<T>& gradients) {
    log_info("=====================================");
    log_info( "Entering Activation Backward Pass ..." );

    aitensor<T> dInput;

    aimatrix<T> input, output, gradient;

    for (int i = 0; i < this->batch_size; ++i) {

        output   = this->output_data.at(i);
        
        input    = this->input_data.at(i); 

        gradient = gradients.at(i); 

        // Perform Gradient
        dInput.push_back(computeGradient(gradient, output, input)); 

    }

    log_detail("Activation Gradient Result" );
    log_matrix(dInput.at(0)); 

    // To support generateDotFormat()
    this->max_dInput = dInput.at(0).maxCoeff();
    this->min_dInput = dInput.at(0).minCoeff();

    return dInput; // this becomes input to the next Node or next Layer.
}
 
template <class T>
std::string Activation<T>::generateDotFormat() {
    std::string dot = "{* Activation (" + activationtype + ")*}|";
    T min_input = 0.0, max_input = 0.0;
    if (this->N != 0 || this->M != 0) 
    try {
       max_input = this->max_dInput; 
       min_input = this->min_dInput; 
    } catch (...) {};
    dot += "{dInput|min=" + scalar_to_string(min_input) + "|max=" + scalar_to_string(max_input) + "}";  
    return dot;
}

/*****************************************************************************************************
* Base Loss Functions
*****************************************************************************************************/

// Mean Squared Error. Returns (scalar)
// Expected input dimensions:  NxW ( N for input size, and W for feature size)
// Expected overall loss: Average along dimensions B and N.
template <class T>
const aiscalar<T> Loss<T>::mse(const aimatrix<T>& predicted, const aimatrix<T>& target) { 

    aimatrix<T> mse_loss = ( predicted.array() - target.array() ).square();

    // Mean along the W dimension.
    airowvector<T> overall_mse_loss = mse_loss.rowwise().mean();

    aiscalar<T> batch_mean_mse_loss = overall_mse_loss.mean();

    return batch_mean_mse_loss;  
}

template <class T>
const aimatrix<T> Loss<T>::mseGradient(const aimatrix<T>& predicted, const aimatrix<T>& target) {
    return 2 * (predicted.array() - target.array());
}

// Mean Squared Error. Returns (scalar)
// Expected input dimensions:  NxW ( N for input size, and W for feature size)
// Expected overall loss: Average along dimensions B and N.
template <class T>
const aiscalar<T> Loss<T>::bce(const aimatrix<T>& predicted, const aimatrix<T>& target) {
    aimatrix<T> bce_loss = -target.array() * predicted.array().log() - (1.0 - target.array()) * (1.0 - predicted.array()).log();

    // Sum along the W dimension for each combination of B and N
    // aimatrix<T> overall_bce_loss = bce_loss.mean(Eigen::array<int, 1>({2})); // Average along dimension W (axis 2)
    airowvector<T> overall_bce_loss = bce_loss.rowwise().mean();

    // aiscalar<T> batch_mean_bce_loss = overall_bce_loss.mean(Eigen::array<int, 1>({0, 1})); // Average along dimensions B and N
    aiscalar<T> batch_mean_bce_loss = overall_bce_loss.mean();

    // aimatrix<T> averageLoss = loss.mean();
    return batch_mean_bce_loss; // aimatrix<T>::Constant(1, 1, averageLoss);
}

template <class T>
const aimatrix<T> Loss<T>::bceGradient(const aimatrix<T>& predicted, const aimatrix<T>& target) {
    aimatrix<T> gradient = (predicted - target).array() / (predicted.array() * (1 - predicted.array()));
    return gradient;
}

// For Loss Categorical Cross Entropy. Usually, we use Softmax.
// If predicted and target has BxNxC dimension where C is number of classes, then result will be BxN.
template <class T>
const aiscalar<T> Loss<T>::cce(const aimatrix<T>& predicted, const aimatrix<T>& target) {

    // Calculate the CCE loss for each batch and instance (log likelihood)
    aimatrix<T> cce_loss = -predicted.array() * predicted.array().log();

    // Calculate the overall CCE loss by averaging along the class dimension (C)
    // aimatrix<T> overall_cce_loss = cce_loss.mean(Eigen::array<int, 1>({2}));
    airowvector<T> overall_cce_loss = cce_loss.rowwise().mean();

    // Calculate the mean loss along the batch (B)
    // aiscalar<T> batch_mean_cce_loss = overall_cce_loss.mean(Eigen::array<int, 1>({0, 1}))(0);
    aiscalar<T> batch_mean_cce_loss = overall_cce_loss.mean();

    return batch_mean_cce_loss;

}

template <class T>
const aimatrix<T> Loss<T>::cceGradient(const aimatrix<T>& predicted, const aimatrix<T>& target) {
    aimatrix<T> gradient = ( predicted.array() - target.array() );
    return gradient;
}

// For Support Vectors (not necessarily for Neural)
template <class T>
const aiscalar<T> Loss<T>::hingeLoss(const aimatrix<T>& predicted, const aimatrix<T>& target) {

    // Calculate the CCE loss for each batch and instance (log likelihood)
    aimatrix<T> hinge_loss = (1.0 - predicted.array() * target.array()).cwiseMax(0.0);

    // Calculate the overall CCE loss by averaging along the class dimension (C)
    // aimatrix<T> overall_hinge_loss = hinge_loss.mean(Eigen::array<int, 1>({2}));
    airowvector<T> overall_hinge_loss = hinge_loss.rowwise().mean();

    // Calculate the mean loss along the batch (B) and instance (N) dimensions
    // aiscalar<T> batch_mean_hinge_loss = overall_hinge_loss.mean(Eigen::array<int, 1>({0, 1}))(0);
    aiscalar<T> batch_mean_hinge_loss = overall_hinge_loss.mean();

    return batch_mean_hinge_loss;

}

// For Support Vectors (not necessarily for Neural)
template <class T>
const aimatrix<T> Loss<T>::hingeLossGradient(const aimatrix<T>& predicted, const aimatrix<T>& target) {
    aimatrix<T> gradient = (predicted.array() * target.array() < 1).select(-target, 0);
    return gradient;
}

template <class T>
const aiscalar<T> Loss<T>::computeLoss(const aitensor<T>& predicted, const aitensor<T>& target) { 
    aiscalar<T> output  = 0.0, total_output = 0.0;

    // if input_data is from linear transform, then dimension is BxNxW
    this->batch_size = predicted.size();
    this->input_size = predicted.at(0).rows();
    this->param_size = predicted.at(0).cols();

    aimatrix<T> batch_predicted, batch_target;

    if (predicted.size()        != target.size() || 
        predicted.at(0).rows()  != target.at(0).rows() || 
        predicted.at(0).cols()  != target.at(0).cols() ) {

        throw AIException("Dimension of Prediction and target do not match ...");

    }

    for (int i = 0; i < this->batch_size; ++i) {

        batch_predicted = predicted.at(i); 
        batch_target    = target.at(i); 

        if (losstype == "mse") {
            output = mse(batch_predicted, batch_target);
        } else
        if (losstype == "bce") {
            output = bce(batch_predicted, batch_target);
        } else
        if (losstype == "cce") {
            output = cce(batch_predicted, batch_target);
        } else
        if (losstype == "hingeLoss") {
            output = hingeLoss(batch_predicted, batch_target);
        } 
        total_output += output;

    }

    total_output = total_output / this->batch_size;

    return total_output; // this becomes input to the next Node or next Layer forward.
}

template <class T>
const aitensor<T> Loss<T>::computeGradients(const aitensor<T>& predicted, const aitensor<T>& target) { 
    aitensor<T> gradients;

    // if input_data is from linear transform, then dimension is BxNxW
    this->batch_size = predicted.size();
    this->input_size = predicted.at(0).rows();
    this->param_size = predicted.at(0).cols();

    aitensor<T> dInput; // (this->batch_size, this->input_size, this->param_size);

    aimatrix<T> batch_predicted, batch_target;

    for (int i = 0; i < this->batch_size; ++i) {

        batch_predicted = predicted.at(i); 
        batch_target    = target.at(i); 

        if (losstype == "mse") {
            dInput.push_back(mseGradient(batch_predicted, batch_target));
        } else
        if (losstype == "bce") {
            dInput.push_back(bceGradient(batch_predicted, batch_target));
        } else
        if (losstype == "cce") {
            dInput.push_back(cceGradient(batch_predicted, batch_target));
        } else
        if (losstype == "hingeLoss") {
            dInput.push_back(hingeLossGradient(batch_predicted, batch_target));
        } 
    }

    return dInput; // this becomes input to the next Node or next Layer backward.
}
  
/************ Basic Operators initialize templates ************/
 
template class Optimizer<float>;  // Instantiate with float
template class Optimizer<double>;  // Instantiate with double

template class Linear<float>;  // Instantiate with float
template class Linear<double>;  // Instantiate with double
 
template class BatchNorm<float>;  // Instantiate with float
template class BatchNorm<double>;  // Instantiate with double

template class LayerNorm<float>;  // Instantiate with float
template class LayerNorm<double>;  // Instantiate with double

template class Activation<float>;  // Instantiate with float
template class Activation<double>;  // Instantiate with double

template class Loss<float>;  // Instantiate with float
template class Loss<double>;  // Instantiate with double
