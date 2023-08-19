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
// SGD optimizer with optional step decay
template <class T>
void Optimizer<T>::sgd(const aimatrix<T>& weights, const aimatrix<T>& gradients, int currentEpoch ,
                bool useStepDecay, T decayRateStep, int decayStep) {
    // Update weights
    weights.array() -= learningRate * gradients.array();

    if (useStepDecay) {
        stepDecay(learningRate, decayRateStep, currentEpoch, decayStep);
    }
}

// Momentum optimizer with optional step decay
template <class T>
void Optimizer<T>::momentum(const aimatrix<T>& weights, const aimatrix<T>& gradients, int currentEpoch ,
                T momentumRate,
                bool useStepDecay, T decayRateStep,  int decayStep) {
    // Initialize Momentum optimizer variables
    //static Eigen::MatrixXd velocity = Eigen::MatrixXd::Zero(weights.rows(), weights.cols());
    if (velocity.cols() == 0 && velocity.rows() == 0) {
        velocity = Eigen::MatrixXd::Zero(weights.rows(), weights.cols());
    }

    // Update velocity
    velocity = momentumRate * velocity + learningRate * gradients;

    // Update weights
    weights.array() -= velocity.array();

    if (useStepDecay) {
        stepDecay(learningRate, decayRateStep, currentEpoch, decayStep);
    }
}

// Adam optimizer with optional step decay
template <class T>
void Optimizer<T>::adam(const aimatrix<T>& weights, const aimatrix<T>& gradients, int currentEpoch ,
                T beta1, T beta2, T epsilon,
                bool useStepDecay, T decayRateStep,  int decayStep) {

    log_info("============================");
    log_info("Entering Adam Optimation ...");

    if (moments.cols() == 0 && moments.rows() == 0) {
        moments = Eigen::MatrixXd::Zero(weights.rows(), weights.cols());
    }
    if (velocity.cols() == 0 && velocity.rows() == 0) {
        velocity = Eigen::MatrixXd::Zero(weights.rows(), weights.cols());
    }

    T beta1_t = std::pow(beta1, currentEpoch + 1);
    T beta2_t = std::pow(beta2, currentEpoch + 1);

    int doublePrecision = std::numeric_limits<T>::digits10;
    std::cout.precision(doublePrecision);

    log_detail( "Beta 1 Calculation: {:2.10f} based on beta1: {}", beta1_t, beta1 );
    log_detail( "Beta 2 Calculation: {:2.10f} based on beta2: {}", beta2_t, beta2 );

    log_detail( "Calculating: beta1 * moments.array()" );
    log_matrix( beta1 * moments.array() );

    log_detail( "Calculating: beta2 * velocity.array()" );
    log_matrix( beta2 * velocity.array() );


    // Update momentum and velocity
    moments = beta1 * moments.array() + (1 - beta1) * gradients.array();
    velocity = beta2 * velocity.array() + (1 - beta2) * (gradients.array() * gradients.array());

    log_detail( "Gradients" );
    log_matrix( gradients.array() );

    log_detail( "power of gradients" );
    log_matrix( gradients.array() * gradients.array() );


    log_detail( "momentum" );
    log_matrix( moments );
    log_detail( "velocity" );
    log_matrix( velocity );

    // Compute bias-corrected moment estimates
    Eigen::MatrixXd m_hat = moments / (1 - beta1_t);
    Eigen::MatrixXd v_hat = velocity / (1 - beta2_t);

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

// RMSprop optimizer with optional step decay
template <class T>
void Optimizer<T>::rmsprop(const aimatrix<T>& weights, const aimatrix<T>& gradients, int currentEpoch ,
                T rho, T epsilon,
                bool useStepDecay, T decayRateStep,  int decayStep) {
    // Initialize RMSprop optimizer variables
    if (rms.cols() == 0 && rms.rows() == 0) {
        rms = Eigen::MatrixXd::Zero(weights.rows(), weights.cols());
    }

    // Update RMSprop cache
    rms = rho * rms + (1 - rho) * (gradients.array() * gradients.array()).matrix();

    // Update weights
    weights.array() -= learningRate * (gradients.array() / (rms.array().sqrt() + epsilon));

    if (useStepDecay) {
        stepDecay(learningRate, decayRateStep, currentEpoch, decayStep);
    }
}

// Adagrad optimizer with optional step decay
template <class T>
void Optimizer<T>::adagrad(const aimatrix<T>& weights, const aimatrix<T>& gradients, int currentEpoch ,
                T epsilon,
                bool useStepDecay, T decayRateStep,  int decayStep) {
    // Initialize Adagrad optimizer variables
    // static Eigen::MatrixXd accum = Eigen::MatrixXd::Zero(weights.rows(), weights.cols());
    if (accum.cols() == 0 && accum.rows() == 0) {
        accum = Eigen::MatrixXd::Zero(weights.rows(), weights.cols());
    }

    // Update sum of squared gradients
    accum.array() += gradients.array() * gradients.array();

    // Update weights
    weights.array() -= learningRate * gradients.array() / (accum.array().sqrt() + epsilon);

    if (useStepDecay) {
        stepDecay(learningRate, decayRateStep, currentEpoch, decayStep);
    }
}

// Adamax optimizer with optional step decay
template <class T>
void Optimizer<T>::adamax(const aimatrix<T>& weights, const aimatrix<T>& gradients, int currentEpoch , 
                T beta1, T beta2, T epsilon,
                bool useStepDecay, T decayRateStep, int decayStep) {
    // Initialize Adamax optimizer variables
    if (moments.cols() == 0 && moments.rows() == 0) {
        moments = Eigen::MatrixXd::Zero(weights.rows(), weights.cols());
    }
    if (nu.cols() == 0 && nu.rows() == 0) {
        nu = Eigen::MatrixXd::Zero(weights.rows(), weights.cols());
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

// Nadam optimizer with optional step decay
template <class T>
void Optimizer<T>::nadam(const aimatrix<T>& weights, const aimatrix<T>& gradients, int currentEpoch , 
                T beta1, T beta2, T epsilon,
                bool useStepDecay, T decayRateStep, int decayStep) {

    if (moments.cols() == 0 && moments.rows() == 0) {
        moments = Eigen::MatrixXd::Zero(weights.rows(), weights.cols());
    }
    if (velocity.cols() == 0 && velocity.rows() == 0) {
        velocity = Eigen::MatrixXd::Zero(weights.rows(), weights.cols());
    }


    T beta1_t = std::pow(beta1, currentEpoch + 1);
    T beta2_t = std::pow(beta2, currentEpoch + 1);

    // Update momentum and velocity
    moments = beta1 * moments + (1 - beta1) * gradients;
    velocity = beta2 * velocity + (1 - beta2) * (gradients.array() * gradients.array()).matrix();

    // Compute bias-corrected moment estimates
    Eigen::MatrixXd m_hat = moments / (1 - beta1_t);
    Eigen::MatrixXd v_hat = velocity / (1 - beta2_t);

    // Update weights
    weights.array() -= learningRate * (beta1 * m_hat + (1 - beta1) * gradients).array()
                / (v_hat.array().sqrt() + epsilon);

    if (useStepDecay) {
        stepDecay(learningRate, decayRateStep, currentEpoch, decayStep);
    }
}

// Step decay for learning rate
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

    // Initialize Weights & Biases
    heInitialization(parameters.weights);
    // heInitialization(parameters.biases);
    parameters.biases.setConstant(T(0.01));

    //gradients.weights.resize(M, W); // allocates memory
    //gradients.biases.resize(W); // allocates memory

    // Initialize Gradients     
    //gradients.weights.setZero();
    //gradients.biases.setZero();

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
const aimatrix<T>& Linear<T>::linearTransform(const aimatrix<T>& input_data) {

    log_info("==================================");
    log_info("Entering Linear Transformation ...");

    // Initialize the parameters.
    setInitialWeights(input_data.cols());
    aimatrix<T>& weights = parameters.weights;
    airowvector<T>& biases = parameters.biases;

    log_detail( "Size of Input: {:d}", input_data.size() );
    log_detail( "Size of Weights: {:d}", weights.size() );
    log_detail( "Size of Biases: {:d}", biases.size() );
    log_detail( "---");
    log_detail( "Input" );
    log_matrix( input_data );
    log_detail( "Weights" );
    log_matrix( weights );
    log_detail( "Matrix Multiplication of Input and Weights (Standard)" );
    log_matrix( input_data * weights );
    log_detail( "Matrix Multiplication of Input and Weights (Using BaseOperator)" );
    log_matrix( BaseOperator::matmul(input_data, weights) );
    log_detail( "bias" );
    log_rowvector( biases );

    aimatrix<T> output;
    
    if (bias == true) {
        // rowwise() means, slice one row at a time and add to bias (which is 1 row horizontally).
        output = BaseOperator::matmul(input_data, weights).rowwise() + biases; // Ax + b = Wx + b; NxW dimension.
    } else {
        output = BaseOperator::matmul(input_data, weights); // Ax   = Wx; NxW dimension.
    }
    return output;
}

template <class T>
const aitensor<T>& Linear<T>::forward(const aitensor<T>& input_data) { 

    log_info("===============================================");
    log_info("Linear Transformation Forward Pass ...");

    // Cache for later back propagation.
    this->input_data = input_data;

    this->batch_size = input_data.dimension(0);
    this->input_size = input_data.dimension(1);
    this->embedding_size = input_data.dimension(2);

    const aitensor<T>& outputs(this->batch_size, this->input_size, this->W); // BxNxW
    aimatrix<T> input;

    for (int i = 0; i < this->batch_size; ++i) {

        input = input_data.chip(i, 0); // input_size x embedding_size

        log_detail( "Size of input:", this->input.size() );

        // Perform Linear Transformation.
        aimatrix<T> output = linearTransform(input);

        outputs.chip(i, 0) = output;
    }

    return outputs; // this becomes input to the next Node or next Layer.
}

template <class T>
OperationParams<T> Linear<T>::gradient_Wrt_Weight_Bias(const aimatrix<T>& new_gradients) {
    int N = new_gradients.rows();

    // initialize gradients for next iteration.
    OperationParams<T> gradients;
    gradients.weights.setZero();
    gradients.biases.setZero();

    log_detail("Computing Gradient with respect to bias:");
    
    log_detail( "Size of gradient: {:d}" , new_gradients.size() );
    log_detail( "Size of input: {:d}" , this->input_data.size() );
    log_detail( "Size of tranposed input: {:d}" , (this->input_data).size() );
    log_detail( "---");
    log_detail( "Before multiply of input and gradients");
    log_detail( "The input:" );
    log_matrix( this->input_data  );
    log_detail( "The gradient:"  );
    log_matrix( new_gradients );
    log_detail( "The initial gradient:" );
    log_matrix( gradients.weights );
    
    log_info("Multiply input and gradient.");
    // Compute the gradient with respect to the weights (dW)
    gradients.weights = BaseOperator::matmul(new_gradients.transpose(), this->input_data).transpose();  // dL/W = (dL/DC.T * x).T
    
    log_detail( "Computing Gradient with respect to weight:");
    log_matrix( gradients.weights );

    log_detail( "Add the bias." );

    // Compute the gradient with respect to the bias (db)
    if (bias == true)  gradients.biases = new_gradients.colwise().sum();

    log_detail(  "gbiases rows: {:d}, cols: {:d}" , gradients.biases.rows(), gradients.biases.cols() );
    log_rowvector( gradients.biases );
    
    log_detail( "Normalize the gradients." );
    // Normalize the gradients by dividing by the number of samples (N)
    gradients.weights /= N;  // dW - gradients of weights (MxW)
    if (bias == true) gradients.biases /= N;   // db - gradients of bias (1xW)

    log_detail( "Normalized gradients and biases:" );
    log_detail( "Gradient weights" );
    log_matrix( gradients.weights );
    log_detail( "Gadient biases" );
    log_rowvector( gradients.biases );

    // Return Gradients
    return gradients;
}

template <class T>
const aimatrix<T>& Linear<T>::gradient_Wrt_Input(const aimatrix<T>& new_gradients) { 
    int N = new_gradients.rows();

    log_detail( "Computing Gradient with respect to weight:" );
    log_detail( "The weight:" );
    log_matrix( parameters.weights );
    log_detail( "The gradient:" ) ;
    log_matrix( new_gradients );
    // Compute the gradient with respect to the input (dInput)
    const aimatrix<T>& dInput = BaseOperator::matmul(parameters.weights, new_gradients.transpose()).transpose();   // dL/x = (W * dL/DC.T)

    log_detail( "The computed gradient with respect to input:" );
    log_matrix( dInput );

    // Normalize the gradients by dividing by the number of samples (N)
    dInput /= N;  // dInput - gradients of input (NxM)

    log_detail( "normalized dInput"  );
    log_matrix(  dInput  );
    return dInput;
}

// Leave the gradients as is. They are cached in the Node. 
// They will be used to update the parameters in next parallel operations.
// the dInput is the gradient we propagate to source Nodes in the graph;
// while the parameter gradients get cached to be used to update the parameters later.
template <class T>
const aitensor<T>& Linear<T>::backward(const aitensor<T>& gradients) { 


    log_info("===========================================");
    log_info( "Entering Linear Gradient Backward pass ..." );

    aimatrix<T>& weights = parameters.weights;
    airowvector<T>& biases = parameters.biases;

    log_detail( "Size of gradients: {:d}" , gradients.size()  );
    log_detail( "Size of weights: {:d}"  , weights.size()  );
    log_detail( "Size of biases: {:d}" , biases.size()  );
    log_detail( "---" );
    log_detail( "Computing Gradient now ..." );

    const aitensor<T>& dInput(this->batch_size, this->input_size, this->embedding_size);

    aimatrix<T> gradient;

    for (int i = 0; i < this->batch_size; ++i) {

        gradient = gradients.chip(i, 0);

        OperationParams<T> wbgradients = gradient_Wrt_Weight_Bias(gradient);

        log_detail( "Computing Delta Error now ..."  );

        gradient = gradient_Wrt_Input(gradient);

        dInput.chip(i, 0) = gradient;

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

        aimatrix<T> pbiases = parameters.biases.matrix(); // convert vector to a 1xW matrix
        aimatrix<T> gbiases = gradients.biases.matrix();  // convert vector to a 1xW matrix

        log_detail( "Assigning ..." );

        log_detail( "pbiases rows: {:d}, cols: {:d}" ,  pbiases.rows() , pbiases.cols() );
        log_detail( "gbiases rows: {:d}, cols: {:d}", gbiases.rows() , gbiases.cols() );

        log_detail( "Gradient weights:" );
        log_matrix( gradients.weights  );
        log_detail( "Gradient biases:" );
        log_rowvector( gradients.biases );

        log_detail( "Before Updated weights:" );
        log_matrix( parameters.weights  );
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
            opt_biases->adam(pbiases, gbiases, iter);
            parameters.biases = pbiases.row(0);
        }

        log_detail( "Updated weights:" );
        log_matrix( parameters.weights );
        log_detail( "Updated biases:" );
        log_rowvector( parameters.biases  );

    }

    // initialize gradients for next iteration.
    vgradients.clear();
}

template <class T>
std::string Linear<T>::generateDotFormat(const std::string& name) {
    std::string dot = "{* Linear Transformation (" + name + ") *}|";  
    aiscalar<T> min_weights = 0.0, max_weights = 0.0;
    aiscalar<T> min_biases = 0.0, max_biases = 0.0;
    if (this->M != 0) // if weights are already initialized.
    try {
        min_weights = parameters.weights.minCoeff();
        max_weights = parameters.weights.maxCoeff();
        min_biases = parameters.biases.minCoeff();
        max_biases = parameters.biases.maxCoeff();
    } catch (...) {};
    dot += "{Weights|min=" + std::to_string(min_weights) + "|max=" + std::to_string(max_weights) + "}|";    
    dot += "{Biases|min=" + std::to_string(min_biases) + "|max=" + std::to_string(max_biases) + "}"; 
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
    heInitialization(scale);
    //heInitialization(shift);
    shift.setConstant(0.01);

    // Initialize Gradients     
    //dScale.setZero();
    //dShift.setZero();
}

template <class T>
std::tuple<aimatrix<T>, aimatrix<T>>& BatchNorm<T>::normalize(const aimatrix<T>& input_data) {

    log_info("=================================");
    log_info( "Entering Batch Normalization ..." );

    setInitialWeights(input_data.cols());

    log_detail( "Input Data:" );
    log_matrix( input_data );

    // Calculate batch mean along the N dimension, but along the M dimension.
    airowvector<T> batchMean = input_data.colwise().mean();

    log_detail( "Computing Mean ..." );
    log_rowvector( batchMean );

    // Calculate: X - mean
    minusMean = input_data.rowwise() - batchMean;

    log_detail( "minusMean" );
    log_matrix( minusMean );

    // Calculate batch variance along the N dimension
    airowvector<T> batchVariance = minusMean.array().square().colwise().mean();

    log_detail( "Variance ..." );
    log_rowvector( batchVariance );

    // Add a small epsilon for numerical stability
    airowvector<T> epsilonVector = airowvector<T>::Constant(batchVariance.size(), epsilon);

    log_detail( "Epsilon ..." );
    log_rowvector( epsilonVector );

    // Calculate batch standard deviation along the N dimension
    batchStdDev = (batchVariance + epsilonVector).cwiseSqrt();

    log_detail(  "stdDev ..." );
    log_rowvector( batchStdDev );

    // Normalize the inputs along the N dimension
    aimatrix<T> normalizedInput = minusMean.array().rowwise()  / batchStdDev.array();

    log_detail( "normalizedInput along the N  ..." );
    log_matrix( normalizedInput );

    // Scale and shift the normalized inputs along the N dimension.
    aimatrix<T> normalizedOutput = (normalizedInput.array().rowwise() * scale.array()).array().rowwise() + shift.array();

    log_detail( "scale ..." );
    log_rowvector( scale );

    log_detail( "shift ..." );
    log_rowvector( shift );

    log_detail( "normalizedOutput scaled ..." );
    log_matrix( normalizedOutput );

    return std::make_tuple(normalizedInput, normalizedOutput);
}

template <class T>
const aitensor<T>& BatchNorm<T>::forward(const aitensor<T>& input_data) { 

    // Cache for later back propagation.
    this->input_data = input_data;

    // if input_data is from liner transform, then dimension is BxNxW
    this->batch_size = input_data.dimension(0);
    this->input_size = input_data.dimension(1);
    this->param_size = input_data.dimension(2); 

    const aitensor<T>& normalizedOutput(this->batch_size, this->input_size, this->param_size);
    aimatrix<T> input, normInput, normOutput;

    for (int i = 0; i < this->input_size; ++i) {

        input = input_data.chip(i, 1); // batch_size x embedding_size

        log_detail( "Size of input:", this->input.size() );

        // Perform Linear Transformation.
        std::tie(normInput, normOutput) = normalize(input);

        normalizedInput.chip(i, 1) = normInput;
        normalizedOutput.chip(i, 1) = normOutput;

    }

    return normalizedOutput; // this becomes input to the next Node or next Layer.
}

// Leave the gradients as is for the scale and shift. They are cached in the Node. 
// They will be used to update the parameters in next parallel operations.
// the dInput is the gradient we propagate to source Nodes in the graph;
// while the parameter gradients get cached to be used to update the parameters later.
template <class T>
const aitensor<T>& BatchNorm<T>::backward(const aitensor<T>& gradients) {

    int N = this->input_data.rows();

    log_info("==============================================");
    log_info("Entering Batch Normalization Backward pass ...");


    const aitensor<T>& dInput(this->batch_size, this->input_size, this->param_size);

    aimatrix<T> gradient, normInput;

    for (int i = 0; i < this->input_size; ++i) {

        gradient = gradients.chip(i, 1);

        normInput = normalizedInput.chip(i, 1);

        // Compute the gradient with respect to the scale parameter (Gamma)
        airowvector<T> dScale = (gradient.array() * normInput.array()).colwise().sum();
        vgscale.push_back(dScale);

        // Compute the gradient with respect to the shift parameter (Beta)
        airowvector<T> dShift = gradient.colwise().sum();
        vgshift.push_back(dShift);

        log_detail( "scale and shift" );
        log_rowvector( scale );
        log_rowvector( shift );

        log_detail( "dScale and dShift gradients" );
        log_rowvector( dScale );
        log_rowvector( dShift );

        // Compute the gradient with respect to the normalized input
        aimatrix<T> dNormalizedInput = gradient.array().rowwise() * scale.array();

        log_detail( "dNormalizedInput" );
        log_matrix( dNormalizedInput );

        log_detail( "dNormalizedInput * minusMean" );
        log_matrix( dNormalizedInput.array() * minusMean.array() );

        log_detail( "dNormalizedInput * minusMean rowwise sum" );
        log_matrix( (dNormalizedInput.array() * minusMean.array()).rowwise().sum() );

        log_detail( "barchStdDev" );
        log_rowvector( batchStdDev.array() );

        log_detail( "layerStdDev * layerStdDev"  );
        log_rowvector( batchStdDev.array() * batchStdDev.array() );

        // Compute the gradient with respect to the batch standard deviation
        airowvector<T> dBatchStdDev = -(dNormalizedInput.array() * minusMean.array()).colwise().sum() / (batchStdDev.array() * batchStdDev.array());

        // Compute the gradient with respect to the batch variance
        airowvector<T> dBatchVariance = 0.5 * (dBatchStdDev.array() / batchStdDev.array());

        log_detail( "dBatchVariance" );
        log_rowvector( dBatchVariance );

        aimatrix<T> dNormMinusMean1 = (dNormalizedInput.array() * (1.0 / (batchStdDev.array())).replicate(N,1)) + 
                                            (2.0 * minusMean.array() * (1.0/N * dBatchVariance.replicate(N, 1)).array());

        log_detail( "istd" );
        log_matrix( (1.0/(batchStdDev.array())).replicate(N,1) );

        log_detail( "dsq" );
        log_matrix( 1.0/N * dBatchVariance.replicate(N, 1) );

        log_detail( "xmu" );
        log_matrix( minusMean  );

        // Compute the gradient with respect to the batch mean
        airowvector<T> dBatchMean = -1.0 * dNormMinusMean1.array().colwise().sum();

        log_detail(  "dBatchMean" );
        log_rowvector( dBatchMean );

        aimatrix<T> dNormMinusMean2 = 1.0/N *  dBatchMean.replicate(N,1).array();

        log_detail( "dNormMinusMean2" );
        log_matrix( dNormMinusMean2 );

        // Compute the gradient with respect to the input
        gradient = dNormMinusMean1.array() + dNormMinusMean2.array();
        log_detail( "gradient" );;
        log_matrix( gradient );

        dInput.chip(i, 1) = gradient;

    }

    return dInput;
}

template <class T>
void BatchNorm<T>::updateParameters(std::string& optimizertype, T& learningRate, int& iter) {

    log_info("==================================================");
    log_info("Entering Batch Normalization Upgrade Parameters...");


    for (int i = 0; i < this->input_size; ++i) {

        aimatrix<T> pscale(1, this->M); 
        aimatrix<T> gscale(1, this->M); 
        aimatrix<T> pshift(1, this->M); 
        aimatrix<T> gshift(1, this->M); 

        pscale.row(0) = this->scale;
        gscale.row(0) = vgscale[i];

        pshift.row(0) = this->shift;
        gshift.row(0) = vgshift[i];

        if (opt_scale == nullptr) {
            opt_scale = new Optimizer<T>(optimizertype, learningRate);
        }

        if (opt_shift == nullptr) {
            opt_shift = new Optimizer<T>(optimizertype, learningRate);
        }

        log_detail( "Updating Scale" );

        opt_scale->adam(pscale, gscale, iter);

        log_detail( "Updating Shift" );

        opt_shift->adam(pshift, gshift, iter);

        this->scale = pscale.row(0);
        this->shift = pshift.row(0);

        log_detail( "Updated scale" );
        log_rowvector( this->scale );
        log_detail( "Updated shift" );
        log_rowvector( this->shift );

    }
    
    // initialize gradients for next iteration.
    vgscale.empty();
    vgshift.empty();
}

template <class T>
std::string BatchNorm<T>::generateDotFormat(const std::string& name) {
    std::string dot = "{* Batch Normalization (" + name + ") *}|";  
    aiscalar<T> min_scale = 0.0, max_scale = 0.0;
    aiscalar<T> min_shift = 0.0, max_shift = 0.0;
    if (this->M != 0) // if weights are already initialized.
    try {
        max_scale = scale.maxCoeff();
        min_scale = scale.minCoeff();
        max_shift = shift.maxCoeff();
        min_shift = shift.minCoeff();
    } catch (...) {};
    dot += "{Shape|min=" + std::to_string(min_scale) + "|max=" + std::to_string(max_scale) + "}|";
    dot += "{Shift|min=" + std::to_string(min_shift) + "|max=" + std::to_string(max_shift) + "}";
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
    heInitialization(scale);
    // heInitialization(shift);
    shift.setConstant(0.01);

    // Initialize Gradients     
    // dScale.setZero();
    // dShift.setZero();
}

template <class T>
std::tuple<aimatrix<T>, aimatrix<T>>& LayerNorm<T>::normalize(const aimatrix<T>& input_data) {
    log_info("=================================");
    log_info( "Entering Layer Normalization ..." );

    setInitialWeights(input_data.rows());

    log_detail(" Input Data ...");
    log_matrix( input_data );

    // Calculate layer mean along the M dimension, but along the N dimension.
    aivector<T> layerMean = input_data.rowwise().mean();

    log_detail( "Mean ..." );
    log_vector( layerMean );

    // Calculate: X - mean
    minusMean = input_data.colwise() - layerMean;

    log_detail( "minusMean" );
    log_matrix( minusMean );

    // Calculate batch variance along the M dimension
    aivector<T> layerVariance = minusMean.array().square().rowwise().mean();

    log_detail( "Variance ..." );
    log_vector( layerVariance );

    // Add a small epsilon for numerical stability
    aivector<T> epsilonVector = Eigen::VectorXd::Constant(layerVariance.size(), epsilon);

    log_detail( "Epsilon ..." );
    log_vector( epsilonVector );

    // Calculate batch standard deviation across the M dimension
    layerStdDev = (layerVariance + epsilonVector).array().sqrt();

    log_detail( "stdDev ..." );
    log_vector( layerStdDev );

    // Normalize the inputs along the M dimension
    aimatrix<T> normalizedInput = minusMean.array().colwise()  / layerStdDev.array();

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

    return std::make_tuple(normalizedInput, normalizedOutput);
}

template <class T>
const aitensor<T>& LayerNorm<T>::forward(const aitensor<T>& input_data) { 

    // Cache for later back propagation.
    this->input_data = input_data;

    // if input_data is from liner transform, then dimension is BxNxW
    this->batch_size = input_data.dimension(0);
    this->input_size = input_data.dimension(1);
    this->param_size = input_data.dimension(2);

    const aitensor<T>& normalizedOutput(this->batch_size, this->input_size, this->param_size);
    aimatrix<T> input, normInput, normOutput;

    for (int i = 0; i < this->batch_size; ++i) {

        input = input_data.chip(i, 0); // input_size x param_size

        log_detail( "Size of input:", this->input.size() );

        // Perform Linear Transformation.
        std::tie(normInput, normOutput) = normalize(input);

        normalizedInput.chip(i, 0) = normInput;
        normalizedOutput.chip(i, 0) = normOutput;

    }

    return normalizedOutput; // this becomes input to the next Node or next Layer.
}

// Leave the gradients as is for the scale and shift. They are cached in the Node. 
// They will be used to update the parameters in next parallel operations.
// the dInput is the gradient we propagate to source Nodes in the graph;
// while the parameter gradients get cached to be used to update the parameters later.
template <class T>
const aitensor<T>& LayerNorm<T>::backward(const aitensor<T>& gradients) {

    int W = this->input_data.cols();

    log_info("==============================================");
    log_info("Entering Layer Normalization Backward pass ...");


    const aitensor<T>& dInput(this->batch_size, this->input_size, this->param_size);

    aimatrix<T> gradient, normInput;

    for (int i = 0; i < this->batch_size; ++i) {

        gradient = gradients.chip(i, 0);

        normInput = normalizedInput.chip(i, 0);

        // Compute the gradient with respect to the scale parameter (Gamma)
        aivector<T> dScale = (gradient.array() * normInput.array()).rowwise().sum();
        vgscale.push_back(dScale);

        // Compute the gradient with respect to the shift parameter (Beta)
        aivector<T> dShift = gradient.rowwise().sum();
        vgshift.push_back(dShift);

        log_detail( "scale and shift gradients" );
        log_vector( scale );
        log_vector( shift );

        log_detail( "dScale and dShift gradients" );
        log_vector( dScale );
        log_vector( dShift );

        // Compute the gradient with respect to the normalized input
        aimatrix<T> dNormalizedInput = gradient.array().colwise() * scale.array();

        log_detail( "dNormalizedInput" );
        log_matrix( dNormalizedInput );

        log_detail( "dNormalizedInput * minusMean" );
        log_matrix( dNormalizedInput.array() * minusMean.array() );

        log_detail( "dNormalizedInput * minusMean rowwise sum" );
        log_matrix( (dNormalizedInput.array() * minusMean.array()).rowwise().sum() );

        log_detail( "layerStdDev" );
        log_vector( layerStdDev.array() );

        log_detail( "layerStdDev * layerStdDev" );
        log_vector( (layerStdDev.array() * layerStdDev.array()) );

        // Compute the gradient with respect to the layer standard deviation
        aivector<T> dLayerStdDev = -(dNormalizedInput.array() * minusMean.array()).rowwise().sum() / (layerStdDev.array() * layerStdDev.array());

        log_detail( "dLayerStdDev" );
        log_vector( dLayerStdDev );

        // Compute the gradient with respect to the layer variance
        aivector<T> dLayerVariance = 0.5 * (dLayerStdDev.array() / layerStdDev.array());

        log_detail( "dLayerVariance" );
        log_vector( dLayerVariance );

        aimatrix<T> dNormMinusMean1 = (dNormalizedInput.array() * (1.0 / (layerStdDev.array())).replicate(1,W)) + 
                                            (2.0 * minusMean.array() * (1.0/W * dLayerVariance.replicate(1,W)).array());

        log_detail( "dxmu1" );
        log_matrix( (dNormalizedInput.array() * (1.0 / (layerStdDev.array())).replicate(1,W)) );
        log_detail( "dxmu2" );
        log_matrix(  (2.0 * minusMean.array() * (1.0/W * dLayerVariance.replicate(1,W)).array())  );

        log_detail( "dNormMinusMean1" );
        log_matrix( dNormMinusMean1  );

        log_detail( "istd" );
        log_matrix( (1.0/(layerStdDev.array() * layerStdDev.array())).replicate(1,W)  );

        log_detail( "dsq" );
        log_matrix( 1.0/N * dLayerVariance.replicate(1, 2) );

        log_detail( "xmu" );
        log_matrix( minusMean );

        // Compute the gradient with respect to the batch mean
        aivector<T> dLayerMean = -1.0 * dNormMinusMean1.array().rowwise().sum();

        log_detail( "dLayerMean" );
        log_vector( dLayerMean );

        aimatrix<T> dNormMinusMean2 = 1.0/W *  dLayerMean.replicate(1,W).array();

        log_detail("dNormMinusMean2" );
        log_matrix( dNormMinusMean2 );

        // Compute the gradient with respect to the input
        gradient = dNormMinusMean1.array() + dNormMinusMean2.array();

        log_detail( "gradient" );
        log_matrix( gradient );

        dInput.chip(i, 0) = gradient;
    
    }

    return dInput;
}

template <class T>
void LayerNorm<T>::updateParameters(std::string& optimizertype, T& learningRate, int& iter) {

    log_info("==================================================");
    log_info("Entering Batch Normalization Upgrade Parameters...");


    for (int i = 0; i < this->batch_size; ++i) {

        aimatrix<T> pscale(1, this->N, 1); 
        aimatrix<T> gscale(1, this->N, 1); 
        aimatrix<T> pshift(1, this->N, 1); 
        aimatrix<T> gshift(1, this->N, 1); 

        pscale.col(0) = this->scale;
        gscale.col(0) = vgscale[i];

        pshift.col(0) = this->shift;
        gshift.col(0) = vgshift[i];

        if (opt_scale == nullptr) {
            opt_scale = new Optimizer<T>(optimizertype, learningRate);
        }

        if (opt_shift == nullptr) {
            opt_shift = new Optimizer<T>(optimizertype, learningRate);
        }

        log_detail( "Updating Scale" );

        opt_scale->adam(pscale, gscale, iter);

        log_detail( "Updating Shift" );

        opt_shift->adam(pshift, gshift, iter);

        this->scale = pscale.col(0);
        this->shift = pshift.col(0);

        log_detail( "Updated scale" );
        log_vector( this->scale );
        log_detail( "Updated shift" );
        log_vector( this->shift );

    }
    
    // initialize gradients for next iteration.
    vgscale.empty();
    vgshift.empty();
}

template <class T>
std::string LayerNorm<T>::generateDotFormat(const std::string& name) {
    std::string dot = "{* Layer Normalization (" + name + ") *}|"; 
    aiscalar<T> min_scale = 0.0, max_scale = 0.0;
    aiscalar<T> min_shift = 0.0, max_shift = 0.0;
    if (this->N != 0) // if weights are already initialized.
    try {
        max_scale = scale.maxCoeff();
        min_scale = scale.minCoeff();
        max_shift = shift.maxCoeff();
        min_shift = shift.minCoeff();
    } catch (...) {};
    dot += "{Shape|min=" + std::to_string(min_scale) + "|max=" + std::to_string(max_scale) + "}|";
    dot += "{Shift|min=" + std::to_string(min_shift) + "|max=" + std::to_string(max_shift) + "}";
    return dot;   
}

/*****************************************************************************************************
* Base Activation Functions
*****************************************************************************************************/
template <class T>
const aitensor<T>& Activation<T>::relu(const aitensor<T>& x) {
    return x.array().max(0.0);
}

/*****************************************************************************************************
 * So, the gradient of the ReLU function with respect to its input (z) can be expressed as follows:
 * dy/dz = propagated_gradient * 1 for z > 0
 * dy/dz = propagated_gradient * 0 for z <= 0
 *****************************************************************************************************/
template <class T>
const aitensor<T>& Activation<T>::reluGradient(const aitensor<T>& gradients) {
   const aitensor<T>& dInput = this->input_data.array().max(T(0.0)).template cast<bool>().template cast<T>() * gradients.array();
    return dInput;
}

template <class T>
const aitensor<T>& Activation<T>::leakyReLU(const aitensor<T>& x, float alpha) {
    return x.array().max(alpha * x.array());
}

/*****************************************************************************************************
 * So, the gradient of the LeakyReLU function with respect to its input (z) can be expressed as follows:
 * dy/dz = propagated_gradient * 1 for z > 0
 * dy/dz = propagated_gradient * alpha for z <= 0
*****************************************************************************************************/
template <class T>
const aitensor<T>& Activation<T>::leakyReluGradient(const aitensor<T>& gradients) {
    const aitensor<T>& dInput = this->input_data.array().max(T(0.0)).template cast<bool>().template cast<T>().max(alpha) * gradients.array();
    return dInput;
}

template <class T>
const aitensor<T>& Activation<T>::gelu(const aitensor<T>& x) {
    return 0.5 * x.array() * (1.0 + ((x.array() * std::sqrt(2.0 / M_PI)).tanh()));
}

/*****************************************************************************************************
 * Gelu Gradient ...
*****************************************************************************************************/
template <class T>
const aitensor<T>& Activation<T>::geluGradient(const aitensor<T>& gradients) {
    // Calculate the coefficient used in the GELU formula
    T coefficient = sqrt(2.0 / M_PI);
    // Compute the cumulative distribution function (CDF) part of the GELU gradient
    const aitensor<T>& cdf = 0.5 * (1.0 + (gradients.array() / coefficient).tanh());
    // Compute the probability density function (PDF) part of the GELU gradient
    const aitensor<T>& pdf = exp(-0.5 * gradients.array().square()) / coefficient;
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
void Activation<T>::setInitSize(const aitensor<T>& input_data) {

    // if size is already set, 
    // it means weights have previously already been set by an initial forward pass
    if (this->N != 0 || this->M != 0) return;

    this->N = input_data.rows();
    this->M = input_data.cols();

    // Initialize scaling and shifting parameters   
    this->dInput.resize(this->N, this->M); // allocates memory
}

template <class T>
const aitensor<T>& Activation<T>::computeActivation(const aitensor<T>& input_data) { 
    const aitensor<T>& output;

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
const aitensor<T>& Activation<T>::computeGradient(const aitensor<T>& gradients) {
    aitensor<T> dInput;
    const aitensor<T>& output_data = this->output_data;

    if (activationtype == "sigmoid") {
        dInput = sigmoidGradient(gradients, output_data);
    } else
    if (activationtype == "tanh") {
        dInput = tanhGradient(gradients, output_data);
    } else
    if (activationtype == "relu") {
        dInput = reluGradient(gradients);
    } else
    if (activationtype == "leakyrelu") {
        dInput = leakyReluGradient(gradients);
    } else
    if (activationtype == "gelu") {
        dInput = geluGradient(gradients);
    } else
    if (activationtype == "softmax") {
        dInput = softmaxGradient(gradients, output_data);
    } 
    return dInput; // this becomes input to the next Node or next Layer backward.
}

template <class T>
const aitensor<T>& Activation<T>::forward(const aitensor<T>& input_data) { 

    log_info( "=====================================" );
    log_info( "Entering Activation Forward Pass ..." );

    // Cache for later back propagation.
    this->input_data = input_data;

    // Perform Activation
    this->output_data = computeActivation(input_data);

    log_detail("Activation Result" );
    log_matrix( this->output_data );

    return this->output_data; // this becomes input to the next Node or next Layer.
}

template <class T>
const aitensor<T>& Activation<T>::backward(const aitensor<T>& gradients) {
    log_info("=====================================");
    log_info( "Entering Activation Backward Pass ..." );

    // Perform Gradient
    aitensor<T> dInput = computeGradient(gradients);

    log_detail("Activation Gradient Result" );
    log_matrix( dInput );

    return dInput; // this becomes input to the next Node or next Layer.
}

template <class T>
std::string Activation<T>::generateDotFormat() {
    std::string dot = "{* Activation (" + activationtype + ")*}|";
    aiscalar<T> min_input = 0.0, max_input = 0.0;
    if (this->N != 0 || this->M != 0) 
    try {
       max_input = this->dInput.maxCoeff();
       min_input = this->dInput.minCoeff();
    } catch (...) {};
    dot += "{dInput|min=" + std::to_string(min_input) + "|max=" + std::to_string(max_input) + "}";  
    return dot;
}

/*****************************************************************************************************
* Base Loss Functions
*****************************************************************************************************/

// Mean Squared Error. Returns (scalar)
// Expected input dimensions:  BxNxW (B for batch size, N for input size, and W for feature size)
// Expected intermediate output: BxN, if taking mse along the W dimension
// Expected overall loss: Average along dimensions B and N.
template <class T>
const aiscalar<T>& Loss<T>::mse(const aitensor<T>& predicted, const aitensor<T>& target) { 

    aitensor<T> mse_loss = ( predicted.array() - target.array() ).square();

    // Sum along the W dimension for each combination of B and N
    aimatrix<T> overall_mse_loss = mse_loss.mean(Eigen::array<int, 1>({2})); // Average along dimension W (axis 2)

    aiscalar<T> batch_mean_mse_loss = overall_mse_loss.mean(Eigen::array<int, 1>({0, 1})); // Average along dimensions B and N

    // aimatrix<T> mse = diff.array().square().rowwise().sum().mean();
    return batch_mean_mse_loss; // aimatrix<T>::Constant(1, 1, mse);
}

template <class T>
const aitensor<T>& Loss<T>::mseGradient(const aitensor<T>& predicted, const aitensor<T>& target) {
    return 2 * (predicted.array() - target.array());
}

// Binary Cross Entropy.  Returns (scalar)
// Expected input dimensions:  BxNxW (B for batch size, N for input size, and W for feature size)
// Expected intermediate output: BxN, if taking mse along the W dimension
// Expected overall loss: Average along dimensions B and N.
template <class T>
const aiscalar<T>& Loss<T>::bce(const aitensor<T>& predicted, const aitensor<T>& target) {
    aitensor<T> bce_loss = -target.array() * predicted.array().log() - (1.0 - target.array()) * (1.0 - predicted.array()).log();

    // Sum along the W dimension for each combination of B and N
    aimatrix<T> overall_bce_loss = bce_loss.mean(Eigen::array<int, 1>({2})); // Average along dimension W (axis 2)

    aiscalar<T> batch_mean_bce_loss = overall_bce_loss.mean(Eigen::array<int, 1>({0, 1})); // Average along dimensions B and N

    // aimatrix<T> averageLoss = loss.mean();
    return batch_mean_bce_loss; // aimatrix<T>::Constant(1, 1, averageLoss);
}

template <class T>
const aitensor<T>& Loss<T>::bceGradient(const aitensor<T>& predicted, const aitensor<T>& target) {
    aitensor<T> gradient = (predicted - target).array() / (predicted.array() * (1 - predicted.array()));
    return gradient;
}

// For Loss Categorical Cross Entropy. Usually, we use Softmax.
// If predicted and target has BxNxC dimension where C is number of classes, then result will be BxN.
template <class T>
const aiscalar<T>& Loss<T>::cce(const aitensor<T>& predicted, const aitensor<T>& target) {

    // Calculate the CCE loss for each batch and instance (log likelihood)
    aitensor<T> cce_loss = -predicted.array() * predicted.array().log();

    // Calculate the overall CCE loss by averaging along the class dimension (C)
    aimatrix<T> overall_cce_loss = cce_loss.mean(Eigen::array<int, 1>({2}));

    // Calculate the mean loss along the batch (B) and instance (N) dimensions
    aiscalar<T> batch_mean_cce_loss = overall_cce_loss.mean(Eigen::array<int, 1>({0, 1}))(0);

    return batch_mean_cce_loss;

}

template <class T>
const aitensor<T>& Loss<T>::cceGradient(const aitensor<T>& predicted, const aitensor<T>& target) {
    aitensor<T> gradient = ( predicted.array() - target.array() );
    return gradient;
}

// For Support Vectors (not necessarily for Neural)
template <class T>
const aiscalar<T>& Loss<T>::hingeLoss(const aitensor<T>& predicted, const aitensor<T>& target) {

    // Calculate the CCE loss for each batch and instance (log likelihood)
    aitensor<T> hinge_loss = (1.0 - predicted.array() * target.array()).cwiseMax(0.0);

    // Calculate the overall CCE loss by averaging along the class dimension (C)
    aimatrix<T> overall_hinge_loss = hinge_loss.mean(Eigen::array<int, 1>({2}));

    // Calculate the mean loss along the batch (B) and instance (N) dimensions
    aiscalar<T> batch_mean_hinge_loss = overall_hinge_loss.mean(Eigen::array<int, 1>({0, 1}))(0);

    return batch_mean_hinge_loss;

}

// For Support Vectors (not necessarily for Neural)
template <class T>
const aitensor<T>& Loss<T>::hingeLossGradient(const aitensor<T>& predicted, const aitensor<T>& target) {
    aitensor<T> gradient = (predicted.array() * target.array() < 1).select(-target, 0);
    return gradient;
}

template <class T>
const aiscalar<T>& Loss<T>::computeLoss(const aitensor<T>& predicted, const aitensor<T>& target) { 
    aiscalar<T> output;
    if (losstype == "mse") {
        output = mse(predicted, target);
    } else
    if (losstype == "bce") {
        output = bce(predicted, target);
    } else
    if (losstype == "cce") {
        output = cce(predicted, target);
    } else
    if (losstype == "hingeLoss") {
        output = hingeLoss(predicted, target);
    } 
    return output; // this becomes input to the next Node or next Layer forward.
}

template <class T>
const aitensor<T>& Loss<T>::computeGradients(const aitensor<T>& predicted, const aitensor<T>& target) { 
    aitensor<T> gradients;
    if (losstype == "mse") {
        gradients = mseGradient(predicted, target);
    } else
    if (losstype == "bce") {
        gradients = bceGradient(predicted, target);
    } else
    if (losstype == "cce") {
        gradients = cceGradient(predicted, target);
    } else
    if (losstype == "hingeLoss") {
        gradients = hingeLossGradient(predicted, target);
    } 
    return gradients; // this becomes input to the next Node or next Layer backward.
}