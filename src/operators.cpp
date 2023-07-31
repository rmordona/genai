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
void Optimizer::sgd(Eigen::MatrixXd& weights, Eigen::MatrixXd& gradients, int currentEpoch ,
                bool useStepDecay, double decayRateStep, int decayStep) {
    // Update weights
    weights.array() -= learningRate * gradients.array();

    if (useStepDecay) {
        stepDecay(learningRate, decayRateStep, currentEpoch, decayStep);
    }
}

// Momentum optimizer with optional step decay
void Optimizer::momentum(Eigen::MatrixXd& weights, Eigen::MatrixXd& gradients, int currentEpoch ,
                double momentumRate,
                bool useStepDecay, double decayRateStep,  int decayStep) {
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
void Optimizer::adam(Eigen::MatrixXd& weights, Eigen::MatrixXd& gradients, int currentEpoch ,
                double beta1, double beta2, double epsilon,
                bool useStepDecay, double decayRateStep,  int decayStep) {

    log_info("============================");
    log_info("Entering Adam Optimation ...");

    if (moments.cols() == 0 && moments.rows() == 0) {
        moments = Eigen::MatrixXd::Zero(weights.rows(), weights.cols());
    }
    if (velocity.cols() == 0 && velocity.rows() == 0) {
        velocity = Eigen::MatrixXd::Zero(weights.rows(), weights.cols());
    }

    double beta1_t = std::pow(beta1, currentEpoch + 1);
    double beta2_t = std::pow(beta2, currentEpoch + 1);

    int doublePrecision = std::numeric_limits<double>::digits10;
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
void Optimizer::rmsprop(Eigen::MatrixXd& weights, Eigen::MatrixXd& gradients, int currentEpoch ,
                double rho, double epsilon,
                bool useStepDecay, double decayRateStep,  int decayStep) {
    // Initialize RMSprop optimizer variables
    // static Eigen::MatrixXd rms = Eigen::MatrixXd::Zero(weights.rows(), weights.cols());
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
void Optimizer::adagrad(Eigen::MatrixXd& weights, Eigen::MatrixXd& gradients, int currentEpoch ,
                double epsilon,
                bool useStepDecay, double decayRateStep,  int decayStep) {
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
void Optimizer::adamax(Eigen::MatrixXd& weights, Eigen::MatrixXd& gradients, int currentEpoch , 
                double beta1, double beta2, double epsilon,
                bool useStepDecay, double decayRateStep, int decayStep) {
    // Initialize Adamax optimizer variables
    //static Eigen::MatrixXd m = Eigen::MatrixXd::Zero(weights.rows(), weights.cols());
    //static Eigen::MatrixXd u = Eigen::MatrixXd::Zero(weights.rows(), weights.cols());
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
void Optimizer::nadam(Eigen::MatrixXd& weights, Eigen::MatrixXd& gradients, int currentEpoch , 
                double beta1, double beta2, double epsilon,
                bool useStepDecay, double decayRateStep, int decayStep) {
    // Initialize Nadam optimizer variables
    //static Eigen::MatrixXd m = Eigen::MatrixXd::Zero(weights.rows(), weights.cols());
    //static Eigen::MatrixXd v = Eigen::MatrixXd::Zero(weights.rows(), weights.cols());
    if (moments.cols() == 0 && moments.rows() == 0) {
        moments = Eigen::MatrixXd::Zero(weights.rows(), weights.cols());
    }
    if (velocity.cols() == 0 && velocity.rows() == 0) {
        velocity = Eigen::MatrixXd::Zero(weights.rows(), weights.cols());
    }


    double beta1_t = std::pow(beta1, currentEpoch + 1);
    double beta2_t = std::pow(beta2, currentEpoch + 1);

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
void Optimizer::stepDecay(double& learningRate, double decayRate, int currentEpoch, int decayStep) {
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
void Linear::setInitialWeights(int M) {

    // if size is already set, 
    // it means weights have previously already been set by an initial forward pass
    if (this->M != 0) return;

    this->M = M;

    parameters.weights.resize(M, W); // allocates memory
    parameters.biases.resize(W); // allocates memory

    // Initialize Weights & Biases
    heInitialization(parameters.weights);
    // heInitialization(parameters.biases);
    parameters.biases.setConstant(0.01);

    gradients.weights.resize(M, W); // allocates memory
    gradients.biases.resize(W); // allocates memory

    // Initialize Gradients     
    gradients.weights.setZero();
    gradients.biases.setZero();

}
 
OperationParams Linear::getParameters() const {
    return parameters;
}
 
OperationParams Linear::getGradients() const {
    return gradients;
}

// While the parameter weight has dimension MxW,  the resulting transformation has dimension of NxW.
// We only need the M dimension from an NxM input to generate parameter matrix.
// where weights is NxW and bias is W.
Eigen::MatrixXd Linear::linearTransform(const Eigen::MatrixXd& input_data) {

    log_info("==================================");
    log_info("Entering Linear Transformation ...");

    // Initialize the parameters.
    setInitialWeights(input_data.cols());
    Eigen::MatrixXd& weights = parameters.weights;
    Eigen::RowVectorXd& biases = parameters.biases;

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

    Eigen::MatrixXd output;
    
    if (bias == true) {
        // rowwise() means, slice one row at a time and add to bias (which is 1 row horizontally).
        output = BaseOperator::matmul(input_data, weights).rowwise() + biases; // Ax + b = Wx + b; NxW dimension.
    } else {
        output = BaseOperator::matmul(input_data, weights); // Ax   = Wx; NxW dimension.
    }
    return output;
}

Eigen::MatrixXd Linear::forward(Eigen::MatrixXd& input_data) { 

    log_info("===============================================");
    log_info("Linear Transformation Forward Pass ...");

    // Cache for later back propagation.
    this->input_data = input_data;

    log_detail( "Size of input:", this->input_data.size() );

    // Perform Linear Transformation.
    Eigen::MatrixXd output = linearTransform(input_data);

    return output; // this becomes input to the next Node or next Layer.
}

void Linear::gradient_Wrt_Weight_Bias(Eigen::MatrixXd& gradient) {
    int N = gradient.rows();

    log_detail("Computing Gradient with respect to bias:");
    
    log_detail( "Size of gradient: {:d}" , gradient.size() );
    log_detail( "Size of input: {:d}" , this->input_data.size() );
    log_detail( "Size of tranposed input: {:d}" , (this->input_data).size() );
    log_detail( "---");
    log_detail( "Before multiply of input and gradients");
    log_detail( "The input:" );
    log_matrix( this->input_data  );
    log_detail( "The gradient:"  );
    log_matrix( gradient );
    log_detail( "The initial gradient:" );
    log_matrix( gradients.weights );
    
    log_info("Multiply input and gradient.");
    // Compute the gradient with respect to the weights (dW)
    gradients.weights = BaseOperator::matmul(gradient.transpose(), this->input_data).transpose();  // dL/W = (dL/DC.T * x).T
    
    log_detail( "Computing Gradient with respect to weight:");
    log_matrix( gradients.weights );

    log_detail( "Add the bias." );

    // Compute the gradient with respect to the bias (db)
    if (bias == true)  gradients.biases = gradient.colwise().sum();

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
}

Eigen::MatrixXd Linear::gradient_Wrt_Input(Eigen::MatrixXd& gradient) { 
    int N = gradient.rows();

    log_detail( "Computing Gradient with respect to weight:" );
    log_detail( "The weight:" );
    log_matrix( parameters.weights );
    log_detail( "The gradient:" ) ;
    log_matrix( gradient );
    // Compute the gradient with respect to the input (dInput)
    Eigen::MatrixXd dInput = BaseOperator::matmul(parameters.weights, gradient.transpose()).transpose();   // dL/x = (W * dL/DC.T)

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
Eigen::MatrixXd Linear::backward(Eigen::MatrixXd& gradients) { 

    log_info("===========================================");
    log_info( "Entering Linear Gradient Backward pass ..." );

    Eigen::MatrixXd& weights = parameters.weights;
    Eigen::RowVectorXd& biases = parameters.biases;

    log_detail( "Size of gradients: {:d}" , gradients.size()  );
    log_detail( "Size of weights: {:d}"  , weights.size()  );
    log_detail( "Size of biases: {:d}" , biases.size()  );
    log_detail( "---" );
    log_detail( "Computing Gradient now ..." );

    gradient_Wrt_Weight_Bias(gradients);

    log_detail( "Computing Delta Error now ..."  );

    Eigen::MatrixXd dInput = gradient_Wrt_Input(gradients);

    log_detail( "Done with Gradients..." );
    return dInput;
}

void Linear::updateParameters(std::string& optimizertype, double& learningRate, int& iter) {

    log_info("=======================================");
    log_info( "Entering Linear Parameter Updates ..." );
    log_detail( "size: {:d}", this->W );

    Eigen::MatrixXd pbiases = parameters.biases.matrix();
    Eigen::MatrixXd gbiases = gradients.biases.matrix(); 

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
        opt_weights = new Optimizer(optimizertype, learningRate);
    }

    if (opt_biases == nullptr && bias == true) {
        opt_biases = new Optimizer(optimizertype, learningRate);
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

    // initialize gradients for next iteration.
    gradients.weights.setZero();
    gradients.biases.setZero();
}

std::string Linear::generateDotFormat(const std::string& name) {
    std::string dot = "{* Linear Transformation (" + name + ") *}|";  
    double min_weights = 0.0, max_weights = 0.0;
    double min_biases = 0.0, max_biases = 0.0;
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
void BatchNorm::setInitialWeights(int M) {

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
    dScale.setZero();
    dShift.setZero();
}

Eigen::MatrixXd BatchNorm::normalize(const Eigen::MatrixXd& input_data) {

    log_info("=================================");
    log_info( "Entering Batch Normalization ..." );

    setInitialWeights(input_data.cols());

    log_detail( "Input Data:" );
    log_matrix( input_data );

    // Calculate batch mean along the N dimension, but along the M dimension.
    Eigen::RowVectorXd batchMean = input_data.colwise().mean();

    log_detail( "Computing Mean ..." );
    log_rowvector( batchMean );

    // Calculate: X - mean
    minusMean = input_data.rowwise() - batchMean;

    log_detail( "minusMean" );
    log_matrix( minusMean );

    // Calculate batch variance along the N dimension
    Eigen::RowVectorXd batchVariance = minusMean.array().square().colwise().mean();

    log_detail( "Variance ..." );
    log_rowvector( batchVariance );

    // Add a small epsilon for numerical stability
    Eigen::RowVectorXd epsilonVector = Eigen::RowVectorXd::Constant(batchVariance.size(), epsilon);

    log_detail( "Epsilon ..." );
    log_rowvector( epsilonVector );

    // Calculate batch standard deviation along the N dimension
    batchStdDev = (batchVariance + epsilonVector).cwiseSqrt();

    log_detail(  "stdDev ..." );
    log_rowvector( batchStdDev );

    // Normalize the inputs along the N dimension
    normalizedInput = minusMean.array().rowwise()  / batchStdDev.array();

    log_detail( "normalizedInput along the N  ..." );
    log_matrix( normalizedInput );

    // Scale and shift the normalized inputs along the N dimension.
    Eigen::MatrixXd normalizedOutput = (normalizedInput.array().rowwise() * scale.array()).array().rowwise() + shift.array();

    log_detail( "scale ..." );
    log_rowvector( scale );

    log_detail( "shift ..." );
    log_rowvector( shift );

    log_detail( "normalizedOutput scaled ..." );
    log_matrix( normalizedOutput );

    return normalizedOutput;
}

void BatchNorm::setScale(const Eigen::VectorXd& newScale) {
    scale = newScale;
}

void BatchNorm::setShift(const Eigen::VectorXd& newShift) {
    shift = newShift;
}

Eigen::MatrixXd BatchNorm::forward(Eigen::MatrixXd& input_data) { 

    // Cache for later back propagation.
    this->input_data = input_data;

    // Perform Linear Transformation.
    Eigen::MatrixXd output = normalize(input_data);

    return output; // this becomes input to the next Node or next Layer.
}

// Leave the gradients as is for the scale and shift. They are cached in the Node. 
// They will be used to update the parameters in next parallel operations.
// the dInput is the gradient we propagate to source Nodes in the graph;
// while the parameter gradients get cached to be used to update the parameters later.
Eigen::MatrixXd BatchNorm::backward(const Eigen::MatrixXd& gradients) {
    int N = input_data.rows();

    log_info("==============================================");
    log_info("Entering Batch Normalization Backward pass ...");

    // Compute the gradient with respect to the scale parameter (Gamma)
    dScale = (gradients.array() * normalizedInput.array()).colwise().sum();

    // Compute the gradient with respect to the shift parameter (Beta)
    dShift = gradients.colwise().sum();

    log_detail( "scale and shift" );
    log_rowvector( scale );
    log_rowvector( shift );

    log_detail( "dScale and dShift gradients" );
    log_rowvector( dScale );
    log_rowvector( dShift );

    // Compute the gradient with respect to the normalized input
    Eigen::MatrixXd dNormalizedInput = gradients.array().rowwise() * scale.array();

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
    Eigen::RowVectorXd dBatchStdDev = -(dNormalizedInput.array() * minusMean.array()).colwise().sum() / (batchStdDev.array() * batchStdDev.array());

    // Compute the gradient with respect to the batch variance
    Eigen::RowVectorXd dBatchVariance = 0.5 * (dBatchStdDev.array() / batchStdDev.array());

    log_detail( "dBatchVariance" );
    log_rowvector( dBatchVariance );

    Eigen::MatrixXd dNormMinusMean1 = (dNormalizedInput.array() * (1.0 / (batchStdDev.array())).replicate(N,1)) + 
                                        (2.0 * minusMean.array() * (1.0/N * dBatchVariance.replicate(N, 1)).array());

    log_detail( "istd" );
    log_matrix( (1.0/(batchStdDev.array())).replicate(N,1) );

    log_detail( "dsq" );
    log_matrix( 1.0/N * dBatchVariance.replicate(N, 1) );

    log_detail( "xmu" );
    log_matrix( minusMean  );

    // Compute the gradient with respect to the batch mean
    Eigen::RowVectorXd dBatchMean = -1.0 * dNormMinusMean1.array().colwise().sum();

    log_detail(  "dBatchMean" );
    log_rowvector( dBatchMean );

    Eigen::MatrixXd dNormMinusMean2 = 1.0/N *  dBatchMean.replicate(N,1).array();

    log_detail( "dNormMinusMean2" );
    log_matrix( dNormMinusMean2 );

    // Compute the gradient with respect to the input
    Eigen::MatrixXd dInput = dNormMinusMean1.array() + dNormMinusMean2.array();
    log_detail( "dInput" );;
    log_matrix( dInput );

    return dInput;
}

void BatchNorm::updateParameters(std::string& optimizertype, double& learningRate, int& iter) {

    log_info("==================================================");
    log_info("Entering Batch Normalization Upgrade Parameters...");

    Eigen::MatrixXd pscale(1, this->M); 
    Eigen::MatrixXd gscale(1, this->M); 
    Eigen::MatrixXd pshift(1, this->M); 
    Eigen::MatrixXd gshift(1, this->M); 

    pscale.row(0) = scale;
    gscale.row(0) = dScale;

    pshift.row(0) = shift;
    gshift.row(0) = dShift;

    if (opt_scale == nullptr) {
        opt_scale = new Optimizer(optimizertype, learningRate);
    }

    if (opt_shift == nullptr) {
        opt_shift = new Optimizer(optimizertype, learningRate);
    }

    log_detail( "Updating Scale" );

    opt_scale->adam(pscale, gscale, iter);

    log_detail( "Updating Shift" );

    opt_shift->adam(pshift, gshift, iter);

    scale = pscale.row(0);
    shift = pshift.row(0);

    log_detail( "Updated scale" );
    log_rowvector( scale );
    log_detail( "Updated shift" );
    log_rowvector( shift );

    // initialize gradients for next iteration.
    dScale.setZero();
    dShift.setZero();

}

std::string BatchNorm::generateDotFormat(const std::string& name) {
    std::string dot = "{* Batch Normalization (" + name + ") *}|";  
    double min_scale = 0.0, max_scale = 0.0;
    double min_shift = 0.0, max_shift = 0.0;
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
void LayerNorm::setInitialWeights(int N) {

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
    dScale.setZero();
    dShift.setZero();
}

Eigen::MatrixXd LayerNorm::normalize(const Eigen::MatrixXd& input_data) {

    log_info("=================================");
    log_info( "Entering Layer Normalization ..." );

    setInitialWeights(input_data.rows());

    log_detail(" Input Data ...");
    log_matrix( input_data );

    // Calculate layer mean along the M dimension, but along the N dimension.
    Eigen::VectorXd layerMean = input_data.rowwise().mean();

    log_detail( "Mean ..." );
    log_vector( layerMean );

    // Calculate: X - mean
    minusMean = input_data.colwise() - layerMean;

    log_detail( "minusMean" );
    log_matrix( minusMean );

    // Calculate batch variance along the M dimension
    Eigen::VectorXd layerVariance = minusMean.array().square().rowwise().mean();

    log_detail( "Variance ..." );
    log_vector( layerVariance );

    // Add a small epsilon for numerical stability
    Eigen::VectorXd epsilonVector = Eigen::VectorXd::Constant(layerVariance.size(), epsilon);

    log_detail( "Epsilon ..." );
    log_vector( epsilonVector );

    // Calculate batch standard deviation across the M dimension
    layerStdDev = (layerVariance + epsilonVector).array().sqrt();

    log_detail( "stdDev ..." );
    log_vector( layerStdDev );

    // Normalize the inputs along the M dimension
    normalizedInput = minusMean.array().colwise()  / layerStdDev.array();

    log_detail( "normalizedInput along the N  ..." );
    log_matrix( normalizedInput );

    // Scale and shift the normalized inputs
    Eigen::MatrixXd normalizedOutput = (normalizedInput.array().colwise() * scale.array()).array().colwise() + shift.array();

    log_detail( "scale ..." );
    log_vector( scale );

    log_detail( "shift ..." );
    log_vector( shift );

    log_detail( "normalizedOutput scaled ..." );
    log_matrix( normalizedOutput );

    return normalizedOutput;
}


void LayerNorm::setScale(const Eigen::VectorXd& newScale) {
    scale = newScale;
}

void LayerNorm::setShift(const Eigen::VectorXd& newShift) {
    shift = newShift;
}

Eigen::MatrixXd LayerNorm::forward(Eigen::MatrixXd& input_data) { 

    // Cache for later back propagation.
    this->input_data = input_data;

    // Perform Linear Transformation.
    Eigen::MatrixXd output = normalize(input_data);
 

    return output; // this becomes input to the next Node or next Layer.
}

// Leave the gradients as is for the scale and shift. They are cached in the Node. 
// They will be used to update the parameters in next parallel operations.
// the dInput is the gradient we propagate to source Nodes in the graph;
// while the parameter gradients get cached to be used to update the parameters later.
Eigen::MatrixXd LayerNorm::backward(const Eigen::MatrixXd& gradients) {
    int W = input_data.cols();

    log_info("===============================================");
    log_info( "Entering Layer Normalization Backward Pass ..." );

    log_detail( "normalizedInput" );
    log_matrix( normalizedInput );
    // Compute the gradient with respect to the scale parameter (Gamma)
    dScale = (gradients.array() * normalizedInput.array()).rowwise().sum();

    // Compute the gradient with respect to the shift parameter (Beta)
    dShift = gradients.rowwise().sum();

    log_detail( "scale and shift gradients" );
    log_vector( scale );
    log_vector( shift );

    log_detail( "dScale and dShift gradients" );
    log_vector( dScale );
    log_vector( dShift );

    // Compute the gradient with respect to the normalized input
    Eigen::MatrixXd dNormalizedInput = gradients.array().colwise() * scale.array();

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
    Eigen::VectorXd dLayerStdDev = -(dNormalizedInput.array() * minusMean.array()).rowwise().sum() / (layerStdDev.array() * layerStdDev.array());

    log_detail( "dLayerStdDev" );
    log_vector( dLayerStdDev );

    // Compute the gradient with respect to the layer variance
    Eigen::VectorXd dLayerVariance = 0.5 * (dLayerStdDev.array() / layerStdDev.array());

    log_detail( "dLayerVariance" );
    log_vector( dLayerVariance );

    Eigen::MatrixXd dNormMinusMean1 = (dNormalizedInput.array() * (1.0 / (layerStdDev.array())).replicate(1,W)) + 
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
    Eigen::VectorXd dLayerMean = -1.0 * dNormMinusMean1.array().rowwise().sum();

    log_detail( "dLayerMean" );
    log_vector( dLayerMean );

    Eigen::MatrixXd dNormMinusMean2 = 1.0/W *  dLayerMean.replicate(1,W).array();

    log_detail("dNormMinusMean2" );
    log_matrix( dNormMinusMean2 );

    // Compute the gradient with respect to the input
    Eigen::MatrixXd dInput = dNormMinusMean1.array() + dNormMinusMean2.array();

    log_detail( "dInput" );
    log_matrix( dInput );

    return dInput;
}

void LayerNorm::updateParameters(std::string& optimizertype, double& learningRate, int& iter) {
    Eigen::MatrixXd pscale(this->N, 1); 
    Eigen::MatrixXd gscale(this->N, 1); 
    Eigen::MatrixXd pshift(this->N, 1); 
    Eigen::MatrixXd gshift(this->N, 1); 

    log_info("====================================================");
    log_info( "Entering Layer Normalization Upgrade Parameters ..." );

    pscale.col(0) = scale;
    gscale.col(0) = dScale;

    pshift.col(0) = shift;
    gshift.col(0) = dShift;

    if (opt_scale == nullptr) {
        opt_scale = new Optimizer(optimizertype, learningRate);
    }

    if (opt_shift == nullptr) {
        opt_shift = new Optimizer(optimizertype, learningRate);
    }

    log_detail( "Updating Layer Normal scale" );
    opt_scale->adam(pscale, gscale, iter);
    log_detail( "Updating Layer Normal shift" );
    opt_shift->adam(pshift, gshift, iter);
    scale = pscale.col(0);
    shift = pshift.col(0);

    log_detail( "Updated scale:" );
    log_vector( scale  );
    log_detail( "Updated shift:" );
    log_vector( shift );

    // initialize gradients for next iteration.
    dScale.setZero();
    dShift.setZero();

}

std::string LayerNorm::generateDotFormat(const std::string& name) {
    std::string dot = "{* Layer Normalization (" + name + ") *}|"; 
    double min_scale = 0.0, max_scale = 0.0;
    double min_shift = 0.0, max_shift = 0.0;
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

// Here, instead of using the term logits, let's just use x.
Eigen::MatrixXd Activation::sigmoid(const Eigen::MatrixXd& x) {
    return (1.0 / (1.0 + (-x).array().exp())).matrix();
}

/*****************************************************************************************************
 * So, the gradient of the sigmoid function with respect to its input (z) can be expressed as follows:
 * dy/dz = propagated_gradient * sigmoid(z) * (1 - sigmoid(z))
 * Or if output (y) is cached, where y = sigmoid(z)
 * then we use the output such that:
 * dy/dz = propagated_gradient * y * (1 - y)
 *****************************************************************************************************/
Eigen::MatrixXd Activation::sigmoidGradient(const Eigen::MatrixXd& gradients, const Eigen::MatrixXd& output_data) {
    Eigen::MatrixXd dInput = gradients.array() * output_data.array() * ( 1 - output_data.array());
    return dInput;
}

Eigen::MatrixXd Activation::tanh(const Eigen::MatrixXd& x) {
    return x.array().tanh();
}

/*****************************************************************************************************
 * So, the gradient of the tanh function with respect to its input (z) can be expressed as follows:
 * dy/dz = propagated_gradient * (1 - tanh(z)^2)
 * Or if output (y) is cached, where y = tanh(z)
 * then we use the output such that:
 * dy/dz = propagated_gradient * y * (1 - y^2)
 *****************************************************************************************************/
Eigen::MatrixXd Activation::tanhGradient(const Eigen::MatrixXd& gradients, const Eigen::MatrixXd& output_data) {
    Eigen::MatrixXd dInput =  gradients.array() *  ( 1 - output_data.array().pow(2));
    return dInput;
}

Eigen::MatrixXd Activation::relu(const Eigen::MatrixXd& x) {
    return x.array().max(0.0);
}

/*****************************************************************************************************
 * So, the gradient of the ReLU function with respect to its input (z) can be expressed as follows:
 * dy/dz = propagated_gradient * 1 for z > 0
 * dy/dz = propagated_gradient * 0 for z <= 0
 *****************************************************************************************************/
Eigen::MatrixXd Activation::reluGradient(const Eigen::MatrixXd& gradients) {
    Eigen::MatrixXd dInput = this->input_data.array().max(0.0).cast<bool>().cast<double>() * gradients.array();
    return dInput;
}

Eigen::MatrixXd Activation::leakyReLU(const Eigen::MatrixXd& x, float alpha) {
    return x.array().max(alpha * x.array());
}

/*****************************************************************************************************
 * So, the gradient of the LeakyReLU function with respect to its input (z) can be expressed as follows:
 * dy/dz = propagated_gradient * 1 for z > 0
 * dy/dz = propagated_gradient * alpha for z <= 0
*****************************************************************************************************/
Eigen::MatrixXd Activation::leakyReluGradient(const Eigen::MatrixXd& gradients) {
    Eigen::MatrixXd dInput = this->input_data.array().max(0.0).cast<bool>().cast<double>().max(alpha) * gradients.array();
    return dInput;
}

Eigen::MatrixXd Activation::gelu(const Eigen::MatrixXd& x) {
    return 0.5 * x.array() * (1.0 + ((x.array() * std::sqrt(2.0 / M_PI)).tanh()));
}

/*****************************************************************************************************
 * Gelu Gradient ...
*****************************************************************************************************/
Eigen::MatrixXd Activation::geluGradient(const Eigen::MatrixXd& gradients) {
    // Calculate the coefficient used in the GELU formula
    double coefficient = sqrt(2.0 / M_PI);
    // Compute the cumulative distribution function (CDF) part of the GELU gradient
    Eigen::MatrixXd cdf = 0.5 * (1.0 + (gradients.array() / coefficient).tanh());
    // Compute the probability density function (PDF) part of the GELU gradient
    Eigen::MatrixXd pdf = exp(-0.5 * gradients.array().square()) / coefficient;
    // Combine the CDF and PDF components to obtain the final gradient values
    // Apply element-wise operations on arrays: add CDF, multiply x by PDF, add a term based on x^3
    return 0.5 * (1.0 + (cdf.array() + gradients.array() * pdf.array() + 0.044715 * gradients.array().cube())).matrix();
}


// This assumes that the input is defined with NxM dimensionality.
// Therefore the size of the parameters and thus gradients will be based on MxW where W is the number of weights to use.
void Activation::setInitSize(const Eigen::MatrixXd& input_data) {

    // if size is already set, 
    // it means weights have previously already been set by an initial forward pass
    if (this->N != 0 || this->M != 0) return;

    this->N = input_data.rows();
    this->M = input_data.cols();

    // Initialize scaling and shifting parameters   
    this->dInput.resize(this->N, this->M); // allocates memory
}

Eigen::MatrixXd Activation::computeActivation(const Eigen::MatrixXd& input_data) { 
    Eigen::MatrixXd output;

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

Eigen::MatrixXd Activation::computeGradient(const Eigen::MatrixXd& gradients, const Eigen::MatrixXd& output_data) {
    Eigen::MatrixXd dInput;
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

Eigen::MatrixXd Activation::forward(Eigen::MatrixXd& input_data) { 

    log_info( "=====================================" );
    log_info( "Entering Activation Forward Pass ..." );

    // Cache for later back propagation.
    this->input_data = input_data;

    // Perform Activation
    Eigen::MatrixXd output = computeActivation(input_data);

    log_detail("Activation Result" );
    log_matrix( output );

    return output; // this becomes input to the next Node or next Layer.
}

Eigen::MatrixXd Activation::backward(const Eigen::MatrixXd& gradient, const Eigen::MatrixXd& output_data) {
    log_info("=====================================");
    log_info( "Entering Activation Backward Pass ..." );

    // Perform Gradient
    this->dInput = computeGradient(gradient, output_data);

    log_detail("Activation Gradient Result" );
    log_matrix( this->dInput );

    return this->dInput; // this becomes input to the next Node or next Layer.
}

std::string Activation::generateDotFormat() {
    std::string dot = "{* Activation (" + activationtype + ")*}|";
    double min_input = 0.0, max_input = 0.0;
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

// Mean Squared Error. Returns 1x1 matrix (scalar)
Eigen::MatrixXd Loss::mse(const Eigen::MatrixXd& predicted, const Eigen::MatrixXd& target) {
    Eigen::VectorXd diff = predicted - target;
    double mse = diff.array().square().rowwise().sum().mean();
    return Eigen::MatrixXd::Constant(1, 1, mse);
}

Eigen::MatrixXd Loss::mseGradient(const Eigen::MatrixXd& predicted, const Eigen::MatrixXd& target) {
    return 2 * (predicted - target);
}

// Binary Cross Entropy.  Returns 1x1 matrix (scalar)
Eigen::MatrixXd Loss::bce(const Eigen::MatrixXd& predicted, const Eigen::MatrixXd& target) {
    Eigen::MatrixXd loss = -target.array() * predicted.array().log() - (1.0 - target.array()) * (1.0 - predicted.array()).log();
    double averageLoss = loss.mean();
    return Eigen::MatrixXd::Constant(1, 1, averageLoss);
}

Eigen::MatrixXd Loss::bceGradient(const Eigen::MatrixXd& predicted, const Eigen::MatrixXd& target) {
    Eigen::MatrixXd gradient = (predicted - target).array() / (predicted.array() * (1 - predicted.array()));
    return gradient;
}


// For Loss Categorial Cross Entropy. Usually, we use Softmax.
// If predicted and target has NxC dimension where C is number of classes, then result will be Nx1.
Eigen::MatrixXd Loss::cce(const Eigen::MatrixXd& predicted, const Eigen::MatrixXd& target) {
    int N = predicted.rows(); // Number of samples
    int C = predicted.cols(); // Number of classes
    Eigen::MatrixXd loss(N, 1);
    for (int i = 0; i < N; ++i) {
        double sampleLoss = 0.0;
        for (int j = 0; j < C; ++j) {
            sampleLoss -= target(i, j) * std::log(predicted(i, j));
        }
        loss(i, 0) = sampleLoss;
    }
    return loss;
}

Eigen::MatrixXd Loss::cceGradient(const Eigen::MatrixXd& predicted, const Eigen::MatrixXd& target) {
    int numClasses = predicted.rows();
    Eigen::MatrixXd gradient(numClasses, 1);
    for (int i = 0; i < numClasses; ++i) {
        gradient(i, 0) = predicted(i, 0) - target(i, 0);
    }
    return gradient;
}

/*
Eigen::MatrixXd hingeLoss(const Eigen::MatrixXd& predicted, const Eigen::MatrixXd& target) {
    return predicted.array().max(0.0) - predicted.array() * target.array() + (1 - target.array()).max(0.0);
} */

// For Support Vectors (not necessarily for Neural)
Eigen::MatrixXd Loss::hingeLoss(const Eigen::MatrixXd& predicted, const Eigen::MatrixXd& target) {
    Eigen::MatrixXd loss = (1.0 - predicted.array() * target.array()).cwiseMax(0.0);
    return loss;
}

Eigen::MatrixXd Loss::hingeLossGradient(const Eigen::MatrixXd& predicted, const Eigen::MatrixXd& target) {
    return (predicted.array() * target.array() < 1).select(-target, 0);
}

Eigen::MatrixXd Loss::computeLoss(const Eigen::MatrixXd& predicted, const Eigen::MatrixXd& target) { 
    Eigen::MatrixXd output;
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

Eigen::MatrixXd Loss::computeGradients(const Eigen::MatrixXd& predicted, const Eigen::MatrixXd& target) { 
    Eigen::MatrixXd gradients;
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