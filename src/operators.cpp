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
    // Initialize Adam optimizer variables
    //static Eigen::MatrixXd m = Eigen::MatrixXd::Zero(weights.rows(), weights.cols());
    // static Eigen::MatrixXd v = Eigen::MatrixXd::Zero(weights.rows(), weights.cols());

    if (moments.cols() == 0 && moments.rows() == 0) {
        moments = Eigen::MatrixXd::Zero(weights.rows(), weights.cols());
    }
    if (velocity.cols() == 0 && velocity.rows() == 0) {
        velocity = Eigen::MatrixXd::Zero(weights.rows(), weights.cols());
    }

    double beta1_t = std::pow(beta1, currentEpoch + 1);
    double beta2_t = std::pow(beta2, currentEpoch + 1);

    // Print the value with the specified precision
    float value1 = 1.23456789f;
    int floatPrecision = std::numeric_limits<float>::digits10;
    std::cout.precision(floatPrecision);
    std::cout << "Value: " << value1 << std::endl;

    // Print the value with the specified precision
    double value2 = 1.23456789;
    int doublePrecision = std::numeric_limits<double>::digits10;
    std::cout.precision(doublePrecision);
    std::cout << "Value: " << value2 << std::endl;

    std::cout << "beta1_t\n";
    std::cout << beta2_t << "\n";
    std::cout << "beta2_t\n";
    std::cout << beta2_t << "\n";

    std::cout << "beta1 * m\n";
    std::cout << beta1 * moments.array() << "\n";

    std::cout << "beta2 * v\n";
    std::cout << beta2 * velocity.array() << "\n";

    // Update momentum and velocity
    moments = beta1 * moments.array() + (1 - beta1) * gradients.array();
    velocity = beta2 * velocity.array() + (1 - beta2) * (gradients.array() * gradients.array());

    std::cout << "gradients\n";
    std::cout << gradients.array()  << "\n";

    std::cout << "power of gradients\n";
    std::cout << gradients.array() * gradients.array() << "\n";


    std::cout << "momentum\n";
    std::cout << moments << "\n";
    std::cout << "velocity\n";
    std::cout << velocity << "\n";

    // Compute bias-corrected moment estimates
    Eigen::MatrixXd m_hat = moments / (1 - beta1_t);
    Eigen::MatrixXd v_hat = velocity / (1 - beta2_t);

    std::cout << "momentum hat\n";
    std::cout << m_hat << "\n";
    std::cout << "velocity hat\n";
    std::cout << v_hat << "\n";

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

    std::cout << "Entering Linear Transformation\n";

    // Initilalize the parameters.
    setInitialWeights(input_data.cols());
    Eigen::MatrixXd& weights = parameters.weights;
    Eigen::RowVectorXd& biases = parameters.biases;
    std::cout << "Size of input: " << input_data.size() << "\n";
    std::cout << "Size of weights: "  << weights.size() << "\n";
    std::cout << "Size of biases: " << biases.size() << "\n";

    std::cout << "xxxxx \n";
    std::cout << "the input ...\n";
    std::cout << input_data  << "\n\n";
    std::cout << "weights\n";
    std::cout << weights << "\n\n";
    std::cout << "matmuls 1\n";
    std::cout << input_data * weights << "\n\n";
    std::cout << "matmuls 2\n";
    std::cout << BaseOperator::matmul(input_data, weights) << "\n\n";
    std::cout << "bias\n";
    std::cout << biases << "\n";
    std::cout << "yyyyy \n";

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

    // Cache for later back propagation.
    this->input_data = input_data;

    std::cout << "Size of input:" << this->input_data.size() << "\n";

    // Perform Linear Transformation.
    Eigen::MatrixXd output = linearTransform(input_data);

    print_string("Linear forward pass ...", true); 

    return output; // this becomes input to the next Node or next Layer.
}

void Linear::gradient_Wrt_Weight_Bias(Eigen::MatrixXd& gradient) {
    int N = gradient.rows();
    
    std::cout << "Size of gradient:" << gradient.size() << "\n";
    std::cout << "Size of input:" << this->input_data.size() << "\n";
    std::cout << "Size of tranposed input:" << (this->input_data).size() << "\n";

    std::cout << "Before multiply of input and gradients\n";
    std::cout << "the input:\n";
    std::cout << this->input_data << "\n";
    std::cout << "the gradient:\n";
    std::cout << gradient << "\n";
    std::cout << "the initial gradient:\n";
    std::cout << gradients.weights << "\n";
    
    std::cout << "Multiply input and gradient.\n";
    // Compute the gradient with respect to the weights (dW)
    gradients.weights = BaseOperator::matmul(gradient.transpose(), this->input_data).transpose();  // dL/W = (dL/DC.T * x).T
    
    std::cout << "the input:\n";
    std::cout << this->input_data << "\n";
    std::cout << "the gradient:\n";
    std::cout << gradient << "\n";
    std::cout << "the computed gradient with respect to weight:\n";
    std::cout << gradients.weights << "\n";

    std::cout << "Add the bias.\n";
    // Compute the gradient with respect to the bias (db)
    if (bias == true)  gradients.biases = gradient.colwise().sum();

    std::cout << "gbiases rows: " <<  gradients.biases.rows() << ", cols: " <<  gradients.biases.cols() << "\n";
    std::cout << gradients.biases << "\n";
    
        std::cout << "Normalize the gradients.\n";
    // Normalize the gradients by dividing by the number of samples (N)
    gradients.weights /= N;  // dW - gradients of weights (MxW)
    if (bias == true) gradients.biases /= N;   // db - gradients of bias (1xW)
    std::cout << "Normalized gradients ...\n";
    std::cout << "gradient weights\n";
    std::cout << gradients.weights << "\n";
    std::cout << "gradient biases\n";
    std::cout << gradients.biases << "\n";
}

Eigen::MatrixXd Linear::gradient_Wrt_Input(Eigen::MatrixXd& gradient) { 
    int N = gradient.rows();
    std::cout << "the weight:\n";
    std::cout << parameters.weights << "\n";
    std::cout << "the gradient:\n";
    std::cout << gradient << "\n";
    // Compute the gradient with respect to the input (dInput)
    Eigen::MatrixXd dInput = BaseOperator::matmul(parameters.weights, gradient.transpose()).transpose();   // dL/x = (W * dL/DC.T)

    std::cout << "the computed gradient with respect to input:\n";
    std::cout << dInput << "\n";

    // Normalize the gradients by dividing by the number of samples (N)
    dInput /= N;  // dInput - gradients of input (NxM)

    std::cout << "normalized dInput\n";
    std::cout << dInput << "\n";
    return dInput;
}

// Leave the gradients as is. They are cached in the Node. 
// They will be used to update the parameters in next parallel operations.
// the dInput is the gradient we propagate to source Nodes in the graph;
// while the parameter gradients get cached to be used to update the parameters later.
Eigen::MatrixXd Linear::backward(Eigen::MatrixXd& gradients) { 
    std::cout << "Entering Linear Gradient backward pass ...\n";
    Eigen::MatrixXd& weights = parameters.weights;
    Eigen::RowVectorXd& biases = parameters.biases;
    std::cout << "Size of gradients: " << gradients.size() << "\n";
    std::cout << "Size of weights: "  << weights.size() << "\n";
    std::cout << "Size of biases: " << biases.size() << "\n";
    std::cout << "Computing Gradient now ....\n";
    gradient_Wrt_Weight_Bias(gradients);
    std::cout << "Computing Delta Error now ....\n";
    Eigen::MatrixXd dInput = gradient_Wrt_Input(gradients);
    std::cout << "Done ...\n";
    return dInput;
}

void Linear::updateParameters(std::string& optimizertype, double& learningRate, int& iter) {
    std::cout << "LInear parameter updating in ...\n";
    std::cout << "size: " << this->W << " \n";
    Eigen::MatrixXd pbiases = parameters.biases.matrix();
    Eigen::MatrixXd gbiases = gradients.biases.matrix(); 
    std::cout << "Assigning ...\n";

    std::cout << "pbiases rows: " << pbiases.rows() << ", cols: " << pbiases.cols() << "\n";
    std::cout << "gbiases rows: " << gbiases.rows() << ", cols: " << gbiases.cols() << "\n";

    std::cout << "Gradient weights:\n";
    std::cout << gradients.weights << "\n";
    std::cout << "Gradient biases:\n";
    std::cout << gradients.biases << "\n";

    std::cout << "Before Updated weights:\n";
    std::cout << parameters.weights << "\n";
    std::cout << "Before Updated biases:\n";
    std::cout << parameters.biases << "\n";

    if (opt_weights == nullptr) {
        opt_weights = new Optimizer(optimizertype, learningRate);
    }

    if (opt_biases == nullptr && bias == true) {
        opt_biases = new Optimizer(optimizertype, learningRate);
    }

    std::cout << "Updating Linear weights  \n";
    opt_weights->adam(parameters.weights, gradients.weights, iter);
    std::cout << "Updating Linear biases \n";

    if (bias == true) {
        opt_biases->adam(pbiases, gbiases, iter);
        parameters.biases = pbiases.row(0);
    }

    std::cout << "Updated weights:\n";
    std::cout << parameters.weights << "\n";
    std::cout << "Updated biases:\n";
    std::cout << parameters.biases << "\n";

    // initialize gradients for next iteration.
    gradients.weights.setZero();
    gradients.biases.setZero();
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

    setInitialWeights(input_data.cols());

    std::cout << input_data << std::endl;

    // Calculate batch mean along the N dimension, but along the M dimension.
    Eigen::RowVectorXd batchMean = input_data.colwise().mean();

    std::cout << "Mean ..." << std::endl;
    std::cout << batchMean << std::endl;

    // Calculate: X - mean
    minusMean = input_data.rowwise() - batchMean;

    std::cout << "minusMean\n";
    std::cout << minusMean << "\n";

    // Calculate batch variance along the N dimension
    Eigen::RowVectorXd batchVariance = minusMean.array().square().colwise().mean();

    std::cout << "Variance ..." << std::endl;
    std::cout << batchVariance << std::endl;

    // Add a small epsilon for numerical stability
    Eigen::RowVectorXd epsilonVector = Eigen::RowVectorXd::Constant(batchVariance.size(), epsilon);

    std::cout << "Epsilon ..." << std::endl;
    std::cout << epsilonVector << std::endl;

    // Calculate batch standard deviation along the N dimension
    batchStdDev = (batchVariance + epsilonVector).cwiseSqrt();

    std::cout << "stdDev ..." << std::endl;
    std::cout << batchStdDev << std::endl;

    // Normalize the inputs along the N dimension
    normalizedInput = minusMean.array().rowwise()  / batchStdDev.array();

    std::cout << "normalizedInput along the N  ..." << std::endl;
    std::cout << normalizedInput << std::endl;

    // Scale and shift the normalized inputs along the N dimension.
    Eigen::MatrixXd normalizedOutput = (normalizedInput.array().rowwise() * scale.array()).array().rowwise() + shift.array();

    std::cout << "scale ..." << std::endl;
    std::cout << scale << std::endl;

    std::cout << "shift ..." << std::endl;
    std::cout << shift << std::endl;

    std::cout << "normalizedOutput scaled ..." << std::endl;
    std::cout << normalizedOutput << std::endl;

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

    print_string("Batch Normalize forward pass ...", true); 

    return output; // this becomes input to the next Node or next Layer.
}

// Leave the gradients as is for the scale and shift. They are cached in the Node. 
// They will be used to update the parameters in next parallel operations.
// the dInput is the gradient we propagate to source Nodes in the graph;
// while the parameter gradients get cached to be used to update the parameters later.
Eigen::MatrixXd BatchNorm::backward(const Eigen::MatrixXd& gradients) {
    int N = input_data.rows();

    // Compute the gradient with respect to the scale parameter (Gamma)
    dScale = (gradients.array() * normalizedInput.array()).colwise().sum();

    // Compute the gradient with respect to the shift parameter (Beta)
    dShift = gradients.colwise().sum();

    std::cout << "scale and shift\n";
    std::cout << scale << "\n\n";
    std::cout << shift << "\n";

    std::cout << "dScale and dShift gradients\n";
    std::cout << dScale << "\n\n";
    std::cout << dShift << "\n";

    // Compute the gradient with respect to the normalized input
    Eigen::MatrixXd dNormalizedInput = gradients.array().rowwise() * scale.array();

    std::cout << "dNormalizedInput\n";
    std::cout << dNormalizedInput << "\n";

    std::cout << "dNormalizedInput * minusMean\n";
    std::cout << dNormalizedInput.array() * minusMean.array()  << "\n";

    std::cout << "dNormalizedInput * minusMean rowwise sum\n";
    std::cout << (dNormalizedInput.array() * minusMean.array()).rowwise().sum() << "\n";;

    std::cout << "barchStdDev\n";
    std::cout << batchStdDev.array() << "\n";

    std::cout << "layerStdDev * layerStdDev\n";
    std::cout << (batchStdDev.array() * batchStdDev.array()) << "\n";

    // Compute the gradient with respect to the batch standard deviation
    Eigen::RowVectorXd dBatchStdDev = -(dNormalizedInput.array() * minusMean.array()).colwise().sum() / (batchStdDev.array() * batchStdDev.array());

    // Compute the gradient with respect to the batch variance
    Eigen::RowVectorXd dBatchVariance = 0.5 * (dBatchStdDev.array() / batchStdDev.array());

    std::cout << "dBatchVariance\n";
    std::cout << dBatchVariance << "\n";

    Eigen::MatrixXd dNormMinusMean1 = (dNormalizedInput.array() * (1.0 / (batchStdDev.array())).replicate(N,1)) + 
                                        (2.0 * minusMean.array() * (1.0/N * dBatchVariance.replicate(N, 1)).array());

    std::cout << "istd\n";
    std::cout << (1.0/(batchStdDev.array())).replicate(N,1)  << "\n";

    std::cout << "dsq\n";
    std::cout << 1.0/N * dBatchVariance.replicate(N, 1)  << "\n";

    std::cout << "xmu\n";
    std::cout << minusMean  << "\n";

    // Compute the gradient with respect to the batch mean
    Eigen::RowVectorXd dBatchMean = -1.0 * dNormMinusMean1.array().colwise().sum();

    std::cout << "dBatchMean\n";
    std::cout << dBatchMean << "\n";

    Eigen::MatrixXd dNormMinusMean2 = 1.0/N *  dBatchMean.replicate(N,1).array();

    std::cout << "dNormMinusMean2\n";
    std::cout << dNormMinusMean2 << "\n";

    // Compute the gradient with respect to the input
    Eigen::MatrixXd dInput = dNormMinusMean1.array() + dNormMinusMean2.array();
    std::cout << "dInput\n";
    std::cout << dInput << "\n";

    return dInput;
}

void BatchNorm::updateParameters(std::string& optimizertype, double& learningRate, int& iter) {
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

    std::cout << "Updating Batch Normal scale  \n";
    opt_scale->adam(pscale, gscale, iter);
    std::cout << "Updating Batch Normal shift  \n";
    opt_shift->adam(pshift, gshift, iter);
    scale = pscale.row(0);
    shift = pshift.row(0);

    std::cout << "Updated scale:\n";
    std::cout << scale << "\n";
    std::cout << "Updated shift:\n";
    std::cout << shift << "\n";

    // initialize gradients for next iteration.
    dScale.setZero();
    dShift.setZero();

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

    setInitialWeights(input_data.rows());

    std::cout << input_data << std::endl;

    // Calculate layer mean along the M dimension, but along the N dimension.
    Eigen::VectorXd layerMean = input_data.rowwise().mean();

    std::cout << "Mean ..." << std::endl;
    std::cout << layerMean << std::endl;

    // Calculate: X - mean
    minusMean = input_data.colwise() - layerMean;

    std::cout << "minusMean\n";
    std::cout << minusMean << "\n";

    // Calculate batch variance along the M dimension
    Eigen::VectorXd layerVariance = minusMean.array().square().rowwise().mean();

    std::cout << "Variance ..." << std::endl;
    std::cout << layerVariance << std::endl;

    // Add a small epsilon for numerical stability
    Eigen::VectorXd epsilonVector = Eigen::VectorXd::Constant(layerVariance.size(), epsilon);

    std::cout << "Epsilon ..." << std::endl;
    std::cout << epsilonVector << std::endl;

    // Calculate batch standard deviation across the M dimension
    layerStdDev = (layerVariance + epsilonVector).array().sqrt();

    std::cout << "stdDev ..." << std::endl;
    std::cout << layerStdDev << std::endl;

    // Normalize the inputs along the M dimension
    normalizedInput = minusMean.array().colwise()  / layerStdDev.array();

    std::cout << "normalizedInput along the N  ..." << std::endl;
    std::cout << normalizedInput << std::endl;

    // Scale and shift the normalized inputs
    Eigen::MatrixXd normalizedOutput = (normalizedInput.array().colwise() * scale.array()).array().colwise() + shift.array();

    std::cout << "scale ..." << std::endl;
    std::cout << scale << std::endl;

    std::cout << "shift ..." << std::endl;
    std::cout << shift << std::endl;

    std::cout << "normalizedOutput scaled ..." << std::endl;
    std::cout << normalizedOutput << std::endl;

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

    print_string("Layer Normalize forward pass ...", true); 

    return output; // this becomes input to the next Node or next Layer.
}

// Leave the gradients as is for the scale and shift. They are cached in the Node. 
// They will be used to update the parameters in next parallel operations.
// the dInput is the gradient we propagate to source Nodes in the graph;
// while the parameter gradients get cached to be used to update the parameters later.
Eigen::MatrixXd LayerNorm::backward(const Eigen::MatrixXd& gradients) {
    int W = input_data.cols();

    std::cout << "normalizedInput \n";
    std::cout << normalizedInput << "\n";
    // Compute the gradient with respect to the scale parameter (Gamma)
    dScale = (gradients.array() * normalizedInput.array()).rowwise().sum();

    // Compute the gradient with respect to the shift parameter (Beta)
    dShift = gradients.rowwise().sum();

    std::cout << "scale and shift gradients\n";
    std::cout << scale << "\n\n";
    std::cout << shift << "\n";

    std::cout << "dScale and dShift gradients\n";
    std::cout << dScale << "\n\n";
    std::cout << dShift << "\n";

    // Compute the gradient with respect to the normalized input
    Eigen::MatrixXd dNormalizedInput = gradients.array().colwise() * scale.array();

    std::cout << "dNormalizedInput\n";
    std::cout << dNormalizedInput << "\n";

    std::cout << "dNormalizedInput * minusMean\n";
    std::cout << dNormalizedInput.array() * minusMean.array()  << "\n";

    std::cout << "dNormalizedInput * minusMean rowwise sum\n";
    std::cout << (dNormalizedInput.array() * minusMean.array()).rowwise().sum() << "\n";;

    std::cout << "layerStdDev\n";
    std::cout << layerStdDev.array() << "\n";

    std::cout << "layerStdDev * layerStdDev\n";
    std::cout << (layerStdDev.array() * layerStdDev.array()) << "\n";

    // Compute the gradient with respect to the layer standard deviation
    Eigen::VectorXd dLayerStdDev = -(dNormalizedInput.array() * minusMean.array()).rowwise().sum() / (layerStdDev.array() * layerStdDev.array());

    std::cout << "dLayerStdDev\n";
    std::cout << dLayerStdDev << "\n";

    // Compute the gradient with respect to the layer variance
    Eigen::VectorXd dLayerVariance = 0.5 * (dLayerStdDev.array() / layerStdDev.array());

    std::cout << "dLayerVariance\n";
    std::cout << dLayerVariance << "\n";

    Eigen::MatrixXd dNormMinusMean1 = (dNormalizedInput.array() * (1.0 / (layerStdDev.array())).replicate(1,W)) + 
                                        (2.0 * minusMean.array() * (1.0/W * dLayerVariance.replicate(1,W)).array());

    std::cout << "dxmu1\n";
    std::cout << (dNormalizedInput.array() * (1.0 / (layerStdDev.array())).replicate(1,W)) << "\n";
    std::cout << "dxmu2\n";
    std::cout <<  (2.0 * minusMean.array() * (1.0/W * dLayerVariance.replicate(1,W)).array()) << "\n";

    std::cout << "dNormMinusMean1\n";
    std::cout << dNormMinusMean1 << "\n";

    std::cout << "istd\n";
    std::cout << (1.0/(layerStdDev.array() * layerStdDev.array())).replicate(1,W)  << "\n";

    std::cout << "dsq\n";
    std::cout << 1.0/N * dLayerVariance.replicate(1, 2)  << "\n";

    std::cout << "xmu\n";
    std::cout << minusMean  << "\n";

    // Compute the gradient with respect to the batch mean
    Eigen::VectorXd dLayerMean = -1.0 * dNormMinusMean1.array().rowwise().sum();

    std::cout << "dLayerMean\n";
    std::cout << dLayerMean << "\n";

    Eigen::MatrixXd dNormMinusMean2 = 1.0/W *  dLayerMean.replicate(1,W).array();

    std::cout << "dNormMinusMean2\n";
    std::cout << dNormMinusMean2 << "\n";

    // Compute the gradient with respect to the input
    Eigen::MatrixXd dInput = dNormMinusMean1.array() + dNormMinusMean2.array();

    std::cout << "dInput\n";
    std::cout << dInput << "\n";

    return dInput;
}

void LayerNorm::updateParameters(std::string& optimizertype, double& learningRate, int& iter) {
    Eigen::MatrixXd pscale(this->N, 1); 
    Eigen::MatrixXd gscale(this->N, 1); 
    Eigen::MatrixXd pshift(this->N, 1); 
    Eigen::MatrixXd gshift(this->N, 1); 

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

    std::cout << "Updating Layer Normal scale  \n";
    opt_scale->adam(pscale, gscale, iter);
    std::cout << "Updating Layer Normal shift  \n";
    opt_shift->adam(pshift, gshift, iter);
    scale = pscale.col(0);
    shift = pshift.col(0);

    std::cout << "Updated scale:\n";
    std::cout << scale << "\n";
    std::cout << "Updated shift:\n";
    std::cout << shift << "\n";

    // initialize gradients for next iteration.
    dScale.setZero();
    dShift.setZero();

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
    std::cout << "leaky Relu Gradient ...\n";
    std::cout << "input:\n";
    std::cout << input_data << "\n";
    std::cout << "gradient:\n";
    std::cout << gradients << "\n";
    Eigen::MatrixXd dInput = this->input_data.array().max(0.0).cast<bool>().cast<double>().max(alpha) * gradients.array();
    std::cout << input_data << "\n\n";
    std::cout << "dInput\n\n";
    std::cout << dInput << "\n";
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

Eigen::MatrixXd Activation::computeActivation(const Eigen::MatrixXd& input_data) { 
    Eigen::MatrixXd output;
    if (activationtype == "sigmoid") {
        output = sigmoid(input_data);
    } else
    if (activationtype == "tanh") {
        output = tanh(input_data);
    } else
    if (activationtype == "relu") {
        std::cout << "Relu\n";
        output = relu(input_data);
    } else
    if (activationtype == "leakyrelu") {
        std::cout << "leakyRelu\n";
        output = leakyReLU(input_data, alpha);
    } else
    if (activationtype == "gelu") {
        output = gelu(input_data);
    } else
    if (activationtype == "softmax") {
        std::cout << "softmax\n";
        output = softmax(input_data);
    }
    std::cout << "Activation output\n";
    std::cout << output << "\n";
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
        std::cout << "relu\n";
        dInput = reluGradient(gradients);
    } else
    if (activationtype == "leakyrelu") {
        std::cout << "leakyrelu\n";
        dInput = leakyReluGradient(gradients);
    } else
    if (activationtype == "gelu") {
        dInput = geluGradient(gradients);
    } else
    if (activationtype == "softmax") {
        std::cout << "softmax\n";
        dInput = softmaxGradient(gradients, output_data);
    } 
    return dInput; // this becomes input to the next Node or next Layer backward.
}

Eigen::MatrixXd Activation::forward(Eigen::MatrixXd& input_data) { 

    // Cache for later back propagation.
    this->input_data = input_data;

    // Perform Activation
    Eigen::MatrixXd output = computeActivation(input_data);

    print_string("Activation forward pass ...", true); 

    return output; // this becomes input to the next Node or next Layer.
}

Eigen::MatrixXd Activation::backward(const Eigen::MatrixXd& gradient, const Eigen::MatrixXd& output_data) {

    // Perform Gradient
    Eigen::MatrixXd dInput = computeGradient(gradient, output_data);

    print_string("Activation backward pass ...", true); 

    return dInput; // this becomes input to the next Node or next Layer.
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
    std::cout << "Predicted ...\n";
    std::cout << predicted << "\n";
    std::cout << "target ...\n";
    std::cout << target << "\n";
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
