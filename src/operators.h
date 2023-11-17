/*
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
 */
 
#ifndef OPERATORS_H
#define OPERATORS_H

#include <any>
#include "logger.h"

class SampleClass {
private:
    std::any privmember;

public:
    template <typename T>
    SampleClass(T value) : privmember(value) {}

    template <typename T>
    T getPrivateMember() {
        return std::any_cast<T>(privmember);
    }
};

/*****************************************************************************************************
* Base Optimizer Class
*****************************************************************************************************/
template<class T>
class Optimizer : public BaseOperator {
private:
    std::string optimizertype = "adam";
    T learningRate = 0.001;
    aimatrix<T> moments;
    aimatrix<T> velocity;
    aimatrix<T> rho;
    aimatrix<T> rms;
    aimatrix<T> accum;
    aimatrix<T> nu;
public:

    Optimizer(const std::string& optimizertype, T& learningRate) {
        this->optimizertype = optimizertype;
        this->learningRate = learningRate; 
        moments.setZero();  
        velocity.setZero();  
        rho.setZero();   
        rms.setZero();   
        accum.setZero();  
        nu.setZero();  
    }

    template <typename UT>
    void update(const std::string& optimizertype, UT& weights, UT& gradients, int currentEpoch = 0) {
        if (optimizertype == "sgd") {
            weights = sgd(weights, gradients, currentEpoch);  
        } else
        if (optimizertype == "momentum") {
            weights = momentum(weights, gradients, currentEpoch);
        } else
        if (optimizertype == "rmsprop") {
            weights = rmsprop(weights, gradients, currentEpoch);
        } else
        if (optimizertype == "adam") {
            weights = adam(weights, gradients, currentEpoch);
        } else
        if (optimizertype == "adagrad") {
            weights = adagrad(weights, gradients, currentEpoch);
        } else
        if (optimizertype == "adamax") {
            weights = adamax(weights, gradients, currentEpoch);
        } else
        if (optimizertype == "nadam") {
            weights = nadam(weights, gradients, currentEpoch);
        }
    }

    // SGD optimizer with optional step decay
    const aimatrix<T> sgd(const aimatrix<T>& weights, const aimatrix<T>& gradients, int currentEpoch = 0);
                
    const aivector<T> sgd(const aivector<T>& weights, const aivector<T>& gradients, int currentEpoch = 0);
                
    const airowvector<T> sgd(const airowvector<T>& weights, const airowvector<T>& gradients, int currentEpoch = 0);

    // Momentum optimizer with optional step decay
    const aimatrix<T> momentum(const aimatrix<T>& weights, const aimatrix<T>& gradients, int currentEpoch = 0,
                    T momentumRate = 0.9);

    const aivector<T> momentum(const aivector<T>& weights, const aivector<T>& gradients, int currentEpoch = 0,
                    T momentumRate = 0.9);
                
    const airowvector<T> momentum(const airowvector<T>& weights, const airowvector<T>& gradients, int currentEpoch = 0,
                    T momentumRate = 0.9);

    // Adam optimizer with optional step decay
    const aimatrix<T> adam(const aimatrix<T>& weights, const aimatrix<T>& gradients, int currentEpoch = 0,
                    const T beta1 = 0.9, const T beta2 = 0.999, const T epsilon = 1e-8);

    const aivector<T> adam(const aivector<T>& weights, const aivector<T>& gradients, int currentEpoch = 0,
                    const T beta1 = 0.9, const T beta2 = 0.999, const T epsilon = 1e-8);
                
    const airowvector<T> adam(const airowvector<T>& weights, const airowvector<T>& gradients, int currentEpoch = 0,
                    const T beta1 = 0.9, const T beta2 = 0.999, const T epsilon = 1e-8);

    // RMSprop optimizer with optional step decay
    const aimatrix<T> rmsprop(const aimatrix<T>& weights, const aimatrix<T>& gradients, int currentEpoch = 0,
                    T rho = 0.9, T epsilon = 1e-8);

    const aivector<T> rmsprop(const aivector<T>& weights, const aivector<T>& gradients, int currentEpoch = 0,
                    T rho = 0.9, T epsilon = 1e-8);
                
    const airowvector<T> rmsprop(const airowvector<T>& weights, const airowvector<T>& gradients, int currentEpoch = 0,
                    T rho = 0.9, T epsilon = 1e-8);

    // Adagrad optimizer with optional step decay
    const aimatrix<T> adagrad(const aimatrix<T>& weights, const aimatrix<T>& gradients, int currentEpoch = 0,
                    T epsilon = 1e-8);

    // Adagrad optimizer with optional step decay
    const aivector<T> adagrad(const aivector<T>& weights, const aivector<T>& gradients, int currentEpoch = 0,
                    T epsilon = 1e-8);

    // Adagrad optimizer with optional step decay
    const airowvector<T> adagrad(const airowvector<T>& weights, const airowvector<T>& gradients, int currentEpoch = 0,
                    T epsilon = 1e-8);

    // Adamax optimizer with optional step decay
    const aimatrix<T> adamax(const aimatrix<T>& weights, const aimatrix<T>& gradients, int currentEpoch = 0, 
                    T beta1 = 0.9, T beta2 = 0.999, T epsilon = 1e-8);

    // Adamax optimizer with optional step decay
    const aivector<T> adamax(const aivector<T>& weights, const aivector<T>& gradients, int currentEpoch = 0, 
                    T beta1 = 0.9, T beta2 = 0.999, T epsilon = 1e-8);

    // Adamax optimizer with optional step decay
    const airowvector<T> adamax(const airowvector<T>& weights, const airowvector<T>& gradients, int currentEpoch = 0, 
                    T beta1 = 0.9, T beta2 = 0.999, T epsilon = 1e-8);

    // Nadam optimizer with optional step decay
    const aimatrix<T> nadam(const aimatrix<T>& weights, const aimatrix<T>& gradients, int currentEpoch = 0, 
                    T beta1 = 0.9, T beta2 = 0.999, T epsilon = 1e-8);

    // Nadam optimizer with optional step decay
    const aivector<T> nadam(const aivector<T>& weights, const aivector<T>& gradients, int currentEpoch = 0, 
                    T beta1 = 0.9, T beta2 = 0.999, T epsilon = 1e-8);

    // Nadam optimizer with optional step decay
    const airowvector<T> nadam(const airowvector<T>& weights, const airowvector<T>& gradients, int currentEpoch = 0, 
                    T beta1 = 0.9, T beta2 = 0.999, T epsilon = 1e-8);

    void forwardPass() {}
    void backwardPass() {}
};

/*****************************************************************************************************
* Base Linear Class
*****************************************************************************************************/
template <class T>
class Linear : public BaseOperator {
private:
    aitensor<T> input_data; // BxNxM samples where B=Batch Size, N=input Size, M=number of features
    aitensor<T> output_data;

    OperationParams<T> parameters; // Learnable Parameters. The core of AI.
    std::vector<OperationParams<T>> vgradients; // inputs to next backward-wise Nodes   (gradients with respect to weights & biases)

    int M = 0; // number of features (embedding vector size)
    int W = 0; // number of weights 
    bool bias = true; // Use bias by default.

    Optimizer<T>* opt_weights = nullptr; // for optimizer
    Optimizer<T>* opt_biases = nullptr; // for optimizer

    int batch_size      = 0;
    int input_size      = 0;
    int embedding_size  = 0;
    int outputHeight    = 0;
    int outputWidth     = 0;

public:
    Linear(int size, bool bias = true)  {
        this->W = size;
        this->bias = bias;
        log_info( "**** Linear instance created ****" );
    }

    // This assumes that the input is defined with NxM dimensionality.
    // Therefore the size of the parameters and thus gradients will be based on MxW where W is the number of weights to use;
    // The output will have NxW dimension.
    void setInitialWeights(int M);

    OperationParams<T> getParameters() const;

    std::vector<OperationParams<T>> getGradients() const;

    // While the parameter weight has dimension MxW,  the resulting transformation has dimension of NxW.
    // We only need the M dimension from an NxM input to generate parameter matrix.
    // where weights is NxW and bias is W.
    const aimatrix<T> linearTransform(const aimatrix<T>& input_data);

    // const aitensor<T> forward(const aitensor<T>& input_data);

    const aitensor<T> forward(const aitensor<T>& input_data);

    // aitensor<T> getOutput() { return this->output_data; }

    OperationParams<T> gradient_Wrt_Weight_Bias(const aimatrix<T>& new_gradients, const aimatrix<T>& input_data);

    const aimatrix<T> gradient_Wrt_Input(const aimatrix<T>& new_gradients);

    // Leave the gradients as is. They are cached in the Node. 
    // They will be used to update the parameters in next parallel operations.
    // the dInput is the gradient we propagate to source Nodes in the graph;
    // while the parameter gradients get cached to be used to update the parameters later.
    const aitensor<T> backward(const aitensor<T>& gradients);

    void updateParameters(std::string& optimizertype, T& learningRate, int& iter);

    std::string generateDotFormat(const std::string& name = "generic", bool operators = false, bool weights = false);

    void forwardPass() {}
    void backwardPass() {}


};

/*****************************************************************************************************
* Base BatchNorm Class
*****************************************************************************************************/
template <class T>
class BatchNorm : public BaseOperator {
private:

    // Cached data for backpropagation.
    aitensor<T> input_data; // BxNxM samples where B=Batch Size, N=input Size, M=number of features

    // Across W dimension
    aivector<T> scale; // (gamma) Learnable parameter.
    aivector<T> shift; // (beta) Learnable parameter.

    std::vector<aivector<T>> vgscale; // gradient for scale (gamma)
    std::vector<aivector<T>> vgshift; // gradient for the shift (beta)


    aitensor<T> normalizedInput; // BxNxM (initial dataset) or  BxNxW after transformation
    aitensor<T> minusMean; 
    aivector<T> batchStdDev;
    int M = 0;

    double epsilon=1e-8;

    Optimizer<T>* opt_scale = nullptr; // for optimizer
    Optimizer<T>* opt_shift = nullptr; // for optimizer


    int batch_size      = 0;
    int input_size      = 0;
    int param_size      = 0;
    int outputHeight    = 0;
    int outputWidth     = 0;

public:
    BatchNorm() {
      // initialize gradients for next iteration.
        vgscale.clear();
        vgshift.clear();
        log_info( "**** Batch normalization instance created ****" );
    }

    // This assumes that the input is defined with NxM dimensionality.
    // Therefore the size of the parameters and thus gradients will be based on MxW where W is the number of weights to use.
    void setInitialWeights(int M);

    std::tuple<aimatrix<T>, aimatrix<T>, aimatrix<T>> normalize(const aimatrix<T>& input_data);

    const aitensor<T> forward(const aitensor<T>& input_data);

    // Leave the gradients as is for the scale and shift. They are cached in the Node. 
    // They will be used to update the parameters in next parallel operations.
    // the dInput is the gradient we propagate to source Nodes in the graph;
    // while the parameter gradients get cached to be used to update the parameters later.
    const aitensor<T> backward(const aitensor<T>& gradients);

    void updateParameters(std::string& optimizertype, T& learningRate, int& iter);

    std::string generateDotFormat(const std::string& name = "generic", bool operators = false, bool weights = false);

    void forwardPass() {}
    void backwardPass() {}

};

/*****************************************************************************************************
* Base LayerNorm Class
*****************************************************************************************************/
template <class T>
class LayerNorm : public BaseOperator {
private:
    aitensor<T> input_data; // BxNxW samples

    // Across W dimension
    aivector<T> scale; // (gamma) Learnable parameter.
    aivector<T> shift; // (beta) Learnable parameter.

    std::vector<aivector<T>> vgscale; // gradient for scale (gamma)
    std::vector<aivector<T>> vgshift; // gradient for the shift (beta)

    // Cached data for backpropagation.
    aitensor<T> normalizedInput; // BxNxW samples 
    aitensor<T> minusMean;
    aivector<T> layerStdDev;
    int N = 0;

    double epsilon=1e-8;

    Optimizer<T>* opt_scale = nullptr; // for optimizer
    Optimizer<T>* opt_shift = nullptr; // for optimizer

    int batch_size      = 0;
    int input_size      = 0;
    int param_size      = 0;
    int outputHeight    = 0;
    int outputWidth     = 0;

public:
    LayerNorm() {
        // initialize gradients for next iteration.
        vgscale.clear();
        vgshift.clear();
        log_info( "**** Layer normalization instance created ****" );
    }

    // This assumes that the input is defined with NxM dimensionality.
    // Therefore the size of the parameters and thus gradients will be based on MxW where W is the number of weights to use.
    void setInitialWeights(int N);

    std::tuple<aimatrix<T>, aimatrix<T>, aimatrix<T>> normalize(const aimatrix<T>& input_data);

    const aitensor<T> forward(const aitensor<T>& input_data);

    // Leave the gradients as is for the scale and shift. They are cached in the Node. 
    // They will be used to update the parameters in next parallel operations.
    // the dInput is the gradient we propagate to source Nodes in the graph;
    // while the parameter gradients get cached to be used to update the parameters later.
    const aitensor<T> backward(const aitensor<T>& gradients);

    void updateParameters(std::string& optimizertype, T& learningRate, int& iter);

    std::string generateDotFormat(const std::string& name = "generic", bool operators = false, bool weights = false);

    void forwardPass() {}
    void backwardPass() {}
};

/*****************************************************************************************************
* Base Activation Class
*****************************************************************************************************/
template <class T>
class Activation : public BaseOperator {
private:

    // Cached data for backpropagation.
    aitensor<T> input_data; // BxNxW samples 
    aitensor<T> output_data; // BxNxW samples 

    // To support generateDotFormat()
    int max_dInput = 0;
    int min_dInput = 0;

    std::string activationtype = "leakyrelu";
    T alpha = 0.01; // for leakyReLU

    int N = 0;
    int M = 0;

    int batch_size;
    int input_size;
    int param_size;

public:

    Activation(const std::string& activationtype = "leakyrelu") {
        this->activationtype = activationtype;
        log_info( "**** Activation instance created ****" );
    }

    Activation(const std::string& activationtype = "leakyrelu", const T alpha=0.01) {
        this->activationtype = activationtype;
        this->alpha = alpha;
        log_info( "**** Activation instance created ****" );
    }

    //  Resize dInput.
    void setInitSize(const aimatrix<T>& input_data);

    // Here, instead of using the term logits, let's just use x.
    const aimatrix<T>  softmax(const aimatrix<T>& x);

    /*****************************************************************************************************
     * So, the gradient of the sigmoid function with respect to its input (z) can be expressed as follows:
     * dy/dz = propagated_gradient * sigmoid(z) * (1 - sigmoid(z))
     * Or if output (y) is cached, where y = sigmoid(z)
     * then we use the output such that:
     * dy/dz = propagated_gradient * y * (1 - y)
     *****************************************************************************************************/
    const aimatrix<T>  softmaxGradient(const aimatrix<T>& gradients, const aimatrix<T>& output_data);

    // Here, instead of using the term logits, let's just use x.
    const aimatrix<T>  sigmoid(const aimatrix<T>& x);

    /*****************************************************************************************************
     * So, the gradient of the sigmoid function with respect to its input (z) can be expressed as follows:
     * dy/dz = propagated_gradient * sigmoid(z) * (1 - sigmoid(z))
     * Or if output (y) is cached, where y = sigmoid(z)
     * then we use the output such that:
     * dy/dz = propagated_gradient * y * (1 - y)
     *****************************************************************************************************/
    const aimatrix<T>  sigmoidGradient(const aimatrix<T>& gradients, const aimatrix<T>& output_data);

    const aimatrix<T>  tanh(const aimatrix<T>& x);

    /*****************************************************************************************************
     * So, the gradient of the tanh function with respect to its input (z) can be expressed as follows:
     * dy/dz = propagated_gradient * (1 - tanh(z)^2)
     * Or if output (y) is cached, where y = tanh(z)
     * then we use the output such that:
     * dy/dz = propagated_gradient * y * (1 - y^2)
     *****************************************************************************************************/
    const aimatrix<T>  tanhGradient(const aimatrix<T>& gradients, const aimatrix<T>& output_data);

    const aimatrix<T> relu(const aimatrix<T>& x);

    /*****************************************************************************************************
     * So, the gradient of the ReLU function with respect to its input (z) can be expressed as follows:
     * dy/dz = propagated_gradient * 1 for z > 0
     * dy/dz = propagated_gradient * 0 for z <= 0
     *****************************************************************************************************/
    const aimatrix<T>  reluGradient(const aimatrix<T>& gradients, const aimatrix<T>& input);  

    const aimatrix<T> leakyReLU(const aimatrix<T>& x, float alpha);

    /*****************************************************************************************************
     * So, the gradient of the LeakyReLU function with respect to its input (z) can be expressed as follows:
     * dy/dz = propagated_gradient * 1 for z > 0
     * dy/dz = propagated_gradient * alpha for z <= 0
    *****************************************************************************************************/
    const aimatrix<T> leakyReluGradient(const aimatrix<T>& gradients, const aimatrix<T>& input);

    const aimatrix<T> gelu(const aimatrix<T>& x);

    /*****************************************************************************************************
     * Gelu Gradient ...
    *****************************************************************************************************/
    const aimatrix<T> geluGradient(const aimatrix<T>& gradients, const aimatrix<T>& input);

    const aimatrix<T> computeActivation(const aimatrix<T>& input_data);

    const aimatrix<T> computeGradient(const aimatrix<T>& gradients, const aimatrix<T>& output, const aimatrix<T>& input);

    const aitensor<T> forward(const aitensor<T>& input_data);

    const aitensor<T> backward(const aitensor<T>& gradients);

    std::string generateDotFormat(const std::string& name = "generic", bool operators = false, bool weights = false);

    void forwardPass() {}
    void backwardPass() {}
};

/*****************************************************************************************************
* Base Dropout Class
* set approximately P% of cells in the matrix to zero.
*****************************************************************************************************/
template <class T>
class Dropout: public BaseOperator {
private:
    float probability = 0.05; // 50% dropout

    int batch_size = 0;
    int input_size = 0;
    int param_size = 0;

    aitensor<T> mask_filter;

    // Using the Fisher-Yates shuffle algorithm
    aimatrix<T> maskedMatrix(int rows, int cols) {

        aimatrix<T> matrix = aimatrix<T>::Zero(rows, cols); 

        // Create a vector containing numbers from 1 to 10
        std::vector<int> numbers;
        for (int i = 1; i <= rows * cols; ++i) {
            numbers.push_back(i);
        }

        // Create a random number generator
        std::random_device rd;
        std::mt19937 generator(rd());
        std::shuffle(numbers.begin(), numbers.end(), generator);

        // Calculate the number of cells to set to 1.
        // If drop probability is 0.20, then  ( 1 - 0.20 ) will be set to 1.
        int num_cells_to_set = static_cast<int>(rows * cols * (1 - this->probability));

        log_detail("probability: {0} size: {1} out of {2}", this->probability, num_cells_to_set, rows * cols );

        // Take the first 'sequenceSize' elements from the shuffled vector
        std::vector<int> sequence(numbers.begin(), numbers.begin() + num_cells_to_set);

        int target = 0;
        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                target ++ ;
                // Use std::find to search for the element
                auto it = std::find(sequence.begin(), sequence.end(), target);
                if (it != sequence.end()) {
                    matrix(i, j) = 1.0;
                }
            }
        }

        return matrix;
    }

    
public:
    Dropout(const float probability = 0.05) {
        this->probability = probability;
    }

    const aitensor<T> forward(const aitensor<T>& input_data);

    const aitensor<T> backward(const aitensor<T>& gradients);

    std::string generateDotFormat(const std::string& name = "generic", bool operators = false, bool weights = false);

    void forwardPass() {}
    void backwardPass() {}

};

/*****************************************************************************************************
* Base Flatten Class
*****************************************************************************************************/
template <class T>
class Flatten: public BaseOperator {
private:
    int input_height = 0;
    int input_width  = 0;
public:

    Flatten() {}

    const aitensor<T> forward(const aitensor<T>& input_data);

    const aitensor<T> backward(const aitensor<T>& gradients);

    std::string generateDotFormat(const std::string& name = "generic", bool operators = false, bool weights = false);

    void forwardPass() {}
    void backwardPass() {}

};

/*****************************************************************************************************
* Base Reduction Class
*****************************************************************************************************/
class Reduction : public BaseOperator {
private:
    Eigen::MatrixXd input_data;
    std::string reducttype = "add";

public:
    Reduction(const std::string& reducttype = "add") {
        this->reducttype = reducttype;
    }

    std::string getType();

    void forwardPass() {}
    void backwardPass() {}

};

/*****************************************************************************************************
* Base Loss Class
*****************************************************************************************************/
template <class T>
class Loss : public BaseOperator {
private:
    // std::string losstype = "mse";

    // int batch_size;
    // int input_size;
    // int param_size;

public:

    // Mean Squared Error. Returns 1x1 matrix (scalar)
    static const aiscalar<T> mse(const aimatrix<T>& predicted, const aimatrix<T>& target);

    static const aimatrix<T> mseGradient(const aimatrix<T>& predicted, const aimatrix<T>& target);

    // Binary Cross Entropy.  Returns 1x1 matrix (scalar)
    static const aiscalar<T> bce(const aimatrix<T>& predicted, const aimatrix<T>& target);

    static const aimatrix<T> bceGradient(const aimatrix<T>& predicted, const aimatrix<T>& target);

    // For Loss Categorial Cross Entropy. Usually, we use Softmax.
    static const aiscalar<T> cce(const aimatrix<T>& predicted, const aimatrix<T>& target);

    static const aimatrix<T> cceGradient(const aimatrix<T>& predicted, const aimatrix<T>& target);

    // For Support Vectors (not necessarily for Neural)
    static const aiscalar<T> hingeLoss(const aimatrix<T>& predicted, const aimatrix<T>& target);

    static const aimatrix<T> hingeLossGradient(const aimatrix<T>& predicted, const aimatrix<T>& target);

    static const aiscalar<T> computeLoss(const std::string& losstype, const aitensor<T>& predicted, const aitensor<T>& target);

    static const aitensor<T> computeGradients(const std::string& losstype, const aitensor<T>& predicted, const aitensor<T>& target);

    void forwardPass() {}
    void backwardPass() {}
};


/*****************************************************************************************************
* Base Metrics Class
* Operational only for classifications where loss function is either BCE or CCE
*****************************************************************************************************/
template <class T>
class Metrics : public BaseOperator {
private:

    static std::tuple<T, T, T> calculateMetrics(const aimatrix<T>& prediction, const aimatrix<T>& target) {
        int rows = target.rows();
        int cols = target.cols();

        T TP = 0.0;
        T FP = 0.0;
        T FN = 0.0;

        std::cout << "Precision Entering Calculation" << std::endl;
        std::cout << "Precision TP:" << TP << " FP:" << FP << " FN:" << FN << std::endl;

        for (int i = 0; i < rows; i++) {
            for (int j = 0; j < cols; j++) {
                T actual = target(i, j);
                T predicted = prediction(i, j);

                std::cout << "Precision Entering Calculation:(" << actual << "):(" << predicted << ")" << std::endl;

                actual = (actual > 0.5) ? 1.0 : 0.0;
                predicted = (predicted > 0.5) ? 1.0 : 0.0;

                if (actual == 1.0 && predicted == 1.0) {
                    TP++;
                } else if (actual == 0.0 && predicted == 1.0) {
                    FP++;
                } else if (actual == 1.0 && predicted == 0.0) {
                    FN++;
                }
                std::cout << "Precision TP:" << TP << " FP:" << FP << " FN:" << FN << std::endl;

            }
        }

        T precision = (TP) / (TP + FP);
        T recall = (TP) / (TP + FN);
        T f1score = (2.0 * precision * recall) / (precision + recall);

        std::cout << "Precision then: " << precision << std::endl;

        return std::make_tuple(precision, recall, f1score);
    }

    static bool findMetrics(const std::vector<std::string>& metricstype, const std::string& target) {
        auto it =  std::find(metricstype.begin(), metricstype.end(), target);
        return (it != metricstype.end());
    }

public:


    static const PerfMetrics<T> computeMetrics(const std::vector<std::string>& metricstype, const aitensor<T>& predicted, const aitensor<T>& target);

    // AUC-ROC. Returns 1x1 matrix (scalar)
    static const aiscalar<T> aucroc(const aimatrix<T>& predicted, const aimatrix<T>& target);

    void forwardPass() {}
    void backwardPass() {}
};

#endif
