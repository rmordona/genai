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

    // SGD optimizer with optional step decay
    void sgd(aimatrix<T>& weights, const aimatrix<T>& gradients, int currentEpoch = 0,
                    bool useStepDecay = false, T decayRateStep = 0.1, int decayStep = 0);

    // Momentum optimizer with optional step decay
    void momentum(aimatrix<T>& weights, const aimatrix<T>& gradients, int currentEpoch = 0,
                    T momentumRate = 0.9,
                    bool useStepDecay = false, T decayRateStep = 0.1,  int decayStep = 0);

    // Adam optimizer with optional step decay
    void adam(aimatrix<T>& weights, const aimatrix<T>& gradients, int currentEpoch = 0,
                    T beta1 = 0.9, T beta2 = 0.999, T epsilon = 1e-8,
                    bool useStepDecay = false, T decayRateStep = 0.1,  int decayStep = 0);

    // RMSprop optimizer with optional step decay
    void rmsprop(aimatrix<T>& weights, const aimatrix<T>& gradients, int currentEpoch = 0,
                    T rho = 0.9, T epsilon = 1e-8,
                    bool useStepDecay = false, T decayRateStep = 0.1,  int decayStep = 0);

    // Adagrad optimizer with optional step decay
    void adagrad(aimatrix<T>& weights, const aimatrix<T>& gradients, int currentEpoch = 0,
                    T epsilon = 1e-8,
                    bool useStepDecay = false, T decayRateStep = 0.1,  int decayStep = 0);

    // Adamax optimizer with optional step decay
    void adamax(aimatrix<T>& weights, const aimatrix<T>& gradients, int currentEpoch = 0, 
                    T beta1 = 0.9, T beta2 = 0.999, T epsilon = 1e-8,
                    bool useStepDecay = false, T decayRateStep = 0.1, int decayStep = 0);

    // Nadam optimizer with optional step decay
    void nadam(aimatrix<T>& weights, const aimatrix<T>& gradients, int currentEpoch = 0, 
                    T beta1 = 0.9, T beta2 = 0.999, T epsilon = 1e-8,
                    bool useStepDecay = false, T decayRateStep = 0.1, int decayStep = 0);

    // Step decay for learning rate
    void stepDecay(T& learningRate, T decayRate, int currentEpoch, int decayStep);

    void forwardPass() {}
    void backwardPass() {}
};

template <class T>
class Linear : public BaseOperator {
private:
    aitensor<T> input_data; // BxNxM samples where B=Batch Size, N=input Size, M=number of features

    OperationParams<T> parameters; // Learnable Parameters. The core of AI.

    // OperationParams gradients;  // inputs to next backward-wise Nodes   (gradients with respect to weights & biases)
    std::vector<OperationParams<T>> vgradients; // inputs to next backward-wise Nodes   (gradients with respect to weights & biases)

    int M = 0; // number of features (embedding vector size)
    int W = 0; // number of weights 
    bool bias = true; // Use bias by default.

    Optimizer<T>* opt_weights = nullptr; // for optimizer
    Optimizer<T>* opt_biases = nullptr; // for optimizer

    int batch_size;
    int input_size;
    int embedding_size;

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

    const aitensor<T> forward(const aitensor<T>& input_data);

    OperationParams<T> gradient_Wrt_Weight_Bias(const aimatrix<T>& new_gradients, const aimatrix<T>& input_data);

    const aimatrix<T> gradient_Wrt_Input(const aimatrix<T>& new_gradients);

    // Leave the gradients as is. They are cached in the Node. 
    // They will be used to update the parameters in next parallel operations.
    // the dInput is the gradient we propagate to source Nodes in the graph;
    // while the parameter gradients get cached to be used to update the parameters later.
    const aitensor<T> backward(const aitensor<T>& gradients);

    void updateParameters(std::string& optimizertype, T& learningRate, int& iter);

    void forwardPass() {}
    void backwardPass() {}
    std::string generateDotFormat(const std::string& name = "generic");

};

template <class T>
class BatchNorm : public BaseOperator {
private:

    // Cached data for backpropagation.
    aitensor<T> input_data; // BxNxM samples where B=Batch Size, N=input Size, M=number of features

    // Across W dimension
    airowvector<T> scale; // (gamma) Learnable parameter.
    airowvector<T> shift; // (beta) Learnable parameter.

    std::vector<airowvector<T>> vgscale; // gradient for scale (gamma)
    std::vector<airowvector<T>> vgshift; // gradient for the shift (beta)


    aitensor<T> normalizedInput; // BxNxW samples 
    aitensor<T> minusMean;
    airowvector<T> batchStdDev;
    int M = 0;

    double epsilon=1e-8;

    Optimizer<T>* opt_scale = nullptr; // for optimizer
    Optimizer<T>* opt_shift = nullptr; // for optimizer

    int batch_size;
    int input_size;
    int param_size;

public:
    BatchNorm() {
      // initialize gradients for next iteration.
        vgscale.empty();
        vgshift.empty();
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

    void forwardPass() {}
    void backwardPass() {}
    std::string generateDotFormat(const std::string& name = "generic");
};

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

    int batch_size;
    int input_size;
    int param_size;

public:
    LayerNorm() {
        // initialize gradients for next iteration.
        vgscale.empty();
        vgshift.empty();
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

    void forwardPass() {}
    void backwardPass() {}
    std::string generateDotFormat(const std::string& name = "generic");
};

template <class T>
class Activation : public BaseOperator {
private:

    // Cached data for backpropagation.
    aitensor<T> input_data; // BxNxW samples 
    aitensor<T> output_data; // BxNxW samples 
    aitensor<T> dInput; // Gradient

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
    //const aimatrix<T>  sigmoid(const aimatrix<T>& x);

    /*****************************************************************************************************
     * So, the gradient of the sigmoid function with respect to its input (z) can be expressed as follows:
     * dy/dz = propagated_gradient * sigmoid(z) * (1 - sigmoid(z))
     * Or if output (y) is cached, where y = sigmoid(z)
     * then we use the output such that:
     * dy/dz = propagated_gradient * y * (1 - y)
     *****************************************************************************************************/
    //const aimatrix<T>  sigmoidGradient(const aimatrix<T>& gradients, const aimatrix<T>& output_data);

    //const aimatrix<T>  tanh(const aimatrix<T>& x);

    /*****************************************************************************************************
     * So, the gradient of the tanh function with respect to its input (z) can be expressed as follows:
     * dy/dz = propagated_gradient * (1 - tanh(z)^2)
     * Or if output (y) is cached, where y = tanh(z)
     * then we use the output such that:
     * dy/dz = propagated_gradient * y * (1 - y^2)
     *****************************************************************************************************/
    //const aimatrix<T>  tanhGradient(const aimatrix<T>& gradients, const aimatrix<T>& output_data);

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

    void forwardPass() {}
    void backwardPass() {}
    std::string generateDotFormat();
};

/*****************************************************************************************************
* Base Loss Functions
*****************************************************************************************************/
template <class T>
class Loss : public BaseOperator {
private:
    std::string losstype = "mse";

    int batch_size;
    int input_size;
    int param_size;

public:

    Loss(const std::string& losstype = "mse") {
        this->losstype = losstype;
    }

    // Mean Squared Error. Returns 1x1 matrix (scalar)
    const aiscalar<T> mse(const aimatrix<T>& predicted, const aimatrix<T>& target);

    const aimatrix<T> mseGradient(const aimatrix<T>& predicted, const aimatrix<T>& target);

    // Binary Cross Entropy.  Returns 1x1 matrix (scalar)
    const aiscalar<T> bce(const aimatrix<T>& predicted, const aimatrix<T>& target);

    const aimatrix<T> bceGradient(const aimatrix<T>& predicted, const aimatrix<T>& target);

    // For Loss Categorial Cross Entropy. Usually, we use Softmax.
    const aiscalar<T> cce(const aimatrix<T>& predicted, const aimatrix<T>& target);

    const aimatrix<T> cceGradient(const aimatrix<T>& predicted, const aimatrix<T>& target);

    // For Support Vectors (not necessarily for Neural)
    const aiscalar<T> hingeLoss(const aimatrix<T>& predicted, const aimatrix<T>& target);

    const aimatrix<T> hingeLossGradient(const aimatrix<T>& predicted, const aimatrix<T>& target);

    const aiscalar<T> computeLoss(const aitensor<T>& predicted, const aitensor<T>& target);

    const aitensor<T> computeGradients(const aitensor<T>& predicted, const aitensor<T>& target);

    void forwardPass() {}
    void backwardPass() {}
};

class Dropout: public BaseOperator {
public:
    Dropout(int size) {
        // Initialize scaling and shifting parameters
    }

    void forwardPass() {}
    void backwardPass() {}
};

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

#endif
