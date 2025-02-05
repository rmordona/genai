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
 */

#include "genai.h"
#include "convolution.h"

/**************************************************************************************************************************
* Convolution Class:
* This class provides the convolution layer of a neural network. 
* This assumes that the input is defined with BxNxM dimensionality.
* Therefore the size of the parameters and thus gradients will be based on MxW where W is the number of weights to use;
* The output will have BxNxW dimension.
***************************************************************************************************************************/
template <class T>
void Convolution<T>::setInitialWeights(int kernel_size) {

    kernel.weights.resize(kernel_size, kernel_size); // allocates memory
    kernel.biases.resize(kernel_size); // allocates memory

    log_detail( "manual: kernel Dimension: {0}x{1}",  kernel.weights.rows(),  kernel.weights.cols());

    // Initialize Weights & Biases
    heInitMatrix(kernel.weights);
    // heInitialization(parameters.biases);
    kernel.biases.setConstant(T(0.01));

    log_detail( "manual: kernel Dimension: {0}x{1}",  kernel.weights.rows(),  kernel.weights.cols());

}

template <class T>
const aitensor<T> Convolution<T>::forward(const aitensor<T>& input_data) {

    log_info("===============================================");
    log_info("Convolution Forward Pass ...");

    this->input_data = input_data;
    this->batch_size   = input_data.size();
    this->inputHeight  = input_data.at(0).rows();
    this->inputWidth   = input_data.at(0).cols();
    this->kernelHeight = kernel.weights.rows();
    this->kernelWidth  = kernel.weights.cols();
    this->outputHeight = (inputHeight - ((kernelHeight - 1) * dilation + 1) + 2 * padding)/stride + 1;
    this->outputWidth = (inputHeight - ((kernelWidth - 1) * dilation + 1) + 2 * padding)/stride + 1;

    aimatrix<T> input;
    aimatrix<T> output(outputHeight, outputWidth);

    log_detail("Convolution input size {0}x{1}", inputHeight, inputWidth);
    log_detail("Convolution kernel size {0}x{1}", kernelHeight, kernelWidth);
    log_detail("Convolution output size {0}x{1}", outputHeight, outputWidth);

    log_detail("Kernel:");
    log_matrix(kernel.weights);

    for (int h = 0; h < batch_size; h++) {

        input = this->input_data.at(h);

        for (int i = 0; i < outputHeight; i++) {
            for (int j = 0; j < outputWidth; j++) {
                // Initialize the convolution result at (i, j)
                double convSum = 0.0;
                // Perform convolution with dilation, stride, and padding
                for (int k = 0; k < kernelHeight; k++) {
                    for (int l = 0; l < kernelWidth; l++) {
                        int inputRow = i * stride - padding + k * dilation;
                        int inputCol = j * stride - padding + l * dilation;
                        if (inputRow >= 0 && inputRow < inputHeight && inputCol >= 0 && inputCol < inputWidth) {
                            if (bias == true) {
                                convSum += input(inputRow, inputCol) * kernel.weights(k, l) + kernel.biases(k);
                            } else {
                                convSum += input(inputRow, inputCol) * kernel.weights(k, l);
                            }
                        }
                    }
                }

                // Store the result
                output(i, j) = convSum;
            }
        }

        log_detail("Output after convolution (batch {0}):", h);
        log_matrix(output);

        this->output_data.push_back(output);
    }

    log_info("End Convolution Forward Pass ...");

    return this->output_data;
}
 
template <class T>
const aitensor<T> Convolution<T>::backward(const aitensor<T>& gradients) {
 
    log_info("===============================================");
    log_info("Convolution Backward Pass ...");

    int batch_size   = this->input_data.size();
    int inputHeight  = this->input_data.at(0).rows();
    int inputWidth   = this->input_data.at(0).cols();
    int outputHeight = gradients.at(0).rows();
    int outputWidth  = gradients.at(0).cols();
    int kernelHeight = kernel.weights.rows();
    int kernelWidth  = kernel.weights.cols();

    // Initialize gradients
    aimatrix<T> input;
    aitensor<T> dInput;
    OperationParams<T> dkernel; 
    dkernel.weights.resize(this->kernel_size, this->kernel_size);
    dkernel.biases.resize(this->kernel_size);

    aimatrix<T> dOut;

    for (int h = 0; h < batch_size; h++) {

        input = this->input_data.at(h);

        aimatrix<T> d_input(inputHeight, inputWidth);
        dOut = gradients.at(h);

        for (int i = 0; i < outputHeight; i++) {
            for (int j = 0; j < outputWidth; j++) {
                // Gradients with respect to output
                T d_output_value = dOut(i, j);

                for (int k = 0; k < kernelHeight; k++) {
                    for (int l = 0; l < kernelWidth; l++) {
                        int inputRow = i * stride - padding + k * dilation;
                        int inputCol = j * stride - padding + l * dilation;
                        if (inputRow >= 0 && inputRow < inputHeight && inputCol >= 0 && inputCol < inputWidth) {
                            // Gradient with respect to input data
                            d_input(inputRow, inputCol) += d_output_value * kernel.weights(k, l);
                            // Gradient with respect to kernel weights
                            dkernel.weights(k, l) += d_output_value * input(inputRow, inputCol);
                        }
                    }
                }
            }
        }

        // Gradients with respect to biases
        if (bias == true) {
            for (int k = 0; k < kernelHeight; k++) {
                dkernel.biases(k) = dOut.row(k).sum();
            }
        }

        log_detail("Gradient with respect to kernel (batch {0}):", h);
        log_matrix(dkernel.weights);

        log_detail("Gradient with respect to Input (batch {0}):", h);
        log_matrix(dkernel.weights);

        vgradients.push_back(dkernel);
        dInput.push_back(d_input);
    }

    log_info("End Convolution ForwBackwardard Pass ...");

    return dInput;
}
 
template <class T>
void Convolution<T>::updateParameters(std::string& optimizertype, T& learningRate, int& iter) {

    log_info("==================================================");
    log_info("Entering Convolution Upgrade Parameters...");

    int batch_size   = this->input_data.size();

    for (int i = 0; i < batch_size; ++i) {

            OperationParams<T>  gradients = vgradients[i];

            if (opt_weights == nullptr) {
                opt_weights = new Optimizer<T>(optimizertype, learningRate);
            }

            if (opt_biases == nullptr && bias == true) {
                opt_biases = new Optimizer<T>(optimizertype, learningRate);
            }

            log_detail( "Updating Linear weights" );
            opt_weights->update(optimizertype, kernel.weights, gradients.weights, iter);
            log_detail( "Updating Linear biases" );

            if (bias == true) {
                opt_biases->update(optimizertype, kernel.biases, gradients.biases, iter);
            }


    }
    this->vgradients.clear();
    this->output_data.clear();
}

template <class T>
Topology Convolution<T>::generateDotFormat(const std::string& name , bool operators, bool weights) {
    Topology topology;
    topology.dot = "{* Convolution (" + name + ") *}|";  
    topology.parameters = 0;
    T min_weights = 0.0, max_weights = 0.0;
    T min_biases = 0.0, max_biases = 0.0;
    try {
        min_weights = kernel.weights.minCoeff();
        max_weights = kernel.weights.maxCoeff();
        min_biases = kernel.biases.minCoeff();
        max_biases = kernel.biases.maxCoeff();
    } catch (...) {};
    int parameters = kernel.weights.rows() * kernel.weights.cols() + kernel.biases.size();
    topology.parameters += parameters;
    topology.dot += "{Parameters=" + std::to_string(parameters) + "|" +
             "BatchSize=" + std::to_string(batch_size) + "}|";
    topology.dot += "{Input=(" + std::to_string(inputHeight) + " x " + std::to_string(inputWidth) + ")|";  
    topology.dot += "Kernel=(" + std::to_string(kernelHeight) + " x " + std::to_string(kernelWidth) + ")|";  
    topology.dot += "Output=(" + std::to_string(outputHeight) + " x " + std::to_string(outputWidth) + ")}|";  
    if (weights == true) {
    topology.dot += "{Kernel Weights|min=" + scalar_to_string(min_weights) + "|max=" + scalar_to_string(max_weights) + "}|";    
    topology.dot += "{Kernel Biases|min=" + scalar_to_string(min_biases) + "|max=" + scalar_to_string(max_biases) + "}"; 
    } else {
        topology.dot.pop_back(); // Remove dangling | character
    }
    return topology;
}

/**********  Convolution Network initialize templates *****************/

template class Convolution<float>;  // Instantiate with float
template class Convolution<double>;  // Instantiate with double

