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

    int batch_size = input_data.size();

    int inputHeight  = input_data.at(0).rows();
    int inputWidth   = input_data.at(0).cols();

    for (int h = 0; h < batch_size; h++) {

        int kernelHeight = kernel.weights.rows();
        int kernelWidth  = kernel.weights.cols();
        int outputHeight = (inputHeight + 2 * padding - dilation * (kernelHeight - 1) - 1) / stride + 1;
        int outputWidth  = (inputWidth + 2 * padding - dilation * (kernelWidth - 1) - 1) / stride + 1;
        aimatrix<T> result(outputHeight, outputWidth);

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
                            convSum += BaseOperator::matmul(input(inputRow, inputCol),  kernel.weights(k, l)).rowwise() + kernel.biases(k);
                        }
                    }
                }

                // Store the result
                result(i, j) = convSum;
            }
        }

        this->output_data.push(result);
    }

    return this->output_data;
}

template <class T>
const aitensor<T> Convolution<T>::backward(const aitensor<T>& gradients) {

    log_info("===============================================");
    log_info("Convolution Backward Pass ...");

    int batch_size = this->input_data.size();

    int inputHeight  = this->input_data.at(0).rows();
    int inputWidth   = this->input_data.at(0).cols();

    int outputHeight  = gradients.at(0).rows();
    int outputWidth   = gradients.at(0).cols();

    int kernelHeight = kernel.weights.rows();
    int kernelWidth  = kernel.weights.cols();

    // Initialize gradients
    aitensor<T> dInput;
    OperationParams<T> dkernel; 
    dkernel.weghts.resize(this->kernel_size, this->kernel_size);
    dkernel.biases.resize(this->kernel_size);

    aimatrix<T> dOut;

    OperationParams<T> dkernel; // Learnable Parameters. The core of AI.

    for (int h = 0; h < batch_size; h++) {

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
                            d_input(inputRow, inputCol) += BaseOperator::matmul(d_output_value, kernel.weights(k, l));

                            // Gradient with respect to kernel weights
                            dkernel.weights(k, l) += BaseOperator::matmul(d_output_value, this->input_data(inputRow, inputCol));
                        }
                    }
                }
            }
        }

        // Gradients with respect to biases
        for (int k = 0; k < kernelHeight; k++) {
            dkernel.biases(k) = dOut.row(k).sum();
        }

        vgradients.push(dkernel);
        dInput.push(d_input);
    }

    return dInput;
}

template <class T>
void Convolution<T>::updateParameters(std::string& optimizertype, T& learningRate, int& iter) {


for (int i = 0; i < this->batch_size; ++i) {

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


    this->vgradients.clear();
    this->output_data.clear();
}

/**********  Convolution Network initialize templates *****************/

template class Convolution<float>;  // Instantiate with float
template class Convolution<double>;  // Instantiate with double

