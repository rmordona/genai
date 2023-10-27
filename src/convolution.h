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

#pragma once
#ifndef CONVOLUTION_H
#define CONVOLUTION_H

#include "operators.h"

/**************************************************************************************************************************
* Convolution Class:
* This class provides the convolution layer of a neural network. 
* This assumes that the input is defined with BxNxM dimensionality.
* Therefore the size of the parameters and thus gradients will be based on MxW where W is the number of weights to use;
* The output will have BxNxW dimension.
***************************************************************************************************************************/
template <class T>
class Convolution : public BaseOperator {
private:
    int kernel_size = 2;
    int stride      = 1;
    int padding     = 1;
    int dilation    = 1;

    bool bias = true; // Use bias by default.

    int batch_size    = 0;
    int inputHeight   = 0;
    int inputWidth    = 0;
    int kernelHeight  = 0;
    int kernelWidth   = 0;
    int outputHeight = 0;
    int outputWidth   = 0;

    aitensor<T> input_data;
    aitensor<T> output_data;

    OperationParams<T> kernel; // Learnable Parameters. The core of AI.
    std::vector<OperationParams<T>> vgradients; // inputs to next backward-wise Nodes   (gradients with respect to weights & biases)

    Optimizer<T>* opt_weights = nullptr; // for optimizer
    Optimizer<T>* opt_biases = nullptr; // for optimizer


public:
    Convolution(int kernel_size = 2, int stride = 1, int padding = 1, int dilation = 1, bool bias = true)   {
        this->kernel_size = kernel_size;
        this->stride      = stride;
        this->padding     = padding;
        this->dilation    = dilation;   
        this->bias        = bias;
        setInitialWeights(this->kernel_size);     
    }

    void setInitialWeights(int kernel_size);
    const aitensor<T> forward(const aitensor<T>& input_data);
    const aitensor<T> backward(const aitensor<T>& gradients);
    void updateParameters(std::string& optimizertype, T& learningRate, int& iter);

    std::string generateDotFormat(const std::string& name = "generic", bool operators = false, bool weights = false);

    void forwardPass() {} // virtual function of BaseOperator (different from those of the RecurrentBase)
    void backwardPass() {} // virtual function of BaseOperator (different from those of the RecurrentBase)
};

#endif
