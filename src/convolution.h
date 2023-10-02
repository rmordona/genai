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

template <class T>
class Convolution : public BaseOperator {
private:
    int kernel_size = 0;
    int stride      = 0;
    int padding     = 0;
    int dilation    = 0;
public:
    Convolution(int kernel_size = 3, int stride = 1, int padding = 1, int dilation = 1)   {
        this->kernel_size = kernel_size;
        this->stride      = stride;
        this->padding     = padding;
        this->dilation    = dilation;        
    }

    const aitensor<T> forward(const aitensor<T>& input_data);
    const aitensor<T> backward(const aitensor<T>& gradients);
    void updateParameters(std::string& optimizertype, T& learningRate, int& iter);

    void forwardPass() {} // virtual function of BaseOperator (different from those of the RecurrentBase)
    void backwardPass() {} // virtual function of BaseOperator (different from those of the RecurrentBase)
};

#endif
