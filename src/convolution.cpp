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


template <class T>
const aitensor<T> Convolution<T>::forward(const aitensor<T>& input_data) {

    int inputHeight  = input_data.rows();
    int inputWidth   = input_data.cols();
    int kernelHeight = kernel.rows();
    int kernelWidth  = kernel.cols();
    int outputHeight = (inputHeight + 2 * padding - dilation * (kernelHeight - 1) - 1) / stride + 1;
    int outputWidth  = (inputWidth + 2 * padding - dilation * (kernelWidth - 1) - 1) / stride + 1;
    MatrixXd result(outputHeight, outputWidth);

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
                        convSum += input(inputRow, inputCol) * kernel(k, l);
                    }
                }
            }

            // Store the result
            result(i, j) = convSum;
        }
    }

    return result;
}

template <class T>
const aitensor<T> Convolution<T>::backward(const aitensor<T>& gradients) {
    return aitensor<T>();
}

template <class T>
void Convolution<T>::updateParameters(std::string& optimizertype, T& learningRate, int& iter) {

}

/**********  Convolution Network initialize templates *****************/

template class Convolution<float>;  // Instantiate with float
template class Convolution<double>;  // Instantiate with double

