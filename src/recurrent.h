

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

#ifndef RECURRENT_H
#define RECURRENT_H

enum RNNType {
    MANY_TO_ONE,
    ONE_TO_MANY,
    MANY_TO_MANY
};

class CellBase {
public:

    class Split {
    private:
        Eigen::MatrixXd A;
        Eigen::MatrixXd B;
    public:
        Split(const Eigen::MatrixXd& XH, int param_size, int hidden_size) {
            A = XH.block(0, 0, param_size, hidden_size);
            B = XH.block(param_size, 0, param_size + hidden_size, hidden_size);
        }
        Eigen::MatrixXd transposedX() {
            return A.transpose();
        }
        Eigen::MatrixXd transposedH() {
            return B.transpose();
        }

    };
};

/************************************************************************************************************
* Note that the design of the Recurrent Network Cells is based on using a matrix with dimension NxP
* where N is the number of words and P is the embedding size (the dimension).
* Each word in N does does not belong to one sequence, rather is a set of sequences.
* Here is a better illustration for an input matrix of 5xP:
* 
* Input Data (5xP):
* step t      step t+1    step t+2        step t+P-1
* word1_embed word2_embed word3_embed ... wordP_embed (sequence 1)
* word1_embed word2_embed word3_embed ... wordP_embed (sequence 2)
* ...
* word1_embed word2_embed word3_embed ... wordP_embed (sequence 5)
*
* This will allow us to process sequences in batches instead of just looping over one step at a time
* for a single sequence. Now, if that's the case, we pad each sequence to achieve uniform sequence lengths.
*************************************************************************************************************/
class RNNCell : public CellBase {
private:
   // Parameters (weights and biases)
    Eigen::MatrixXd W;  // Weight for the Input   (p x h)
    Eigen::MatrixXd U;  // Weight for the Hidden State  (h x h)  (rows x columns)
    // Eigen::MatrixXd V;  // Weight for the predicted output  (h x o)
    Eigen::RowVectorXd bh; // Hidden bias
    // Eigen::RowVectorXd bo; // Bias for the predicted output
    Eigen::MatrixXd H;  // Hidden state  (n x h) where n = number of words, h = hidden size

    Eigen::MatrixXd dX; // Gradient with respect to Input
    Eigen::MatrixXd dH; // Gradient with respect to Hidden state

    // Eigen::MatrixXd input_data;

    // Stored data for backward pass.
    Eigen::MatrixXd X; // Input dimension (n x p) where n = number of words, p = embedding size (features)

    // Y.hat, target
    // Eigen::MatrixXd Yhat, Y; // (n x o)

    int input_size = 0;
    int param_size = 0;
    int hidden_size;
    // int output_size;

    double learning_rate; // Learning rate

public:
    // RNNCell(int input_size, int param_size, int hidden_size, int output_size, double learning_rate);
    RNNCell(int hidden_size, double learning_rate);
    const Eigen::MatrixXd& getHiddenState();
    void setInitialWeights(int N, int P);
    const Eigen::MatrixXd& forward(const Eigen::MatrixXd& input_data);
    const Eigen::MatrixXd& backward(const Eigen::MatrixXd& input_data, const Eigen::MatrixXd& dnext_h);

};
 
class LSTMCell : public CellBase {
private:
    // Parameters (weights and biases)
    Eigen::MatrixXd Wf;      // Weight matrix for input gate from input x         (p + h) x h
    Eigen::MatrixXd Wi;      // Weight matrix for input gate from input x         (p + h) x h
    Eigen::MatrixXd Wo;      // Weight matrix for output gate from input x        (p + h) x h
    Eigen::MatrixXd Wg;      // Weight matrix for candidate state from input x    (p + h) x h
    // Eigen::MatrixXd V;       // Weight for Output (h x o)

    Eigen::RowVectorXd bf;   // Bias vector for input gate        (1xh)
    Eigen::RowVectorXd bi;   // Bias vector for input gate        (1xh)
    Eigen::RowVectorXd bo;   // Bias vector for output gate       (1xh)
    Eigen::RowVectorXd bg;   // Bias vector for candidate state   (1xh)
    // Eigen::RowVectorXd by;   // Bias for the predicted output

    Eigen::MatrixXd Ft;      // Forget Gate       (nxh)
    Eigen::MatrixXd It;      // Input Gate        (nxh)
    Eigen::MatrixXd Ot;      // Output Gate       (nxh)
    Eigen::MatrixXd Gt;      // Candidate State   (nxh)

    Eigen::MatrixXd H;       // Hidden state (n x h)
    Eigen::MatrixXd C;       // Cell state   (n x h)

    // Eigen::MatrixXd input_data;

    Eigen::MatrixXd X;       // (n x p)
    // Eigen::MatrixXd Yhat, Y; // (n x o)
 
    Eigen::MatrixXd XH;      // Concatenate X and H
    Eigen::MatrixXd dX;      // Gradient with respect to Input
    Eigen::MatrixXd dH;      // Gradient with respect to Hidden state
    Eigen::MatrixXd dC;      // Gradient with respect to Cell state

    int input_size;
    int param_size;
    int hidden_size;
    // int output_size;

    double learning_rate;    // Learning rate for parameter updates

public:
    LSTMCell(int param_size, double learning_rate);
    const Eigen::MatrixXd& getHiddenState();
    const Eigen::MatrixXd& getCellState();
    void setInitialWeights(int N, int P);
    const Eigen::MatrixXd& forward(const Eigen::MatrixXd& input_data);
    const Eigen::MatrixXd& backward(const Eigen::MatrixXd& input_data, const Eigen::MatrixXd& dnext_h);

};

class GRUCell : public CellBase {
private:
    // Weights for the input-to-hidden connections
    Eigen::MatrixXd Wz;      // Weight matrix for the update gate               (p + h) x h
    Eigen::MatrixXd Wr;      // Weight matrix for the reset gate                (p + h) x h
    Eigen::MatrixXd Wg;      // Weight matrix for the candidate hidden state    (p + h) x h
    // Eigen::MatrixXd V;       // Weight for Output (h x o)

    // Biases for the hidden units
    Eigen::RowVectorXd bz;   // Bias vector for the update gate              (1xh)
    Eigen::RowVectorXd br;   // Bias vector for the reset gate               (1xh)
    Eigen::RowVectorXd bg;   // Bias vector for the candidate hidden state   (1xh)
    // Eigen::RowVectorXd bo;   // Bias for the predicted output

    Eigen::MatrixXd Zt;      // Forget Gate       (nxh)
    Eigen::MatrixXd Rt;      // Input Gate        (nxh)
    Eigen::MatrixXd Gt;      // Candidate State   (nxh)

    // Eigen::MatrixXd input_data;

    Eigen::MatrixXd X;       // (n x p)
    Eigen::MatrixXd H;         // Hidden state (n x h)

    // Eigen::MatrixXd Yhat, Y; // (n x o)

    Eigen::MatrixXd XH;      // Concatenate X and H

    Eigen::MatrixXd dX; // Gradient with respect to Input
    Eigen::MatrixXd dH;      // Gradient with respect to Hidden state

    int input_size;
    int param_size;
    int hidden_size;
    // int output_size;

    double learning_rate;    // Learning rate for parameter updates

public:
    GRUCell(int hidden_size, double learning_rate);
    const Eigen::MatrixXd& getHiddenState();
    void setInitialWeights(int N, int P);
    const Eigen::MatrixXd& forward(const Eigen::MatrixXd& input_data);
    const Eigen::MatrixXd& backward(const Eigen::MatrixXd& input_data, const Eigen::MatrixXd& dnext_h);

};

class RNN : public BaseOperator {
private:
    std::vector<RNNCell> fcells;
    std::vector<RNNCell> bcells;
    std::vector<RNNCell> cells;
    Eigen::Tensor<double, 3> input_data;
    int num_directions = 2;
    int sequence_length;
    int initialized = false;
    int hidden_size;
    int output_size;
    int num_layers;
    bool bidirectional;
    RNNType rnntype;
    ActivationType otype;
    ReductionType rtype;

    std::vector<Eigen::MatrixXd> foutput, boutput, outputs;
    std::vector<Eigen::MatrixXd> V;      // Weight for the predicted output  (h x o)
    std::vector<Eigen::RowVectorXd> bo;  // Bias for the predicted output
    std::vector<Eigen::MatrixXd> Yhat;
    std::vector<Eigen::MatrixXd> gradients;
    std::vector<Eigen::MatrixXd> dV;      // Weight for the predicted output  (h x o)
    std::vector<Eigen::RowVectorXd> dbo;  // Bias for the predicted output

public:
    RNN(int hidden_size, int output_size, double learning_rate, int num_layers, 
        bool bidirectional, RNNType rnntype) 
        : hidden_size(hidden_size), output_size(output_size), num_layers(num_layers), bidirectional(bidirectional), rnntype(rnntype) {
      //  : RecurrentNetwork(num_layers, bidirectional, rnntype) {

        for (int i = 0; i < num_layers; ++i) {
            RNNCell cell(hidden_size, learning_rate);
            fcells.push_back(cell);
        }
        for (int i = 0; i < num_layers; ++i) {
            RNNCell cell(hidden_size, learning_rate);
            bcells.push_back(cell);
        }
    }
    const std::vector<Eigen::MatrixXd>&  forward(const Eigen::Tensor<double, 3>& input_data);
    const std::vector<Eigen::MatrixXd>&  backward(const std::vector<Eigen::MatrixXd>& gradients);
    void updateParameters(std::string& optimizertype, double& learningRate, int& iter);

    void forwardPass() {}
    void backwardPass() {}

};

class LSTM : public BaseOperator {
private:
    std::vector<LSTMCell> fcells;
    std::vector<LSTMCell> bcells;
    std::vector<LSTMCell> cells;
    Eigen::Tensor<double, 3> input_data;
    int num_directions = 2;
    int sequence_length;
    int initialized = false;
    int hidden_size;
    int output_size;
    int num_layers;
    bool bidirectional;
    RNNType rnntype;
    ActivationType otype;
    ReductionType rtype;

    std::vector<Eigen::MatrixXd> foutput, boutput, outputs;
    std::vector<Eigen::MatrixXd> V;      // Weight for the predicted output  (h x o)
    std::vector<Eigen::RowVectorXd> bo;  // Bias for the predicted output
    std::vector<Eigen::MatrixXd> Yhat;
    std::vector<Eigen::MatrixXd> gradients;
    std::vector<Eigen::MatrixXd> dV;      // Weight for the predicted output  (h x o)
    std::vector<Eigen::RowVectorXd> dbo;  // Bias for the predicted output
public:
    LSTM(int hidden_size, int output_size, double learning_rate, int num_layers, 
        bool bidirectional, RNNType rnntype) 
        : hidden_size(hidden_size), output_size(output_size), num_layers(num_layers), bidirectional(bidirectional), rnntype(rnntype) {
      //  : RecurrentNetwork(num_layers, bidirectional, rnntype) {

        for (int i = 0; i < num_layers; ++i) {
            LSTMCell cell(hidden_size, learning_rate);
            fcells.push_back(cell);
        }
        for (int i = 0; i < num_layers; ++i) {
            LSTMCell cell(hidden_size, learning_rate);
            bcells.push_back(cell);
        }
    }
    const std::vector<Eigen::MatrixXd>&  forward(const Eigen::Tensor<double, 3>& input_data);
    const std::vector<Eigen::MatrixXd>&  backward(const std::vector<Eigen::MatrixXd>& gradients);
    void updateParameters(std::string& optimizertype, double& learningRate, int& iter);

    void forwardPass() {}
    void backwardPass() {}
};

class GRU : public BaseOperator {
private:
    std::vector<GRUCell> fcells;
    std::vector<GRUCell> bcells;
    std::vector<GRUCell> cells;
    Eigen::Tensor<double, 3> input_data;
    int num_directions = 2;
    int sequence_length;
    int initialized = false;
    int hidden_size;
    int output_size;
    int num_layers;
    bool bidirectional;
    RNNType rnntype;
    ActivationType otype;
    ReductionType rtype;

    std::vector<Eigen::MatrixXd> foutput, boutput, outputs;
    std::vector<Eigen::MatrixXd> V;      // Weight for the predicted output  (h x o)
    std::vector<Eigen::RowVectorXd> bo;  // Bias for the predicted output
    std::vector<Eigen::MatrixXd> Yhat;
    std::vector<Eigen::MatrixXd> gradients;
    std::vector<Eigen::MatrixXd> dV;      // Weight for the predicted output  (h x o)
    std::vector<Eigen::RowVectorXd> dbo;  // Bias for the predicted output
public:
    GRU(int hidden_size, int output_size, double learning_rate, int num_layers, 
        bool bidirectional, RNNType rnntype) 
        : hidden_size(hidden_size), output_size(output_size), num_layers(num_layers), bidirectional(bidirectional), rnntype(rnntype) {
      //  : RecurrentNetwork(num_layers, bidirectional, rnntype) {

        for (int i = 0; i < num_layers; ++i) {
            GRUCell cell(hidden_size, learning_rate);
            fcells.push_back(cell);
        }
        for (int i = 0; i < num_layers; ++i) {
            GRUCell cell(hidden_size, learning_rate);
            bcells.push_back(cell);
        }
    }
    const std::vector<Eigen::MatrixXd>&  forward(const Eigen::Tensor<double, 3>& input_data);
    const std::vector<Eigen::MatrixXd>&  backward(const std::vector<Eigen::MatrixXd>& gradients);
    void updateParameters(std::string& optimizertype, double& learningRate, int& iter);

    void forwardPass() {}
    void backwardPass() {}
};


template <typename CellType>
std::tuple<Eigen::MatrixXd, Eigen::RowVectorXd> setInitialWeights(int step, Eigen::MatrixXd out, CellType& rnn);

template <typename CellType>
const std::vector<Eigen::MatrixXd>& processOutputs(CellType& rnn);

template <typename CellType>
void processGradients(const std::vector<Eigen::MatrixXd>& gradients, CellType& rnn);

template <typename CellType>
const std::vector<Eigen::MatrixXd>& forwarding(const Eigen::Tensor<double, 3>& input_data, CellType* rnn);

template <typename CellType>
const std::vector<Eigen::MatrixXd>& backprop(const std::vector<Eigen::MatrixXd>& gradients, CellType* rnn);


#endif