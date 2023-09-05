

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

template <class T>
class CellBase {
public:

    // Here we are splitting the concatenated XH where dimension is Nx(P+H)
    // such that we split XH into two matrices and get the original dimensions for X and H
    // so that X will have NxP and H will have NxH

    static std::tuple<aimatrix<T>, aimatrix<T>> split(const aimatrix<T>& XH, int param_size, int hidden_size) {
        aimatrix<T> A = XH.block(0, 0, param_size, hidden_size);
        aimatrix<T> B = XH.block(param_size, 0, param_size + hidden_size, hidden_size);
        return std::make_tuple(A.transpose(), B.transpose());
    }

    class Split {
    private:
        aimatrix<T> A;
        aimatrix<T> B;
    public:
        Split(const aimatrix<T>& XH, int param_size, int hidden_size) {
            A = XH.block(0, 0, param_size, hidden_size);
            B = XH.block(param_size, 0, param_size + hidden_size, hidden_size);
        }
        aimatrix<T> transposedX() {
            return (aimatrix<T>) (A.transpose());
        }
        aimatrix<T> transposedH() {
            return (aimatrix<T>) (B.transpose());
        }

    };
    /*
    class Split {  
    private:
        aimatrix<T> X;
        aimatrix<T> H;
    public:
        Split(const aimatrix<T>& XH, int param_size, int hidden_size) {
            int input_size = XH.dimension(0);

            // Create TensorMap for tensor X and H
            this->X = XH.slice(Eigen::array<int, 2>{0, 0}, Eigen::array<int, 2>{input_size, param_size});
            this->H = XH.slice(Eigen::array<int, 2>{0, param_size}, Eigen::array<int, 2>{input_size, hidden_size});
        }

        aimatrix<T> transposedX() {
            return  (this->X).shuffle(Eigen::array<int, 2>{1, 0});
        }
        aimatrix<T> transposedH() {
            return (this->H).shuffle(Eigen::array<int, 2>{1, 0});
        }

    };
    */
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
template <class T>
class RNNCell : public CellBase<T> {
private:
   // Parameters (weights and biases)
    aimatrix<T> W;  // Weight for the Input   (p x h)
    aimatrix<T> U;  // Weight for the Hidden State  (h x h)  (rows x columns)
    // aimatrix<T> V;  // Weight for the predicted output  (h x o)
    airowvector<T> bh; // Hidden bias
    // airowvector<T> bo; // Bias for the predicted output
    aimatrix<T> H;  // Hidden state  (n x h) where n = number of words, h = hidden size

    aimatrix<T> dX; // Gradient with respect to Input
    aimatrix<T> dH; // Gradient with respect to Hidden state

    // aimatrix<T> input_data;

    // Stored data for backward pass.
    aimatrix<T> X; // Input dimension (n x p) where n = number of words, p = embedding size (features)

    // Y.hat, target
    // aimatrix<T> Yhat, Y; // (n x o)

    int input_size = 0;
    int param_size = 0;
    int hidden_size;
    // int output_size;

    T learning_rate; // Learning rate

public:
    RNNCell(int hidden_size, T learning_rate);
    const aimatrix<T>& getHiddenState();
    void setInitialWeights(int N, int P);
    const aimatrix<T>& forward(const aimatrix<T>& input_data);
    const aimatrix<T>& backward(const aimatrix<T>& input_data, const aimatrix<T>& dnext_h);

};

template <class T>
class LSTMCell : public CellBase<T> {
private:
    // Parameters (weights and biases)
    aimatrix<T> Wf;      // Weight matrix for input gate from input x         (p + h) x h
    aimatrix<T> Wi;      // Weight matrix for input gate from input x         (p + h) x h
    aimatrix<T> Wo;      // Weight matrix for output gate from input x        (p + h) x h
    aimatrix<T> Wg;      // Weight matrix for candidate state from input x    (p + h) x h
    // aimatrix<T> V;       // Weight for Output (h x o)

    airowvector<T> bf;   // Bias vector for input gate        (1xh)
    airowvector<T> bi;   // Bias vector for input gate        (1xh)
    airowvector<T> bo;   // Bias vector for output gate       (1xh)
    airowvector<T> bg;   // Bias vector for candidate state   (1xh)
    // airowvector<T> by;   // Bias for the predicted output

    aimatrix<T> Ft;      // Forget Gate       (nxh)
    aimatrix<T> It;      // Input Gate        (nxh)
    aimatrix<T> Ot;      // Output Gate       (nxh)
    aimatrix<T> Gt;      // Candidate State   (nxh)

    aimatrix<T> H;       // Hidden state (n x h)
    aimatrix<T> C;       // Cell state   (n x h)

    // aimatrix<T> input_data;

    aimatrix<T> X;       // (n x p)
    // aimatrix<T> Yhat, Y; // (n x o)
 
    aimatrix<T> XH;      // Concatenate X and H
    aimatrix<T> dX;      // Gradient with respect to Input
    aimatrix<T> dH;      // Gradient with respect to Hidden state
    aimatrix<T> dC;      // Gradient with respect to Cell state

    int input_size;
    int param_size;
    int hidden_size;
    // int output_size;

    T learning_rate;    // Learning rate for parameter updates

public:
    LSTMCell(int param_size, T learning_rate);
    const aimatrix<T>& getHiddenState();
    const aimatrix<T>& getCellState();
    void setInitialWeights(int N, int P);
    const aimatrix<T>& forward(const aimatrix<T>& input_data);
    const aimatrix<T>& backward(const aimatrix<T>& input_data, const aimatrix<T>& dnext_h);

};

template <class T>
class GRUCell : public CellBase<T> {
private:
    // Weights for the input-to-hidden connections
    aimatrix<T> Wz;      // Weight matrix for the update gate               (p + h) x h
    aimatrix<T> Wr;      // Weight matrix for the reset gate                (p + h) x h
    aimatrix<T> Wg;      // Weight matrix for the candidate hidden state    (p + h) x h
    // aimatrix<T> V;       // Weight for Output (h x o)

    // Biases for the hidden units
    airowvector<T> bz;   // Bias vector for the update gate              (1xh)
    airowvector<T> br;   // Bias vector for the reset gate               (1xh)
    airowvector<T> bg;   // Bias vector for the candidate hidden state   (1xh)
    // airowvector<T> bo;   // Bias for the predicted output

    aimatrix<T> Zt;      // Forget Gate       (nxh)
    aimatrix<T> Rt;      // Input Gate        (nxh)
    aimatrix<T> Gt;      // Candidate State   (nxh)

    // aimatrix<T> input_data;

    aimatrix<T> X;       // (n x p)
    aimatrix<T> H;         // Hidden state (n x h)

    // aimatrix<T> Yhat, Y; // (n x o)

    aimatrix<T> XH;      // Concatenate X and H

    aimatrix<T> dX; // Gradient with respect to Input
    aimatrix<T> dH;      // Gradient with respect to Hidden state

    int input_size;
    int param_size;
    int hidden_size;
    // int output_size;

    T learning_rate;    // Learning rate for parameter updates

public:
    GRUCell(int hidden_size, T learning_rate);
    const aimatrix<T>& getHiddenState();
    void setInitialWeights(int N, int P);
    const aimatrix<T>& forward(const aimatrix<T>& input_data);
    const aimatrix<T>& backward(const aimatrix<T>& input_data, const aimatrix<T>& dnext_h);

};

template <class T>
class RNN : public BaseOperator {
private:
    std::vector<RNNCell<T>> fcells;
    std::vector<RNNCell<T>> bcells;
    std::vector<RNNCell<T>> cells;
    
    aitensor<T> input_data;

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

    std::vector<aimatrix<T>> foutput, boutput, outputs;
    std::vector<aimatrix<T>> V;      // Weight for the predicted output  (h x o)
    std::vector<airowvector<T>> bo;  // Bias for the predicted output

    std::vector<aimatrix<T>> dV;      // Weight for the predicted output  (h x o)
    std::vector<airowvector<T>> dbo;  // Bias for the predicted output

    aitensor<T> gradients;
    aitensor<T> Yhat;

public:
    RNN(int hidden_size, int output_size, T learning_rate, int num_layers, 
        bool bidirectional, RNNType rnntype) 
        : hidden_size(hidden_size), output_size(output_size), num_layers(num_layers), bidirectional(bidirectional), rnntype(rnntype) {
      //  : RecurrentNetwork(num_layers, bidirectional, rnntype) {

        for (int i = 0; i < num_layers; ++i) {
            RNNCell<T> cell(hidden_size, learning_rate);
            fcells.push_back(cell);
        }
        for (int i = 0; i < num_layers; ++i) {
            RNNCell<T> cell(hidden_size, learning_rate);
            bcells.push_back(cell);
        }
    }
    const aitensor<T>&  forward(const aitensor<T>& input_data);
    const aitensor<T>&  backward(const aitensor<T>& gradients);
    void updateParameters(std::string& optimizertype, T& learningRate, int& iter);

    void forwardPass() {}
    void backwardPass() {}

};

template <class T>
class LSTM : public BaseOperator {
private:
    std::vector<LSTMCell<T>> fcells;
    std::vector<LSTMCell<T>> bcells;
    std::vector<LSTMCell<T>> cells;
    aitensor<T> input_data;
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

    std::vector<aimatrix<T>> foutput, boutput, outputs;
    std::vector<aimatrix<T>> V;      // Weight for the predicted output  (h x o)
    std::vector<airowvector<T>> bo;  // Bias for the predicted output

    std::vector<aimatrix<T>> dV;      // Weight for the predicted output  (h x o)
    std::vector<airowvector<T>> dbo;  // Bias for the predicted output

    aitensor<T> gradients;
    aitensor<T> Yhat;

public:
    LSTM(int hidden_size, int output_size, T learning_rate, int num_layers, 
        bool bidirectional, RNNType rnntype) 
        : hidden_size(hidden_size), output_size(output_size), num_layers(num_layers), bidirectional(bidirectional), rnntype(rnntype) {
      //  : RecurrentNetwork(num_layers, bidirectional, rnntype) {

        for (int i = 0; i < num_layers; ++i) {
            LSTMCell<T> cell(hidden_size, learning_rate);
            fcells.push_back(cell);
        }
        for (int i = 0; i < num_layers; ++i) {
            LSTMCell<T> cell(hidden_size, learning_rate);
            bcells.push_back(cell);
        }
    }
    const aitensor<T>&  forward(const aitensor<T>& input_data);
    const aitensor<T>&  backward(const aitensor<T>& gradients);
    void updateParameters(std::string& optimizertype, T& learningRate, int& iter);

    void forwardPass() {}
    void backwardPass() {}
};

template <class T>
class GRU : public BaseOperator {
private:
    std::vector<GRUCell<T>> fcells;
    std::vector<GRUCell<T>> bcells;
    std::vector<GRUCell<T>> cells;
    aitensor<T> input_data;
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

    std::vector<aimatrix<T>> foutput, boutput, outputs;
    std::vector<aimatrix<T>> V;      // Weight for the predicted output  (h x o)
    std::vector<airowvector<T>> bo;  // Bias for the predicted output

    std::vector<aimatrix<T>> dV;      // Weight for the predicted output  (h x o)
    std::vector<airowvector<T>> dbo;  // Bias for the predicted output

    aitensor<T> gradients;
    aitensor<T> Yhat;

public:
    GRU(int hidden_size, int output_size, T learning_rate, int num_layers, 
        bool bidirectional, RNNType rnntype) 
        : hidden_size(hidden_size), output_size(output_size), num_layers(num_layers), bidirectional(bidirectional), rnntype(rnntype) {
      //  : RecurrentNetwork(num_layers, bidirectional, rnntype) {

        for (int i = 0; i < num_layers; ++i) {
            GRUCell<T> cell(hidden_size, learning_rate);
            fcells.push_back(cell);
        }
        for (int i = 0; i < num_layers; ++i) {
            GRUCell<T> cell(hidden_size, learning_rate);
            bcells.push_back(cell);
        }
    }
    const aitensor<T>& forward(const aitensor<T>& input_data);
    const aitensor<T>&  backward(const aitensor<T>& gradients);
    void updateParameters(std::string& optimizertype, T& learningRate, int& iter);

    void forwardPass() {}
    void backwardPass() {}
};


template <typename CellType, class T>
std::tuple<aimatrix<T>, airowvector<T>> setInitialWeights(int step, aimatrix<T> out, CellType& rnn);

template <typename CellType, class T>
const std::vector<aimatrix<T>>& processOutputs(CellType& rnn);

template <typename CellType, class T>
void processGradients(const aitensor<T>& gradients, CellType& rnn);

template <typename CellType, class T>
const aitensor<T>& forwarding(const aitensor<T>& input_data, CellType* rnn);

template <typename CellType, class T>
const aitensor<T>&backprop(const aitensor<T>& gradients, CellType* rnn);


/**********  Recurrent Network initialize templates *****************/

template class CellBase<float>;  // Instantiate with float
template class CellBase<double>;  // Instantiate with double

template class RNNCell<float>;  // Instantiate with float
template class RNNCell<double>;  // Instantiate with double

template class LSTMCell<float>;  // Instantiate with float
template class LSTMCell<double>;  // Instantiate with double

template class GRUCell<float>;  // Instantiate with float
template class GRUCell<double>;  // Instantiate with double

template class RNN<float>;  // Instantiate with float
template class RNN<double>;  // Instantiate with double

template class LSTM<float>;  // Instantiate with float
template class LSTM<double>;  // Instantiate with double

template class GRU<float>;  // Instantiate with float
template class GRU<double>;  // Instantiate with double

#endif