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

#include <genai.h>


/***************************************************************************************************************************
*********** IMPLEMENTING RNNCell
****************************************************************************************************************************/

class CellBase {
    /*  Empty class, but used to identify the type of cell later during forward and backward propagation.*/
};

class RNNCell : public CellBase {

private:
    Eigen::MatrixXd W;  // Weight for the Input   (p x h)
    Eigen::MatrixXd U;  // Weight for the Hidden State  (h x h)  (rows x columns)
    Eigen::MatrixXd V;  // Weight for the predicted output  (h x o)
    Eigen::RowVectorXd bh; // Hidden bias
    Eigen::RowVectorXd bo; // Bias for the predicted output
    Eigen::MatrixXd H;  // Hidden state  (n x h) where n = number of words, h = hidden size

    Eigen::MatrixXd dX; // Gradient with respect to Input
    Eigen::MatrixXd dH; // Gradient with respect to Hidden state

    double learning_rate; // Learning rate

    // Stored data for backward pass.
    Eigen::MatrixXd X; // Input dimension (n x p) where n = number of words, p = embedding size (features)

    // Y.hat, target
    Eigen::MatrixXd Yhat, Y; // (n x o)

public:
    RNNCell(int input_size, int param_size, int hidden_size, int output_size, double learning_rate) : 
            learning_rate(learning_rate) {
        // Initialize parameters, gates, states, etc.
        W.resize(param_size, hidden_size);
        U.resize(hidden_size, hidden_size);
        V.resize(hidden_size, output_size);

        BaseOperator::heInitialization(W);
        BaseOperator::heInitialization(U);
        BaseOperator::heInitialization(V);

        bh = Eigen::RowVectorXd::Zero(hidden_size);
        bo = Eigen::RowVectorXd::Zero(output_size);

        H    = Eigen::MatrixXd::Zero(input_size, hidden_size);
        Yhat = Eigen::MatrixXd::Zero(input_size, output_size);

        learning_rate = 0.01;
    }

    const Eigen::MatrixXd& forward(const Eigen::MatrixXd& input_data) {

        // Store input for backward pass.
        // this can be the prev_hidden_state
        X = input_data;

        // Compute hidden state.
        //     (nxh) =  (nxp) * (pxh) + (nxh) * (hxh) + h
        H = BaseOperator::tanh(X * W + H * U + bh);

        // Compute Yhat.
        //     (nxo) =  (nxh) * (hxo) + h
        Yhat = BaseOperator::softmax(H * V + bo);

        // We return the prediction for layer forward pass.
        // For the Hidden State output, we cache it instead for time step forward pass.
        return Yhat; 
    }

    const Eigen::MatrixXd& backward(const Eigen::MatrixXd& dnext_h) {

        // Compute gradient with respect to tanh
        Eigen::MatrixXd dtanh = (1.0 - H.array().square()).matrix() * dnext_h;

        // Compute gradient with respect to hidden-to-hidden weights Whh.
        Eigen::MatrixXd dU = H.transpose() * dtanh;

        // Compute gradient with respect to input-to-hidden weights Wxh.
        Eigen::MatrixXd dW = X.transpose() * dtanh;

        // Compute gradient with respect to hidden-state.
        dH = dtanh * U.transpose();

        // Compute gradient with respect to input (dInput).
        dX = dtanh * W.transpose();

        // Compute gradient with respect to hidden bias bh.
        Eigen::RowVectorXd dbh = dtanh.colwise().sum(); 

        // Compute gradient with respect to output bias bo.
        Eigen::RowVectorXd dbo = Y.colwise().sum();

        // Update weights and biases.
        W -= learning_rate * dW;
        U -= learning_rate * dU;
        bh -= learning_rate * dbh;
        bo -= learning_rate * dbo;

        // Return gradient with respect to input.
        // We return the gradient wrt X for each layer, not wrt H for each time step.
        // for the gradient wrt H, we just cache it for time step propagation, but layer propagation.
        return dX;
    }

    const Eigen::MatrixXd& getOutput() const {
        return Yhat;
    }


};

/***************************************************************************************************************************
*********** IMPLEMENTING LSTMCell
****************************************************************************************************************************/
class LSTMCell : public CellBase {
private:
    // Parameters (weights and biases)
    Eigen::MatrixXd Wf;   // Weight matrix for input gate from input x         (p + h) x h
    Eigen::MatrixXd Wi;   // Weight matrix for input gate from input x         (p + h) x h
    Eigen::MatrixXd Wo;   // Weight matrix for output gate from input x        (p + h) x h
    Eigen::MatrixXd Wc;   // Weight matrix for candidate state from input x    (p + h) x h
    Eigen::MatrixXd V;    // Weight for Output (h x o)

    Eigen::RowVectorXd bf;   // Bias vector for input gate        (1xh)
    Eigen::RowVectorXd bi;   // Bias vector for input gate        (1xh)
    Eigen::RowVectorXd bo;   // Bias vector for output gate       (1xh)
    Eigen::RowVectorXd bc;   // Bias vector for candidate state   (1xh)

    Eigen::MatrixXd Ft;   // Forget Gate       (nxh)
    Eigen::MatrixXd It;   // Input Gate        (nxh)
    Eigen::MatrixXd Ot;   // Output Gate        (nxh)
    Eigen::MatrixXd Gt;   // Candidate State    (nxh)

    Eigen::Matrix H;         // Hidden state (n x h)
    Eigen::Matrix C;         // Cell state (n x h)

    Eigen::MatrixXd X;       // (n x p)
    Eigen::MatrixXd Yhat, Y; // ( n x o)

    Eigen::MatrixXd XH;  

    double learning_rate; // Learning rate for parameter updates

    Eigen::MatrixXd dH; // Gradient with respect to Hidden state
    Eigen::MatrixXd dC; // Gradient with respect to Cell state

public:
    LSTMCell(int input_size, int param_size, int hidden_size, int output_size, double learning_rate);
    const Eigen::MatrixXd& forward(const Eigen::MatrixXd& input_data);
    const Eigen::MatrixXd& backward(const Eigen::MatrixXd& dnext_h);

};

LSTMCell::LSTMCell(int input_size, int param_size, int hidden_size, int output_size, double learning_rate)
    : learning_rate(learning_rate) {
    // Initialize parameters (weights and biases)
    // Note: Initialize these parameters according to your initialization strategy
    Wf.resize(param_size + hidden_size, hidden_size);
    Wi.resize(param_size + hidden_size, hidden_size);
    Wo.resize(param_size + hidden_size, hidden_size);
    Wg.resize(param_size + hidden_size, hidden_size);
    V.resize(hidden_size, output_size);

    BaseOperator::heInitialization(Wf);
    BaseOperator::heInitialization(Wi);
    BaseOperator::heInitialization(Wo);
    BaseOperator::heInitialization(Wg);

    bf = Eigen::RowVectorXd::Zero(hidden_size);
    bi = Eigen::RowVectorXd::Zero(hidden_size);
    bo = Eigen::RowVectorXd::Zero(hidden_size);
    bg = Eigen::RowVectorXd::Zero(hidden_size);

    H    = Eigen::MatrixXd::Zero(input_size, hidden_size);
    C    = Eigen::MatrixXd::Zero(input_size, hidden_size);
    Yhat = Eigen::MatrixXd::Zero(input_size, output_size);

    XH   = Eigen::MatrixXd::Zero(input_size, param_size + hidden_size); // concatenate X and H

}

const Eigen::MatrixXd& LSTMCell::forward(const Eigen::MatrixXd& input_data) {

    // Store input for backward pass.
    // this can be the prev_hidden_state
    X = input_data;

    // Concatenate input and hidden state.
    XH << X, H;

    // Calculate the forget gate values
    Ft = BaseOperator::sigmoid(XH * Wf + bf);

    // Calculate the input gate values
    It = BaseOperator::sigmoid(XH * Wi + bi);

    // Calculate the output gate values
    Ot = BaseOperator::sigmoid(XH * Wo + bo);

    // Calculate the candidate state values
    Gt = BaseOperator::tanh(XH * Wc + bc);

    // Calculate the new cell state by updating based on the gates
    C = Ft * C + It + Gt;

    // Calculate the new hidden state by applying the output gate and tanh activation
    H = BaseOperator::tanh(C) * Ot;

    // Return the updated hidden state and stored_cell_state
    return std::make_tuple(hidden_state, stored_cell_state);

    // Compute Yhat.
    //     (nxo) =  (nxh) * (hxo) + h
    Yhat = BaseOperator::softmax(H * V + bo);

    // We return the prediction for layer forward pass.
    // For the Cell State and Hidden State output, we cache them instead for time step forward pass.
    return Yhat; 

}

std::tuple <const Eigen::MatrixXd& X, const Eigen::MatrixXd& H> LSTMCell::Split(const Eigen::MatrixXd& XH) {
    return std::make_tuple(XH.block(0, 0, param_size, hidden_size), 
                           XH.block(param_size, 0, param_size + hidden_size, hidden_size));
}

const Eigen::MatrixXd& LSTMCell::backward(const Eigen::MatrixXd& dnext_h, const Eigen::MatrixXd& dnext_c) {
    // Backpropagation logic for LSTM


    // Compute gradients with respect to the gates
    Eigen::MatrixXd dFt = (dC.array() * C.array()) * Ft.array() * (1 - Ft;
    Eigen::MatrixXd dIt = (dC.array() * Gt.array()) * It.array() * (1 - It);
    Eigen::MatrixXd dGt = (dC.array() * It.array()) * (1 - Gt.array().square());
    Eigen::MatrixXd dOt = dnext_h.array() * BaseOperator::tanh(Ct) * Ot.array() * ( 1 - Ot);

    // Compute gradients with respect to hidden-to-hidden weights Wf, Wi, Wo, Wc
    Eigen::MatrixXd dWf = XH.tranpose() * dFt;
    Eigen::MatrixXd dWi = XH.tranpose() * dIt;
    Eigen::MatrixXd dWg = XH.tranpose() * dGt;
    Eigen::MatrixXd dWo = XH.tranpose() * dOt;


    // Compute gradients with respect to hidden biases bf, bi, bo, bc
    Eigen::VectorXd dbf = dFt.colwise().sum();
    Eigen::VectorXd dbi = dIt.colwise().sum();
    Eigen::VectorXd dbo = dOt.colwise().sum();
    Eigen::VectorXd dbg = dGt.colwise().sum();

    // Compute gradient with respect to the cell state C (dC)
    Eigen::VectorXd dC = dC.array() * Ft.array();

    // Compute gradient with respect to input (dInput)
    Eigen::VectorXd dX = dFt * Split(Wf).H.tranpose() 
                       + dIt * Split(Wi).H.tranpose()  
                       + dOt * Split(Wo).H.tranpose()
                       + dgt * Split(Wg).H.tranpose();

    // Compute gradient with respect to input (dInput)
    Eigen::VectorXd dH = dFt * Split(Wf).X.tranpose() 
                       + dIt * Split(Wi).X.tranpose()  
                       + dOt * Split(Wo).X.tranpose()
                       + dgt * Split(Wg).X.tranpose();

    // Update parameters and stored states
    Wf -= learning_rate * dWf;
    Wi -= learning_rate * dWi;
    Wo -= learning_rate * dWo;
    Wg -= learning_rate * dWg;
    bf -= learning_rate * dbf;
    bi -= learning_rate * dbi;
    bo -= learning_rate * dbo;
    bc -= learning_rate * dbc;

    H -= learning_rate * dH;
    C -= learning_rate * dC;

    // Return gradient with respect to input.
    // We return the gradient wrt X for each layer, not wrt H & C for each time step.
    // for the gradient wrt H & C, we just cache it for time step propagation, but layer propagation.
    return dX;
}

/***************************************************************************************************************************
 *********** IMPLEMENTING GRUCell
****************************************************************************************************************************/
class GRUCell : public CellBase {
private:
    // Weights for the input-to-hidden connections
    Eigen::MatrixXd Wxz;  // Weight matrix for the update gate
    Eigen::MatrixXd Wxr;  // Weight matrix for the reset gate
    Eigen::MatrixXd Wxh;  // Weight matrix for the candidate hidden state

    // Weights for the hidden-to-hidden connections
    Eigen::MatrixXd Whz;  // Weight matrix for the update gate
    Eigen::MatrixXd Whr;  // Weight matrix for the reset gate
    Eigen::MatrixXd Whh;  // Weight matrix for the candidate hidden state

    // Biases for the hidden units
    Eigen::VectorXd bz;   // Bias vector for the update gate
    Eigen::VectorXd br;   // Bias vector for the reset gate
    Eigen::VectorXd bh;   // Bias vector for the candidate hidden state

    Eigen::VectorXd stored_dprev_h; // Gradient with respect to Hidden state

    // Learning rate for weight updates during backpropagation
    double learning_rate;

    // Stored data for backward pass.
    Eigen::MatrixXd input_data;

    // Stored values from the forward pass for backpropagation
    Eigen::VectorXd stored_update_gate;  // Stored update gate values
    Eigen::VectorXd stored_reset_gate;   // Stored reset gate values
    Eigen::VectorXd stored_prev_h_bar;   // Stored previous hidden bar values
    Eigen::VectorXd hidden_state;         // Current hidden state of the cell

    // Helper activation functions
    Eigen::VectorXd sigmoid(const Eigen::VectorXd& x);  // Sigmoid activation function
    Eigen::VectorXd tanh(const Eigen::VectorXd& x);     // Hyperbolic tangent activation function

public:
    GRUCell(int input_size, int hidden_size, double learning_rate)
        : Wxz(hidden_size, input_size), Wxr(hidden_size, input_size), Wxh(hidden_size, input_size),
          Whz(hidden_size, hidden_size), Whr(hidden_size, hidden_size), Whh(hidden_size, hidden_size),
          bz(hidden_size), br(hidden_size), bh(hidden_size),
          learning_rate(learning_rate), stored_update_gate(hidden_size), stored_reset_gate(hidden_size),
          stored_prev_h_bar(hidden_size), hidden_state(hidden_size) {
        // Initialize weights and biases
        Wxz.setRandom(hidden_size, input_size);
        Wxr.setRandom(hidden_size, input_size);
        Wxh.setRandom(hidden_size, input_size);
        Whz.setRandom(hidden_size, input_size);
        Whr.setRandom(hidden_size, input_size);
        Whh.setRandom(hidden_size, input_size);
        bz.setZero();
        br.setZero();
        bh.setZero();
    }

    GRUCell(int input_size, int hidden_size, double learning_rate);
    Eigen::VectorXd forward(const Eigen::VectorXd& x);
    void backward(const Eigen::VectorXd& dnext_h, const Eigen::VectorXd& dnext_c);
};

GRUCell::GRUCell(int input_size, int hidden_size, double learning_rate)
    : Wxz(hidden_size, input_size), Wxr(hidden_size, input_size), Wxh(hidden_size, input_size),
        Whz(hidden_size, hidden_size), Whr(hidden_size, hidden_size), Whh(hidden_size, hidden_size),
        bz(hidden_size), br(hidden_size), bh(hidden_size),
        learning_rate(learning_rate), stored_update_gate(hidden_size), stored_reset_gate(hidden_size),
        stored_prev_h_bar(hidden_size), hidden_state(hidden_size) {
    // Initialize weights and biases
    Wxz.setRandom();
    Wxr.setRandom();
    Wxh.setRandom();
    Whz.setRandom();
    Whr.setRandom();
    Whh.setRandom();
    bz.setZero();
    br.setZero();
    bh.setZero();
}

Eigen::VectorXd GRUCell::sigmoid(const Eigen::VectorXd& x) {
    return 1.0 / (1.0 + (-x.array()).exp());
}

Eigen::VectorXd GRUCell::tanh(const Eigen::VectorXd& x) {
    return x.array().tanh();
}

Eigen::MatrixXd GRUCell::forward(const Eigen::MatrixXd& input_data) {
    // Calculate the update gate using input, hidden state, and biases
    stored_update_gate = sigmoid(Wxz * input_data + Whz * hidden_state + bz);

    // Calculate the reset gate using input, hidden state, and biases
    stored_reset_gate = sigmoid(Wxr * input_data + Whr * hidden_state + br);

    // Calculate the candidate hidden state using input, reset gate, hidden state, and biases
    stored_prev_h_bar = tanh(Wxh * input_data + Whh * (stored_reset_gate.cwiseProduct(hidden_state)) + bh);

    // Calculate the new hidden state using update gate, previous hidden state, and candidate hidden state
    hidden_state = (1.0 - stored_update_gate).cwiseProduct(hidden_state) + stored_update_gate.cwiseProduct(stored_prev_h_bar);

    // cache input
    this->input_data = input_data;

    // Return the updated hidden state
    return hidden_state;
}

Eigen::MatrixXd GRUCell::backward(const Eigen::MatrixXd& dnext_h) {
    // Backpropagation logic for GRU
    // Compute gradients with respect to hidden-to-hidden weights Whz, Whr, Whh
    Eigen::MatrixXd dWhz = dnext_h * stored_reset_gate.transpose();
    Eigen::MatrixXd dWhr = dnext_h * hidden_state_bar.transpose();
    Eigen::MatrixXd dWhh = dnext_h * (1.0 - stored_update_gate).transpose();

    // Compute gradients with respect to input-to-hidden weights Wxz, Wxr, Wxh
    Eigen::MatrixXd dWxz = dnext_h * stored_reset_gate * stored_x.transpose();
    Eigen::MatrixXd dWxr = dnext_h * hidden_state * stored_x.transpose();
    Eigen::MatrixXd dWxh = dnext_h * (1.0 - stored_update_gate) * stored_x.transpose();

    // Compute gradients with respect to hidden biases bz, br, bh
    Eigen::VectorXd dbz = dnext_h * stored_reset_gate;
    Eigen::VectorXd dbr = dnext_h * hidden_state_bar;
    Eigen::VectorXd dbh = dnext_h * (1.0 - stored_update_gate);

    // Compute gradients with respect to reset gate r, update gate z, and candidate state h_bar
    Eigen::VectorXd dr = (dnext_h.cwiseProduct(hidden_state - hidden_state_bar))
        .cwiseProduct(hidden_state_bar).cwiseProduct(stored_reset_gate.cwiseProduct(1.0 - stored_reset_gate));
    Eigen::VectorXd dz = (dnext_h.cwiseProduct(hidden_state_bar - hidden_state))
        .cwiseProduct(hidden_state_bar).cwiseProduct(stored_update_gate.cwiseProduct(1.0 - stored_update_gate));
    Eigen::VectorXd dh_bar = dnext_h.cwiseProduct(stored_update_gate);

    // Compute gradient with respect to input (dInput)
    Eigen::VectorXd dInput = Wxz.transpose() * (dnext_h * stored_update_gate * stored_update_gate.cwiseProduct(sigmoid(Wxz * stored_update_gate))) +
                             Wxr.transpose() * (dnext_h * hidden_state * stored_reset_gate.cwiseProduct(sigmoid(Wxr * stored_reset_gate))) +
                             Wxh.transpose() * (dnext_h * (1.0 - stored_update_gate) * stored_prev_h_bar.cwiseProduct(tanh(Wxh * stored_prev_h_bar)));


    // Update parameters
    Whz -= learning_rate * dWhz;
    Whr -= learning_rate * dWhr;
    Whh -= learning_rate * dWhh;
    Wxz -= learning_rate * dWxz;
    Wxr -= learning_rate * dWxr;
    Wxh -= learning_rate * dWxh;
    bz -= learning_rate * dbz;
    br -= learning_rate * dbr;
    bh -= learning_rate * dbh;

    // Update stored states and gates
    stored_reset_gate -= learning_rate * dr;
    stored_update_gate -= learning_rate * dz;
    stored_prev_h_bar -= learning_rate * dh_bar;

    // Return gradient with respect to input.
    return dInput;
}


/**********************************************************************************************
* Implement RNN
**********************************************************************************************/

enum class RNNType {
    MANY_TO_ONE,
    ONE_TO_MANY,
    MANY_TO_MANY
};

class RecurrentNetwork : public BaseOperator {
private:
    bool bidirectional;

    int input_size;
    int hidden_size;
    int output_size;
    int sequence_length;
    int num_layers;
    double learning_rate;
    bool bidirectional;
    RNNtype rnntype = RNNType::MANY_TO_ONE;

    Eigen::MatrixXd input_data;

    bool isBidirectional = false; // Default to non-bidirectional

    Eigen::MatrixXd reverseInput(const Eigen::MatrixXd& input_data) {
            // Create a separate copy of the input data for reversal
        Eigen::MatrixXd reversed_input_data = input_data;

        // Reverse each input step in the reversed input data
        for (int i = 0; i < reversed_input_data.rows(); ++i) {
            Eigen::VectorXd input_step = reversed_input_data.row(i);
            std::reverse(input_step.data(), input_step.data() + input_step.size());
        }
        return reversed_input_data;
    }

public:
    RecurrentNetwork(int input_size, int hidden_size, int num_layers, double learning_rate,
        bool bidirectional, RNNType rnntype) : bidirectional(bidirectional), rnntype(rnntype) {}


    template <typename CellType>
    Eigen::MatrixXd forwardManyToMany(const Eigen::MatrixXd& input_data, CellType& rnn);

    template <typename CellType>
    Eigen::MatrixXd forwardOneToMany(const  Eigen::VectorX& input_data, CellType&  rnn);

    template <typename CellType>
    Eigen::VectorXd forwardManyToOne(const Eigen::MatrixXd& input_data, CellType&  rnn);

    template <typename CellType>
    Eigen::MatrixXd backwardManyToOne(Eigen::MatrixXd& gradients, CellType&  rnn);

    template <typename CellType>
    Eigen::MatrixXd backwardManyToMany(Eigen::MatrixXd& gradients, CellType&  rnn);

    template <typename CellType>
    Eigen::VectorXd backwardOneToMany(Eigen::VectorXd& gradients, CellType&  rnn);

    void RecurrentNetwork::updateParameters(std::string& optimizertype, double& learningRate, int& iter);

    RNNtype getType() {
        return this->rnntype;
    }

};

class RNN : public RecurrentNetwork {
private:
    std::vector<RNNCell> fcells;
    std::vector<RNNCell> bcells;
    std::vector<RNNCell> cells;
public:
    RNN(int input_size, int hidden_size, int num_layers, double learning_rate,
    bool bidirectional, RNNType rnntype) : bidirectional(bidirectional), rnntype(rnntype) {
        for (int i = 0; i < num_layers; ++i) {
            fcells.push_back(std::make_unique<RNNCell>(hidden_size, hidden_size, learning_rate));
        }
        for (int i = 0; i < num_layers; ++i) {
            bcells.push_back(std::make_unique<RNNCell>(hidden_size, hidden_size, learning_rate));
        }
    }

    Eigen::VectorXd forward(const Eigen::MatrixXd& input_data);
    Eigen::MatrixXd backward(Eigen::MatrixXd& gradients);

};

class LSTM : public RecurrentNetwork {
private:
    std::vector<LSTMCell> fcells;
    std::vector<LSTMCell> bcells;
    std::vector<LSTMCell> cells;
public:
    RNN(int input_size, int hidden_size, int num_layers, double learning_rate,
    bool bidirectional, RNNType rnntype) : bidirectional(bidirectional), rnntype(rnntype) {
        for (int i = 0; i < num_layers; ++i) {
            fcells.push_back(std::make_unique<LSTMCell>(hidden_size, hidden_size, learning_rate));
        }
        for (int i = 0; i < num_layers; ++i) {
            bcells.push_back(std::make_unique<LSTMCell>(hidden_size, hidden_size, learning_rate));
        }
    }
};

class GRU : public RecurrentNetwork {
private:
    std::vector<GRUCell> fcells;
    std::vector<GRUCell> bcells;
    std::vector<GRUCell> cells;
public:
    RNN(int input_size, int hidden_size, int num_layers, double learning_rate,
    bool bidirectional, RNNType rnntype) : bidirectional(bidirectional), rnntype(rnntype) {
        for (int i = 0; i < num_layers; ++i) {
            fcells.push_back(std::make_unique<GRUCell>(hidden_size, hidden_size, learning_rate));
        }
        for (int i = 0; i < num_layers; ++i) {
            bcells.push_back(std::make_unique<GRUCell>(hidden_size, hidden_size, learning_rate));
        }
    }
};


Eigen::VectorXd RNN::forward(const Eigen::MatrixXd& input_data) {
    switch(this->rnntype) {
        MANY_TO_ONE:  return forwardManyToOne(input_data, this);  break;
        ONE_TO_MANY:  return forwardOneToMany(input_data, this);  break;
        MANY_TO_MANY: return forwardManyToMany(input_data, this);  break;
        default:
            return nullptr;
    }
}

Eigen::MatrixXd RNN::backward(Eigen::MatrixXd& gradients) {
    switch(this->rnntype) {
        MANY_TO_ONE: return backwardManyToOne(gradients, this); break;
        ONE_TO_MANY: return backwardOneToMany(gradients, this); break;
        MANY_TO_MANY: return backwardManyToMany(gradients, this); break;
        default:
            return nullptr;
    }
}

Eigen::VectorXd LSTM::forward(const Eigen::MatrixXd& input_data) {
    switch(this->rnntype) {
        MANY_TO_ONE:  return forwardManyToOne(input_data, this);  break;
        ONE_TO_MANY:  return forwardOneToMany(input_data, this);  break;
        MANY_TO_MANY: return forwardManyToMany(input_data, this);  break;
        default:
            return nullptr;
    }
}

Eigen::MatrixXd LSTM::backward(Eigen::MatrixXd& gradients) {
    switch(this->rnntype) {
        MANY_TO_ONE: return backwardManyToOne(gradients, this); break;
        ONE_TO_MANY: return backwardOneToMany(gradients, this); break;
        MANY_TO_MANY: return backwardManyToMany(gradients, this); break;
        default:
            return nullptr;
    }
}

Eigen::VectorXd GRU::forward(const Eigen::MatrixXd& input_data) {
    switch(this->rnntype) {
        MANY_TO_ONE:  return forwardManyToOne(input_data, this);  break;
        ONE_TO_MANY:  return forwardOneToMany(input_data, this);  break;
        MANY_TO_MANY: return forwardManyToMany(input_data, this);  break;
        default:
            return nullptr;
    }
}

Eigen::MatrixXd GRU::backward(Eigen::MatrixXd& gradients) {
    switch(this->rnntype) {
        MANY_TO_ONE: return backwardManyToOne(gradients, this); break;
        ONE_TO_MANY: return backwardOneToMany(gradients, this); break;
        MANY_TO_MANY: return backwardManyToMany(gradients, this); break;
        default:
            return nullptr;
    }
}


// Many-To-One Scenario
template <typename CellType>
Eigen::VectorXd RecurrentNetwork::forwardManyToOne(const Eigen::MatrixXd& input_data, CellType& rnn) {
    Eigen::VectorXd last_output(num_directions);

    for (int direction = 0; direction < num_directions; ++direction) {
        rnn->cells = (direction == 0) ? rnn->fcells : rnn->bcells;

        Eigen::VectorXd input_step;
        // Forward pass: Run from first to last time step
        for (int step = 0; step < sequence_length; ++step) {
            if (direction == 0) {
                input_step = input_data.row(step);
            } else {
                input_step = input_data.row(sequence_length - step - 1);
            }

            // Forward pass through each layer of the RNN
            for (int layer = 0; layer < num_layers; ++layer) {
                input_step = rnn->cells[layer].forward(input_step);
            }

            // Store the output of the last time step
            if (step == sequence_length - 1) {
                last_output[direction] = input_step;
            }
        }
    }
    return last_output;
}

// Many-To-Many Scenario
template <typename CellType>
Eigen::MatrixXd RecurrentNetwork::forwardManyToMany(const Eigen::MatrixXd& input_data, CellType& rnn) {
    Eigen::MatrixXd all_outputs(sequence_length, num_directions);

    for (int direction = 0; direction < num_directions; ++direction) {
        rnn->cells  = (direction == 0) ? rnn->fcells : rnn->bcells;

        Eigen::VectorXd input_step;
        Eigen::VectorXd output_step;

        // Forward pass: Run from first to last time step
        for (int step = 0; step < sequence_length; ++step) {
            if (direction == 0) {
                input_step = input_data.row(step);
            } else {
                input_step = input_data.row(sequence_length - step - 1);
            }

            // Forward pass through each layer of the RNN
            for (int layer = 0; layer < num_layers; ++layer) {
                output_step = rnn->cells[layer].forward(input_step);
                input_step = output_step;  // Use the output as input for the next layer
            }

            // Store the output of the current time step
            all_outputs(step, direction) = output_step;
        }
    }
    return all_outputs;
}

// One-To-Many Scenario
template <typename CellType>
Eigen::MatrixXd RecurrentNetwork::forwardOneToMany(const Eigen::VectorXd& input_data, CellType& rnn) {
    Eigen::MatrixXd all_outputs(sequence_length, num_directions);

    for (int direction = 0; direction < num_directions; ++direction) {
        rnn->cells  = (direction == 0) ? rnn->fcells : rnn->bcells;

        Eigen::VectorXd input_step = input_data;

        // Forward pass: Generate outputs for each time step
        for (int step = 0; step < sequence_length; ++step) {
            Eigen::VectorXd output_step;

            // Forward pass through each layer of the RNN
            for (int layer = 0; layer < num_layers; ++layer) {
                output_step = rnn->cells[layer].forward(input_step);
                input_step = output_step;  // Use the output as input for the next layer
            }

            // Store the output of the current time step
            all_outputs(step, direction) = output_step;
        }
    }
    return all_outputs;
}

// Many-To-One Scenario: Backward Pass
template <typename CellType>
Eigen::MatrixXd RecurrentNetwork::backwardManyToOne(Eigen::MatrixXd& gradients, CellType& rnn) {
    Eigen::VectorXd loss_gradient = gradients;

    for (int direction = 0; direction < num_directions; ++direction) {
        rnn->cells  = (direction == 0) ? rnn->fcells : rnn->bcells;

        for (int step = sequence_length - 1; step >= 0; --step) {
            Eigen::VectorXd input_step = direction == 0 ? input_data.row(step) : input_data.row(sequence_length - step - 1);

            // Backward pass through each layer of the RNN
            for (int layer = num_layers - 1; layer >= 0; --layer) {
                loss_gradient = rnn->cells[layer].backward(loss_gradient);
            }
        }
    }

    return loss_gradient;
}

// Many-To-Many Scenario: Backward Pass
template <typename CellType>
Eigen::MatrixXd RecurrentNetwork::backwardManyToMany(Eigen::MatrixXd& gradients, const Eigen::MatrixXd& input_data, CellType& rnn) {
    Eigen::MatrixXd loss_gradients(sequence_length, num_directions);

    for (int direction = 0; direction < num_directions; ++direction) {
        rnn->cells  = (direction == 0) ? rnn->fcells : rnn->bcells;

        for (int step = sequence_length - 1; step >= 0; --step) {
            Eigen::VectorXd input_step = direction == 0 ? input_data.row(step) : input_data.row(sequence_length - step - 1);

            // Backward pass through each layer of the RNN
            for (int layer = num_layers - 1; layer >= 0; --layer) {
                loss_gradients(step, direction) = loss_gradient;
                loss_gradient = rnn->cells[layer].backward(loss_gradient);
            }
        }
    }

    return loss_gradients;
}

// One-To-Many Scenario: Backward Pass
template <typename CellType>
Eigen::VectorXd RecurrentNetwork::backwardOneToMany(Eigen::VectorXd& gradient, CellType& rnn) {
    Eigen::VectorXd loss_gradient = gradient;

    for (int direction = 0; direction < num_directions; ++direction) {
        rnn->cells  = (direction == 0) ? rnn->fcells : rnn->bcells;

        Eigen::VectorXd input_step = input_data;

        // Backward pass: Compute gradients for each time step
        for (int step = sequence_length - 1; step >= 0; --step) {
            Eigen::VectorXd output_step;  // Output of the RNNCell's forward pass

            // Backward pass through each layer of the RNN
            for (int layer = num_layers - 1; layer >= 0; --layer) {
                output_step = rnn->cells[layer].forward(input_step);  // Forward pass for the current step
                loss_gradient = rnn->cells[layer].backward(loss_gradient);
                input_step = output_step;  // Use the output as input for the next layer
            }

            // Store the loss gradient with respect to the input of the current step
            if (direction == 0) {
                loss_gradients(step) = loss_gradient;
            } else {
                loss_gradients(sequence_length - step - 1) = loss_gradient;
            }
        }
    }

    return loss_gradients;
}

void RecurrentNetwork::updateParameters(std::string& optimizertype, double& learningRate, int& iter) {
}

