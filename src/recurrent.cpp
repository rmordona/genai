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

#include <Eigen/Dense>
#include <vector>
#include <string>
#include <iostream>


/***************************************************************************************************************************
*********** IMPLEMENTING RNNCell
****************************************************************************************************************************/

class RNNCell {

private:
    Eigen::MatrixXd Whh; // Hidden-to-hidden weights
    Eigen::MatrixXd Wxh; // Input-to-hidden weights
    Eigen::VectorXd bh;  // Hidden bias

    Eigen::VectorXd hidden_state; // Hidden state (h_t)
    Eigen::VectorXd output;       // Output for this step (h_t)

    double learning_rate; // Learning rate

    // Stored data for backward pass.
    Eigen::VectorXd input_data;

    // Activation function.
    Eigen::VectorXd tanh(const Eigen::VectorXd& x) {
        return x.array().tanh();
    }

public:
    RNNCell(int input_size, int hidden_size) {
        // Initialize parameters, gates, states, etc.
        Whh = Eigen::MatrixXd::Random(hidden_size, hidden_size);
        Wxh = Eigen::MatrixXd::Random(hidden_size, input_size);
        bh = Eigen::VectorXd::Zero(hidden_size);

        hidden_state = Eigen::VectorXd::Zero(hidden_size);
        output = Eigen::VectorXd::Zero(hidden_size);

        learning_rate = 0.01;
    }

    void forward(const Eigen::VectorXd& input_data) {
        // Compute hidden state.
        hidden_state = (Whh * hidden_state + Wxh * x + bh).unaryExpr(tanh);

        // Compute output for this step.
        output = hidden_state;

        // Store input for backward pass.
        this->input_data = input_data;
    }

    void backward(const Eigen::VectorXd& dnext_h) {
        // Compute gradient with respect to hidden-to-hidden weights Whh.
        Eigen::MatrixXd dWhh = dnext_h.cwiseProduct(1.0 - hidden_state.array().square()) * hidden_state.transpose();

        // Compute gradient with respect to input-to-hidden weights Wxh.
        Eigen::MatrixXd dWxh = dnext_h.cwiseProduct(1.0 - hidden_state.array().square()) * this->input_data.transpose();

        // Compute gradient with respect to hidden bias bh.
        Eigen::VectorXd dbh = dnext_h.cwiseProduct(1.0 - hidden_state.array().square());

        // Update weights and biases.
        Whh -= learning_rate * dWhh;
        Wxh -= learning_rate * dWxh;
        bh -= learning_rate * dbh;
    }

    const Eigen::VectorXd& getOutput() const {
        return output;
    }


};

/***************************************************************************************************************************
*********** IMPLEMENTING LSTMCell
****************************************************************************************************************************/
class LSTMCell {
private:
    // Parameters (weights and biases)
    Eigen::MatrixXd Wxf;   // Weight matrix for input gate from input x
    Eigen::MatrixXd Wxi;   // Weight matrix for input gate from input x
    Eigen::MatrixXd Wxo;   // Weight matrix for output gate from input x
    Eigen::MatrixXd Wxc;   // Weight matrix for candidate state from input x
    Eigen::MatrixXd Whf;   // Weight matrix for input gate from previous hidden state
    Eigen::MatrixXd Whi;   // Weight matrix for input gate from previous hidden state
    Eigen::MatrixXd Who;   // Weight matrix for output gate from previous hidden state
    Eigen::MatrixXd Whc;   // Weight matrix for candidate state from previous hidden state
    Eigen::VectorXd bf;    // Bias vector for input gate
    Eigen::VectorXd bi;    // Bias vector for input gate
    Eigen::VectorXd bo;    // Bias vector for output gate
    Eigen::VectorXd bc;    // Bias vector for candidate state

    double learning_rate;  // Learning rate for parameter updates

    // Stored states and gates for use in backward pass
    Eigen::VectorXd stored_input_gate;      // Stored input gate values
    Eigen::VectorXd stored_forget_gate;     // Stored forget gate values
    Eigen::VectorXd stored_output_gate;     // Stored output gate values
    Eigen::VectorXd stored_candidate_state; // Stored candidate state values
    Eigen::VectorXd stored_cell_state;      // Stored cell state values
    Eigen::VectorXd hidden_state;           // Hidden state of the cell

    // Helper activation functions
    Eigen::VectorXd sigmoid(const Eigen::VectorXd& x);  // Sigmoid activation function
    Eigen::VectorXd tanh(const Eigen::VectorXd& x);     // Hyperbolic tangent activation function


public:
    LSTMCell(int input_size, int hidden_size, double learning_rate);
    Eigen::VectorXd forward(const Eigen::VectorXd& x);
    void backward(const Eigen::VectorXd& dnext_h, const Eigen::VectorXd& dnext_c);

};

LSTMCell::LSTMCell(int input_size, int hidden_size, double learning_rate)
    : learning_rate(learning_rate) {
    // Initialize parameters (weights and biases)
    // Note: Initialize these parameters according to your initialization strategy
    Wxf = Eigen::MatrixXd::Random(hidden_size, input_size);
    Wxi = Eigen::MatrixXd::Random(hidden_size, input_size);
    Wxo = Eigen::MatrixXd::Random(hidden_size, input_size);
    Wxc = Eigen::MatrixXd::Random(hidden_size, input_size);
    Whf = Eigen::MatrixXd::Random(hidden_size, hidden_size);
    Whi = Eigen::MatrixXd::Random(hidden_size, hidden_size);
    Who = Eigen::MatrixXd::Random(hidden_size, hidden_size);
    Whc = Eigen::MatrixXd::Random(hidden_size, hidden_size);
    bf = Eigen::VectorXd::Zero(hidden_size);
    bi = Eigen::VectorXd::Zero(hidden_size);
    bo = Eigen::VectorXd::Zero(hidden_size);
    bc = Eigen::VectorXd::Zero(hidden_size);
}

Eigen::VectorXd LSTMCell::sigmoid(const Eigen::VectorXd& x) {
    return 1.0 / (1.0 + (-x.array()).exp());
}

Eigen::VectorXd LSTMCell::tanh(const Eigen::VectorXd& x) {
    return x.array().tanh();
}

Eigen::VectorXd LSTMCell::forward(const Eigen::VectorXd& x) {
    // Calculate the input gate values
    stored_input_gate = sigmoid(Wxi * x + Whi * hidden_state + bi);

    // Calculate the forget gate values
    stored_forget_gate = sigmoid(Wxf * x + Whf * hidden_state + bf);

    // Calculate the output gate values
    stored_output_gate = sigmoid(Wxo * x + Who * hidden_state + bo);

    // Calculate the candidate state values
    stored_candidate_state = tanh(Wxc * x + Whc * hidden_state + bc);

    // Calculate the new cell state by updating based on the gates
    stored_cell_state = stored_forget_gate.cwiseProduct(stored_cell_state) +
                        stored_input_gate.cwiseProduct(stored_candidate_state);

    // Calculate the new hidden state by applying the output gate and tanh activation
    hidden_state = stored_output_gate.cwiseProduct(stored_cell_state.unaryExpr(tanh));

    // Return the updated hidden state
    return hidden_state;
}

void LSTMCell::backward(const Eigen::VectorXd& dnext_h, const Eigen::VectorXd& dnext_c) {
    // Backpropagation logic for LSTM
    // Compute gradients with respect to hidden-to-hidden weights Whf, Whi, Who, Whc
    Eigen::MatrixXd dWhf = dnext_h * stored_cell_state.unaryExpr(tanh).cwiseProduct(dnext_c).transpose();
    Eigen::MatrixXd dWhi = dnext_h * stored_candidate_state.cwiseProduct(dnext_c).transpose();
    Eigen::MatrixXd dWho = dnext_h * stored_cell_state.unaryExpr(tanh).transpose();
    Eigen::MatrixXd dWhc = dnext_h * stored_input_gate.cwiseProduct(dnext_c).transpose();

    // Compute gradients with respect to input-to-hidden weights Wxf, Wxi, Wxo, Wxc
    Eigen::MatrixXd dWxf = dnext_h * stored_cell_state.unaryExpr(tanh).cwiseProduct(dnext_c) * x.transpose();
    Eigen::MatrixXd dWxi = dnext_h * stored_candidate_state.cwiseProduct(dnext_c) * x.transpose();
    Eigen::MatrixXd dWxo = dnext_h * stored_cell_state.unaryExpr(tanh) * x.transpose();
    Eigen::MatrixXd dWxc = dnext_h * stored_input_gate.cwiseProduct(dnext_c) * x.transpose();

    // Compute gradients with respect to hidden biases bf, bi, bo, bc
    Eigen::VectorXd dbf = dnext_h * stored_cell_state.unaryExpr(tanh).cwiseProduct(dnext_c);
    Eigen::VectorXd dbi = dnext_h * stored_candidate_state.cwiseProduct(dnext_c);
    Eigen::VectorXd dbo = dnext_h * stored_cell_state.unaryExpr(tanh);
    Eigen::VectorXd dbc = dnext_h * stored_input_gate.cwiseProduct(dnext_c);

    // Compute gradient with respect to the cell state C (dC)
    Eigen::VectorXd dC = dnext_c + dnext_h.cwiseProduct(stored_output_gate).cwiseProduct(stored_cell_state.unaryExpr(tanh).cwiseProduct(dnext_c));

    // Compute gradients with respect to previous hidden state h and input x
    Eigen::VectorXd dprev_h = stored_output_gate.cwiseProduct(dnext_h.cwiseProduct(stored_cell_state.unaryExpr(tanh))) + 
                              stored_output_gate.cwiseProduct(dC).cwiseProduct(stored_cell_state.unaryExpr(tanh));
    Eigen::MatrixXd dWprev_hf = dnext_h.cwiseProduct(dC).cwiseProduct(stored_cell_state.unaryExpr(tanh)).transpose();
    Eigen::MatrixXd dWprev_hi = dnext_h.cwiseProduct(dC).cwiseProduct(stored_candidate_state).transpose();
    Eigen::MatrixXd dWprev_ho = dnext_h.cwiseProduct(dnext_h.cwiseProduct(stored_cell_state.unaryExpr(tanh))).transpose();
    Eigen::MatrixXd dWprev_hc = dnext_h.cwiseProduct(dnext_h.cwiseProduct(stored_output_gate)).transpose();
    Eigen::MatrixXd dWprev_xf = dnext_h.cwiseProduct(dC).cwiseProduct(stored_cell_state.unaryExpr(tanh)) * x.transpose();
    Eigen::MatrixXd dWprev_xi = dnext_h.cwiseProduct(dC).cwiseProduct(stored_candidate_state) * x.transpose();
    Eigen::MatrixXd dWprev_xo = dnext_h.cwiseProduct(dnext_h.cwiseProduct(stored_cell_state.unaryExpr(tanh))) * x.transpose();
    Eigen::MatrixXd dWprev_xc = dnext_h.cwiseProduct(dnext_h.cwiseProduct(stored_output_gate)) * x.transpose();

    // Update parameters and stored states
    Whf -= learning_rate * dWhf;
    Whi -= learning_rate * dWhi;
    Who -= learning_rate * dWho;
    Whc -= learning_rate * dWhc;
    Wxf -= learning_rate * dWxf;
    Wxi -= learning_rate * dWxi;
    Wxo -= learning_rate * dWxo;
    Wxc -= learning_rate * dWxc;
    bf -= learning_rate * dbf;
    bi -= learning_rate * dbi;
    bo -= learning_rate * dbo;
    bc -= learning_rate * dbc;

    hidden_state -= learning_rate * dprev_h;
}

/***************************************************************************************************************************
 *********** IMPLEMENTING GRUCell
****************************************************************************************************************************/
class GRUCell {
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

    // Learning rate for weight updates during backpropagation
    double learning_rate;

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

Eigen::VectorXd GRUCell::forward(const Eigen::VectorXd& x) {
    // Calculate the update gate using input, hidden state, and biases
    stored_update_gate = sigmoid(Wxz * x + Whz * hidden_state + bz);

    // Calculate the reset gate using input, hidden state, and biases
    stored_reset_gate = sigmoid(Wxr * x + Whr * hidden_state + br);

    // Calculate the candidate hidden state using input, reset gate, hidden state, and biases
    stored_prev_h_bar = tanh(Wxh * x + Whh * (stored_reset_gate.cwiseProduct(hidden_state)) + bh);

    // Calculate the new hidden state using update gate, previous hidden state, and candidate hidden state
    hidden_state = (1.0 - stored_update_gate).cwiseProduct(hidden_state) + stored_update_gate.cwiseProduct(stored_prev_h_bar);

    // Return the updated hidden state
    return hidden_state;
}

void GRUCell::backward(const Eigen::VectorXd& dnext_h) {
    // Compute gradients with respect to hidden-to-hidden weights
    Eigen::MatrixXd dWhz = dnext_h * stored_prev_h_bar.transpose();
    Eigen::MatrixXd dWhr = dnext_h * hidden_state.transpose();
    Eigen::MatrixXd dWhh = dnext_h * (1.0 - stored_update_gate).transpose();

    // Compute gradients with respect to input-to-hidden weights
    Eigen::MatrixXd dWxz = dnext_h * stored_update_gate * stored_update_gate.cwiseProduct(sigmoid(Wxz * stored_update_gate));
    Eigen::MatrixXd dWxr = dnext_h * hidden_state * stored_reset_gate.cwiseProduct(sigmoid(Wxr * stored_reset_gate));
    Eigen::MatrixXd dWxh = dnext_h * (1.0 - stored_update_gate) * stored_prev_h_bar.cwiseProduct(tanh(Wxh * stored_prev_h_bar));

    // Compute gradients with respect to hidden biases
    Eigen::VectorXd dbz = dnext_h * stored_update_gate.cwiseProduct(sigmoid(Wxz * stored_update_gate));
    Eigen::VectorXd dbr = dnext_h * stored_reset_gate.cwiseProduct(sigmoid(Wxr * stored_reset_gate));
    Eigen::VectorXd dbh = dnext_h * (1.0 - stored_update_gate).cwiseProduct(tanh(Wxh * stored_prev_h_bar));

    // Compute gradient with respect to previous hidden state h_bar
    Eigen::VectorXd dh_bar = dnext_h.cwiseProduct(stored_update_gate);

    // Update weights and biases
    Whz -= learning_rate * dWhz;
    Whr -= learning_rate * dWhr;
    Whh -= learning_rate * dWhh;

    Wxz -= learning_rate * dWxz;
    Wxr -= learning_rate * dWxr;
    Wxh -= learning_rate * dWxh;

    bz -= learning_rate * dbz;
    br -= learning_rate * dbr;
    bh -= learning_rate * dbh;

    stored_prev_h_bar -= learning_rate * dh_bar;
}

void GRUCell::backward(const Eigen::VectorXd& dnext_h) {
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
    hidden_state_bar -= learning_rate * dh_bar;
}

/**********************************************************************************************
* Implement Auxiliary Classes
**********************************************************************************************/

class MeanSquaredErrorLoss {
public:
    // Calculate the mean squared error loss between predictions and targets
    static double calculate(const Eigen::VectorXd& predictions, const Eigen::VectorXd& targets) {
        Eigen::VectorXd squared_errors = (predictions - targets).array().square();
        return squared_errors.mean();
    }

    // Calculate the gradient of the loss with respect to predictions
    // A more generic calculation of gradient for MSE loss function
    static Eigen::VectorXd gradient(const Eigen::VectorXd& predictions, const Eigen::VectorXd& targets) {
        return (predictions - targets) / targets.size();
    }

    // Compute loss gradients for non-bidirectional recurrent network
    // more specific for network such as one that is not bidirectional RNN.
    static Eigen::VectorXd compute_loss_gradients(const Eigen::VectorXd& predicted_output,
                                            const Eigen::VectorXd& target_output) {
        // Compute the gradients of the loss function with respect to the predicted output.
        Eigen::VectorXd loss_gradients(predicted_output.size());
        loss_gradients = 2.0 * (predicted_output - target_output); // For example, using mean squared error loss

        return loss_gradients;
    }
};

/**********************************************************************************************
* Implement RecurrentNetwork
**********************************************************************************************/

class RecurrentNetwork {
private:
    int input_size;
    int hidden_size;
    int output_size;
    int sequence_length;
    int num_layers;
    double learning_rate;
    bool bidirectional;

    RNN rnn;
    LSTM lstm;
    GRU gru;

    bool isBidirectional = false; // Default to non-bidirectional

    // Compute loss gradients for non-bidirectional recurrent network
    Eigen::VectorXd compute_loss_gradients(const Eigen::VectorXd& predicted_output,
                                            const Eigen::VectorXd& target_output) {
        // Compute the gradients of the loss function with respect to the predicted output.
        Eigen::VectorXd loss_gradients(predicted_output.size());
        loss_gradients = 2.0 * (predicted_output - target_output); // For example, using mean squared error loss

        return loss_gradients;
    }

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

    class RNN {
    private:
        std::vector<RNNCell> rnn_forward_cells;
        std::vector<RNNCell> rnn_backward_cells;
        bool bidirectional;

    public:
        RNN(int input_size, int hidden_size, int num_layers, double learning_rate,
            bool bidirectional) : bidirectional(bidirectional) {
            // Initialize RNN cells for the forward pass
            for (int i = 0; i < num_layers; ++i) {
                rnn_forward_cells.push_back(RNNCell(hidden_size, hidden_size, learning_rate));
            }

            // If bidirectional, initialize RNN cells for the backward pass
            if (bidirectional) {
                for (int i = 0; i < num_layers; ++i) {
                    rnn_backward_cells.push_back(RNNCell(hidden_size, hidden_size, learning_rate));
                }
            }
        }

        // Train function for RNN
        void train(const Eigen::MatrixXd& input_data, const Eigen::MatrixXd& target_data, int num_epochs) {
            int num_directions = bidirectional ? 2 : 1; // Number of directions (1 for unidirectional, 2 for bidirectional)

            Eigen::MatrixXd reversed_input, reversed_target;
            if (bidrectional) {
                reversed_input  = reverseInput(input_data);
                reversed_target = reverseInput(target_data);
            }

            for (int epoch = 0; epoch < num_epochs; ++epoch) {
                for (int direction = 0; direction < num_directions; ++direction) {
                    // Choose the appropriate forward/backward cells based on the direction
                    std::vector<RNNCell>& rnn_cells = (direction == 0) ? rnn_forward_cells : rnn_backward_cells;

                    for (int layer = 0; layer < num_layers; ++layer) {
                        // Perform forward and backward passes for each layer's RNNCell
                        for (int step = 0; step < sequence_length; ++step) {
                            Eigen::VectorXd input_step = (direction == 0) ? input_data.row(step) : reverse_input.row(step);
                            
                            // Forward pass
                            Eigen::VectorXd output = rnn_cells[layer].forward(input_step);

                            // Calculate loss using the MSE loss function
                            Eigen::VectorXd target_step = (direction == 0) ? target_data.row(step): reversed_target.row(step);
                            double loss = MeanSquaredErrorLoss::calculate(output, target_step);

                            total_loss += loss;

                            // Backward pass
                            Eigen::VectorXd dnext_h;
                            if (layer < num_layers - 1) {
                                dnext_h = rnn_cells[layer + 1].stored_dprev_h;
                            } else {
                                // Calculate gradient of loss with respect to predictions
                                Eigen::VectorXd loss_gradient = MeanSquaredErrorLoss::gradient(output, target_step);
                                // Propagate the gradient through the RNNCell's backward pass
                                dnext_h = loss_gradient;
                            }
                            rnn_cells[layer].backward(dnext_h);

                            // Update the stored dprev_h for the next iteration
                            rnn_cells[layer].stored_dprev_h = rnn_cells[layer].dX;
                        }
                    }
                }
            }
        }

    };

    class LSTM {
    private:
        std::vector<LSTMCell> lstm_forward_cells;
        std::vector<LSTMCell> lstm_backward_cells;
        bool bidirectional;

    public:
        LSTM(int input_size, int hidden_size, int num_layers, double learning_rate,
            bool bidirectional) : bidirectional(bidirectional) {
            // Initialize LSTM cells for the forward pass
            for (int i = 0; i < num_layers; ++i) {
                lstm_forward_cells.push_back(LSTMCell(input_size, hidden_size, learning_rate));
            }

            // If bidirectional, initialize LSTM cells for the backward pass
            if (bidirectional) {
                for (int i = 0; i < num_layers; ++i) {
                    lstm_backward_cells.push_back(LSTMCell(input_size, hidden_size, learning_rate));
                }
            }
        }

        void train(const Eigen::MatrixXd& input_data, const Eigen::MatrixXd& target_data, int num_epochs) {
            int num_directions = bidirectional ? 2 : 1;
            int num_layers = lstm_forward_cells.size();
            int sequence_length = input_data.rows();

            Eigen::MatrixXd reversed_input, reversed_target;
            if (bidrectional) {
                reversed_input  = reverseInput(input_data);
                reversed_target = reverseInput(target_data);
            }

            for (int epoch = 0; epoch < num_epochs; ++epoch) {
                for (int direction = 0; direction < num_directions; ++direction) {
                    std::vector<LSTMCell>& lstm_cells = direction == 0 ? lstm_forward_cells : lstm_backward_cells;

                    for (int layer = 0; layer < num_layers; ++layer) {
                        for (int step = 0; step < sequence_length; ++step) {
                            Eigen::VectorXd input_step = (direction == 0) ? input_data.row(step) : reverse_input.row(step);

                            // Forward pass
                            Eigen::VectorXd output = lstm_cells[layer].forward(input_step);

                            // Calculate loss using the Mean Squared Error loss function
                            Eigen::VectorXd target_step = (direction == 0) ? target_data.row(step): reversed_target.row(step);
                            double loss = MeanSquaredErrorLoss::calculate(output, target_step);

                            total_loss += loss;

                            // Backward pass
                            Eigen::VectorXd dnext_h;
                            if (layer < num_layers - 1) {
                                dnext_h = lstm_cells[layer + 1].stored_dprev_h;
                            } else {
                                Eigen::VectorXd loss_gradient = MeanSquaredErrorLoss::gradient(output, target_step);
                                dnext_h = loss_gradient;
                            }
                            lstm_cells[layer].backward(dnext_h);

                            // Update the stored dprev_h for the next iteration
                            lstm_cells[layer].stored_dprev_h = lstm_cells[layer].dX;
                        }
                    }
                }
            }
        }

    };

    class GRU {
    private:
        std::vector<GRUCell> gru_forward_cells;
        std::vector<GRUCell> gru_backward_cells;
        bool bidirectional;

    public:
        GRU(int input_size, int hidden_size, int num_layers, double learning_rate,
            bool bidirectional)
            : bidirectional(bidirectional) {
            // Initialize GRU cells for the forward pass
            for (int i = 0; i < num_layers; ++i) {
                gru_forward_cells.push_back(GRUCell(input_size, hidden_size, learning_rate));
            }

            // If bidirectional, initialize GRU cells for the backward pass
            if (bidirectional) {
                for (int i = 0; i < num_layers; ++i) {
                    gru_backward_cells.push_back(GRUCell(input_size, hidden_size, learning_rate));
                }
            }
        }

        // Train function for GRU
        void train(const Eigen::MatrixXd& input_data, const Eigen::MatrixXd& target_data, int num_epochs) {
            int num_directions = bidirectional ? 2 : 1;
            int num_layers = gru_forward_cells.size();
            int sequence_length = input_data.rows();

            Eigen::MatrixXd reversed_input, reversed_target;
            if (bidrectional) {
                reversed_input  = reverseInput(input_data);
                reversed_target = reverseInput(target_data);
            }

            for (int epoch = 0; epoch < num_epochs; ++epoch) {
                for (int direction = 0; direction < num_directions; ++direction) {
                    std::vector<GRUCell>& gru_cells = direction == 0 ? gru_forward_cells : gru_backward_cells;

                    for (int layer = 0; layer < num_layers; ++layer) {
                        for (int step = 0; step < sequence_length; ++step) {
                            Eigen::VectorXd input_step = (direction == 0) ? input_data.row(step) : reverse_input.row(step);

                            // Forward pass
                            Eigen::VectorXd output = gru_cells[layer].forward(input_step);

                            // Calculate loss using the Mean Squared Error loss function
                            Eigen::VectorXd target_step = (direction == 0) ? target_data.row(step): reversed_target.row(step);
                            double loss = MeanSquaredErrorLoss::calculate(output, target_step);

                            total_loss += loss;

                            // Backward pass
                            Eigen::VectorXd dnext_h;
                            if (layer < num_layers - 1) {
                                dnext_h = gru_cells[layer + 1].stored_dprev_h;
                            } else {
                                Eigen::VectorXd loss_gradient = MeanSquaredErrorLoss::gradient(output, target_step);
                                dnext_h = loss_gradient;
                            }
                            gru_cells[layer].backward(dnext_h);

                            // Update the stored dprev_h for the next iteration
                            gru_cells[layer].stored_dprev_h = gru_cells[layer].dX;
                        }
                    }
                }
            }
        }
    };


/*************/

    std::wstring generate(const std::wstring& input, int num_steps) {
        // Convert input sequence to suitable format.
        Eigen::VectorXd input_vector(input.size());
        for (size_t i = 0; i < input.size(); ++i) {
            input_vector(i) = static_cast<double>(input[i]); // Convert wchar_t to double
        }

        // Initialize output sequence.
        std::wstring generated_output;
        generated_output.reserve(num_steps);

        // Perform forward passes to generate output sequence.
        Eigen::VectorXd current_input = input_vector;
        for (int step = 0; step < num_steps; ++step) {
            Eigen::VectorXd predicted_output;
            if (isBidirectional) {
                predicted_output = bidirectional_forward(current_input);
            } else {
                predicted_output = forward(current_input);
            }

            // Convert the predicted output back to wstring and append to the generated_output.
            for (int i = 0; i < predicted_output.size(); ++i) {
                wchar_t char_value = static_cast<wchar_t>(predicted_output(i)); // Convert double to wchar_t
                generated_output += char_value;
            }

            // Set the current input for the next step.
            current_input = predicted_output;
        }

        return generated_output;
    }

};
