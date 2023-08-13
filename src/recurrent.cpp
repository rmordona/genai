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
#include "recurrent.h"
 
/***************************************************************************************************************************
*********** IMPLEMENTING RNNCell
****************************************************************************************************************************/
RNNCell::RNNCell(int input_size, int param_size, int hidden_size, int output_size, double learning_rate) 
    : input_size(input_size), param_size(param_size), hidden_size(hidden_size), output_size(output_size), 
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

const Eigen::MatrixXd& RNNCell::forward(const Eigen::MatrixXd& input_data) {

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

const Eigen::MatrixXd& RNNCell::backward(const Eigen::MatrixXd& dnext_h) {

    // Compute gradient with respect to tanh
    Eigen::MatrixXd dtanh = (1.0 - H.array().square()).array() * dnext_h.array();

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
 
/***************************************************************************************************************************
*********** IMPLEMENTING LSTMCell
****************************************************************************************************************************/
LSTMCell::LSTMCell(int input_size, int param_size, int hidden_size, int output_size, double learning_rate)
    : input_size(input_size), param_size(param_size), hidden_size(hidden_size), output_size(output_size), 
      learning_rate(learning_rate) {
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
    by = Eigen::RowVectorXd::Zero(output_size);

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
    Gt = BaseOperator::tanh(XH * Wg + bg);

    // Calculate the new cell state by updating based on the gates
    C = Ft.array() * C.array() + It.array() * Gt.array();

    // Calculate the new hidden state by applying the output gate and tanh activation
    H = BaseOperator::tanh(C).array() * Ot.array();

    // Compute Yhat.
    //     (nxo) =  (nxh) * (hxo) + h
    Yhat = BaseOperator::softmax(H * V + by);

    // We return the prediction for layer forward pass.
    // For the Cell State and Hidden State output, we cache them instead for time step forward pass.
    return Yhat; 

}

const Eigen::MatrixXd& LSTMCell::backward(const Eigen::MatrixXd& dnext_h) {
    // Backpropagation logic for LSTM

    // Compute gradients with respect to the gates
    dC = (Ot.array() * (1 - BaseOperator::tanh(C.array()).array().square()) * dnext_h.array()) + dC.array();
    Eigen::MatrixXd dFt = dC.array() * C.array() * Ft.array() * (1 - Ft.array());
    Eigen::MatrixXd dIt = dC.array() * Gt.array() * It.array() * (1 - It.array());
    Eigen::MatrixXd dGt = dC.array() * It.array().array() * (1 - Gt.array().square().array());
    Eigen::MatrixXd dOt = dnext_h.array() * BaseOperator::tanh(C).array() * Ot.array() * (1 - Ot.array());

    // Concatenate input and hidden state.
    XH << X, H;

    // Compute gradients with respect to hidden-to-hidden weights Wf, Wi, Wo, Wc
    Eigen::MatrixXd dWf = XH.transpose() * dFt;
    Eigen::MatrixXd dWi = XH.transpose() * dIt;
    Eigen::MatrixXd dWg = XH.transpose() * dGt;
    Eigen::MatrixXd dWo = XH.transpose() * dOt;


    // Compute gradients with respect to hidden biases bf, bi, bo, bc

    Eigen::RowVectorXd dbf = dFt.colwise().sum();
    Eigen::RowVectorXd dbi = dIt.colwise().sum();
    Eigen::RowVectorXd dbo = dOt.colwise().sum();
    Eigen::RowVectorXd dbg = dGt.colwise().sum();

    // Compute gradient with respect to the cell state C (dC)
    dC = dC.array() * Ft.array();

    // Compute gradient with respect to hidden state H (dH)
    int ps = param_size, hs = hidden_size;
    dH  = dFt * Split(Wf, ps, hs).transposedX() 
        + dIt * Split(Wi, ps, hs).transposedX()  
        + dOt * Split(Wo, ps, hs).transposedX() 
        + dGt * Split(Wg, ps, hs).transposedX();

    // Compute gradient with respect to input (dInput)
    dX  = dFt * Split(Wf, ps, hs).transposedH() 
        + dIt * Split(Wi, ps, hs).transposedH()  
        + dOt * Split(Wo, ps, hs).transposedH()
        + dGt * Split(Wg, ps, hs).transposedH();

    // Compute gradient with respect to output bias bo.
    Eigen::RowVectorXd dby = Y.colwise().sum();

    // Update parameters and stored states
    Wf -= learning_rate * dWf;
    Wi -= learning_rate * dWi;
    Wo -= learning_rate * dWo;
    Wg -= learning_rate * dWg;
    bf -= learning_rate * dbf;
    bi -= learning_rate * dbi;
    bo -= learning_rate * dbo;
    bg -= learning_rate * dbg;
    by -= learning_rate * dby;

    // Update hidden state  and cell state
    H -= learning_rate * dH;
    C -= learning_rate * dC;

    // Update stored states and gates
    Ft -= learning_rate * dFt;
    It -= learning_rate * dIt;
    Ot -= learning_rate * dOt;
    Gt -= learning_rate * dGt;

    // Return gradient with respect to input.
    // We return the gradient wrt X for each layer, not wrt H & C for each time step.
    // for the gradient wrt H & C, we just cache it for time step propagation, but layer propagation.
    return dX;
}

/***************************************************************************************************************************
 *********** IMPLEMENTING GRUCell
****************************************************************************************************************************/
GRUCell::GRUCell(int input_size, int param_size, int hidden_size, int output_size, double learning_rate) 
        : input_size(input_size), param_size(param_size), hidden_size(hidden_size), output_size(output_size), 
          learning_rate(learning_rate) {
    // Initialize parameters (weights and biases)
    // Note: Initialize these parameters according to your initialization strategy
    Wz.resize(param_size + hidden_size, hidden_size);
    Wr.resize(param_size + hidden_size, hidden_size);
    Wg.resize(param_size + hidden_size, hidden_size);
    V.resize(hidden_size, output_size);

    BaseOperator::heInitialization(Wz);
    BaseOperator::heInitialization(Wr);
    BaseOperator::heInitialization(Wg);
    BaseOperator::heInitialization(V);

    bz = Eigen::RowVectorXd::Zero(hidden_size);
    br = Eigen::RowVectorXd::Zero(hidden_size);
    bg = Eigen::RowVectorXd::Zero(hidden_size);
    bo = Eigen::RowVectorXd::Zero(hidden_size);

    H    = Eigen::MatrixXd::Zero(input_size, hidden_size);
    Yhat = Eigen::MatrixXd::Zero(input_size, output_size);

    XH   = Eigen::MatrixXd::Zero(input_size, param_size + hidden_size); // concatenate X and H

}

const Eigen::MatrixXd& GRUCell::forward(const Eigen::MatrixXd& input_data) {

    // Store input for backward pass.
    // this can be the prev_hidden_state
    X = input_data;

    // Concatenate input and hidden state.
    XH << X, H;

    // Calculate the update gate using input, hidden state, and biases
    Zt =  BaseOperator::sigmoid(XH * Wz + bz);

    // Calculate the reset gate using input, hidden state, and biases
    Rt =  BaseOperator::sigmoid(XH * Wr + br);

    // Calculate the candidate hidden state using input, reset gate, hidden state, and biases
    Eigen::MatrixXd RH = Rt.array() * H.array();
    Eigen::MatrixXd rXH(input_size, hidden_size + hidden_size);
    rXH << X, RH;
    Rt = BaseOperator::tanh(XH * Wg + bg);

    // Calculate Ghat


    // Calculate the new hidden state using update gate, previous hidden state, and candidate hidden state
    H = (1 - Zt.array()).array() * Gt.array() + Zt.array() * H.array(); 

    // Compute Yhat.
    //     (nxo) =  (nxh) * (hxo) + h
    Yhat = BaseOperator::softmax(H * V + bo);

    // We return the prediction for layer forward pass.
    // For the Hidden State output, we cache them instead for time step forward pass.
    return Yhat; 
}

const Eigen::MatrixXd& GRUCell::backward(const Eigen::MatrixXd& dnext_h) {
    // Backpropagation logic for GRU
    // Compute gradients with respect to hidden-to-hidden weights Wz, Wr, Wg

    int ps = param_size, hs = hidden_size;

    // Compute gradients with respect to the gates
    Eigen::MatrixXd dZt = dnext_h.array() * ( H.array() - Gt.array()) * Zt.array() * ( 1 - Zt.array());
    Eigen::MatrixXd dGt = dnext_h.array() * ( 1 - Zt.array()) * ( 1 - Gt.array().square());
    Eigen::MatrixXd dRt = ( dGt.array() * (H * Split(Wr, ps, hs).transposedH()).array() 
                          * ( 1 - Rt.array().square()) )
                          * Rt.array() * ( 1 - Rt.array());

    // Concatenate input and hidden state.
    XH << X, H;

    // Compute gradients with respect to hidden-to-hidden weights Wz, Wr, Wg
    Eigen::MatrixXd dWz = XH.transpose() * dZt;
    Eigen::MatrixXd dWr = XH.transpose() * dRt;
    Eigen::MatrixXd dWg = XH.transpose() * dGt;

    // Compute gradients with respect to hidden biases bz, br, bg
    Eigen::RowVectorXd dbz = dZt.colwise().sum();
    Eigen::RowVectorXd dbr = dGt.colwise().sum();
    Eigen::RowVectorXd dbg = dRt.colwise().sum();

    // Compute gradient with respect to hidden state H (dH)

    dH  = dZt * Split(Wz, ps, hs).transposedX() 
        + dRt * Split(Wr, ps, hs).transposedX()  
        + dGt * Split(Wg, ps, hs).transposedX();

    // Compute gradient with respect to input (dInput)
    dX  = dZt * Split(Wz, ps, hs).transposedH() 
        + dRt * Split(Wr, ps, hs).transposedH()  
        + dGt * Split(Wg, ps, hs).transposedH();

    // Compute gradient with respect to output bias bo.
    Eigen::RowVectorXd dbo = Y.colwise().sum();

    // Update parameters
    Wz -= learning_rate * dWz;
    Wr -= learning_rate * dWr;
    Wg -= learning_rate * dWg;
    bz -= learning_rate * dbz;
    br -= learning_rate * dbr;
    bg -= learning_rate * dbg;
    bo -= learning_rate * dbo;

    // Update hidden state  
    H -= learning_rate * dH;

    // Update stored states and gates
    Zt -= learning_rate * dZt;
    Rt -= learning_rate * dRt;
    Gt -= learning_rate * dGt;

    // Return gradient with respect to input.
    // We return the gradient wrt X for each layer, not wrt H & C for each time step.
    // for the gradient wrt H & C, we just cache it for time step propagation, but layer propagation.
    return dX;
}


/**********************************************************************************************
* Implement Recurrent Network
**********************************************************************************************/
const Eigen::MatrixXd& RNN::forward(const Eigen::MatrixXd& input_data) {
    RNNType rnntype = this->rnntype;
    if (rnntype == MANY_TO_ONE) {
        output = forwardManyToOne(input_data, this); 
    } else
    if (rnntype == ONE_TO_MANY) {
        output = forwardOneToMany(input_data, this); 
    } else {// MANYTOMANY 
        output = forwardManyToMany(input_data, this); 
    }
    return output;
}

const Eigen::MatrixXd& RNN::backward(const Eigen::MatrixXd& gradients) {
    RNNType rnntype = this->rnntype;
    if (rnntype == MANY_TO_ONE) {
        output = backwardManyToOne(gradients, this); 
    } else
    if (rnntype == ONE_TO_MANY) {
        output = backwardOneToMany(gradients, this); 
    } else {// MANYTOMANY 
        output = backwardManyToMany(gradients, this); 
    }
    return output;
}

const Eigen::MatrixXd& LSTM::forward(const Eigen::MatrixXd& input_data) {
    RNNType rnntype = this->rnntype;
    if (rnntype == MANY_TO_ONE) {
        output = forwardManyToOne(input_data, this); 
    } else
    if (rnntype == ONE_TO_MANY) {
        output = forwardOneToMany(input_data, this); 
    } else {// MANYTOMANY 
        output = forwardManyToMany(input_data, this); 
    }
    return output;
}

const Eigen::MatrixXd& LSTM::backward(const Eigen::MatrixXd& gradients) {
    RNNType rnntype = this->rnntype;
    if (rnntype == MANY_TO_ONE) {
        output = backwardManyToOne(gradients, this); 
    } else
    if (rnntype == ONE_TO_MANY) {
        output = backwardOneToMany(gradients, this); 
    } else {// MANYTOMANY 
        output = backwardManyToMany(gradients, this); 
    }
    return output;
}

const Eigen::MatrixXd& GRU::forward(const Eigen::MatrixXd& input_data) {
    RNNType rnntype = this->rnntype;
    if (rnntype == MANY_TO_ONE) {
        output = forwardManyToOne(input_data, this); 
    } else
    if (rnntype == ONE_TO_MANY) {
        output = forwardOneToMany(input_data, this); 
    } else {// MANYTOMANY 
        output = forwardManyToMany(input_data, this); 
    }
    return output;
}

const Eigen::MatrixXd& GRU::backward(const Eigen::MatrixXd& gradients) {
    RNNType rnntype = this->rnntype;
    if (rnntype == MANY_TO_ONE) {
        output = backwardManyToOne(gradients, this); 
    } else
    if (rnntype == ONE_TO_MANY) {
        output = backwardOneToMany(gradients, this); 
    } else {// MANYTOMANY 
        output = backwardManyToMany(gradients, this); 
    }
    return output;
}


// Many-To-One Scenario
template <typename CellType>
const Eigen::MatrixXd& forwardManyToOne(const Eigen::MatrixXd& input_data, CellType& rnn) {
    Eigen::VectorXd last_output(rnn->num_directions);

    rnn->input_data = input_data;

    for (int direction = 0; direction < rnn->num_directions; ++direction) {
        rnn->cells = (direction == 0) ? rnn->fcells : rnn->bcells;

        Eigen::VectorXd input_step;
        // Forward pass: Run from first to last time step
        for (int step = 0; step < rnn->sequence_length; ++step) {
            if (direction == 0) {
                input_step = input_data.row(step);
            } else {
                input_step = input_data.row(rnn->sequence_length - step - 1);
            }

            // Forward pass through each layer of the RNN
            for (int layer = 0; layer < rnn->num_layers; ++layer) {
                input_step = rnn->cells[layer].forward(input_step);
            }

            // Store the output of the last time step
            if (step == rnn->sequence_length - 1) {
                last_output[direction] = input_step;
            }
        }
    }
    return last_output;
}

// One-To-Many Scenario
template <typename CellType>
const Eigen::MatrixXd& forwardOneToMany(const Eigen::MatrixXd& input_data, CellType& rnn) {
    Eigen::MatrixXd all_outputs(rnn->sequence_length, rnn->num_directions);

    rnn->input_data = input_data;

    for (int direction = 0; direction < rnn->num_directions; ++direction) {
        rnn->cells  = (direction == 0) ? rnn->fcells : rnn->bcells;

        Eigen::VectorXd input_step = input_data;

        // Forward pass: Generate outputs for each time step
        for (int step = 0; step < rnn->sequence_length; ++step) {
            Eigen::VectorXd output_step;

            // Forward pass through each layer of the RNN
            for (int layer = 0; layer < rnn->num_layers; ++layer) {
                output_step = rnn->cells[layer].forward(input_step);
                input_step = output_step;  // Use the output as input for the next layer
            }

            // Store the output of the current time step
            all_outputs(step, direction) = output_step;
        }
    }
    return all_outputs;
}

// Many-To-Many Scenario
template <typename CellType>
const Eigen::MatrixXd& forwardManyToMany(const Eigen::MatrixXd& input_data, CellType& rnn) {
    Eigen::MatrixXd all_outputs(rnn->sequence_length, rnn->num_directions);

    rnn->input_data = input_data;

    for (int direction = 0; direction < rnn->num_directions; ++direction) {
        rnn->cells  = (direction == 0) ? rnn->fcells : rnn->bcells;

        Eigen::VectorXd input_step;
        Eigen::VectorXd output_step;

        // Forward pass: Run from first to last time step
        for (int step = 0; step < rnn->sequence_length; ++step) {
            if (direction == 0) {
                input_step = input_data.row(step);
            } else {
                input_step = input_data.row(rnn->sequence_length - step - 1);
            }

            // Forward pass through each layer of the RNN
            for (int layer = 0; layer < rnn->num_layers; ++layer) {
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
const Eigen::MatrixXd& backwardManyToOne(const Eigen::MatrixXd& gradients, CellType& rnn) {
    Eigen::MatrixXd loss_gradients(rnn->sequence_length, rnn->num_directions);

    for (int direction = 0; direction < rnn->num_directions; ++direction) {
        rnn->cells  = (direction == 0) ? rnn->fcells : rnn->bcells;

        for (int step = rnn->sequence_length - 1; step >= 0; --step) {
           // Eigen::VectorXd input_step = direction == 0 ? input_data.row(step) : input_data.row(rnn->sequence_length - step - 1);

            // Backward pass through each layer of the RNN
            for (int layer = rnn->num_layers - 1; layer >= 0; --layer) {
                loss_gradients = rnn->cells[layer].backward(loss_gradients);
            }
        }
    }

    return loss_gradients;
}

// One-To-Many Scenario: Backward Pass
template <typename CellType>
const Eigen::MatrixXd& backwardOneToMany(const Eigen::MatrixXd& gradient, CellType& rnn) {
    Eigen::MatrixXd loss_gradients(rnn->sequence_length, rnn->num_directions);

    for (int direction = 0; direction < rnn->num_directions; ++direction) {
        rnn->cells  = (direction == 0) ? rnn->fcells : rnn->bcells;

        Eigen::VectorXd input_step = rnn->input_data;

        // Backward pass: Compute gradients for each time step
        for (int step = rnn->sequence_length - 1; step >= 0; --step) {
            Eigen::VectorXd output_step;  // Output of the RNNCell's forward pass

            // Backward pass through each layer of the RNN
            for (int layer = rnn->num_layers - 1; layer >= 0; --layer) {
                output_step = rnn->cells[layer].forward(input_step);  // Forward pass for the current step
                loss_gradients = rnn->cells[layer].backward(loss_gradients);
                input_step = output_step;  // Use the output as input for the next layer
            }

            // Store the loss gradient with respect to the input of the current step
            if (direction == 0) {
                loss_gradients(step) = loss_gradients;
            } else {
                loss_gradients(rnn->sequence_length - step - 1) = loss_gradients;
            }
        }
    }

    return loss_gradients;
}

// Many-To-Many Scenario: Backward Pass
template <typename CellType>
const Eigen::MatrixXd& backwardManyToMany(const Eigen::MatrixXd& gradients, const Eigen::MatrixXd& input_data, CellType& rnn) {
    Eigen::MatrixXd loss_gradients(rnn->sequence_length, rnn->num_directions);

    for (int direction = 0; direction < rnn->num_directions; ++direction) {
        rnn->cells  = (direction == 0) ? rnn->fcells : rnn->bcells;

        for (int step = rnn->sequence_length - 1; step >= 0; --step) {
           // Eigen::VectorXd input_step = direction == 0 ? input_data.row(step) : input_data.row(rnn->sequence_length - step - 1);

            // Backward pass through each layer of the RNN
            for (int layer = rnn->num_layers - 1; layer >= 0; --layer) {
                loss_gradients(step, direction) = loss_gradients;
                loss_gradients = rnn->cells[layer].backward(loss_gradients);
            }
        }
    }

    return loss_gradients;
}

void RNN::updateParameters(std::string& optimizertype, double& learningRate, int& iter) {
}

void LSTM::updateParameters(std::string& optimizertype, double& learningRate, int& iter) {
}

void GRU::updateParameters(std::string& optimizertype, double& learningRate, int& iter) {
}

