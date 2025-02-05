/*
 * GENAI - A framework that can help build LLMs
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
#include "topology.h"
#include "model.h"

namespace py = pybind11;
using namespace py::literals;

template <class T>
void BaseModel<T>::setGraph(std::shared_ptr<Graph<T>>  graph) {

    // auto mygraph = dynamic_cast<Graph<T>*>(graph);
    // this->graph = mygraph;
    this->graph = graph;
    log_detail( "SetGraph ... Node count: {:d}", this->graph->getNodes().size() );
}

template <class T>
std::shared_ptr<Graph<T>> BaseModel<T>::getGraph( ) {
    return this->graph;
}

template <class T>
void BaseModel<T>::setLoss(std::string& losstype) {
    this->losstype = losstype;
}

// The output is assumed to have BxNxM where B=batch/sequence size, N=number of samples, M=embedding vector size
// This allows to compute for the output size,  MxW where W is the number of weights (features) to use.
template <class T>
void BaseModel<T>::setTarget(const py::array_t<T>& target, const bool normalize) {
    this->target = ConvertData::totensor(target);

    if (normalize == true) {
        this->target = BaseOperator::standardize(this->target);
    }
}

template <class T>
aitensor<T> BaseModel<T>::getTarget() {
    return this->target;
}

template <class T>
void BaseModel<T>::useCrossEntropy() {

}

/*****************************************************************************************************************************
******************************************************************************************************************************
* BaseModel::train
* This is the core function to fit a model
* Temporarily store np.array (passed as dtype = np.float32) to a double pointer (this->input_fdata).
*
* If the cross-entropy loss for your decoder transformer is not converging and remains around a particular value (e.g., 4.0), 
* there are several potential reasons for this behavior. Here are some common considerations to troubleshoot and improve convergence:
*
* Learning Rate:
*
* Check the learning rate used in your optimizer. If the learning rate is too high, the optimization process might oscillate or 
* fail to converge. Conversely, if the learning rate is too low, the model may converge very slowly. Experiment with different 
* learning rates to find an appropriate value.
*
* Model Capacity:
*
* Consider the capacity of your model. If the model is too small for the complexity of the task, it may struggle to learn 
* meaningful representations. Consider increasing the model size (number of layers, hidden dimensions) if necessary.
*
* Gradient Clipping:
* 
* If you are using gradient clipping, ensure that the clipping threshold is appropriate. Gradient clipping can be beneficial 
* for preventing exploding gradients, but an excessively low threshold may hinder learning.
*
* Data Issues:
*
* Check your input data and preprocessing. Ensure that your input sequences are correctly formatted, and the preprocessing
* does not introduce issues. Check for class imbalances or any anomalies in your dataset.
*
* Initialization:
*
* Ensure that your model parameters are initialized properly. Poor initialization can lead to slow convergence or convergence 
* to suboptimal solutions. Consider using techniques like Xavier/Glorot initialization.
* 
* Regularization:
*
* Experiment with regularization techniques such as dropout or layer normalization. Regularization can help prevent overfitting 
* and improve generalization.
*
* Training Duration:
* 
* Allow the model to train for a sufficient number of epochs. If the loss is stable but high, it might need more training to 
* converge to a better solution.
*
* Validation Loss:
*
* Monitor the validation loss in addition to the training loss. If the training loss is low but the validation loss is high, 
* the model might be overfitting. Adjust regularization or consider early stopping.
*
* Check for NaN Values:
*
* Check for NaN (Not a Number) values in the gradients or loss. This can occur if there are numerical stability issues. 
* Monitor the loss and gradients during training.
*
* Debugging Techniques:
*
* Gradually simplify your model and training pipeline for debugging. Ensure that a simpler version of your model can learn from 
* the data before introducing complexity.
*******************************************************************************************************************************
******************************************************************************************************************************/
template <class T>
std::vector<float> BaseModel<T>::train(std::string& losstype, std::vector<std::string>& metricstype, std::string& optimizertype, 
                int batch_size, const int max_epoch, const T learn_rate , const bool use_step_decay, const T decay_rate) {

    // Initialize MPI
    //MPI_Init(NULL, NULL);

    this->losstype      = losstype;
    this->optimizertype = optimizertype;
    this->metricstype   = metricstype;
    this->learningRate  = learn_rate;
    this->useStepDecay  = use_step_decay;
    this->decayRate     = decay_rate;
    this->batch_size    = batch_size;

    std::vector<float> losses = {};

    int mod_epoch = std::ceil(max_epoch * 0.10);

    log_info( "******************************************************************************************" );
    log_info( "********************************* Start Training *****************************************" );
    log_info( "******************************************************************************************" );
    log_detail( "Number of Graph Nodes: {:d}", this->graph->getNodes().size() );

    aiscalar<T> stopping_criteria = 1e-4;
    // aiscalar<T> old_loss = inf();
  
    auto start_time        = std::chrono::system_clock::now();
    auto end_time          = std::chrono::system_clock::now();
    std::time_t next_time  = std::chrono::system_clock::to_time_t(end_time);
    double total_seconds = 0.0, total_mins = 0.0, total_hrs = 0.0, total_days = 0.0, duration = 0.0;
    int    total_iteration = 0;
    std::chrono::duration<double> elapsed_seconds;

    py_cout << "Fitting the model ..." << std::endl;

    aitensor<T> batch_target = {}; // Y
    aitensor<T> batch_output = {}; // Y-hat

    // We assume that model.setData and model.setTarget have been called to set
    // the input data and target data prior to invoking the train function.
    int target_size = this->target.size();

    py_cout << "Target Size " << target_size << std::endl;

    if ((int) batch_size >= (int) target_size) {
        this->batch_size = target_size;
    }

    py_cout << "Batch Size: " << batch_size << " Target Size: " << target_size << std::endl;
 
    for (int iter = 1; iter <= max_epoch; iter++) {

        log_detail( "<<<<<<<<<<<<<<<<<<<<<<<<< Process batch (iteration {:d})  >>>>>>>>>>>>>>>>>>>>>>>>>>", (iter) );

        // Note that batch processing is done at the node level.

        // int total_loss = 0, 
        int tot_metrics_precision = 0, tot_metrics_recall = 0, tot_metrics_f1score = 0;

        // int total_loss_cnt = 0;
        int total_metrics_cnt = 0;

        this->start_index = std::rand() % (target_size - this->batch_size + 1);
        batch_target = BaseOperator::getBatch(this->target, this->start_index, this->batch_size);
        log_detail( "Entering Forward Propagation ..." );
        batch_output = this->graph->forwardPropagation(this->start_index, this->batch_size);
        log_detail( "Forward Output: Tensor Size {0} {1}x{2}", batch_output.size(), batch_output.at(0).rows(), batch_output.at(0).cols() );
        log_matrix( batch_output );

        log_detail( "Computing Loss ..." );
        this->loss = this->graph->computeLoss(this->losstype, batch_output, batch_target); 
        log_detail( "Computing Loss Gradient ..." );
        this->gradients = this->graph->computeGradients(this->losstype, batch_output, batch_target);
        log_matrix( this->gradients );
        log_detail( "Entering Backward Propagation ..." );
        this->gradients = this->graph->backwardPropagation(this->gradients); 
        log_detail( "Updating Parameters ..." );
        this->graph->updateParameters(this->optimizertype, this->learningRate, iter);
        if (this->losstype == "bce" || this->losstype == "cce") {
            log_detail( "Calculate Performance Metrics ...");
            this->metrics = this->graph->computeMetrics(metricstype, batch_output, batch_target);
            tot_metrics_precision += this->metrics.precision;
            tot_metrics_recall += this->metrics.recall;
            tot_metrics_f1score += this->metrics.f1score;
            total_metrics_cnt ++;
        }
        // Calculate Time, then display loss
        end_time = std::chrono::system_clock::now();
        elapsed_seconds = end_time - start_time;
        next_time = std::chrono::system_clock::to_time_t(end_time);
        start_time = end_time;

        total_seconds += elapsed_seconds.count();
        duration += elapsed_seconds.count();
        total_iteration++;
        losses.push_back(this->loss);
        
        if (this->losstype == "bce" || this->losstype == "cce") {
            tot_metrics_precision = tot_metrics_precision / total_metrics_cnt;
            tot_metrics_recall    = tot_metrics_recall / total_metrics_cnt;
            tot_metrics_f1score   = tot_metrics_f1score / total_metrics_cnt;
        }
        // Print Progress
        if (iter == 1 || iter % mod_epoch == 0 || iter == max_epoch) {

            // Use Step Decay  
            if (this->useStepDecay) {
                this->learningRate = this->learningRate * (this->decayRate);
            }
 
            py_cout << "Epoch " << iter << "/" << max_epoch << " ... ";
            py_cout << "Loss: " << this->loss;

            if (this->losstype == "bce" || this->losstype == "cce") {
                if (this->metrics.isprecision) {
                    py_cout << " ... Acc (P): " << tot_metrics_precision;
                }
                if (this->metrics.isrecall) {
                    py_cout << "... Acc (R): " << tot_metrics_recall;
                }
                if (this->metrics.isf1score) {
                    py_cout << "... Acc (F1): " << tot_metrics_f1score;
                }
            }

            double avg_microseconds = (total_seconds / total_iteration) * 1000000;
            py_cout << " ... Avg Elapsed " << avg_microseconds << "us";
            py_cout << " at " << std::ctime(&next_time) << std::endl;

            // Also, log the result if Logging INFO is enabled
            log_detail( "Epoch {}/{} ... Loss: {:8.5f} ... Acc (P): {:8.5f} ... Avg Elapsed {}us at {}", iter, max_epoch, 
                this->loss, tot_metrics_precision, avg_microseconds, std::ctime(&next_time) );

            total_seconds = 0.0;
            total_iteration = 0;

        }
        if (this->loss <= stopping_criteria) break;

    }
    
    total_mins = std::floor( duration / 60 );
    total_seconds = duration - total_mins * 60;
    total_hrs = std::floor(total_mins / 60 );
    total_mins = total_mins - total_hrs * 60;
    total_days = std::floor(total_hrs / 24 );
    total_hrs = total_hrs - total_days * 24;

    log_detail( "Training done" );

    py_cout << " Duration " << "D: " << total_days << ", HR: " << total_hrs << ", MN: " << total_mins << ", SC: " << total_seconds << std::endl;

    // Finalize MPI
    //MPI_Finalize();

    return losses;
}

/**************************************************************************************************
* BaseModel::predict
* Temporarily store np.array (passed as dtype = np.float32) to a double pointer (this->input_fdata).
* Upon prediction entry, the double pointer will be transformed to an aitensor and handed over
* to the Node class.
**************************************************************************************************/
template <class T>
aitensor<T> BaseModel<T>::predict(int sequence_length) {
 
    // Initialize MPI
    //MPI_Init(NULL, NULL);

    log_info( "******************************************************************************************" );
    log_info( "********************************* Start Prediction ***************************************" );
    log_info( "******************************************************************************************" );
    log_detail( "Number of Graph Nodes: {:d}", this->graph->getNodes().size() );

    int target_size = this->graph->getDataSize();

    auto start_time = std::chrono::system_clock::now();

    aitensor<T> predicted;

    int tot_metrics_precision = 0, tot_metrics_recall = 0, tot_metrics_f1score = 0;

    for (int iter = 0; iter <= sequence_length; iter++) {

        predicted = this->graph->forwardPropagation(0, target_size);

        log_detail( "Predicted Result: Tensor Size {0}", predicted.size() );
        log_matrix( predicted );

        if (this->losstype == "bce" || this->losstype == "cce") {
            this->metrics = this->graph->computeMetrics(metricstype, predicted, this->target);
            tot_metrics_precision += this->metrics.precision;
            tot_metrics_recall += this->metrics.recall;
            tot_metrics_f1score += this->metrics.f1score;
        }

    }

    // Calculate Time, then display loss
    auto end_time = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end_time - start_time;
    std::time_t next_time = std::chrono::system_clock::to_time_t(end_time);
    start_time = end_time;

    // Print Progress


    if (this->losstype == "bce" || this->losstype == "cce") {
        if (this->metrics.isprecision) {
            py_cout << " Acc (P): " << tot_metrics_precision / sequence_length;
        }
        if (this->metrics.isrecall) {
            py_cout << "Acc (R): " << tot_metrics_recall / sequence_length;
        }
        if (this->metrics.isf1score) {
            py_cout << "Acc (F1): " << tot_metrics_f1score / sequence_length;
        }
        py_cout << " ... ";
    }
    py_cout << "Elapsed " <<  elapsed_seconds.count() * 1000000 << "us";
    py_cout << " at " << std::ctime(&next_time) << std::endl;

    // Also, log the result if Logging INFO is enabled
    log_detail( "Acc (P): {:8.5f} ... Elapsed {}us at {}", 
            tot_metrics_precision / sequence_length, elapsed_seconds.count() * 1000000, std::ctime(&next_time) );

    log_detail( "Prediction done ..." );

    return predicted;

    // Finalize MPI
    //MPI_Finalize();

}

/**************************************************************************************************
* ModelNode::setData and setDecoderData
* Temporarily store np.array (passed as dtype = np.float32) to a double pointer (this->input_fdata).
* Upon training entry, the double pointer will be transformed to an aitensor and handed over
* to the Node class.
**************************************************************************************************/
void ModelNode::setData(const py::array& data, const bool normalize, const bool positional) {
    try {

        py::buffer_info buf = data.request();

        if (buf.format == py::format_descriptor<double>::format()) {
            if (this->datatype == DataType::float32) {
                throw AIException("Precision used in data is 'double' but the model uses 'float' ...");
            }
            this->input_ddata = ConvertData::totensor(data.cast<py::array_t<double>>());
        } 
        else if (buf.format == py::format_descriptor<float>::format()) {
            if (this->datatype == DataType::float64) {
                throw AIException("Precision used in data is 'float' but the model uses 'double' ...");
            }
            this->input_fdata = ConvertData::totensor(data.cast<py::array_t<float>>());
        } 
        else {
            throw std::runtime_error("Unsupported dtype!");
        }
        this->normalize   = normalize;
        this->positional  = positional;

    } catch (const AIException& e) {
        std::cerr << "(Model:setData) Error: " << e.what() << std::endl;
    } catch (const std::exception& e) {
        // Catch standard exceptions
        std::cerr << "(Model:setData) Standard Error: " << e.what() << " at " << __LINE__ << std::endl;
    } catch (...) {
        // Catch all other exceptions
        std::cerr << "(Model:setData) Unknown Error:" << std::endl;
    }
}

/**************************************************************************************************
* ModelNode::setDecoderDataDouble and setDecoderDataFloat
* Temporarily store np.array (passed as dtype = np.float32) to a double pointer (this->input_fdata).
* Upon training entry, the double pointer will be transformed to an aitensor and handed over
* to the Node class.
**************************************************************************************************/
void ModelNode::setDecoderData(const py::array& data, bool const normalize, const bool positional) {

    try {

        py::buffer_info buf = data.request();

        if (buf.format == py::format_descriptor<double>::format()) {
            if (this->datatype == DataType::float32) {
                throw AIException("Precision used in decoder data is 'double' but the model uses 'float' ...");
            }
            py_cout << " Set Decoder Data in float64 format ... " << std::endl;
            this->decoder_ddata = ConvertData::totensor(data.cast<py::array_t<double>>());
        } 
        else if (buf.format == py::format_descriptor<float>::format()) {
            if (this->datatype == DataType::float64) {
                throw AIException("Precision used in decoder data is 'float' but the model uses 'double' ...");
            }
            py_cout << " Set Decoder Data in float32 format ... " << std::endl;
            this->decoder_fdata = ConvertData::totensor(data.cast<py::array_t<float>>());
        } 
        else {
            throw std::runtime_error("Unsupported dtype!");
        }

        this->normalize   = normalize;
        this->positional  = positional;

    } catch (const AIException& e) {
        std::cerr << "(Model:setDecoderData) Error: " << e.what() << std::endl;
    } catch (const std::exception& e) {
        // Catch standard exceptions
        std::cerr << "(Model:setDecoderData) Standard Error: " << e.what() << " at " << __LINE__ << std::endl;
    } catch (...) {
        // Catch all other exceptions
        std::cerr << "(Model:setDecoderData) Unknown Error:" << std::endl;
    }
}

/**************************************************************************************************
* ModelNode::setEncoderData and setEncoderData
* Temporarily store np.array (passed as dtype = np.float64) to a double pointer (this->input_ddata).
* Upon training entry, the double pointer will be transformed to an aitensor and handed over
* to the Node class.
**************************************************************************************************/
void ModelNode::setEncoderData(const py::array& data, bool const normalize, const bool positional) {

    try {
        py::buffer_info buf = data.request();

        if (buf.format == py::format_descriptor<double>::format()) {
            if (this->datatype == DataType::float32) {
                throw AIException("Precision used in encoder data is 'double' but the model uses 'float' ...");
            }
            py_cout << " Set Encoder Data in float64 format ... " << std::endl;
            this->encoder_ddata = ConvertData::totensor(data.cast<py::array_t<double>>());
        } 
        else if (buf.format == py::format_descriptor<float>::format()) {
            if (this->datatype == DataType::float64) {
                throw AIException("Precision used in encoder data is 'float' but the model uses 'double' ...");
            }
            py_cout << " Set Encoder Data in float32 format ... " << std::endl;
            this->encoder_fdata = ConvertData::totensor(data.cast<py::array_t<float>>());
        } 
        else {
            throw std::runtime_error("Unsupported dtype!");
        }

        this->normalize   = normalize;
        this->positional  = positional;

    } catch (const AIException& e) {
        std::cerr << "(Model:setEncoderData) Error: " << e.what() << std::endl;
    } catch (const std::exception& e) {
        // Catch standard exceptions
        std::cerr << "(Model:setEncoderData) Standard Error: " << e.what() << " at " << __LINE__ << std::endl;
    } catch (...) {
        // Catch all other exceptions
        std::cerr << "(Model:setEncoderData) Unknown Error:" << std::endl;
    }
}


/**************************************************************************************************
* ModelNode::setOperations
* Captures operations for the node and temporarily caches into the ModelNode instance. Later, it will
* be transfered to the main Node class as part of preparation for Model.train().
**************************************************************************************************/
void ModelNode::setOperations(std::vector<std::shared_ptr<BaseOperator>>& operations) {
    // Let's now seed with operations

    for (auto& op : operations) {
        // Check the dynamic type of the object using dynamic_cast
        if (auto linear = std::dynamic_pointer_cast<ModelLinear>(op)) {
            if (this->datatype == DataType::float32) {
                Linear<float>* newop = new Linear<float>(
                                                            linear->getSize(), 
                                                            linear->getBias()
                                                        );
                this->operations.push_back(newop);
            } else 
            if (this->datatype == DataType::float64) {
                Linear<double>* newop = new Linear<double>(
                                                            linear->getSize(), 
                                                            linear->getBias()
                                                        );
                this->operations.push_back(newop);
            }
        } else
        if (auto batchnorm = std::dynamic_pointer_cast<ModelBatchNorm>(op)) {
            if (this->datatype == DataType::float32) {
                BatchNorm<float>* newop = new BatchNorm<float>();
                this->operations.push_back(newop);
            } else 
            if (this->datatype == DataType::float64) {
                BatchNorm<double>* newop = new BatchNorm<double>();
                this->operations.push_back(newop);
            }
        } else
        if (auto layernorm = std::dynamic_pointer_cast<ModelLayerNorm>(op)) {
            if (this->datatype == DataType::float32) {
                LayerNorm<float>* newop = new LayerNorm<float>();
                this->operations.push_back(newop);
            } else 
            if (this->datatype == DataType::float64) {
                LayerNorm<double>* newop = new LayerNorm<double>();
                this->operations.push_back(newop);
            } 
        } else
        if (auto activate = std::dynamic_pointer_cast<ModelActivation>(op)) {
            if (this->datatype == DataType::float32) {
                Activation<float>* newop = new Activation<float>(
                                                            activate->getActivationType(), 
                                                            activate->getAlpha()
                                                        );
                this->operations.push_back(newop);
            } else 
            if (this->datatype == DataType::float64) { 
                Activation<double>* newop = new Activation<double>(
                                                            activate->getActivationType(), 
                                                            activate->getAlpha()
                                                        );
                this->operations.push_back(newop);
            }
        }  else  
        if (auto dropout = std::dynamic_pointer_cast<ModelDropout>(op)) {
            if (this->datatype == DataType::float32) {
                Dropout<float>* newop = new Dropout<float>(
                                                            dropout->getProbability()
                                                        );
                this->operations.push_back(newop);
            } else 
            if (this->datatype == DataType::float64) {
                Dropout<double>* newop = new Dropout<double>(
                                                            dropout->getProbability()
                                                        );
                this->operations.push_back(newop);
            }
        }  else
        if (auto flatten = std::dynamic_pointer_cast<ModelFlatten>(op)) {
            if (this->datatype == DataType::float32) {
                Flatten<float>* newop = new Flatten<float>();
                this->operations.push_back(newop);
            } else 
            if (this->datatype == DataType::float64) {
                Flatten<double>* newop = new Flatten<double>();
                this->operations.push_back(newop);
            }
        }  else
        if (auto convolution = std::dynamic_pointer_cast<ModelConvolution>(op)) {
            if (this->datatype == DataType::float32) {
                Convolution<float>* newop = new Convolution<float>(
                                                            convolution->getKernelSize(),
                                                            convolution->getStride(),
                                                            convolution->getPadding(),
                                                            convolution->getDilation(),
                                                            convolution->getBias()                                                                                                                                                                                                                                             
                                                        );
                this->operations.push_back(newop);
            } else 
            if (this->datatype == DataType::float64) { 
                Convolution<double>* newop = new Convolution<double>(
                                                            convolution->getKernelSize(),
                                                            convolution->getStride(),
                                                            convolution->getPadding(),
                                                            convolution->getDilation(),
                                                            convolution->getBias()   
                                                        );
                this->operations.push_back(newop);
            }
        }  else  // Transformer Component
        if (auto attention = std::dynamic_pointer_cast<ModelAttention>(op)) {
            if (this->datatype == DataType::float32) {
                Attention<float>* newop = new Attention<float>(
                                                            attention->getSize(),
                                                            attention->getBias()
                                                        );
                this->operations.push_back(newop);
            } else 
            if (this->datatype == DataType::float64) {
                Attention<double>* newop = new Attention<double>(
                                                            attention->getSize(),
                                                            attention->getBias(),
                                                            attention->getMasked()
                                                        );
                this->operations.push_back(newop);
            } 
        } else  // Transformer Component 
        if (auto feedforward = std::dynamic_pointer_cast<ModelFeedForward>(op)) {
            if (this->datatype == DataType::float32) {
                FeedForward<float>* newop = new FeedForward<float>(
                                                            feedforward->getSize(),
                                                            feedforward->getBias(),
                                                            feedforward->getActivationType(),
                                                            feedforward->getAlpha()
                                                        );
                this->operations.push_back(newop);
            } else 
            if (this->datatype == DataType::float64) {
                FeedForward<double>* newop = new FeedForward<double>(
                                                            feedforward->getSize(),
                                                            feedforward->getBias(),
                                                            feedforward->getActivationType(),
                                                            feedforward->getAlpha()
                                                        );
                this->operations.push_back(newop);
            }
        } else  // Transformer Component 
        if (auto multiheadattention = std::dynamic_pointer_cast<ModelMultiHeadAttention>(op)) {
            if (this->datatype == DataType::float32) {
                MultiHeadAttention<float>* newop = new MultiHeadAttention<float>(
                                                            multiheadattention->getHead(),
                                                            multiheadattention->getAttentionSize(),
                                                            multiheadattention->getBias(),
                                                            multiheadattention->getMasked()
                                                        );
                this->operations.push_back(newop);
            } else 
            if (this->datatype == DataType::float64) {
                MultiHeadAttention<double>* newop = new MultiHeadAttention<double>(
                                                            multiheadattention->getHead(),
                                                            multiheadattention->getAttentionSize(),
                                                            multiheadattention->getBias(),
                                                            multiheadattention->getMasked()
                                                        );
                this->operations.push_back(newop);
            }
        } else   // Transformer Component
        if (auto encoder = std::dynamic_pointer_cast<ModelEncoder>(op)) {
            if (this->datatype == DataType::float32) {
                EncoderLayer<float>* newop = new EncoderLayer<float>(
                                                            encoder->getHead(),
                                                            encoder->getAttentionSize(),
                                                            encoder->getFeedSize(),
                                                            encoder->getLayers(),
                                                            encoder->getBias(),
                                                            encoder->getActivationType(),
                                                            encoder->getAlpha()
                                                        );
                this->operations.push_back(newop);
            } else 
            if (this->datatype == DataType::float64) {
                EncoderLayer<double>* newop = new EncoderLayer<double>(
                                                            encoder->getHead(),
                                                            encoder->getAttentionSize(),
                                                            encoder->getFeedSize(),
                                                            encoder->getLayers(),
                                                            encoder->getBias(),
                                                            encoder->getActivationType(),
                                                            encoder->getAlpha()
                                                        );
                this->operations.push_back(newop);
            }
        } else   // Transformer Component 
        if (auto decoder = std::dynamic_pointer_cast<ModelDecoder>(op)) {
            if (this->datatype == DataType::float32) {
                DecoderLayer<float>* newop = new DecoderLayer<float>(
                                                            decoder->getHead(),
                                                            decoder->getAttentionSize(),
                                                            decoder->getFeedSize(),
                                                            decoder->getLayers(),
                                                            decoder->getBias(),
                                                            decoder->getActivationType(),
                                                            decoder->getAlpha()
                                                        );
                this->operations.push_back(newop);
            } else 
            if (this->datatype == DataType::float64) { 
                DecoderLayer<double>* newop = new DecoderLayer<double>(
                                                            decoder->getHead(),
                                                            decoder->getAttentionSize(),
                                                            decoder->getFeedSize(),
                                                            decoder->getLayers(),
                                                            decoder->getBias(),
                                                            decoder->getActivationType(),
                                                            decoder->getAlpha()
                                                        );
                this->operations.push_back(newop);
            }
        }  else   // Recurrent Network Component
        if (auto rnn = std::dynamic_pointer_cast<ModelRNN>(op)) {
            if (this->datatype == DataType::float32) {
                RNN<float>* newop = new RNN<float>(         rnn->getHiddenSize(),
                                                            rnn->getOuputSize(),
                                                            rnn->getOutputSequenceLength(), // For One-TO-Many Scenario
                                                            rnn->getNumLayers(),
                                                            rnn->getBiDirection(),
                                                            rnn->getRNNType()
                                                        );
                this->operations.push_back(newop);
            } else 
            if (this->datatype == DataType::float64) {
                RNN<double>* newop = new RNN<double>(       rnn->getHiddenSize(),
                                                            rnn->getOuputSize(),
                                                            rnn->getOutputSequenceLength(), // For One-TO-Many Scenario
                                                            rnn->getNumLayers(),
                                                            rnn->getBiDirection(),
                                                            rnn->getRNNType()
                                                        );
                this->operations.push_back(newop);
            }
        } else   // Recurrent Network Component
        if (auto lstm = std::dynamic_pointer_cast<ModelLSTM>(op)) {
            if (this->datatype == DataType::float32) {
                LSTM<float>* newop = new LSTM<float>(       lstm->getHiddenSize(),
                                                            lstm->getOuputSize(),
                                                            lstm->getOutputSequenceLength(), // For One-TO-Many Scenario
                                                            lstm->getNumLayers(),
                                                            lstm->getBiDirection(),
                                                            lstm->getRNNType()
                                                        );
                this->operations.push_back(newop);
            } else 
            if (this->datatype == DataType::float64) {
                LSTM<double>* newop = new LSTM<double>(     lstm->getHiddenSize(),
                                                            lstm->getOuputSize(),
                                                            lstm->getOutputSequenceLength(), // For One-TO-Many Scenario
                                                            lstm->getNumLayers(),
                                                            lstm->getBiDirection(),
                                                            lstm->getRNNType()
                                                        );
                this->operations.push_back(newop);
            }
        } else   // Recurrent Network Component
        if (auto gru = std::dynamic_pointer_cast<ModelGRU>(op)) {
            if (this->datatype == DataType::float32) {
                GRU<float>* newop = new GRU<float>(         gru->getHiddenSize(),
                                                            gru->getOuputSize(),
                                                            gru->getOutputSequenceLength(), // For One-TO-Many Scenario
                                                            gru->getNumLayers(),
                                                            gru->getBiDirection(),
                                                            gru->getRNNType()
                                                        );
                this->operations.push_back(newop);
            } else 
            if (this->datatype == DataType::float64) {
                GRU<double>* newop = new GRU<double>(       gru->getHiddenSize(),
                                                            gru->getOuputSize(),
                                                            gru->getOutputSequenceLength(), // For One-TO-Many Scenario
                                                            gru->getNumLayers(),
                                                            gru->getBiDirection(),
                                                            gru->getRNNType()
                                                        );
                this->operations.push_back(newop);
            }
        } 
    }
}

/************************************************************************************************
* Model::Model
* This is the Model Constructor Implementation - Note that this is only a meta model.
* The actual model is the BaseModel Class.
*************************************************************************************************/
Model::Model(DataType dtype, int seed) {

    this->datatype = dtype;

    if (this->datatype == DataType::float32) {
        std::shared_ptr<BaseModel<float>> bmodelf = std::make_shared<BaseModel<float>>(seed);
        std::shared_ptr<Graph<float>> graphXf = std::make_unique<Graph<float>>();
        bmodelf->setGraph(graphXf);
        this->modelXf = bmodelf;
    } else if (this->datatype == DataType::float64) {
        std::shared_ptr<BaseModel<double>> bmodeld = std::make_shared<BaseModel<double>>(seed);
        std::shared_ptr<Graph<double>> graphXd = std::make_unique<Graph<double>>();
        bmodeld->setGraph(graphXd);
        this->modelXd = bmodeld;
    } else {
        throw std::invalid_argument("Unsupported datatype");
    }
}

/************************************************************************************************
* Model::addNode
* First, we create a meta node to support python API. Alongside, we also create the node 
* and register to the graph.
*************************************************************************************************/
std::shared_ptr<ModelNode> Model::addNode(std::string name, NodeType ntype) {
    if (this->datatype == DataType::float32) {
        if (!isNode(name)) { 
            std::shared_ptr<ModelNode> node = std::make_shared<ModelNode>(name, ntype, this->datatype);
            this->modelXf->getGraph()->createNode(name, ntype);
            this->nodes.push_back(node);
            return node;
        } else {
            std::cerr << "Node already exists. Use another name ..." << std::endl;
        }
    } else if (this->datatype == DataType::float64) {
        if (!isNode(name)) {
            std::shared_ptr<ModelNode> node = std::make_shared<ModelNode>(name, ntype, this->datatype);
            this->modelXd->getGraph()->createNode(name, ntype);
            this->nodes.push_back(node);
            return node;
        } else {
            std::cerr << "Node already exists. Use another name ..." << std::endl;
        }
    } else {
        throw std::invalid_argument("Unsupported datatype");
    }
    return nullptr;
}

/************************************************************************************************
* Model::connect
* Connect nodes in the graph.
*************************************************************************************************/
void Model::connect(std::shared_ptr<ModelNode> from,std::shared_ptr<ModelNode> to) {
    try {
        if (from == nullptr) {
        throw AIException(" Source node is missing ...");
        }
        if (to == nullptr) {
        throw AIException(" Target node is missing ...");
        }
        if (this->datatype == DataType::float32) {
            this->modelXf->getGraph()->connect(from->getName(), to->getName());
        } else
        if (this->datatype == DataType::float64) {
            this->modelXd->getGraph()->connect(from->getName(), to->getName());
        }
    } catch (const AIException& e) {
        std::cerr << "(Model:connect) Error: " << e.what() << std::endl;
    } catch (const std::exception& e) {
        // Catch standard exceptions
        std::cerr << "(Model:connect) Standard Error: " << e.what() << " at " << __LINE__ << std::endl;
    } catch (...) {
        // Catch all other exceptions
        std::cerr << "(Model:connect) Unknown Error:" << std::endl;
    }
}

void Model::connect(std::vector<std::shared_ptr<ModelNode>> from_nodes, std::shared_ptr<ModelNode> to) {
    for (auto& from : from_nodes) {
        this->connect(from, to);
    }
}

void Model::connect(std::shared_ptr<ModelNode> from, std::vector<std::shared_ptr<ModelNode>> to_nodes) {
    for (auto& to : to_nodes) {
        this->connect(from, to);
    }
}

/************************************************************************************************
* Model::seedNodes
* Here, we begin to fill the nodes with data and operations as assigned to them.
*************************************************************************************************/

void Model::seedNodes(bool setOps) {
    for (auto& node: nodes) {
        // First, let's seed with data
        ssize_t size = node->getDataSize();

        // First, let's seed with decoder data (For Transformer Decoders)
        ssize_t dsize = node->getDecoderDataSize();
        ssize_t esize = node->getEncoderDataSize();

        // set Node Input Data
        if (size != 0) {
            if (this->datatype == DataType::float32) {
                this->modelXf->getGraph()->setData(node->getName(), node->getDataFloat(), node->getNormalize(), node->getPositional());
            } else
            if (datatype == DataType::float64) {
                this->modelXd->getGraph()->setData(node->getName(), node->getDataDouble(), node->getNormalize(), node->getPositional());
            }
        }

        // set Node Decoder Data (For Transformer Decoders)
        if (dsize != 0) {
            if (this->datatype == DataType::float32) {
                this->modelXf->getGraph()->setDecoderData(node->getName(), node->getDecoderDataFloat(), node->getNormalize(), node->getPositional());
            } else
            if (this->datatype == DataType::float64) {
                this->modelXd->getGraph()->setDecoderData(node->getName(), node->getDecoderDataDouble(), node->getNormalize(), node->getPositional());
            }
        }

        // set Node Decoder Data (For Transformer Decoders)
        if (esize != 0) {
            if (this->datatype == DataType::float32) {
                this->modelXf->getGraph()->setEncoderData(node->getName(), node->getEncoderDataFloat(), node->getNormalize(), node->getPositional());
            } else
            if (this->datatype == DataType::float64) {
                this->modelXd->getGraph()->setEncoderData(node->getName(), node->getEncoderDataDouble(), node->getNormalize(), node->getPositional());
            }
        }

        if (setOps == true) {
            if (this->datatype == DataType::float32) {
                this->modelXf->getGraph()->setOperations(node->getName(), node->getOperations());
            } else
            if (this->datatype == DataType::float64) {
                this->modelXd->getGraph()->setOperations(node->getName(), node->getOperations());
            }
        }
    } 
}

/************************************************************************************************
* Model::setTarget
* We use modelXf.setTarget to convert the python array to aitensor<float> and store
* the tensor inside the model.
*************************************************************************************************/
void Model::setTarget(const py::array& target, const bool normalize) {
    py::buffer_info buf = target.request();
    try {
        if (buf.format == py::format_descriptor<double>::format()) {
            if (this->datatype == DataType::float32) {
                throw AIException("Precision used in target data is 'double' but the model uses 'float' ...");
            }
            this->modelXd->setTarget(target.cast<py::array_t<double>>(), normalize);
        } 
        else if (buf.format == py::format_descriptor<float>::format()) {
            if (this->datatype == DataType::float64) {
                throw AIException("Precision used in target data is 'float' but the model uses 'double' ...");
            }
            this->modelXf->setTarget(target.cast<py::array_t<float>>(), normalize);
        } 
        else {
            throw std::runtime_error("Unsupported dtype!");
        }
    } catch (const AIException& e) {
        std::cerr << "(Model:setTargetFloat) Error: " << e.what() << std::endl;
    } catch (const std::exception& e) {
        // Catch standard exceptions
        std::cerr << "(Model:setTargetFloat) Standard Error: " << e.what() << " at " << __LINE__ << std::endl;
    } catch (...) {
        // Catch all other exceptions
        std::cerr << "(Model:setTargetFloat) Unknown Error:" << std::endl;
    }
}

/************************************************************************************************
* Model::predictFloat
* We use modelXd.setTarget to convert the python array to aitensor<double> and store
* the tensor inside the model.
*************************************************************************************************/
py::array Model::predict(int sequence_length) {
    py_cout << "Entering Prediction Float ...";

    // Check the dtype of the input argument
    // std::string dtype_name = dtype.attr("name").cast<std::string>();
    // std::cout << "Requested dtype: " << dtype_name << std::endl;

    try {
        this->seedNodes(false);
        if (this->datatype == DataType::float32) {
            aitensor<float> tensor = this->modelXf->predict(sequence_length); 
            return ConvertData::topyarray(tensor); 
        } else
        if (this->datatype == DataType::float64) {
            aitensor<double> tensor = this->modelXd->predict(sequence_length);
            return ConvertData::topyarray(tensor); 
        }

    } catch (const AIException& e) {
        std::cerr << "(Model:predictFloat) Error: " << e.what() << std::endl;
    } catch (const std::exception& e) {
        // Catch standard exceptions
        std::cerr << "(Model:predictFloat)Standard Error: " << e.what() << " at " << __LINE__ << std::endl;
    } catch (...) {
        // Catch all other exceptions
        std::cerr << "(Model:predictFloat) Unknown Error:" << std::endl;
    }
    return py::array_t<float>();
}
  
/*
py::array_t<double> Model::predictDouble(int sequence_length, const py::dtype dtype) {
    py_cout << "Entering Prediction Double ...";

    // Check the dtype of the input argument
    std::string dtype_name = dtype.attr("name").cast<std::string>();
    std::cout << "Requested dtype: " << dtype_name << std::endl;

    try {
        if (datatype == "float") {
            throw AIException("Precision used in target data is 'double' but the model uses 'float' ...");
        }

        this->seedNodes(false);
        aitensor<double> tensor = this->modelXd->predict(sequence_length);  
        return ConvertData::topyarray(tensor); 

    } catch (const AIException& e) {
        std::cerr << "(Model:predictDouble) Error: " << e.what() << std::endl;
    } catch (const std::exception& e) {
        // Catch standard exceptions
        std::cerr << "(Model:predictDouble)Standard Error: " << e.what() << " at " << __LINE__ << std::endl;
    } catch (...) {
        // Catch all other exceptions
        std::cerr << "(Model:predictDouble) Unknown Error:" << std::endl;
    }
    return py::array_t<double>();
}
*/


/************************************************************************************************
* Model::compile
* This is where configuration begins.
*************************************************************************************************/
 
/************************************************************************************************
* Model::generateDotFormat
* Show the graph of the neural network in Dot Format
*************************************************************************************************/
Topology Model::generateDotFormat(bool operators, bool weights) {
        if (this->datatype == DataType::float32) {
            return this->modelXf->getGraph()->generateDotFormat(operators, weights);
        } else
        if (this->datatype == DataType::float64) {
            return this->modelXd->getGraph()->generateDotFormat(operators, weights);
        }
        return Topology();
}

/************************************************************************************************
* Model::train
* This is where training begins. We train the actual model by passing hyperparameters.
*************************************************************************************************/
std::vector<float> Model::train(std::string& losstype, std::vector<std::string>& metricstype, std::string& optimizertype, int batch_size, 
                const int max_epoch, const double learn_rate , const bool use_step_decay, const double decay_rate) {
    try {
        this->seedNodes(true);
        if (this->datatype == DataType::float32) {
            return this->modelXf->train(losstype, metricstype, optimizertype, batch_size,  max_epoch,  
                                  static_cast<float>(learn_rate), use_step_decay, static_cast<float>(decay_rate));
        } else
        if (this->datatype == DataType::float64) {
            return this->modelXd->train(losstype, metricstype, optimizertype, batch_size,  max_epoch, 
                                  static_cast<double>(learn_rate), use_step_decay, static_cast<double>(decay_rate));
        }
    } catch (const AIException& e) {
        std::cerr << "(Model::train) Error: " << e.what() << std::endl;
    } catch (const std::exception& e) {
        // Catch standard exceptions
        std::cerr << "(Model:train) Standard Error: " << e.what() << " at " << __LINE__ << std::endl;
    } catch (...) {
        // Catch all other exceptions
        std::cerr << "(Model:train) Unknown Error:" << std::endl;
    }

    return std::vector<float>();
}


/*
void trainModel(const py::list& repeatSequentially, const py::list& repeatCounts) {

    std::vector<bool> repeatSeq;
    std::vector<int> repeatCnt;

    for (const auto& item : repeatSequentially) {
        repeatSeq.push_back(py::cast<bool>(item));
    }

    for (const auto& item : repeatCounts) {
        repeatCnt.push_back(py::cast<int>(item));
    }

    std::vector<Node*> nodes = this->getNodes();

    // Sort nodes topologically
    std::vector<Node*> sortedNodes = topologicalSort(nodes);

    // Group sorted nodes into sequential regions
    std::vector<std::vector<Node*>> regions = groupNodes(sortedNodes);

    // Repeat regions based on repeatSequentially and repeatCounts
    std::vector<std::vector<Node*>> repeatedRegions = repeatRegions(regions, repeatSeq, repeatCnt);

        // Initialize MPI
    MPI_Init(NULL, NULL);

    // Execute Pipeline
    //executePipelineParallel(repeatedRegions);

    executePipelineParallel();

    // Perform data exchange between regions based on connections
    // performDataExchange(graph, repeatedRegions);

    // Gather gradients from all regions
    // std::vector<double> aggregatedGradients(nodes.size());
    // gatherGradients(repeatedRegions, aggregatedGradients);

    // Aggregate gradients and perform parameter update
    // aggregateGradients(aggregatedGradients);
    // performParameterUpdate(nodes);

    // Finalize MPI
    MPI_Finalize();

}
*/


// Function that processes any NumPy array and doubles its values
template <typename T>
py::array_t<T> process_array_t(py::array_t<T> arr) {
    py::buffer_info buf = arr.request();
    size_t size = buf.size;

    // Create a new NumPy array for results
    auto result = py::array_t<T>({(pybind11::ssize_t) size});
    py::buffer_info res_buf = result.request();

    // Get raw pointers
    T* input_ptr = static_cast<T*>(buf.ptr);
    T* result_ptr = static_cast<T*>(res_buf.ptr);

    // Process: Multiply each element by 2
    for (size_t i = 0; i < (size_t) size; ++i) {
        result_ptr[i] = input_ptr[i] * 2;
    }

    return result;
}

// Generic wrapper to dispatch based on dtype
py::array Model::process_array(py::array arr) {
    py::buffer_info buf = arr.request();

    if (buf.format == py::format_descriptor<double>::format()) {
        return process_array_t<double>(arr.cast<py::array_t<double>>());
    } 
    else if (buf.format == py::format_descriptor<float>::format()) {
        return process_array_t<float>(arr.cast<py::array_t<float>>());
    } 
    else if (buf.format == py::format_descriptor<int>::format()) {
        return process_array_t<int>(arr.cast<py::array_t<int>>());
    } 
    else {
        throw std::runtime_error("Unsupported dtype!");
    }
}

/*
// Generic wrapper to dispatch based on dtype
py::array_t<double> Model::process_xarray(py::array_t<double> arr) {
    py::buffer_info buf = arr.request();

    if (buf.format == py::format_descriptor<double>::format()) {
        return process_array_t<double>(arr.cast<py::array_t<double>>());
    } 
    else if (buf.format == py::format_descriptor<float>::format()) {
        return process_array_t<float>(arr.cast<py::array_t<float>>());
    } 
    else if (buf.format == py::format_descriptor<int>::format()) {
        return process_array_t<int>(arr.cast<py::array_t<int>>());
    } 
    else {
        throw std::runtime_error("Unsupported dtype!");
    }
}

py::array Model::process_xtype(py::array arr) {
    py::buffer_info buf = arr.request();

    if (buf.format == py::format_descriptor<double>::format()) {
        return process_array_t<double>(arr.cast<py::array_t<double>>());
    } 
    else if (buf.format == py::format_descriptor<float>::format()) {
        return process_array_t<float>(arr.cast<py::array_t<float>>());
    } 
    else if (buf.format == py::format_descriptor<int>::format()) {
        return process_array_t<int>(arr.cast<py::array_t<int>>());
    } 
    else {
        throw std::runtime_error("Unsupported dtype!");
    }
}

// Function to return an array with the dtype passed
py::array Model::array_with_dtype(py::dtype dtype) {
    // Check the dtype of the input argument
    std::string dtype_name = dtype.attr("name").cast<std::string>();
    std::cout << "Requested dtype: " << dtype_name << std::endl;

    if (dtype_name == DataType::float32) {
        // Create an array of np.float32
        py::array_t<float> result = py::array_t<float>({5}); // Array size of 5 for example
        auto buf = result.request();
        float* ptr = static_cast<float*>(buf.ptr);
        
        // Initialize values (for example, fill with 1.0)
        for (size_t i = 0; i < 5; ++i) {
            ptr[i] = 1.0f;
        }
        return result;
    }
    else if (dtype_name == DataType::float64) {
        // Create an array of np.float64
        py::array_t<double> result = py::array_t<double>({5}); // Array size of 5 for example
        auto buf = result.request();
        double* ptr = static_cast<double*>(buf.ptr);
        
        // Initialize values (for example, fill with 1.0)
        for (size_t i = 0; i < 5; ++i) {
            ptr[i] = 1.0;
        }
        return result;
    }
    else if (dtype_name == "int32") {
        // Create an array of np.int32
        py::array_t<int> result = py::array_t<int>({5}); // Array size of 5 for example
        auto buf = result.request();
        int* ptr = static_cast<int*>(buf.ptr);
        
        // Initialize values (for example, fill with 1)
        for (size_t i = 0; i < 5; ++i) {
            ptr[i] = 1;
        }
        return result;
    }
    else {
        throw std::runtime_error("Unsupported dtype!");
    }
}
*/