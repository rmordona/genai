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

namespace py = pybind11;
using namespace py::literals;


template <class T>
void BaseModel<T>::setGraph(Graph<T>* graph) {

    auto mygraph = dynamic_cast<Graph<T>*>(graph);
    this->graph = mygraph;
    log_detail( "SetGraph ... Node count: {:d}", this->graph->getNodes().size() );
}

template <class T>
Graph<T>* BaseModel<T>::getGraph( ) {
    return this->graph;
}

template <class T>
void BaseModel<T>::setLoss(std::string& losstype) {
    this->losstype = losstype;
}

// The input is assumed to have NxM where N=number of samples, M=embedding vector size
// This allows to compute for the output size,  MxW where W is the number of weights (features) to use.
template <class T>
void BaseModel<T>::setTarget(const py::array_t<T>& target) {

    log_info("=================");
    log_info( "Setting Data (Tensor) ..." );

    // request a buffer descriptor from Python
    py::buffer_info buffer_info = target.request();

    // extract data an shape of input array
    T* data = static_cast<T *>(buffer_info.ptr);
    std::vector<ssize_t> shape = buffer_info.shape;

    // Get the dimensions of the py::array_t
    ssize_t dim0 = shape[0]; // Batch Size
    ssize_t dim1 = shape[1]; // Input Size
    ssize_t dim2 = shape[2]; // Parameter / Embedding Size

    // Create an Eigen::Map to map the raw data to an Eigen::Tensor
    // Eigen::Map<Eigen::Tensor<T,3>> tensor_map(data, dim0, dim1, dim2);

    aitensor<T> tensor(dim0, dim1, dim2);

    // auto ptr = static_cast<T*>(info.ptr);
    for (int i = 0; i < dim0; ++i) {
        for (int j = 0; j < dim1; ++j) {
            for (int k = 0; k < dim2; ++k) {
                tensor(i, j, k) = *data++;
            }
        }
    }
    this->target = tensor;

}


template <class T>
aitensor<T> BaseModel<T>::getTarget() {
    return target;
}

template <class T>
void BaseModel<T>::useCrossEntropy() {

}

template <class T>
void BaseModel<T>::train(std::string& losstype, std::string& optimizertype, T& learningRate , int& itermax) {

        // Initialize MPI
    //MPI_Init(NULL, NULL);

    this->losstype = losstype;
    this->optimizertype = optimizertype;
    this->learningRate = learningRate;

    log_info( "******************************************************************************************" );
    log_info( "********************************* Start Training *****************************************")
    log_info( "******************************************************************************************" );
    log_detail( "Number of Graph Nodes: {:d}", this->graph->getNodes().size() );

    aiscalar<T> epsilon = 1e-3;
    aiscalar<T> old_loss = inf();

    int iter = 0;

    auto start_time = std::chrono::system_clock::now();

    py_cout << "Starting Iteration ..." << std::endl;

    do {

        log_detail( "<<<<<<<<<<<<<<<<<<<<<<<<< Process batch (iteration {:d})  >>>>>>>>>>>>>>>>>>>>>>>>>>", (iter + 1) );
        this->graph->nextBatch();

        log_detail( "Entering Forward Propagation ..." );
        this->predicted = this->graph->forwardPropagation();

        log_detail( "Predicted Result" );
        log_matrix( this->predicted );

        this->loss = this->graph->computeLoss(this->losstype, this->predicted, this->target); 

        log_detail( "Compute Gradient ..." );
        this->gradients = this->graph->computeGradients(this->predicted, this->target);
        log_matrix( this->gradients );

        log_detail( "Entering Backward Propagation ..." );
        this->gradients = this->graph->backwardPropagation(this->gradients); 

        log_detail( "Updating Parameters ..." );
        this->graph->updateParameters(this->optimizertype, this->learningRate, iter);

        // Calculate Time, then display loss
        auto end_time = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end_time - start_time;
        std::time_t next_time = std::chrono::system_clock::to_time_t(end_time);
        start_time = end_time;
        py_cout << "-------> Compute Loss:";
        py_cout << this->loss;
        py_cout << " ... elapsed " <<  elapsed_seconds.count();
        py_cout << " at " << std::ctime(&next_time) << std::endl;

        // Also, log the result if Logging INFO is enabled
        log_detail( "Compute Loss ... {:8.5f} ... Elapsed {} at {}", this->loss,  elapsed_seconds.count(), std::ctime(&next_time) );

        iter ++;

        if (iter >= itermax) break;

    } while (abs(old_loss - this->loss) > epsilon);

    log_detail( "Training done ..." );

    // Finalize MPI
    //MPI_Finalize();

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
