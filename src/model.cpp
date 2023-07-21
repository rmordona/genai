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


void BaseModel::setGraph(Graph* graph) {

    auto mygraph = dynamic_cast<Graph*>(graph);
    this->graph = mygraph;
    std::cout << "(setGraph) See the size ...\n";
    std::cout << this->graph->getNodes().size() << " all size account ... \n";
}

void BaseModel::setLoss(std::string& losstype) {
    this->losstype = losstype;
}

// The input is assumed to have NxM where N=number of samples, M=embedding vector size
// This allows to compute for the output size,  MxW where W is the number of weights (features) to use.
void BaseModel::setTarget(py::array_t<double> target) {

    std::cout << "Set Target ...\n";
    std::cout << target << "\n";

    // Convert values to C++ array
    py::buffer_info values_info = target.request();
    double* data = static_cast<double*>(values_info.ptr);
    int v_rows = values_info.shape[0]; // N
    int v_cols = values_info.shape[1]; // M
    // Convert a py::array_t row-major order to an Eigen::MatrixXd column-major order.
    this->target = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(data, v_rows, v_cols);

    std::cout << this->target << "\n";
}

Eigen::MatrixXd BaseModel::getTarget() {
    return target;
}

void BaseModel::useCrossEntropy() {

}

void BaseModel::train(std::string& losstype, std::string& optimizertype, double learnrate , int itermax) {

        // Initialize MPI
    //MPI_Init(NULL, NULL);

    this->losstype = losstype;
    this->optimizertype = optimizertype;
    this->learningRate = learnrate;

    std::cout << "(train) See the size ...\n";
    std::cout << this->graph->getNodes().size() << " all size account ... \n";

    double epsilon = 1e-3;
    double old_loss = inf();

    int iter = 0;

    auto start_time = std::chrono::system_clock::now();

    do {

        std::cout << "<<<<<<<<<<<<<<<<<<<<<<<<< **************** Process batch (iteration " << (iter+1) << ") ************* >>>>>>>>>>>>>>>>>>>>>>>>>> \n";
        this->graph->nextBatch();

        std::cout << "Entering Forward Propagation ...\n";
        predicted = this->graph->forwardPropagation();

        std::cout << "Predicted Result: \n";
        std::cout << predicted << "\n";

        std::cout << "Compute Loss ...\n";
        loss = this->graph->computeLoss(losstype, predicted, target);

        auto end_time = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end_time - start_time;
        std::time_t next_time = std::chrono::system_clock::to_time_t(end_time);
        start_time = end_time;
        std::cout << "---------------------------------------------------------------------------------> Loss:" 
                    << loss 
                    << " ... elapsed " <<  elapsed_seconds.count() 
                    << " at " << std::ctime(&next_time) << "\n";

        std::cout << "Compute Gradient ...\n";
        gradients = this->graph->computeGradients(losstype, predicted, target);
        std::cout << gradients << "\n";

        std::cout << "Entering Backward Propagation ...\n";
        gradients = this->graph->backwardPropagation(gradients); 

        std::cout << "\n\n\n";
        std::cout << "Updating Parameters ...\n";
        this->graph->updateParameters(this->optimizertype, this->learningRate, iter);

        iter ++;

        if (iter >= itermax) break; 



    } while (abs(old_loss - loss(0,0)) > epsilon);

    std::cout << "Training done ...\n";

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
