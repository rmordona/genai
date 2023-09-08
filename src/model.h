/*
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

#include "topology.h"

#ifndef BASEMODEL_H
#define BASEMODEL_H


template <class T>
class BaseModel {
private:
    Graph<T>* graph;
    aitensor<T> target;
    aitensor<T> predicted;
    aitensor<T> gradients;
    aiscalar<T> loss;
    std::string losstype = "mse";
    std::string optimizertype = "adam";
    T learningRate = 0.01;
    int itermax = 1;
public:
    BaseModel(const std::string& losstype = "mse", const std::string& optimizertype = "adam", 
          const T learningRate = 0.01, const int itermax = 1) {
        this->losstype = losstype;
        this->optimizertype = optimizertype;
        this->learningRate = learningRate;
        this->itermax = itermax;
    }

    void setGraph(Graph<T>* graph);

    Graph<T>* getGraph();

    void setLoss(std::string& losstype);

    // The input is assumed to have NxM where N=number of samples, M=embedding vector size
    // This allows to compute for the output size,  MxW where W is the number of weights (features) to use.
    void setTarget(const py::array_t<T>& target);

    aitensor<T> getTarget();

    void useCrossEntropy();

    void train(std::string& losstype, std::string& optimizertype, T& learningRate = 0.01, int& itermax = 1);

};

class Model {
private:
    std::string losstype = "mse";
    std::string optimizertype = "adam";
    double learningRate = 0.01;
    int itermax = 1;
    std::string datatype = "float";
    Graph<float> graphXf;
    Graph<double> graphXd;
    BaseModel<float> modelXf;
    BaseModel<double> modelXd;
public:
    Model(const std::string& losstype = "mse", const std::string& optimizertype = "adam", 
          const double learningRate = 0.01, const int itermax = 1, const std::string& datatype = "float") {
        this->losstype = losstype;
        this->optimizertype = optimizertype;
        this->learningRate = learningRate;
        this->itermax = itermax;
        this->datatype = datatype;
        if (datatype == "float") {
            BaseModel<float> mymodelf(losstype, optimizertype, (float) learningRate, itermax);
            this->modelXf = mymodelf;
        } else if (datatype == "double") {
            BaseModel<double> mymodeld(losstype, optimizertype, (double) learningRate, itermax);
            this->modelXd = mymodeld;
        } else {
            throw std::invalid_argument("Unsupported datatype");
        }
        createGraph();
    }

    void createGraph() {
        if (datatype == "float") {
            graphXf = Graph<float>();
            modelXf.setGraph(&graphXf);
        } else if (datatype == "double") {
            graphXd = Graph<double>();
            modelXd.setGraph(&graphXd);
        } else {
            throw std::invalid_argument("Unsupported datatype");
        }
    }

    void addNode(const std::string& name, NodeType ntype, std::vector<std::shared_ptr<BaseOperator>>& operations) {
        if (datatype == "float") {
            Node<float>* node1 =  (modelXf.getGraph())->createNode(name, ntype);
            node1->setOperations(operations);
        } else if (datatype == "double") {
            Node<double>* node1 =  (modelXd.getGraph())->createNode(name, ntype);
            node1->setOperations(operations);
        } else {
            throw std::invalid_argument("Unsupported datatype");
        }
    }

    void setData(const std::string& name, NodeType ntype, const py::array_t<float>& input_dataf, const py::array_t<double>& input_datad) {
        if (datatype == "float") {
            Node<float>* node1 =  (modelXf.getGraph())->createNode(name, ntype);
            //node1->setData(input_dataf);
            node1->setData(input_dataf);
        } else if (datatype == "double") {
            Node<double>* node1 =  (modelXd.getGraph())->createNode(name, ntype);
            //node1->setData(input_datad);
            node1->setData(input_datad);
        } else {
            throw std::invalid_argument("Unsupported datatype");
        }
    }

    template <class T>
    void connect(Node<T>* from, Node<T>* to) {
        addConnection(new Connection<T>(from, to));
    }

    template <class T>
    void connect(Node<T>* from, Node<T>* to, std::vector<std::shared_ptr<BaseOperator>>& operations) {
        to->setOperations(operations);
        addConnection(new Connection<T>(from, to));
    }

    template <class T>
    void connect(std::vector<Node<T>*> from_nodes, Node<T>* to) {
        for (Node<T>* from : from_nodes) {
            addConnection(new Connection<T>(from, to));
        }
    }


};

/************ Base Model initialize templates ************/

template class BaseModel<float>;  // Instantiate with float
template class BaseModel<double>;  // Instantiate with double


#endif
