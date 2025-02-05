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

#ifndef EMBEDDINGS_H
#define EMBEDDINGS_H

#include <sqlite3.h>
#include "logger.h"
#include <random>

#define TK_SPACE_ L"<s>"    
#define TK_MASK_  L"<MASK>"
#define TK_UNK_   L"<UNK>"
#define TK_PAD_   L"<PAD>"
#define TK_SOS_   L"<SOS>"
#define TK_EOS_   L"<EOS>"

/************************************************************************************************************************************
* Base Distance Class
* This class provides different methods to calculate the distance between two vectors.
************************************************************************************************************************************/
template <class T>
class Distance {
private:
    aimatrix<T> wordEmbeddings;
public:
    Distance() {}

    // Euclidean Distance
    static aiscalar<T> euclidean(airowvector<T> token1, airowvector<T> token2) {
        return (token1 - token2).norm(); 
    }

    // Euclidean Distance
    static aiscalar<T> manhattan(airowvector<T> token1, airowvector<T> token2) {
        return  0;
    }

    // FINGER - Fast Inference for Graph-based Apprxoimate Nearest Neighbor
    static aiscalar<T> finger(airowvector<T> token1, airowvector<T> token2) {
        return 0;
    }

};


/************************************************************************************************************************************
********  HNSW represents the Hierarchical Navigational Small World graph ***********************************************************
*
*  Intricacies of the Heirarchical Navigable Small World (HNSW)

* Layered Graph Structure: HNSW organizes data into multiple layers, and each layer is a graph that represents a 
*   different level of granularity. Nodes in lower layers act as shortcuts to nodes in higher layers, creating a hierarchical 
*   structure.
*
* Neighbor Exploration: During insertion, HNSW uses a process called "neighbor exploration" to find and connect 
*   nodes with similar embeddings. This process involves traversing the graph efficiently to identify potential neighbors.
*
* Link Pruning: HNSW employs strategies to prune links between nodes based on distances, optimizing the graph structure. 
*   Pruning helps maintain the navigability of the graph while reducing the number of connections.
*
* Heuristic Search: During the nearest neighbor search, HNSW uses heuristics to efficiently explore the graph and find 
*   approximate nearest neighbors without exhaustive search.
*
* Dynamic Update: HNSW supports dynamic updates, allowing the graph to adapt to changes in the dataset, 
*   such as insertions or deletions of nodes.
*
* Tuning Parameters: HNSW involves tuning parameters such as the number of neighbors for each node, the number of 
*   layers, and the level of link pruning. Optimal parameter settings depend on the characteristics of the data and the specific
*    use case.
*
* Memory Efficiency: In practice, memory efficiency is crucial. HNSW is designed to be memory-efficient, 
*   considering factors like the size of the graph and the amount of memory required for each node.
*
* Parallelization: To enhance performance, some implementations of HNSW leverage parallelization techniques 
*   for certain operations.
*
* Concurrency Control: In concurrent scenarios (e.g., in a multi-threaded environment), HNSW may require additional 
*   considerations for thread safety.
*
* Optimization for Specific Metrics: Depending on the application, HNSW may be optimized for specific distance metrics, 
*   such as Euclidean distance or cosine similarity.
*
************************************************************************************************************************************/

template <class T>
class HNSW {
private:
    struct Node {
        airowvector<T> embedding;
        std::vector<int> neighbors;
        int layer;
    };

    int embedding_size_;
    int max_connections_;
    std::vector<Node> nodes_;

    // Choose a layer for a new node
    int chooseLayer(int nodeId) {
        // Placeholder implementation for simplicity
        // In a real HNSW, this would involve more sophisticated strategies

        // Example: Assign layers based on node density
        const int densityThreshold = 5;  // Adjust the density threshold as needed

        // Calculate the density of nodes in the current layer
        int currentLayer = nodeId % 2;  // Alternating layers
        int nodesInCurrentLayer = countNodesInLayer(currentLayer);

        // If the density in the current layer exceeds the threshold, switch to the other layer
        if (nodesInCurrentLayer > densityThreshold) {
            return (currentLayer + 1) % 2;  // Switch to the other layer
        } else {
            return currentLayer;  // Stay in the current layer
        }
    }

    // Count the number of nodes in a given layer
    int countNodesInLayer(int layer) {
        int count = 0;
        for (const auto& node : nodes_) {
            if (node.layer == layer) {
                ++count;
            }
        }
        return count;
    }

    // Build connections for a newly added node
    void buildConnections(int nodeId) {

        // Neighbor exploration
        for (int i = 0; i < nodeId; ++i) {
            double distance = Distance<T>::euclidean(nodes_[nodeId].embedding, nodes_[i].embedding);

            // Link pruning
            if (distance < threshold(nodes_[nodeId].layer)) {
                nodes_[nodeId].neighbors.push_back(i);
                nodes_[i].neighbors.push_back(nodeId);
            }
        }

        // Heuristic search
        sortNeighborsByDistance(nodeId);

        // Limit connections
        limitConnections(nodeId);
    }

    // Compute Euclidean distance between two Eigen vectors
    double computeEuclideanDistance(const Eigen::VectorXd& v1, const Eigen::VectorXd& v2) {
        return (v1 - v2).norm();
    }

    // Calculate threshold based on layer
    double threshold(int layer) {
        return 0.5;  // Adjust as needed
    }

    // Sort neighbors by distance with additional optimizations
    void sortNeighborsByDistance(int nodeId) {
        auto& currentNode = nodes_[nodeId];

        // Early stopping threshold for the number of comparisons
        const int maxComparisons = std::min(100, static_cast<int>(currentNode.neighbors.size()));

        // Adaptive threshold based on the number of neighbors
        double adaptiveThreshold = 0.1 * std::sqrt(currentNode.neighbors.size());

        // Create a vector to store distances, layers, and indices for sorting
        std::vector<std::tuple<int, double, int>> layerDistanceIndexTuples;
        layerDistanceIndexTuples.reserve(currentNode.neighbors.size());

        // Compute layers, distances, and store in the vector
        for (int neighborId : currentNode.neighbors) {
            int neighborLayer = nodes_[neighborId].layer;
            double distance = Distance<T>::euclidean(currentNode.embedding, nodes_[neighborId].embedding);
            layerDistanceIndexTuples.emplace_back(neighborLayer, distance, neighborId);
        }

        // Example of a sampling strategy: Randomly shuffle the vector before sorting
        // std::random_shuffle - deprecated in gcc++14, removed in gcc++17, use std::shuffle

        // Create a random number generator
        std::random_device rd;  // Non-deterministic random seed
        std::mt19937 g(rd());   // Mersenne Twister PRNG

        std::shuffle(layerDistanceIndexTuples.begin(), layerDistanceIndexTuples.end(), g);

        // Sort using a lambda function for custom comparison and adaptive threshold
        std::partial_sort(layerDistanceIndexTuples.begin(), layerDistanceIndexTuples.begin() + maxComparisons,
                          layerDistanceIndexTuples.end(),
                          [&](const auto& a, const auto& b) {
                              // Sort by layer first, then by distance
                              return std::tie(std::get<0>(a), std::get<1>(a)) <
                                     std::tie(std::get<0>(b), std::get<1>(b)) &&
                                     std::get<1>(a) < adaptiveThreshold;
                          });

        // Clear the neighbors vector and reserve space for the sorted neighbors
        currentNode.neighbors.clear();
        currentNode.neighbors.reserve(layerDistanceIndexTuples.size());

        // Copy the sorted neighbors back to the original neighbors vector
        for (const auto& tuple : layerDistanceIndexTuples) {
            currentNode.neighbors.push_back(std::get<2>(tuple));
        }

        // Additional optimizations for neighbor sorting
        // ...

        // You may want to consider more advanced optimizations, adaptive thresholds based on properties of the data, etc.
    }


    // Limit the number of connections per node
    void limitConnections(int nodeId) {
        if ((int) nodes_[nodeId].neighbors.size() > max_connections_) {
            nodes_[nodeId].neighbors.resize(max_connections_);
        }
    }

    // Sophisticated search algorithm implementation
    std::vector<int> searchNearestNeighbors(const Eigen::VectorXd& query, int k = 1) const {
        std::vector<int> result;

        // Placeholder for additional data structures and parameters
        // You may need additional structures like priority queues, distances, etc.

        // Initialize a priority queue to store candidate nodes
        std::priority_queue<std::tuple<double, int, int>> candidateQueue;

        // Initialize a set to keep track of visited nodes
        std::unordered_set<int> visitedNodes;

        // Initialize a variable to keep track of the best distance found so far
        double bestDistance = std::numeric_limits<double>::infinity();

        // Push the starting node (root) into the priority queue with layer information
        candidateQueue.push({Distance<T>::euclidian_distance(query, nodes_[0].embedding), 0, nodes_[0].layer});

        // Sophisticated search algorithm
        while (!candidateQueue.empty()) {
            // Get the node with the smallest distance and layer from the priority queue
            int currentNodeId = std::get<1>(candidateQueue.top());
            double currentDistance = std::get<0>(candidateQueue.top());
            int currentLayer = std::get<2>(candidateQueue.top());
            candidateQueue.pop();

            // Check if this node has already been visited or is in a different layer
            if (visitedNodes.count(currentNodeId) > 0 || currentLayer != nodes_[currentNodeId].layer) {
                continue;
            }

            // Mark the node as visited
            visitedNodes.insert(currentNodeId);

            // Update the result if the current node is closer than the best distance so far
            if (currentDistance < bestDistance) {
                result.push_back(currentNodeId);
                bestDistance = currentDistance;

                // If we found enough neighbors, break out of the loop
                if ((int) result.size() >= k) {
                    break;
                }
            }

            // Explore neighbors in a sophisticated manner (layered, heuristics, etc.)
            for (int neighborId : nodes_[currentNodeId].neighbors) {
                if (visitedNodes.count(neighborId) == 0) {
                    double neighborDistance = Distance<T>::euclidean(query, nodes_[neighborId].embedding);

                    // Additional heuristics and optimizations go here

                    // Push the neighbor into the priority queue with layer information
                    candidateQueue.push({neighborDistance, neighborId, nodes_[neighborId].layer});
                }
            }
        }

        return result;
    }   

public:
    HNSW(int embedding_size, int max_connections = 5) : embedding_size_(embedding_size), max_connections_(max_connections) {}

    // Add a node to the graph
    void addNode(const Eigen::VectorXd& embedding) {
        int nodeId = nodes_.size();
        int layer = chooseLayer(nodeId);
        nodes_.push_back({embedding, {}, layer});
        buildConnections(nodeId);
    }

    // Print the neighbors of each node
    void printGraph() const {
        for (int i = 0; i < (int) nodes_.size(); ++i) {
            std::cout << "Node " << i << " (Layer " << nodes_[i].layer << ") neighbors: ";
            for (int neighbor : nodes_[i].neighbors) {
                std::cout << neighbor << " ";
            }
            std::cout << "\n";
        }
    }

};

template <class T>
class Embeddings {
private:
    std::unordered_map<std::wstring, int> vocab;
    aimatrix<T> wordEmbeddings;
    aivector<T> wordBiases; // bias term for  word - each unique word in vocab has a bias term

    int vocabSize = 0;
    int embeddingSize = 5;

    // Global Vector
    std::unordered_map<std::wstring, int> comatrix;
    std::unordered_map<std::wstring, std::vector<std::wstring>> tokens;

    // const char* dbFileName = "data.db";
    const char* dbFileName = ":memory:";
    sqlite3* db = NULL;

    int dbopened = 0;

    // Create the token hash-to-index mapping and index-to-token mapping
    std::unordered_map<std::string,unsigned long> tokenHashToIndex;
    // std::vector<std::wstring> indexToToken;

    struct Record {

        // For the Vocabulary
        std::wstring token;
        int frequency;
        int tokenIndex;

        // For the Embedding
        std::string hashKey;
        airowvector<T> embedding;
        T bias;
    };

    void sqlite3_assert_db(int rc, const std::string& errormsg) {
        if (rc != SQLITE_OK) {
            throw AIException(errormsg + " : " + sqlite3_errmsg(this->db));
        }
    }

    void sqlite3_assert(int rc, const std::string& errormsg) {
        if (rc != SQLITE_OK) {
            throw AIException(errormsg + " : " + sqlite3_errmsg(this->db));
        }
    }

    void sqlite3_assert_done(int rc, const std::string& errormsg) {
        if (rc != SQLITE_DONE) {
            throw AIException(errormsg + " : " + sqlite3_errmsg(this->db));
        }
    }

    /************************************************************************************************
    * Embeddings::openDB
    * Function to open a SQLITE database.
    *************************************************************************************************/
    void openDB(sqlite3** db = NULL) {
        int rc = 0;
      //  if (*db == NULL) {
            rc = sqlite3_open_v2(this->dbFileName, &this->db, SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE, NULL);
            this->dbopened = rc;
            sqlite3_assert_db(rc, "Error opening database (Embedding::openDB)");
     /*
        } else {
            rc = sqlite3_open_v2(this->dbFileName, db, SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE, NULL);
            sqlite3_assert_db(rc, "Error opening database (Embedding::openDB)");        
        }
     */
    }

    void closeDB(sqlite3** db = NULL) {
       // if (*db == NULL) {
            if (this->dbopened == SQLITE_OK) {
                sqlite3_close(this->db);
                this->db = NULL;
            }
        /*
        } else {
            sqlite3_close(*db);
            *db = NULL;           
        }
        */
    }

public:

    typedef Record RecordStruct;

    Embeddings(int embeddingSize) {  
        this->embeddingSize = embeddingSize;
        try {
            openDB();
            initializeVectorDB();
        } catch (const AIException& e) {
            std::cerr << "Error (Embeddings::updateEmbeddingsInDatabase): " << e.what() << std::endl;
            closeDB();
        }
    }

    ~Embeddings() {
        closeDB();
    }

    // Initialize Vector DB
    void initializeVectorDB();

    // Function to create the vector table
    void createVectorTable();

    // Function to create the vocabulary table
    void createVocabularyTable();
 
    // Seed Vector DB
    void seedVectorDB();

    // Seed Vocabulary DB
    void seedVocabularyDB(std::unordered_map<std::wstring, int>& vocabulary);

    // Function to update a record into the vocabulary table
    void saveVocabulary(sqlite3* db, const Record& record);

    // Function to insert a record into the vector table
    void saveEmbeddings(sqlite3* db, const Record& record);

    // Function to retrieve an embedding from the database based on the hash key
    bool retrieveEmbeddings(const std::string& hashKey, Record& record);

    // Function to retrieve a record from the database based on the hash key
    // TODO: To use some kind of memcache
    bool retrieveVocabulary(const std::wstring& token, Record& record);

    // Function to retrieve a record from the database based on the hash key
    // TODO: To use some kind of memcache
    bool isInVocabulary(const std::wstring& token);

    bool isInVocabulary(const std::wstring& token, Record& record);

    // Function to get the vocabulary
    const std::unordered_map<std::wstring, int>& getVocabulary() const { return this->vocab; }

    // Initialize Embeddings in Cache
    void initializeEmbeddingsinCache(int row, int col);

    // Create Initial Vocabulary (Saving into the database)
    void createInitialVocabulary(std::unordered_map<std::wstring, int>& vocabulary);

    // Create Initial Embeddings (Saving into the database)
    void createInitialEmbeddings(int embeddingSize, std::unordered_map<std::wstring,int> vocabulary);

    void initializeVectorandVocabMetadata(std::unordered_map<std::wstring, int>& vocabulary);

    // Cross Reference Vocabulary
    void crossReferenceVocabularyinDBandCache(std::unordered_map<std::wstring, int> vocabulary);

    // Function to fetch the embeddings for tokens in the current corpus from the vector database
    void prefetchVocabularyToCache(const std::vector<std::vector<std::wstring>>& corpus);

    // Generate Token Indices
    void generateTokenIndices();

    // Function to fetch the embeddings for tokens in the cached vocabulary (instead of corpus) from the vector database to cache
    int fetchEmbeddings(sqlite3* db, std::string query, int currentIndex);
    void prefetchEmbeddingsToCache();

    // Update Embeddings in the Database
    void updateEmbeddingsInDatabase(const aimatrix<T>& wordEmbeddings, const aivector<T>& wordBiases);

    // Get Vocab Size and Embedding Size
    int getVocabSize() { return this->wordEmbeddings.rows(); }
    int getEmbeddingSize() { return this->wordEmbeddings.cols(); }

    // Get the word Embeddings 
    const aimatrix<T> getWordEmbeddings() { return this->wordEmbeddings; }

    // Get the bias terms for word Embeddings 
    const aivector<T> getWordBiases() { return this->wordBiases; }

    const std::unordered_map<std::string, unsigned long>& getTokenHashIndex() { return this->tokenHashToIndex; }

    unsigned long getTokenIndex(std::string token) {
        unsigned long index = 4294967295; // max
        try {
            index = this->tokenHashToIndex.at(token);
        } catch (const std::out_of_range& e) {}
        return index;
    }

    // Required by GloVe training to construct the co-occurrence matrix
    void buildCoMatrix(const std::vector<std::vector<std::wstring>>& corpus, int batchSize);

    // Get the constructed co-occurrence matrix
    std::unordered_map<std::wstring, int> getCoMatrix() { return this->comatrix; }

    std::unordered_map<std::wstring, std::vector<std::wstring>> getTokens() { return this->tokens; }

    std::vector<std::wstring> listTokens() {
        std::vector<std::wstring> tks;
        for (const auto& token : this->tokens) {
            tks.push_back(token.first);
        }
        return tks;
    }

};


#include <sqlite3.h>

struct TrieNode {
    std::unordered_map<wchar_t, TrieNode*> children;
    bool isEndOfToken;
};

#endif
