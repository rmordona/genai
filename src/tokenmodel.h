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

#include "embeddings.h"
#include "logger.h"

#ifndef TOKENMODEL_H
#define TOKENMODEL_H


/*************************************************************************************************
 * BaseTokenModel is the actual structure we use to perform tokenization.
 * underlying operations needed to train a model.
 *************************************************************************************************/
template <class T>
class BaseTokenModel {
private:
    std::string losstype      = "mse";
    std::string optimizertype = "adagrad";
    int maxIteration = 1;
    T learningRate    = 0.01;
    T regularization  = 1.0;
    T clipThreshold   = 5.0;

    Embeddings<T>* embeddings;
    int vocabSize     = 0;
    int embeddingSize = 5;
    aimatrix<T> wordEmbeddings;

    // Initialize the gradients for the batch
    aimatrix<T> batchGradients;

public:

    int merges        = 1000;

    std::unordered_map<std::wstring, int> vocab;

    bool resetVocab = false;

    struct Record {

        // For the Vocabulary
        std::wstring token;
        int frequency;
        int tokenIndex;

        // For the Embedding
        std::string hashKey;
        aivector<T> vectorValue;

    };

    BaseTokenModel() {}

    // Initialize Embeddings
    void initializeEmbeddings(int embeddingSize) {
        this->embeddingSize = embeddingSize;
        this->embeddings = new Embeddings<T>(embeddingSize);
    }

    // Get the Embedding
    Embeddings<T>* getEmbeddings() { return this->embeddings; }

    // set Vocabulary Size
    void setVocabSize(int vocabSize) { this->vocabSize = vocabSize; }

    // Helper function to split string into words.
    std::vector<std::wstring> splitString(const std::wstring& str);

    // Tokenize new sentences. The tokens are compared against the vocabulary to generate the final tokens.
    // The final tokens are then used to generate the initial word embeddings for later training (e.g. trainGloVe())
    std::vector<std::wstring> tokenize(const std::wstring& sentence);

    // Overload function to perform the same tokenization but for multiple sentences.
    std::vector<std::vector<std::wstring>> tokenize(const std::vector<std::wstring>& sentences);

    // Prefetch vocabulary and vector for given corpus
    void prefetchEmbeddings(const std::vector<std::wstring>& sentences, int token_limit = 4000, int batchSize = 1);

    // Now train a GloVe model
    void train(std::vector<std::wstring>& sentences, int batchSize = 2, 
            const std::string& losstype = "mse", const std::string& optimizertype = "adagrad",
            T learningRate = 0.01, int maxIteration = 1, T clipThreshold = 5.0, T regularization = 1.0);

    // Function to print the vocabulary
    void printVocabulary(int rows);

    // Function to print the word embedding
    void printWordEmbeddings(int rows);

};

/************************************************************************************************
* This is the actual BPE Tokenizer that we use to tokenize sentences.
*************************************************************************************************/
template <class T>
class BPETokenizer : public BaseTokenModel<T> {
private:
    // Embeddings<T>* embeddings;
    TrieNode* root;

    // std::unordered_map<std::wstring, int> vocab;
    // int vocabSize = 0;

public:
    BPETokenizer() {
        root = new TrieNode();
    }

    // Build or Reset Vocabulary.
    void buildVocabulary(const std::vector<std::wstring>& sentences, int numMerges);

    void createEmbeddings(int embeddingSize);

    // Helper function to determine if suffix exists in a string.
    bool endsWith(const std::wstring& str, const std::wstring& suffix);

    // Tokenize the corpus. The result is fed to the mergeTokens to construct the vocabulary.
    std::vector<std::wstring> tokenize_to_char(const std::vector<std::wstring>& corpus);

    // Part of Byte Pair Encoding is to merge tokens that have the highest frequency.
    std::unordered_map<std::wstring, int>  mergeTokens(std::vector<std::wstring>& tokens, int numMerges);

    // Preload BPE Tokenizer
    void preload(const std::vector<std::wstring>& sentences, int numMerges,  int embeddingSize = 0);

    // Merge BPE Tokenizer
    void merge(const std::vector<std::wstring>& sentences, int numMerges);

};

/************************************************************************************************
* We use BPE Tokenizer  class as a meta model only for use  as entry point for python API.
* The actual model is the BPETokenizer class.
* All other meta models include the Model operations.
*************************************************************************************************/
class TokenModel {
private:
    std::string losstype = "mse";
    std::string optimizertype = "adam";
    int max_epoch = 1;
    std::string datatype = "float";
    double learningRate = 0.01;
    double clipThreshold = 0.01;
    double regularization = 0.01;

    std::string tokenizer = "bpetokenizer";

    std::shared_ptr<BPETokenizer<float>> tokenizerf;
    std::shared_ptr<BPETokenizer<double>> tokenizerd;

public: 

    TokenModel(const std::string& tokenizer = "bpetokenizer", const std::string& datatype = "float");

    // set Tokenizer
    // void setTokenizer(const std::string& name);

    // Pretrain BPE Tokenizer
    void preload(const std::vector<std::wstring>& sentences, int numMerges,  int embeddingSize);

    // Merge BPE Tokenizer
    void merge(const std::vector<std::wstring>& sentences, int numMerges);

    // tokenize: Function to tokenize a sentence
    std::vector<std::wstring> tokenize(const std::wstring& sentence);

    // tokenize: Function to tokenize sentences
    std::vector<std::vector<std::wstring>> tokenize(const std::vector<std::wstring>& sentences);

    // train: Function to train corpus using GloVe
    void train(std::vector<std::wstring>& sentences, int batchSize = 2, 
            const std::string& losstype = "mse", const std::string& optimizertype = "adagrad",
            double learningRate = 0.01, int maxIteration = 1, double clipThreshold = 5.0, double regularization = 1.0);

/*
    // Overload the train function
    void train(std::vector<std::wstring>& sentences, int batchSize = 2, 
            const std::string& losstype = "mse", const std::string& optimizertype = "adagrad",
            double learningRate = 0.01, int maxIteration = 1) {
        train(sentences, batchSize, losstype, optimizertype,  learningRate, maxIteration, (double) 5.0, (double) 1.0);
    }
*/

};

#endif
