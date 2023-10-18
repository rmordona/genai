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

    Embeddings<T>* embeddings;
    std::string losstype = "mse";
    std::string optimizertype = "adagrad";
    T learningRate = 0.01;
    int maxIterations = 1;
    T regularization = 1.0;
    T clipThreshold = 5.0;

public:

    // Embeddings<T>* embeddings;

    std::unordered_map<std::wstring, int> vocab;
    aimatrix<T> wordEmbeddings;
    aivector<T> wordBiases;
    int vocabSize = 0;
    int embeddingSize = 5;

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

    BaseTokenModel(const std::string& losstype = "mse", const std::string& optimizertype = "adagrad",
          const T learningRate = 0.01, T regularization = 1.0,
          const int maxIterations = 1,  T clipThreshold = 5.0) {

        this->losstype = losstype;
        this->optimizertype = optimizertype;
        this->learningRate = learningRate;
        this->maxIterations = maxIterations;
        this->clipThreshold = clipThreshold;

        this->embeddings = new Embeddings<T>();
    }

    // Helper function to split string into words.
    std::vector<std::wstring> splitString(const std::wstring& str);

    // Tokenize new sentences. The tokens are compared against the vocabulary to generate the final tokens.
    // The final tokens are then used to generate the initial word embeddings for later training (e.g. trainGloVe())
    std::vector<std::wstring> tokenize(const std::wstring& sentence);

    // Overload function to perform the same tokenization but for multiple sentences.
    std::vector<std::vector<std::wstring>> tokenize(const std::vector<std::wstring>& sentences);

    // Now train a GloVe model
    void trainGloVe(std::vector<std::wstring>& sentences, int batchSize = 2, T learningRate = 0.01, int maxIteration = 1);

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
    Embeddings<T>* embeddings;
    TrieNode* root;
public:
    BPETokenizer() {
        root = new TrieNode();
    }

    // Build or Reset Vocabulary.
    void buildVocabulary(const std::vector<std::wstring>& sentences, int numMerges);

    // Helper function to determine if suffix exists in a string.
    bool endsWith(const std::wstring& str, const std::wstring& suffix);

    // Tokenize the corpus. The result is fed to the mergeTokens to construct the vocabulary.
    std::vector<std::wstring> tokenizeCorpus(const std::vector<std::wstring>& corpus);

    // Part of Byte Pair Encoding is to merge tokens that have the highest frequency.
    void mergeTokens(std::vector<std::wstring>& tokens, int numMerges);

    // Pretrain BPE Tokenizer
    void pretrain(const std::vector<std::wstring>& sentences, int numMerges,  int embeddingSize);

    // Train BPE Tokenizer
    void train(const std::vector<std::wstring>& sentences, int numMerges);

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

    std::string tokenizer = "bpetokenier";

    std::shared_ptr<BaseTokenModel<float>> modelXf;
    std::shared_ptr<BaseTokenModel<double>> modelXd;

    std::shared_ptr<BPETokenizer<float>> tokenizerf;
    std::shared_ptr<BPETokenizer<double>> tokenizerd;
public: 

    TokenModel(const std::string& losstype = "mse", const std::string& optimizertype = "adam", 
          const double learningRate = 0.01, const int itermax = 1, const std::string& datatype = "float");

    // set Tokenizer
    void setTokenizer(const std::string& name );

    // Pretrain BPE Tokenizer
    void pretrain(const std::vector<std::wstring>& sentences, int numMerges,  int embeddingSize);

    // Train BPE Tokenizer
    void train(const std::vector<std::wstring>& sentences, int numMerges);

    // tokenize: Function to tokenize a sentence
    std::vector<std::wstring> tokenize(const std::wstring& sentence);

    // tokenize: Function to tokenize sentences
    std::vector<std::vector<std::wstring>> tokenize(const std::vector<std::wstring>& sentences);

    // trainGloVe: Function to train corpus using GloVe
    void trainGloVe(std::vector<std::wstring>& sentences, int batchSize = 2, double learningRate = 0.01, int maxIteration = 1);

};

#endif
