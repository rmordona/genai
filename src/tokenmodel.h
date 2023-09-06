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

template <class T>
class TokenModel {
private:

    std::string losstype = "mse";
    std::string optimizertype = "adagrad";
    T learningRate = 0.01;
    int maxIterations = 1;
    T regularization = 1.0;
    T clipThreshold = 5.0;

public:

    Embeddings<T>* embeddings;

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

    TokenModel(const std::string& losstype = "mse", const std::string& optimizertype = "adagrad",
          const T learningRate = 0.01, T regularization = 1.0,
          const int maxIterations = 1,  T clipThreshold = 5.0) {

        this->losstype = losstype;
        this->optimizertype = optimizertype;
        this->learningRate = learningRate;
        this->maxIterations = maxIterations;
        this->clipThreshold = clipThreshold;
    }

    // Function to print the vocabulary
    void printVocabulary(int rows);

    // Function to print the word embedding
    void printWordEmbeddings(int rows);

    // Helper function to split string into words.
    std::vector<std::wstring> splitString(const std::wstring& str);

    // Tokenize new sentences. The tokens are compared against the vocabulary to generate the final tokens.
    // The final tokens are then used to generate the initial word embeddings for later training (e.g. trainGloVe())
    std::vector<std::wstring> tokenize(const std::wstring& sentence);

    // Overload function to perform the same tokenization but for multiple sentences.
    std::vector<std::vector<std::wstring>> tokenize(const std::vector<std::wstring>& sentences);

    // Now train a GloVe model
    void trainGloVe(std::vector<std::wstring>& sentences, int batchSize = 2, T learningRate = 0.01, int maxIteration = 1);
};

template <class T>
class BPETokenizer : public TokenModel<T> {
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

#endif
