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

template <class T>
class Embeddings {
private:
    std::unordered_map<std::wstring, int> vocab;
    aimatrix<T> wordEmbeddings;
    aivector<T> wordBiases;
    int vocabSize = 0;
    int embeddingSize = 5;

    const std::string dbFileName = "data.db";
    sqlite3* db = nullptr;

    struct Record {

        // For the Vocabulary
        std::wstring token;
        int frequency;
        int tokenIndex;

        // For the Embedding
        std::string hashKey;
        aivector<T> vectorValue;
        double bias;
    };

    // Create the token hash-to-index mapping and index-to-token mapping
    std::unordered_map<std::string, int> tokenHashToIndex;
    // std::vector<std::wstring> indexToToken;

public:

    Embeddings() {  initializeVectorDB();  }

    // Initialize Vector DB
    void initializeVectorDB();

    // Seed Vector DB
    void seedVectorDB(std::unordered_map<std::wstring, int>& vocabulary);

    // Function to create simple SQLite DB
    void createVectorDB();

    // Function to create the vector table
    void createVectorTable();

    // Function to create the vocabulary table
    void createVocabularyTable();

    // Function to update a record into the vocabulary table
    void saveVocabulary(const Record& record);

    // Function to insert a record into the vector table
    void saveEmbeddings(const Record& record);

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

    // Function to seed the Vector and Vocabulary Tables
    void initializeVectorandVocabMetadata(std::unordered_map<std::wstring, int>& vocabulary, int embeddingSize);

    // Cross Reference Vocabulary
    void crossReferenceVocabularyinDBandCache(std::unordered_map<std::wstring, int>& vocabulary);

    // Function to fetch the embeddings for tokens in the current corpus from the vector database
    void prefetchVocabularyToCache(const std::vector<std::vector<std::wstring>>& corpus);

    // Function to fetch the embeddings for tokens in the cached vocabulary (instead of corpus) from the vector database to cache
    void prefetchEmbeddingsToCache();

    // Update Embeddings in the Database
    void updateEmbeddingsInDatabase(const aimatrix<T>& wordEmbeddings,
                                    const aivector<T>& wordBiases);

    // Get Vocab Size and Embedding Size
    int getVocabSize() { return this->wordEmbeddings.rows(); }
    int getEmbeddingSize() { return this->wordEmbeddings.cols(); }

    // Get Embeddings and indcies
    const aimatrix<T> getWordEmbeddings() { return this->wordEmbeddings; }
    const aivector<T> getWordBiases() { return this->wordBiases; }
    std::unordered_map<std::string, int>& getTokenIndex() { return this->tokenHashToIndex; }
};


#include <sqlite3.h>

struct TrieNode {
    std::unordered_map<wchar_t, TrieNode*> children;
    bool isEndOfToken;
};

#endif

