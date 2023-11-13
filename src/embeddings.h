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

#define TK_SPACE_ L"<SPC>"
#define TK_MASK_  L"<MASK>"
#define TK_UNK_   L"<UNK>"
#define TK_PAD_   L"<PAD>"
#define TK_SOS_   L"<SOS>"
#define TK_EOS_   L"<EOS>"

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

    struct Record {

        // For the Vocabulary
        std::wstring token;
        int frequency;
        int tokenIndex;

        // For the Embedding
        std::string hashKey;
        aivector<T> embedding;
        T bias;
    };

    // Create the token hash-to-index mapping and index-to-token mapping
    std::unordered_map<std::string,int> tokenHashToIndex;
    // std::vector<std::wstring> indexToToken;



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
    void openDB() {
        int rc = sqlite3_open_v2(this->dbFileName, &this->db, SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE, NULL);
        this->dbopened = rc;
        sqlite3_assert_db(rc, "Error opening database (Embedding::openDB)");
    }

    void closeDB() {
        if (this->dbopened == SQLITE_OK) {
            sqlite3_close(this->db);
            this->db = NULL;
        }
    }

public:

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

    // Get Embeddings and indcies
    const aimatrix<T> getWordEmbeddings() { return this->wordEmbeddings; }
    const aivector<T> getWordBiases() { return this->wordBiases; }
    const std::unordered_map<std::string,int>& getTokenHashIndex() { return this->tokenHashToIndex; }

    int getTokenIndex(std::string token) {
        int index = -1;
        try {
            index = this->tokenHashToIndex.at(token);
        } catch (const std::out_of_range& e) {}
        return index;
    }

    void buildCoMatrix(const std::vector<std::vector<std::wstring>>& corpus, int batchSize);
    std::unordered_map<std::wstring, int> getCoMatrix() { return this->comatrix; }
    std::unordered_map<std::wstring, std::vector<std::wstring>> getTokens() { return this->tokens; }


};


#include <sqlite3.h>

struct TrieNode {
    std::unordered_map<wchar_t, TrieNode*> children;
    bool isEndOfToken;
};

#endif

