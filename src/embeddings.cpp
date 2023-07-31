/*
 * GENAI - A distributed framework that can help build LLMs
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


// Initialize Vector dB
void Embeddings::initializeVectorDB() {
   
    // Create Vector DB
    createVectorDB();

    // Create Vector Table
    createVectorTable();

    // Create Vocabulary Table
    createVocabularyTable();
}

// Seed Vector dB
void Embeddings::seedVectorDB(std::unordered_map<std::wstring, int>& vocabulary) {

    int currentIndex = 0;
    for (const auto& entry : vocabulary) {
        Record record;

        record.hashKey = sha256(entry.first);
        record.vectorValue = this->wordEmbeddings.row(currentIndex);
        saveEmbeddings(record);

        record.token = entry.first;
        record.frequency = entry.second;
        record.tokenIndex = currentIndex;
        saveVocabulary(record);

        currentIndex++;
    }

}

// Function to create a simple SQLite vector DB
void Embeddings::createVectorDB() {
    int rc = sqlite3_open(dbFileName.c_str(), &this->db);
    if (rc) {
        log_warning( "Error opening database: {}", sqlite3_errmsg(this->db) );
        std::cerr << "Error opening database: " << sqlite3_errmsg(this->db) << std::endl;
        sqlite3_close(this->db);
    }
}

// Function to create a vector table and a vocabulary lookup table
void Embeddings::createVectorTable() {
    const char* createTableSQL = "CREATE TABLE IF NOT EXISTS corpus_embeddings ("
                                 "hash_key TEXT PRIMARY KEY, "
                                 "vector_value BLOB, "
                                 "bias REAL"
                                 ");";

    char* errorMsg;
    int rc = sqlite3_exec(this->db, createTableSQL, 0, 0, &errorMsg);
    if (rc != SQLITE_OK) {
        log_warning( "SQL error: {}", errorMsg );
        std::cerr << "SQL error: " << errorMsg << std::endl;
        sqlite3_free(errorMsg);
    }
}

// Function to create a vocabulary lookup table
void Embeddings::createVocabularyTable() {

    const char* createTableSQL = "CREATE TABLE IF NOT EXISTS vocabulary ("
                                 "token TEXT PRIMARY KEY, "
                                 "frequency INTEGER, "
                                 "tokenIndex INTEGER"
                                 ");";

    char* errorMsg;
    int rc = sqlite3_exec(this->db, createTableSQL, 0, 0, &errorMsg);
    if (rc != SQLITE_OK) {
        log_warning( "SQL error: {}", errorMsg );
        std::cerr << "SQL error: " << errorMsg << std::endl;
        sqlite3_free(errorMsg);
    }

    const char* createIndexSQL = "CREATE UNIQUE INDEX vocab_index ON vocabulary (tokenIndex);";

    rc = sqlite3_exec(this->db, createIndexSQL, 0, 0, &errorMsg);
    if (rc != SQLITE_OK) {
        log_warning( "SQL error: {}", errorMsg );
        std::cerr << "SQL error: " << errorMsg << std::endl;
        sqlite3_free(errorMsg);
    }

}

// Function to update a record into the vocabulary table
void Embeddings::saveVocabulary(const Record& record) {
    std::stringstream insertSQL;
    // The tokenIndex column is defined with AUTOINCREEMNT
    insertSQL << "INSERT OR REPLACE INTO vocabulary (token, frequency) VALUES (?, ?);";

    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(this->db, insertSQL.str().c_str(), -1, &stmt, 0);
    if (rc != SQLITE_OK) {
        log_warning( "Error preparing statement: {}", sqlite3_errmsg(this->db));
        std::cerr << "Error preparing statement: " << sqlite3_errmsg(db) << std::endl;
        return;
    }

    // Bind the token
    std::string utf8Str = wstringToUtf8(record.token);
    sqlite3_bind_text(stmt, 1, utf8Str.c_str(), -1, SQLITE_STATIC);

    // Bind the frequency
    sqlite3_bind_int(stmt, 2, record.frequency);

    rc = sqlite3_step(stmt);
    if (rc != SQLITE_DONE) {
        log_warning( "Error inserting data: {}", sqlite3_errmsg(this->db));
        std::cerr << "Error inserting data: " << sqlite3_errmsg(this->db) << std::endl;
    }

    sqlite3_finalize(stmt);
}

// Function to insert a record into the vector table
void Embeddings::saveEmbeddings(const Record& record) {
    std::stringstream insertSQL;
    insertSQL << "INSERT OR REPLACE INTO corpus_embeddings (hash_key, vector_value, bias) VALUES (?, ?, ?);";

    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(this->db, insertSQL.str().c_str(), -1, &stmt, 0);
    if (rc != SQLITE_OK) {
        log_warning( "Error preparing statement: {}", sqlite3_errmsg(this->db));
        std::cerr << "Error preparing statement: " << sqlite3_errmsg(db) << std::endl;
        return;
    }

    // Bind the hash key
    sqlite3_bind_text(stmt, 1, record.hashKey.c_str(), -1, SQLITE_STATIC);

    // Bind the vector embedding
    sqlite3_bind_blob(stmt, 2, record.vectorValue.data(), record.vectorValue.size() * sizeof(double), SQLITE_STATIC);

    // Prepare statement and bind bias...
    sqlite3_bind_double(stmt, 3, record.bias);

    rc = sqlite3_step(stmt);
    if (rc != SQLITE_DONE) {
        log_warning( "Error inserting data: {}", sqlite3_errmsg(this->db));
        std::cerr << "Error inserting data: " << sqlite3_errmsg(this->db) << std::endl;
    }

    sqlite3_finalize(stmt);
}

// Function to retrieve an embedding from the database based on the hash key
bool Embeddings::retrieveEmbeddings(const std::string& hashKey, Record& record) {
    std::stringstream selectSQL;
    selectSQL << "SELECT vector_value, bias FROM corpus_embeddings WHERE hash_key = ?;";

    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(this->db, selectSQL.str().c_str(), -1, &stmt, 0);
    if (rc != SQLITE_OK) {
        log_warning( "Error preparing statement: {}", sqlite3_errmsg(this->db));
        std::cerr << "Error preparing statement: " << sqlite3_errmsg(this->db) << std::endl;
        return false;
    }

    // Bind the hash key
    sqlite3_bind_text(stmt, 1, hashKey.c_str(), -1, SQLITE_STATIC);

    rc = sqlite3_step(stmt);
    if (rc == SQLITE_ROW) {
        record.hashKey = hashKey;

        // Get the vector value from the result and store it in the record
        const void* data = sqlite3_column_blob(stmt, 0);
        int size = sqlite3_column_bytes(stmt, 0);
        record.vectorValue.resize(size / sizeof(double));
        std::memcpy(record.vectorValue.data(), data, size);

        record.bias = sqlite3_column_double(stmt, 1);

        sqlite3_finalize(stmt);
        return true;
    }

    sqlite3_finalize(stmt);
    return false;
}

// Function to retrieve a record from the database based on the hash key
// TODO: To use some kind of memcache
bool Embeddings::retrieveVocabulary(const std::wstring& token, Record& record) {
    std::stringstream selectSQL;
    selectSQL << "SELECT frequency FROM vocabulary WHERE token = ?;";

    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(this->db, selectSQL.str().c_str(), -1, &stmt, 0);
    if (rc != SQLITE_OK) {
        log_warning( "Error preparing statement: {}", sqlite3_errmsg(this->db));
        std::cerr << "Error preparing statement: " << sqlite3_errmsg(this->db) << std::endl;
        return false;
    }

    // Bind the hash key
    std::string utf8Str = wstringToUtf8(token);
    sqlite3_bind_text(stmt, 1, utf8Str.c_str(), -1, SQLITE_STATIC);

    rc = sqlite3_step(stmt);
    if (rc == SQLITE_ROW) {
        record.token = token;
        // Get the frequency from the result and store it in the record
        int frequency = sqlite3_column_int(stmt, 0);
        record.frequency = frequency;

        sqlite3_finalize(stmt);
        return true;
    }
    sqlite3_finalize(stmt);
    return false;
}

// Function to retrieve a record from the database based on the hash key
// TODO: To use some kind of memcache
bool Embeddings::isInVocabulary(const std::wstring& token) {
    std::stringstream selectSQL;
    selectSQL << "SELECT frequency FROM vocabulary WHERE token = ?;";

    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(this->db, selectSQL.str().c_str(), -1, &stmt, 0);
    if (rc != SQLITE_OK) {
        log_warning( "Error preparing statement: {}", sqlite3_errmsg(this->db));
        std::cerr << "Error preparing statement: " << sqlite3_errmsg(this->db) << std::endl;
        return false;
    }

    // Bind the hash key
    std::string utf8Str = wstringToUtf8(token);
    sqlite3_bind_text(stmt, 1, utf8Str.c_str(), -1, SQLITE_STATIC);

    rc = sqlite3_step(stmt);
    if (rc == SQLITE_ROW) {
        sqlite3_finalize(stmt);
        return true;
    }

    sqlite3_finalize(stmt);
    return false;
}

bool Embeddings::isInVocabulary(const std::wstring& token, Record& record)  {
    return retrieveVocabulary(token,record);
}


// Function to generate the initial word embedding. We require the size of the constructed vocabulary
void Embeddings::initializeVectorandVocabMetadata(std::unordered_map<std::wstring, int>& vocabulary, int embeddingSize) {
    this->vocab = vocabulary;
    this->vocabSize = vocabulary.size();
    this->embeddingSize = embeddingSize;
    this->wordEmbeddings = Eigen::MatrixXd::Random(this->vocabSize, embeddingSize);
    this->wordBiases = Eigen::VectorXd::Zero(this->vocabSize);

    // Initialize word Embeddings
    BaseOperator::heInitialization(this->wordEmbeddings);

    // Seed VectorDB
    seedVectorDB(this->vocab);
}

// Cross Reference Vocabulary between memory and DB
// Sync in-memory vocab into DB.
void Embeddings::crossReferenceVocabularyinDBandCache(std::unordered_map<std::wstring, int>& vocabulary) {

    log_info( "===============================================================" );
    log_info( "Entering Cross Reference of Vocabulary between DB and Cache ..." );

    this->vocab = vocabulary;
    for (const auto& entry : this->vocab) {
        Record record;
        if (isInVocabulary(entry.first, record)) {
            record.frequency += entry.second;
        } else {
            record.frequency = entry.second;
        }
        record.token = entry.first;
        saveVocabulary(record);
        log_wdetail( "{} {:d}", wstringToUtf8(record.token).c_str(), record.frequency );
    }
}

// Function to fetch the embeddings for tokens in the current corpus from the vector database
void Embeddings::prefetchVocabularyToCache(const std::vector<std::vector<std::wstring>>& corpus) {
    // Assuming you have a SQLite database connection and a table called "corpus_embeddings"
    // with columns "token_hash" (INTEGER) and "embedding" (BLOB)

    log_info( "===============================================" );
    log_info( "Entering Prefetching of Vocabulary to Cache ..." );

    // Open the database and prepare the query
    sqlite3* db;
    sqlite3_stmt* stmt;
    int rc = sqlite3_open(dbFileName.c_str(), &db);
    if (rc != SQLITE_OK) {
        // Handle database open error
        log_warning( "Error opening database: {}", sqlite3_errmsg(this->db) );
        std::cerr << "Error opening database: " << sqlite3_errmsg(this->db) << std::endl;
        return;
    }

    // Prepare the query to fetch embeddings for tokens in the corpus
    std::string query = "SELECT frequency, tokenIndex, token FROM vocabulary WHERE token IN (";
    for (const auto& sentence : corpus) {
        for (const auto& token : sentence) { 
            std::string utf8Str = wstringToUtf8(token);
            query += utf8Str + ",";
        }
    }
    query.pop_back(); // Remove the last comma
    query += ");";

    // Execute the query and fetch the embeddings
    rc = sqlite3_prepare_v2(db, query.c_str(), -1, &stmt, 0);
    if (rc != SQLITE_OK) {
        // Handle query execution error
        log_warning( "No vocabulary found ..." );
        sqlite3_close(db);
        return;
    }

    // Fetch and store embeddings in the dynamic embeddings data structure
    // Also create a token hash index structure for fast lookup
    while (sqlite3_step(stmt) == SQLITE_ROW) {

        int frequency = sqlite3_column_int(stmt, 0);
        // int tokenIndex = sqlite3_column_int(stmt, 1); not used for now
        const unsigned char* tokenBytes = sqlite3_column_text(stmt, 2);
        std::string utf8Str(reinterpret_cast<const char*>(tokenBytes));

        std::wstring token = utf8ToWstring(utf8Str);
        this->vocab[token] += frequency;
    }

    // Close the database
    sqlite3_finalize(stmt);
    sqlite3_close(db);
}

// Function to fetch the embeddings for tokens in the cached vocabulary (instead of corpus) 
// from the vector database to cache
void Embeddings::prefetchEmbeddingsToCache() {
    // Assuming you have a SQLite database connection and a table called "corpus_embeddings"
    // with columns "token_hash" (INTEGER) and "embedding" (BLOB)

    log_info( "==============================================" );
    log_info( "Entering Prefetching of Embedding to Cache ...");

    // Open the database and prepare the query
    sqlite3* db;
    sqlite3_stmt* stmt;
    int rc = sqlite3_open(dbFileName.c_str(), &db);
    if (rc != SQLITE_OK) {
        // Handle database open error
        log_warning( "Error opening database: {}", sqlite3_errmsg(this->db) );
        std::cerr << "Error opening database: " << sqlite3_errmsg(this->db) << std::endl;
        return;
    }

    // Prepare the query to fetch embeddings for tokens in the corpus
    std::string query = "SELECT hash_key, vector_value, bias FROM corpus_embeddings WHERE hash_key IN (";

    // Use the cached vocabulary
    for (const auto& token : this->vocab) {
            std::string tokenHash = sha256(token.first);  
            query += "'" + tokenHash + "',";
    }

    query.pop_back(); // Remove the last comma
    query += ");";

    // Execute the query and fetch the embeddings
    rc = sqlite3_prepare_v2(db, query.c_str(), -1, &stmt, 0);
    if (rc != SQLITE_OK) {
        // Handle query execution error
        log_warning( "No embeddings found ..." );
        sqlite3_close(db);
        return;
    }
 
    // Fetch and store embeddings in the dynamic embeddings data structure
    // Also create a token hash index structure for fast lookup
    int currentIndex = 0;
    this->tokenHashToIndex.clear();
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        const unsigned char* tokenHashBytes = sqlite3_column_text(stmt, 0);
        // Convert the token hash from const unsigned char* to std::string
        std::string tokenHash(reinterpret_cast<const char*>(tokenHashBytes));

        const void* embeddingBlob = sqlite3_column_blob(stmt, 1);
        int embeddingSizeBytes = sqlite3_column_bytes(stmt, 1);

        // Convert the BLOB data to Eigen VectorXd (assuming float64 for the embeddings)
        Eigen::VectorXd embeddings(embeddingSizeBytes / sizeof(double));
        std::memcpy(embeddings.data(), embeddingBlob, embeddingSizeBytes);

        double bias = sqlite3_column_double(stmt, 2);

        // Create the token hash-to-index mapping and index-to-token mapping
        if (this->tokenHashToIndex.find(tokenHash) == this->tokenHashToIndex.end()) {
            this->tokenHashToIndex[tokenHash] = currentIndex;
            this->wordEmbeddings.row(currentIndex) = embeddings;
            this->wordBiases(currentIndex) = bias;
            currentIndex++;
        }

    }

    // Close the database
    sqlite3_finalize(stmt);
    sqlite3_close(db);
}

// Update Embeddings in the Database
void Embeddings::updateEmbeddingsInDatabase(const Eigen::MatrixXd& wordEmbeddings,
                                           const Eigen::VectorXd& wordBiases) {


    log_tag("Embeddings");
    log_info( "==========================================" );
    log_info( "Entering Parameter Update in Embedding ...");

    log_detail( "Word Embeddings" );
    log_matrix( wordEmbeddings );
    for (const auto& indexTokenPair : this->tokenHashToIndex) {
        const std::string& tokenHash = indexTokenPair.first;
        int index = indexTokenPair.second;
        Record record;
        record.hashKey = tokenHash;
        record.vectorValue = wordEmbeddings.row(index);
        record.bias = wordBiases(index);
        saveEmbeddings(record);
    }

}
