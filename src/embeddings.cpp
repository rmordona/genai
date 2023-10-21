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
#include "logger.h"
#include "embeddings.h"

namespace py = pybind11;
using namespace py::literals;

/************************************************************************************************
* Helper Function: strreplace
* Function to replace a substring. Borrowed from stack-overflow (question: 3418231 - Michael Mrozek)
*************************************************************************************************/
bool replace(std::string& str, const std::string& from, const std::string& to) {
    size_t start_pos = str.find(from);
    if(start_pos == std::string::npos)
        return false;
    str.replace(start_pos, from.length(), to);
    return true;
}
void replaceAll(std::string& str, const std::string& from, const std::string& to) {
    if(from.empty())
        return;
    size_t start_pos = 0;
    while((start_pos = str.find(from, start_pos)) != std::string::npos) {
        str.replace(start_pos, from.length(), to);
        start_pos += to.length(); // In case 'to' contains 'from', like replacing 'x' with 'yx'
    }
}
/************************************************************************************************
* Embeddings::initializeVectorDB
* Function to initialize Vector dB
*************************************************************************************************/
template <class T>
void Embeddings<T>::initializeVectorDB() {
   
    // Create Vector DB
    createVectorDB();

    // Create Vector Table
    createVectorTable();

    // Create Vocabulary Table
    createVocabularyTable();
}

/************************************************************************************************
* Embeddings::seedVectorDB
* Function to seed the Vector dB with hashed values.
*************************************************************************************************/

template <class T>
void Embeddings<T>::seedVectorDB(std::unordered_map<std::wstring, int>& vocabulary) {

    int currentIndex = 0;
    for (const auto& entry : vocabulary) {
        Record record;

        record.hashKey = sha256(entry.first);
        record.vectorValue = this->wordEmbeddings.row(currentIndex);
        record.bias = 0.0;
        saveEmbeddings(record);

        record.token = entry.first;
        record.frequency = entry.second;
        record.tokenIndex = currentIndex;
        saveVocabulary(record);

        currentIndex++;
    }

}

/************************************************************************************************
* Embeddings::createVectorDB
* Function to create a simple SQLite vector DB
*************************************************************************************************/
template <class T>
void Embeddings<T>::createVectorDB() {
    int rc = sqlite3_open(dbFileName.c_str(), &this->db);
    if (rc) {
        log_warning( "Error opening database: {0}", sqlite3_errmsg(this->db) );
        std::cerr << "Error opening database: " << sqlite3_errmsg(this->db) << std::endl;
        sqlite3_close(this->db);
        throw AIException("Error Opening database: Embeddings::createVectorDB()");
    }
}

/************************************************************************************************
* Embeddings::createVectorTable
* Function to create a vector table and a vocabulary lookup table
*************************************************************************************************/
template <class T>
void Embeddings<T>::createVectorTable() {
    const char* createTableSQL = "CREATE TABLE IF NOT EXISTS corpus_embeddings ("
                                 "hash_key TEXT PRIMARY KEY, "
                                 "vector_value BLOB, "
                                 "bias REAL"
                                 ");";

    char* errorMsg;
    int rc = sqlite3_exec(this->db, createTableSQL, 0, 0, &errorMsg);
    if (rc != SQLITE_OK) {
        log_warning( "SQL error: {0}", errorMsg );
        std::cerr << "SQL error: " << errorMsg << std::endl;
        sqlite3_free(errorMsg);
        throw AIException("Error Creating (corpus_embeddings) table");
    }
}


/************************************************************************************************
* Embeddings::createVocabularyTable
* Function to create a vocabulary lookup table
*************************************************************************************************/
template <class T>
void Embeddings<T>::createVocabularyTable() {

    const char* createTableSQL = "CREATE TABLE IF NOT EXISTS vocabulary ("
                                 "token TEXT PRIMARY KEY, "
                                 "frequency INTEGER, "
                                 "tokenIndex INTEGER"
                                 ");";

    char* errorMsg;
    int rc = sqlite3_exec(this->db, createTableSQL, 0, 0, &errorMsg);
    if (rc != SQLITE_OK) {
        log_warning( "SQL error: {0}", errorMsg );
        std::cerr << "SQL error: " << errorMsg << std::endl;
        sqlite3_free(errorMsg);
        throw AIException("Error Creating (vocabulary) table");
    }

    const char* createIndexSQL = "CREATE UNIQUE INDEX vocab_index ON vocabulary (tokenIndex);";

    rc = sqlite3_exec(this->db, createIndexSQL, 0, 0, &errorMsg);
    if (rc != SQLITE_OK) {
        log_warning( "SQL error: {0}", errorMsg );
        std::cerr << "SQL error: " << errorMsg << std::endl;
        sqlite3_free(errorMsg);
    }

}

/************************************************************************************************
* Embeddings::saveVocabulary
* Function to insert into or replace a record in the vocabulary table
* Note that vocabulary table only intends to build a list of vocabulary and therefore
* does not represent any embedding.
*************************************************************************************************/
template <class T>
void Embeddings<T>::saveVocabulary(const Record& record) {
    std::stringstream insertSQL;
    // The tokenIndex column is defined with AUTOINCREEMNT
    insertSQL << "INSERT OR REPLACE INTO vocabulary (token, frequency) VALUES (?, ?);";

    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(this->db, insertSQL.str().c_str(), -1, &stmt, 0);
    if (rc != SQLITE_OK) {
        log_warning( "Error preparing statement: {0}", sqlite3_errmsg(this->db));
        std::cerr << "Error preparing statement: " << sqlite3_errmsg(db) << std::endl;
        throw AIException("Error Inserrting vocabulary (Embeddings::saveVocabulary)");
    }

    // Bind the token
    std::string utf8Str = wstringToUtf8(record.token);
    sqlite3_bind_text(stmt, 1, utf8Str.c_str(), -1, SQLITE_STATIC);

    // Bind the frequency
    sqlite3_bind_int(stmt, 2, record.frequency);

    rc = sqlite3_step(stmt);
    if (rc != SQLITE_DONE) {
        log_warning( "Error inserting data: {0}", sqlite3_errmsg(this->db));
        std::cerr << "Error inserting data: " << sqlite3_errmsg(this->db) << std::endl;
    }

    sqlite3_finalize(stmt);
}

/************************************************************************************************
* Embeddings::saveEmbeddings
* Function to insert into or replace an embedding into the vector table
*************************************************************************************************/
template <class T>
void Embeddings<T>::saveEmbeddings(const Record& record) {
    std::stringstream insertSQL;
    insertSQL << "INSERT OR REPLACE INTO corpus_embeddings (hash_key, vector_value, bias) VALUES (?, ?, ?);";

    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(this->db, insertSQL.str().c_str(), -1, &stmt, 0);
    if (rc != SQLITE_OK) {
        log_warning( "Error preparing statement: {0}", sqlite3_errmsg(this->db));
        std::cerr << "Error preparing statement: " << sqlite3_errmsg(db) << std::endl;
        return;
    }

    // Bind the hash key
    sqlite3_bind_text(stmt, 1, record.hashKey.c_str(), -1, SQLITE_STATIC);

    // Bind the vector embedding
    sqlite3_bind_blob(stmt, 2, record.vectorValue.data(), record.vectorValue.size() * sizeof(T), SQLITE_STATIC);

    // Prepare statement and bind bias...
    sqlite3_bind_double(stmt, 3, record.bias);

    rc = sqlite3_step(stmt);
    if (rc != SQLITE_DONE) {
        log_warning( "Error inserting data: {0}", sqlite3_errmsg(this->db));
        std::cerr << "Error inserting data: " << sqlite3_errmsg(this->db) << std::endl;
    }

    sqlite3_finalize(stmt);
}

/************************************************************************************************
* Embeddings::retrieveEmbeddings
* Function to retrieve an embedding from the database based on the hash key
*************************************************************************************************/

template <class T>
bool Embeddings<T>::retrieveEmbeddings(const std::string& hashKey, Record& record) {
    std::stringstream selectSQL;
    selectSQL << "SELECT vector_value, bias FROM corpus_embeddings WHERE hash_key = ?;";

    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(this->db, selectSQL.str().c_str(), -1, &stmt, 0);
    if (rc != SQLITE_OK) {
        log_warning( "Error preparing statement: {0}", sqlite3_errmsg(this->db));
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
        record.vectorValue.resize(size / sizeof(double)); // let column hold double regardless of T (typeclass)
        std::memcpy(record.vectorValue.data(), data, size);

        record.bias = sqlite3_column_double(stmt, 1);

        sqlite3_finalize(stmt);
        return true;
    }

    sqlite3_finalize(stmt);
    return false;
}


/************************************************************************************************
* Embeddings::retrieveVocabulary
* Function to retrieve a record from the database based on the hash key
* TODO :  To use some kind of memcache
*************************************************************************************************/
template <class T>
bool Embeddings<T>::retrieveVocabulary(const std::wstring& token, Record& record) {
    std::stringstream selectSQL;
    selectSQL << "SELECT frequency FROM vocabulary WHERE token = ?;";

    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(this->db, selectSQL.str().c_str(), -1, &stmt, 0);
    if (rc != SQLITE_OK) {
        log_warning( "Error preparing statement: {0}", sqlite3_errmsg(this->db));
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


/************************************************************************************************
* Embeddings::isInVocabulary
* Function to find if a token exists in the vocabulary table
* TODO : To use some kind of memcache
*************************************************************************************************/
template <class T>
bool Embeddings<T>::isInVocabulary(const std::wstring& token) {

    sqlite3* db;
    sqlite3_stmt* stmt;
    int rc1 = sqlite3_open(dbFileName.c_str(), &db);

    if (rc1 != SQLITE_OK) {
        // Handle database open error
        log_warning( "Error opening database: {0}", sqlite3_errmsg(db) );
        std::cerr << "Error opening database: " << sqlite3_errmsg(db) << std::endl;
        throw AIException("Error opening database: Embeddings::isInvocabulary()");
        return false;
    }

    std::stringstream selectSQL;
    selectSQL << "SELECT frequency FROM vocabulary WHERE token = ?;";

    int rc = sqlite3_prepare_v2(db, selectSQL.str().c_str(), -1, &stmt, 0);
    if (rc != SQLITE_OK) {
        log_warning( "Error preparing statement: {0}", sqlite3_errmsg(db));
        std::cerr << "Error preparing statement: " << sqlite3_errmsg(db) << std::endl;
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

    sqlite3_close(db);

    return false;
}

template <class T>
bool Embeddings<T>::isInVocabulary(const std::wstring& token, Record& record)  {
    return retrieveVocabulary(token,record);
}

/************************************************************************************************
* Embeddings::initializeVectorandVocabMetadata
* Function to generate the initial word embedding. We require the size of the
* constructed vocabulary
*************************************************************************************************/
template <class T>
void Embeddings<T>::initializeEmbeddings(int vocabSize) {

    this->wordEmbeddings = aimatrix<T>::Random(vocabSize, this->embeddingSize);
    this->wordBiases = aivector<T>::Zero(vocabSize);

    // Initialize word Embeddings
    BaseOperator::heInitMatrix(this->wordEmbeddings);

}


template <class T>
void Embeddings<T>::initializeVectorandVocabMetadata(std::unordered_map<std::wstring, int>& vocabulary) {

    this->vocabSize = vocabulary.size();
    this->vocab = vocabulary;

    initializeEmbeddings(this->vocabSize);

    // Seed VectorDB
    this->seedVectorDB(this->vocab);
}

/************************************************************************************************
* Embeddings::crossReferenceVocabularyinDBandCache
* Cross Reference Vocabulary between memory and DB.  Sync in-memory vocab into DB.
* If vocabulary is found in the DB, then increment the frequency; otherwise,
* save the new vocabulary with initial count. 
*************************************************************************************************/
template <class T>
void Embeddings<T>::crossReferenceVocabularyinDBandCache(std::unordered_map<std::wstring, int>& vocabulary) {

    log_info( "===============================================================" );
    log_info( "Entering Cross Reference of Vocabulary between DB and Cache ..." );

    this->vocab = vocabulary;
    for (const auto& entry : this->vocab) {
        Record record;
        if (this->isInVocabulary(entry.first, record)) { 
            record.frequency += entry.second;
        } else {
            record.frequency = entry.second;
        }
        record.token = entry.first;
        saveVocabulary(record);
        log_wdetail( "Saved tokens: {0} {1}", wstringToUtf8(record.token).c_str(), record.frequency );
    }
    log_detail("Completed Cross Reference of Vocabulary between DB and Cache");
}

/************************************************************************************************
* Embeddings::prefetchVocabularyToCache
* Function to fetch the embeddings from the vector database for given tokens in the current corpus.
*************************************************************************************************/
template <class T>
void Embeddings<T>::prefetchVocabularyToCache(const std::vector<std::vector<std::wstring>>& corpus) {
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
        log_warning( "Error opening database: {0}", sqlite3_errmsg(this->db) );
        std::cerr << "Error opening database: " << sqlite3_errmsg(this->db) << std::endl;
        return;
    }

    // Prepare the query to fetch embeddings for tokens in the corpus
    std::string query = "SELECT frequency, tokenIndex, token FROM vocabulary WHERE token IN (";
    for (const auto& sentence : corpus) {
        for (const auto& token : sentence) { 
            std::string utf8Str =  wstringToUtf8(token);
            replaceAll( utf8Str, "'", "''"); // handle escape characters.
            query += "'" + utf8Str + "',";
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

    // Fetch and store embeddings in the dynamic embeddings data structure in cache
    // Also create a token hash index structure for fast lookup
    while (sqlite3_step(stmt) == SQLITE_ROW) {

        int frequency = sqlite3_column_int(stmt, 0);
        // int tokenIndex = sqlite3_column_int(stmt, 1); not used for now
        const unsigned char* tokenBytes = sqlite3_column_text(stmt, 2);
        std::string utf8Str(reinterpret_cast<const char*>(tokenBytes));

        std::wstring token = utf8ToWstring(utf8Str);

        std::wcout << "vocab token: " << token << std::endl;

        this->vocab[token] += frequency;
    }

    // Close the database
    sqlite3_finalize(stmt);
    sqlite3_close(db);

    log_detail("Completed Prefetching of Vocabulary to Cache ...");

}

/************************************************************************************************
* Embeddings::prefetchEmbeddingsToCache
* Function to fetch the embeddings for tokens in the cached vocabulary (instead of corpus) 
* from the vector database to cache
*************************************************************************************************/
template <class T>
void Embeddings<T>::prefetchEmbeddingsToCache() {
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
        log_warning( "Error opening database: {0}", sqlite3_errmsg(this->db) );
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

    log_detail("query: {0}", query);

    // Execute the query and fetch the embeddings
    rc = sqlite3_prepare_v2(db, query.c_str(), -1, &stmt, 0);
    if (rc != SQLITE_OK) {
        // Handle query execution error
        log_warning( "No embeddings found ..." );
        sqlite3_close(db);
        return;
    }
 
    log_detail("Start to Cache token hash ...");

    // If cache is unitilialized, then let's initialize
    if (this-wordEmbeddings.size() == 0) {
        initializeEmbeddings(this->vocabSize);
    }

    log_detail("Size of Word Embedding and Bias: {0}x{1} {2}", this->wordEmbeddings.rows(), this->wordEmbeddings.cols(), this->wordBiases.size());

    // Fetch and store embeddings in the dynamic embeddings data structure
    // Also create a token hash index structure for fast lookup
    int currentIndex = 0;
    this->tokenHashToIndex.clear();
    while (sqlite3_step(stmt) == SQLITE_ROW) {
        const unsigned char* tokenHashBytes = sqlite3_column_text(stmt, 0);

        log_detail("1 ...");
        // Convert the token hash from const unsigned char* to std::string
        std::string tokenHash(reinterpret_cast<const char*>(tokenHashBytes));

        const void* embeddingBlob = sqlite3_column_blob(stmt, 1);
        int embeddingSizeBytes = sqlite3_column_bytes(stmt, 1);

        log_detail("2 ...");

        // Convert the BLOB data to Eigen VectorXd (assuming float64 for the embeddings)
        aivector<T> embeddings(embeddingSizeBytes / sizeof(double));
        std::memcpy(embeddings.data(), embeddingBlob, embeddingSizeBytes);

        double bias = sqlite3_column_double(stmt, 2);

        log_detail("3 ...");

        // Find the token based on token hash.
        if (this->tokenHashToIndex.find(tokenHash) == this->tokenHashToIndex.end()) {

            log_detail("3a ...");

            // Create the token hash-to-index mapping
            this->tokenHashToIndex[tokenHash] = currentIndex;

            log_detail("3b ...");

            // Create index-to-token mapping
            this->wordEmbeddings.row(currentIndex) = embeddings;

            log_detail("3c ...{0} {1}", currentIndex, this->wordBiases.size());

            // Add some bias
            this->wordBiases(currentIndex) = bias;

            log_detail("3d ... ");

            currentIndex++;
        }

        log_detail("4 ...");

    }

    // Close the database
    sqlite3_finalize(stmt);
    sqlite3_close(db);

    log_detail("Completed Prefetching of Embedding to Cache ...");

}

/************************************************************************************************
* Embeddings::updateEmbeddingsInDatabase
* Update Embeddings in the Database
*************************************************************************************************/
template <class T>
void Embeddings<T>::updateEmbeddingsInDatabase(const aimatrix<T>& wordEmbeddings,
                                           const aivector<T>& wordBiases) {
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
        record.vectorValue = this->wordEmbeddings.row(index);
        record.bias = this->wordBiases(index);
        saveEmbeddings(record);
    }

}

/************ Tokenizer / Embeddings initialize template ************/

template class Embeddings<float>;  // Instantiate with float
template class Embeddings<double>;  // Instantiate with double