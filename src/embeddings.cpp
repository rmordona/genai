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
   
    try {

        //openDB();

        // Create Vector Table
        createVectorTable();

        // Create Vocabulary Table
        createVocabularyTable();

        //closeDB();

    } catch (const AIException& e) {
        std::cerr << "Error (Embeddings::initializeVectorDB): " << e.what() << std::endl;
        //closeDB();
    } catch (const std::exception& e) {
        // Catch standard exceptions
        std::cerr << "Standard Error (Embeddings::initializeVectorDB): " << e.what() << " at " << __LINE__ << std::endl;
        //closeDB();
    } catch (...) {
        // Catch all other exceptions
        std::cerr << "Unknown Error (Embeddings::initializeVectorDB):" << std::endl;
        //closeDB();
    }
}
 
/************************************************************************************************
* Embeddings::seedVocabularyDB
* Function to seed the Vocabulary DB
*************************************************************************************************/

template <class T>
void Embeddings<T>::seedVocabularyDB(std::unordered_map<std::wstring, int>& vocabulary) {
    try {

        //openDB();

        int currentIndex = 0;
        for (const auto& entry : vocabulary) {

            Record record;
            record.token = entry.first;
            record.frequency = entry.second;
            record.tokenIndex = currentIndex;
            saveVocabulary(this->db, record);
            currentIndex++;
        }

        //closeDB();

    } catch (const AIException& e) {
        std::cerr << "Error (Embeddings::seedVocabularyDB): " << e.what() << std::endl;
        //closeDB();
    } catch (const std::exception& e) {
        // Catch standard exceptions
        std::cerr << "Standard Error (Embeddings::seedVocabularyDB): " << e.what() << " at " << __LINE__ << std::endl;
        //closeDB();
    } catch (...) {
        // Catch all other exceptions
        std::cerr << "Unknown Error (Embeddings::seedVocabularyDB):" << std::endl;
        //closeDB();
    }

}

template <class T>
void Embeddings<T>::createInitialVocabulary(std::unordered_map<std::wstring, int>& vocabulary) {

    log_info( "=======================================" );
    log_info( "Entering Creation of Initial Vocabulary  ..." );

    log_info( "Size of initial vocabulary: {0}", vocabulary.size());

    this->vocabSize = vocabulary.size();
    this->vocab = vocabulary;

    seedVocabularyDB(vocabulary);
}

/************************************************************************************************
* Embeddings::seedVectorDB
* Function to seed the Vector dB with hashed values.
*************************************************************************************************/

template <class T>
void Embeddings<T>::seedVectorDB() {

    try {

        //openDB();

        int currentIndex = 0;
        for (const auto& entry : this->vocab) {
            Record record;

            record.hashKey = sha256(entry.first);
            record.embedding = this->wordEmbeddings.row(currentIndex);
            record.bias = this->wordBiases[currentIndex];
            saveEmbeddings(this->db, record);

            currentIndex++;
        }

        //closeDB();

    } catch (const AIException& e) {
        std::cerr << "Error (Embeddings::seedVectorDB): " << e.what() << std::endl;
        //closeDB();
    } catch (const std::exception& e) {
        // Catch standard exceptions
        std::cerr << "Standard Error (Embeddings::seedVectorDB): " << e.what() << " at " << __LINE__ << std::endl;
        //closeDB();
    } catch (...) {
        // Catch all other exceptions
        std::cerr << "Unknown Error (Embeddings::seedVectorDB):" << std::endl;
        //closeDB();
    }

}




/************************************************************************************************
* Embeddings::createVectorTable
* Function to create a vector table and a vocabulary lookup table
*************************************************************************************************/
template <class T>
void Embeddings<T>::createVectorTable() {
    std::cout << "Creating corpus embeddings ..." << std::endl;
    const char* createTableSQL = "CREATE TABLE IF NOT EXISTS corpus_embeddings ("
                                 "hash_key TEXT PRIMARY KEY, "
                                 "embedding BLOB, "
                                 "bias real "
                                 ");";

    char* errorMsg;
    int rc = sqlite3_exec(this->db, createTableSQL, 0, 0, &errorMsg);
    if (rc != SQLITE_OK) {
        log_warning( "SQL error (Embeddings::createVectorTable): {0}", errorMsg );
        std::cerr << "SQL error (Embeddings::createVectorTable): " << errorMsg << std::endl;
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
        log_warning( "SQL error (Embeddings::createVocabularyTable): {0}", errorMsg );
        std::cerr << "SQL error (Embeddings::createVocabularyTable): " << errorMsg << std::endl;
        sqlite3_free(errorMsg);
        throw AIException("Error Creating (vocabulary) table");
    }

    const char* createIndexSQL = "CREATE UNIQUE INDEX vocab_index ON vocabulary (tokenIndex);";

    rc = sqlite3_exec(this->db, createIndexSQL, 0, 0, &errorMsg);
    if (rc != SQLITE_OK) {
        // No need to log issue.
        log_warning( "SQL error Embeddings::createVocabularyTable):: {0} {1}",  rc, errorMsg );
        std::cerr << "SQL error Embeddings::createVocabularyTable):: " << errorMsg << std::endl;
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
void Embeddings<T>::saveVocabulary(sqlite3* db, const Record& record) {
    std::stringstream insertSQL;
    // The tokenIndex column is defined with AUTOINCREEMNT
    insertSQL << "INSERT OR REPLACE INTO vocabulary (token, frequency) VALUES (?, ?);";

    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db, insertSQL.str().c_str(), -1, &stmt, 0);
    if (rc != SQLITE_OK) {
        sqlite3_finalize(stmt); // Release the stmt memory
    }
    sqlite3_assert(rc, "Error preparing statement (Embeddings::saveVocabulary)");

    // Bind the token
    std::string utf8Str = wstringToUtf8(record.token);
    sqlite3_bind_text(stmt, 1, utf8Str.c_str(), -1, SQLITE_STATIC);

    // Bind the frequency
    sqlite3_bind_int(stmt, 2, record.frequency);

    rc = sqlite3_step(stmt);
    sqlite3_assert_done(rc, "Error inserting data (Embeddings::saveVocabulary)");

    sqlite3_finalize(stmt);
}

/************************************************************************************************
* Embeddings::saveEmbeddings
* Function to insert into or replace an embedding into the vector table
*************************************************************************************************/
template <class T>
void Embeddings<T>::saveEmbeddings(sqlite3* db, const Record& record) {
    std::stringstream insertSQL;
    insertSQL << "INSERT OR REPLACE INTO corpus_embeddings (hash_key, embedding, bias) VALUES (?, ?, ?);";
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db, insertSQL.str().c_str(), -1, &stmt, 0);
    if (rc != SQLITE_OK) {
        sqlite3_finalize(stmt); // Release the stmt memory
    }
    sqlite3_assert(rc, "Error preparing statement (Embeddings::saveEmbeddings)");

    // Bind the hash key
    sqlite3_bind_text(stmt, 1, record.hashKey.c_str(), -1, SQLITE_STATIC);
    if (rc != SQLITE_OK) {
        sqlite3_finalize(stmt); // Release the stmt memory
    }
 
    // Bind the vector embedding
    sqlite3_bind_blob(stmt, 2, record.embedding.data(), record.embedding.size() * sizeof(T), SQLITE_STATIC);
    if (rc != SQLITE_OK) {
        sqlite3_finalize(stmt); // Release the stmt memory
    }

    // Bind the bias term
    sqlite3_bind_double(stmt, 3, static_cast<double>(record.bias));
    if (rc != SQLITE_OK) {
        sqlite3_finalize(stmt); // Release the stmt memory
    }

    rc = sqlite3_step(stmt);
    sqlite3_assert_done(rc, "Error inserting data (Embeddings::saveEmbeddings)");

    sqlite3_finalize(stmt);

}

/************************************************************************************************
* Embeddings::getEmbeddings
* Function to retrieve an embedding from the database based on the hash key
*************************************************************************************************/

template <class T>
bool Embeddings<T>::retrieveEmbeddings(const std::string& hashKey, Record& record) {
    std::stringstream selectSQL;
    selectSQL << "SELECT embedding, bias FROM corpus_embeddings WHERE hash_key = ?;";

    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(this->db, selectSQL.str().c_str(), -1, &stmt, 0);
    if (rc != SQLITE_OK) {
        log_warning( "Error preparing statement (Embeddings::retrieveEmbeddings): {0}", sqlite3_errmsg(this->db));
        std::cerr << "Error preparing statement (Embeddings::retrieveEmbeddings): " << sqlite3_errmsg(this->db) << std::endl;
        sqlite3_finalize(stmt); // Release the stmt memory
        return false;
    }

    // Bind the hash key
    sqlite3_bind_text(stmt, 1, hashKey.c_str(), -1, SQLITE_STATIC);

    rc = sqlite3_step(stmt);
    if (rc == SQLITE_ROW) {
        record.hashKey = hashKey;

        // Get the embedding value from the result and store it in the record
        const void* data = sqlite3_column_blob(stmt, 0);
        int size = sqlite3_column_bytes(stmt, 0);
        record.embedding.resize(size / sizeof(T)); // let column hold double regardless of T (typeclass)
        std::memcpy(record.embedding.data(), data, size);

        // Get the bias term
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
        log_warning( "Error preparing statement (Embeddings::retrieveVocabulary): {0}", sqlite3_errmsg(this->db));
        std::cerr << "Error preparing statement (Embeddings::retrieveVocabulary): " << sqlite3_errmsg(this->db) << std::endl;
        sqlite3_finalize(stmt); // Release the stmt memory
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

    sqlite3_stmt* stmt;

    try {

        //openDB();

        std::stringstream selectSQL;
        selectSQL << "SELECT frequency FROM vocabulary WHERE token = ?;";

        int rc = sqlite3_prepare_v2(this->db, selectSQL.str().c_str(), -1, &stmt, 0);
        if (rc != SQLITE_OK) {
            sqlite3_finalize(stmt); // Release the stmt memory
        }
        sqlite3_assert(rc, "Error preparing statement (Embeddings::isInVocabulary)");

        // Bind the hash key
        std::string utf8Str = wstringToUtf8(token);

        sqlite3_bind_text(stmt, 1, utf8Str.c_str(), -1, SQLITE_STATIC);

        rc = sqlite3_step(stmt);

        if (rc == SQLITE_ROW) {

            sqlite3_finalize(stmt);

            //closeDB();

            return true;
        }

        sqlite3_finalize(stmt);

        //closeDB();

    } catch (const AIException& e) {
        std::cerr << "Error (Embeddings::isInVocabulary): " << e.what() << std::endl;
        //closeDB();
        return false;
    } catch (const std::exception& e) {
        // Catch standard exceptions
        std::cerr << "Standard Error (Embeddings::isInVocabulary): " << e.what() << " at " << __LINE__ << std::endl;
        //closeDB();
        return false;
    } catch (...) {
        // Catch all other exceptions
        std::cerr << "Unknown Error (Embeddings::isInVocabulary):" << std::endl;
        //closeDB();
        return false;
    }

    return false;
}

template <class T>
bool Embeddings<T>::isInVocabulary(const std::wstring& token, Record& record)  {
    return retrieveVocabulary(token,record);
}

/************************************************************************************************
* Embeddings::initializeEmbeddingsinCache 
* Function to generate the initial word embedding. We require the size of the
* constructed vocabulary
*************************************************************************************************/
template <class T>
void Embeddings<T>::initializeEmbeddingsinCache(int row, int col) {

    this->wordEmbeddings = aimatrix<T>::Zero(row, col);
    this->wordBiases = aivector<T>::Zero(row);  // bias term for target word

    // Initialize word Embeddings
    BaseOperator::heInitMatrix(this->wordEmbeddings);

}

/************************************************************************************************
* Embeddings::createInitialEmbeddings
* Function to generate or create the initial word embedding. We require the size of the
* constructed vocabulary. We assume that the vocabulary is already pre-populated and
* fetched into cache (this->vocab)
*************************************************************************************************/
template <class T>
void Embeddings<T>::createInitialEmbeddings(int embeddingSize, std::unordered_map<std::wstring,int> vocabulary) {

    log_info( "===============================================================" );
    log_info( "Entering Creation of Initial Embeddings ..." );

    this->vocabSize = vocabulary.size();
    this->vocab = vocabulary;

    log_detail("Initializing Embeddings ... using embedding size {0} and vocab of size: {1}", embeddingSize, this->vocab.size());

    if (embeddingSize > 0 && this->vocab.size() > 0) {

        this->embeddingSize = embeddingSize;

        // Initialize Embeddings
        initializeEmbeddingsinCache( this->vocabSize, this->embeddingSize );

        // Seed Embeddings / Vector DB
        this->seedVectorDB();
    } 
}
 
/************************************************************************************************
* Embeddings::crossReferenceVocabularyinDBandCache
* Cross Reference Vocabulary between memory and DB.  Sync in-memory vocab into DB.
* If vocabulary is found in the DB, then increment the frequency; otherwise,
* save the new vocabulary with initial count. 
*************************************************************************************************/
template <class T>
void Embeddings<T>::crossReferenceVocabularyinDBandCache(std::unordered_map<std::wstring,int> vocabulary) {

    log_info( "===============================================================" );
    log_info( "Entering Cross Reference of Vocabulary between DB and Cache ..." );

    try {

        //openDB();

        this->vocabSize = vocabulary.size();
        this->vocab = vocabulary;
        for (const auto& entry : this->vocab) {
            Record record;
            if (this->isInVocabulary(entry.first, record)) { 
                record.frequency += entry.second;
            } else {
                record.frequency = entry.second;
            }
            record.token = entry.first;
            saveVocabulary(this->db, record);
            log_wdetail( "Saved tokens: {0} {1}", wstringToUtf8(record.token).c_str(), record.frequency );
        }
        log_detail("Completed Cross Reference of Vocabulary between DB and Cache");

        //closeDB();
        //sqlite3_close(db); 
        //this->db = nullptr;

    } catch (const AIException& e) {
        std::cerr << "Error (Embeddings::crossReferenceVocabularyinDBandCache): " << e.what() << std::endl;
        //closeDB();
    } catch (const std::exception& e) {
        // Catch standard exceptions
        std::cerr << "Standard Error (Embeddings::crossReferenceVocabularyinDBandCache): " << e.what() << " at " << __LINE__ << std::endl;
        //closeDB();
    } catch (...) {
        // Catch all other exceptions
        std::cerr << "Unknown Error (Embeddings::crossReferenceVocabularyinDBandCache):" << std::endl;
        //closeDB();
    }

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

    sqlite3_stmt* stmt;

    try {

        //openDB();

        // Prepare the query to fetch embeddings for tokens in the corpus
        for (const auto& sentence : corpus) {

            std::string query = "SELECT frequency, tokenIndex, token FROM vocabulary WHERE token IN (";
            for (const auto& token : sentence) { 
                std::string utf8Str =  wstringToUtf8(token);
                replaceAll( utf8Str, "'", "''"); // handle escape characters.
                query += "'" + utf8Str + "',";
            }

            query.pop_back(); // Remove the last comma
            query += ");";

            // Execute the query and fetch the embeddings
            int rc = sqlite3_prepare_v2(this->db, query.c_str(), -1, &stmt, 0);
            sqlite3_assert(rc, "No vocabulary found ...");

            // Fetch and store embeddings in the dynamic embeddings data structure in cache
            // Also create a token hash index structure for fast lookup

            while (sqlite3_step(stmt) == SQLITE_ROW) {

                int frequency = sqlite3_column_int(stmt, 0);
                const unsigned char* tokenBytes = sqlite3_column_text(stmt, 2);
                std::string utf8Str(reinterpret_cast<const char*>(tokenBytes));

                std::wstring token = utf8ToWstring(utf8Str);

                this->vocab[token] += frequency;

            }

            sqlite3_finalize(stmt);

        }

        //closeDB();
        // sqlite3_close(db); 

    } catch (const AIException& e) {
        std::cerr << "Error (Embeddings::prefetchVocabularyToCache): " << e.what() << std::endl;
        //closeDB();
    } catch (const std::exception& e) {
        // Catch standard exceptions
        std::cerr << "Standard Error (Embeddings::prefetchVocabularyToCache): " << e.what() << " at " << __LINE__ << std::endl;
        //closeDB();
    } catch (...) {
        // Catch all other exceptions
        std::cerr << "Unknown Error (Embeddings::prefetchVocabularyToCache):" << std::endl;
        //closeDB();
    }


    log_detail("Completed Prefetching of Vocabulary to Cache ...");

}

/************************************************************************************************
* Embeddings::generateTokenIndices
* Function to pre-geneate the token indices
*************************************************************************************************/

template <class T>
void Embeddings<T>::generateTokenIndices() {

    this->tokenHashToIndex.clear();

    log_detail("Initial size of TokenIndex: {0}", this->tokenHashToIndex.size());

    int currentIndex = 0;  
    // Let's generate the Token Indices. Note that this->tokens is generated from buildCoMatrix().
    for (const auto& token : this->tokens) {

        // Prepare the query to fetch embeddings for tokens in the corpus
        std::string tokenHash = sha256(token.first);  

        // Create the token hash-to-index mapping
        this->tokenHashToIndex[tokenHash] = currentIndex;   // This is an in-memory index only

        currentIndex ++;
    }

    log_detail("Final size of TokenIndex: {0}", this->tokenHashToIndex.size());

}

/************************************************************************************************
* Embeddings::fetchEmbeddings
* Function to fetch the embeddings for tokens in the cached vocabulary (instead of corpus) 
* from the vector database to cache
*************************************************************************************************/
template <class T>
int Embeddings<T>::fetchEmbeddings(sqlite3* db, std::string query, int currentIndex) {
    // Execute the query and fetch the embeddings
    sqlite3_stmt* stmt;
    int rc = sqlite3_prepare_v2(db, query.c_str(), -1, &stmt, 0);
    sqlite3_assert(rc, "No embeddings found ");
 
    // Fetch and store embeddings in the dynamic embeddings data structure
    // Also create a token hash index structure for fast lookup

    int fetched = 0;
    // For every token that exists in the Embeddings DB, update the initialized wordEmbedding in Cache
    // otherwise, each entry in the Cache takes the initial he Initialization value. See prefetchEmbeddingsToCache().
    while (sqlite3_step(stmt) == SQLITE_ROW) {

        fetched++;
        const unsigned char* tokenHashBytes = sqlite3_column_text(stmt, 0);

        // Convert the token hash from const unsigned char* to std::string
        std::string tokenHash(reinterpret_cast<const char*>(tokenHashBytes));

        const void* embeddingBlob = sqlite3_column_blob(stmt, 1);
        int embeddingSizeBytes = sqlite3_column_bytes(stmt, 1);

        // Convert the BLOB data to Eigen VectorXd (assuming float64 for the embeddings)
        aivector<T> embeddings(embeddingSizeBytes / sizeof(double));
        std::memcpy(embeddings.data(), embeddingBlob, embeddingSizeBytes);

        // Get the bias term
        T bias = sqlite3_column_double(stmt, 2);

        // Check if missing token based on token hash, then create.
        if (this->tokenHashToIndex.find(tokenHash) == this->tokenHashToIndex.end()) {

            // Create index-to-token mapping
            this->wordEmbeddings.row(currentIndex) = embeddings;
            this->wordBiases[currentIndex] = bias;

            currentIndex++;

        } 
    }

    sqlite3_finalize(stmt);

    return currentIndex;

}

template <class T>
void Embeddings<T>::prefetchEmbeddingsToCache() {
    // Assuming you have a SQLite database connection and a table called "corpus_embeddings"
    // with columns "token_hash" (INTEGER) and "embedding" (BLOB)

    log_info( "==============================================" );
    log_info( "Entering Prefetching of Embedding to Cache ...");

    // Open the database and prepare the query

    try {

        //openDB();

        log_detail("Before Size of Word Embeddings: {0}x{1}", this->wordEmbeddings.rows(), this->wordEmbeddings.cols());

        // This assumes the Tokens are already built for the corpus generated.
        // Therefore, initialize the word embedding in Cache then use the list to update the embedding
        // from DB if it exists. Rest of the list remains initialized using he-initialization.
        initializeEmbeddingsinCache(this->tokens.size(), this->embeddingSize);

        log_detail("Size of Initialized Word Embeddings: {0}x{1}", this->wordEmbeddings.rows(), this->wordEmbeddings.cols());
        log_detail("Size of Bias: {0}", this->wordBiases.size());

        // Use the cached vocabulary
        // fetch every 20
        // int fetch = 1, fetch_limit = 20;

        int currentIndex = 0;  // This is an in-memory index only
        std::vector<std::string> vquery;
        std::string query; 

        log_detail("Size of Tokens: {0} ", this->tokens.size());

        // Note that this->tokens is generated from buildCoMatrix().
        for (const auto& token : this->tokens) {
            Record record;
            bool row = retrieveEmbeddings(sha256(token.first), record);
            // if token is not found in the stored embedding (in db), then let it stay initialized
            // otherwise, we pin retrieved embeddings into cache.
            if (row) {
                this->wordEmbeddings.row(currentIndex) = record.embedding;
                this->wordBiases[currentIndex] = record.bias;
            }
            currentIndex++;

        }

        log_detail( "Final Current Index: {0}", currentIndex );

        //closeDB();

    } catch (const AIException& e) {
        std::cerr << "Error (Embeddings::prefetchEmbeddingsToCache): " << e.what() << std::endl;
        //closeDB();
    } catch (const std::exception& e) {
        // Catch standard exceptions
        std::cerr << "Standard Error (Embeddings::prefetchEmbeddingsToCache): " << e.what() << " at " << __LINE__ << std::endl;
        //closeDB();
    } catch (...) {
        // Catch all other exceptions
        std::cerr << "Unknown Error (Embeddings::prefetchEmbeddingsToCache):" << std::endl;
        //closeDB();
    }

    log_detail("Completed Prefetching of Embedding to Cache ...");

}
 
/************************************************************************************************
* Embeddings::updateEmbeddingsInDatabase
* Update Embeddings in the Database
*************************************************************************************************/
template <class T>
void Embeddings<T>::updateEmbeddingsInDatabase(const aimatrix<T>& wordEmbeddings, const aivector<T>& wordBiases) {

    try {
 
        Record record;

        for (const auto& indexTokenPair : this->tokenHashToIndex) {
            int index = indexTokenPair.second;
            record.hashKey   = indexTokenPair.first;
            record.embedding = this->wordEmbeddings.row(index);
            record.bias      = this->wordBiases[index];
            saveEmbeddings(this->db, record);
        }

        this->wordEmbeddings.array() = wordEmbeddings.array();
        this->wordBiases.array()     = wordBiases.array(); 

        //closeDB();

    } catch (const AIException& e) {
        std::cerr << "Error (Embeddings::updateEmbeddingsInDatabase): " << e.what() << std::endl;
        //closeDB();
    } catch (const std::exception& e) {
        // Catch standard exceptions
        std::cerr << "Standard Error (Embeddings::updateEmbeddingsInDatabase): " << e.what() << " at " << __LINE__ << std::endl;
        //closeDB();
    } catch (...) {
        // Catch all other exceptions
        std::cerr << "Unknown Error (Embeddings::updateEmbeddingsInDatabase):" << std::endl;
        //closeDB();
    }

}

// This is batch level co-occurrence
template <class T>
void Embeddings<T>::buildCoMatrix(const std::vector<std::vector<std::wstring>>& corpus, int batchSize) {

    log_info( "==========================================" );
    log_info( "Entering buildCoMatrices ...");

    int corpus_size = static_cast<int>(corpus.size());

    // Global Vector
    std::unordered_map<std::wstring, int> comatrix;
    std::unordered_map<std::wstring, std::vector<std::wstring>> tokens; 

    for (int batchStart = 0; batchStart < corpus_size; batchStart += batchSize) {
        int batchEnd = std::min(batchStart + batchSize, corpus_size);

        // Iterate over each sentence in the batch
        for (int i = batchStart; i < batchEnd; ++i) {
            const auto& sentence = corpus[i];

            // Iterate over each token in the sentence
            for (int j = 0; j < (int) sentence.size(); j++) {
                std::wstring targetWord = sentence[j];

                // Get the context window for the target word
                // context window being 3 words
                int start = std::max(0, j - 3);
                int end = std::min(corpus_size - 1, j + 3);

                // Iterate over the context token
                for (int k = start; k <= end && k < (int) sentence.size(); k++) {
                    // Skip the target token itself
                    if (j == k) continue;
                    std::wstring contextWord = sentence[k]; 

                    if (targetWord == TK_SOS_) continue;
                    if (targetWord == TK_EOS_) continue;
                    if (targetWord == TK_PAD_) continue;

                    if (contextWord == TK_SOS_) continue;
                    if (contextWord == TK_EOS_) continue;
                    if (contextWord == TK_PAD_) continue;

                    // include only pairs with co-occurrence, eliminating building a sparse matrix
                    std::wstring cooccur = targetWord + contextWord;
                    comatrix[ cooccur ] ++;  
                    if (comatrix[cooccur] < 2) { // capture only unique context words
                        //x++;
                        tokens[ targetWord ].push_back(contextWord);
                    }

                }

            }
        }
    }

    this->comatrix = comatrix;
    this->tokens = tokens;

    this->tokens[TK_SOS_] = {}; //.push_back(TK_SOS_);
    this->tokens[TK_EOS_] = {}; //.push_back(TK_EOS_);
    this->tokens[TK_PAD_] = {}; //.push_back(TK_PAD_);

    log_detail("Size of comatrices: {0}", this->comatrix.size());
    //log_detail("Size of comatrices: {0}", x);

}

/************ Tokenizer / Embeddings initialize template ************/

template class Distance<float>;  // Instantiate with float
template class Distance<double>;  // Instantiate with double

template class Embeddings<float>;  // Instantiate with float
template class Embeddings<double>;  // Instantiate with double