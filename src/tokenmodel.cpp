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

/******************************************************************************************************************************
 * Notes (From bag of words to becoming part of a vocabulary for word embedding lookup):
 * 
 * 1. It begins by creating a vocabulary using BPE as an example.
 * 
 *  The size of the initial word embedding matrix in Transformer frameworks is typically determined by the vocabulary size and 
 *  the desired embedding dimension.
 *
 *  The vocabulary size is the total number of unique tokens in the training corpus. Each token is assigned an index in the 
 *  vocabulary, ranging from 0 to (vocab_size - 1). The embedding dimension is the size of the vector representation for each token.
 * 
 *  For example, if you have a vocabulary size of 50,000 tokens and you want each token to be represented by a 768-dimensional 
 *  vector, then your initial word embedding matrix would have dimensions of 50,000 x 768.
 * 
 *  The entries in the embedding matrix are randomly initialized before training begins. These random values will be updated 
 *  during the training process to learn meaningful representations for the tokens based on the task and the context in which 
 *  they appear.
 * 
 * 2. The randomly initialized embedding matrix is then trained (either using Word2Vec, GloVe, or fastText). Otherwise,
 *    learned from scatch.
 * 
 *  In order to train the word embedding matrix, it is common to use a large corpus of text data. The embedding matrix is 
 *  randomly initialized as a starting point, and then the model is trained on the corpus to optimize the embeddings based 
 *  on the specific task at hand.
 * 
 *  During the training process, the model learns to adjust the values of the embedding matrix based on the patterns and 
 *  relationships it discovers in the training data. By exposing the model to a diverse and representative corpus, it can 
 *  learn meaningful representations of words that capture their semantic and syntactic properties.
 *
 *  The training objective typically involves optimizing the model's parameters, including the embedding matrix, to minimize 
 *  a loss function. This loss function is task-dependent and can vary based on the specific NLP task being addressed, such 
 *  as language modeling, sentiment analysis, or machine translation.
 *
 *  By training the embedding matrix on a large corpus, the model can learn to encode useful information about word 
 *  meanings and relationships. These embeddings can then be used for downstream tasks, such as text classification or 
 *  information retrieval, where the learned representations can help capture semantic similarities and contextual information.
 *
 *  It's important to note that the size and quality of the training corpus can significantly impact the performance and 
 *  generalization of the word embeddings. A larger and more diverse corpus can lead to better embeddings, as the model 
 *  has more exposure to different contexts and linguistic patterns.
 *
 * 3. The pre-trained embedding matrix becomes a lookup-up along with the vocabulary to generate word embeddings for 
 *    modeling LLMs.
 * 
 *  Once the word embedding matrix is trained, it can serve as a lookup table for word representations in a vocabulary. 
 *  Each word in the vocabulary is associated with a vector representation, which is obtained by looking up the corresponding 
 *  row in the trained embedding matrix.
 *
 *  The word representations obtained from the embedding matrix are often further enhanced by incorporating positional 
 *  information using positional embedding methods. In transformer-based models, positional encoding is commonly used to 
 *  provide the model with information about the relative positions of words within a sequence.
 *
 *  Positional encoding is typically achieved by adding position-specific vectors to the word embeddings. These positional 
 *  vectors capture information about the order or position of words in the input sequence, allowing the model to differentiate 
 *  between words based on their positions. The specific method for generating positional encodings can vary, but popular 
 *  approaches include sine and cosine functions or learned positional embeddings.
 *
 *  By combining the word embeddings with positional encodings, the model can capture both the meaning of individual 
 *  words and their relative positions within the input sequence. This allows the model to encode contextual information 
 *  and capture dependencies between words effectively, enabling it to understand the sequential nature of the text data.
 * 
 * Summary: 
 * 
 * The process of obtaining a 768-dimensional embedding for a unique token, e.g. 'Moon', involves using a pre-trained word 
 * embedding model or learning one from scratch. The vocabulary provides the mapping between the tokens and their 
 * corresponding indices, but it doesn't directly contain the embeddings themselves.
 *
 * Typically, word embeddings are learned from large corpora using techniques like Word2Vec, GloVe, or fastText. These methods 
 * aim to capture semantic and contextual information about words by analyzing the distributional patterns of words in the
 * training data. The embeddings are learned in such a way that words with similar meanings or usage contexts have similar 
 * vector representations.
 *
 * Once the word embeddings are learned, they are stored as a separate matrix or embedding layer in the model. Each row 
 * of the embedding matrix corresponds to a word in the vocabulary, and the 768-dimensional vector associated with the 
 * word is retrieved by its index. So, when the token "Moon" is looked up in the vocabulary and its index is determined to 
 * be 1, the corresponding row in the embedding matrix is retrieved, which represents the 768-dimensional embedding for 
 * the word "Moon".
 * 
 * In summary, the embeddings themselves are derived from pre-training or training on large corpora using specific algorithms, 
 * and the vocabulary serves as a mapping between the tokens and their indices, allowing the embeddings to be efficiently 
 * retrieved during the model's operation.
 * 
 * Notes 1: 
 * 
 * Having a large and diverse corpus can indeed be beneficial when training word embeddings or building models for 
 * natural language understanding. The size and diversity of the corpus allow the model to capture a broader range of 
 * linguistic patterns, contextual relationships, and semantic information.
 *
 * In the case of word embeddings, a larger corpus provides more examples of word co-occurrences and usage contexts, 
 * enabling the model to learn richer representations for words. By exposing the model to a wide variety of texts, 
 * including different domains, genres, and languages, it becomes more capable of understanding and generalizing across 
 * various linguistic phenomena.
 *
 * However, it's important to note that the size of the corpus is not the sole determinant of the model's performance. 
 * The quality of the data, the preprocessing steps, the training algorithm, and other factors also play crucial roles 
 * in achieving effective and meaningful representations.
 *
 * While a large corpus can be beneficial, it's worth considering that the size of the corpus used for training is often 
 * limited by practical considerations such as computational resources and time constraints. Researchers and practitioners 
 * often strike a balance between the available resources and the desired level of performance when choosing the corpus 
 * size and composition for their specific applications.
 *
 * Ultimately, the goal is to have a representative and diverse corpus that covers a wide range of language patterns and 
 * usages, enabling the model to learn robust and generalizable representations.
 * 
 * Note 2:
 * 
 * Having different sets of vocabulary and corresponding embedding matrices for different perspectives or domains can be 
 * beneficial in various applications. This approach is often referred to as domain-specific or domain-adapted word embeddings.
 *
 * By creating separate vocabularies and embedding matrices for different domains, you can capture domain-specific semantics, 
 * concepts, and contextual relationships more effectively. This allows the model to tailor its understanding and representation 
 * of words to specific domains, improving the performance and relevance of the embeddings within those domains.
 *
 * For example, in natural language processing tasks related to specific domains such as biology, finance, or legal 
 * documents, using domain-specific embeddings can help the model better understand and handle the specialized terminology 
 * and context-specific language patterns in those domains.
 *
 * Creating domain-specific embeddings typically involves training word embeddings on domain-specific corpora or datasets that 
 * are specific to the target domain. This allows the embeddings to capture the specific nuances and characteristics of the 
 * domain, resulting in more accurate and relevant representations.
 * 
 * By employing different sets of vocabulary and embedding matrices for different perspectives or domains, you can enhance the 
 * model's ability to handle diverse and specialized language use cases, leading to improved performance and domain-specific 
 * understanding.
*******************************************************************************************************************************/
#include "genai.h"

namespace py = pybind11;
using namespace py::literals;

#define TK_SPACE_ L"<SPC>"
#define TK_MASK_  L"<MASK>"
#define TK_UNK_   L"<UNK>"
#define TK_PAD_   L"<PAD>"
#define TK_SOS_   L"<SOS>"
#define TK_EOS_   L"<EOS>"

std::vector<std::wstring> TokenModel::splitString(const std::wstring& str) {
    std::vector<std::wstring> words;
    size_t start = 0;
    size_t end = str.find(L' ');
    while (end != std::wstring::npos) {
        std::wstring word = str.substr(start, end - start);
        words.push_back(word);
        start = end + 1;
        end = str.find(L' ', start);
    }
    // Add the last word (or the only word if there are no spaces)
    std::wstring lastWord = str.substr(start);
    if (!lastWord.empty()) {
        words.push_back(lastWord);
    }
    return words;
}

std::vector<std::wstring> BPETokenizer::tokenizeCorpus(const std::vector<std::wstring>& corpus) {
    std::vector<std::wstring> tokens;

    log_info( "===========================" );
    log_info( "Tokenizing into Corpus  ..." );

    for (const std::wstring& sentence : corpus) {
        for (auto it = sentence.begin(); it != sentence.end(); ++it) {
            wchar_t ch = *it;
            std::wstring token(1, ch);
            // Access the next character using the iterator
            // Check if the iterator has reached the second-to-last character
            if (ch == L' ') {
                continue;
            } else
            if (std::next(it) != sentence.end()) {
                wchar_t nextCh = *(std::next(it));
                if (nextCh == L' ') {
                    token += TK_SPACE_;
                }
            } else {
                token += TK_SPACE_;
            }

            tokens.push_back(token);
            this->vocab[token]++;
        }
    }

    return tokens;
}

bool BPETokenizer::endsWith(const std::wstring& str, const std::wstring& suffix) {
    if (str.length() < suffix.length()) {
        return false;
    }
    return str.substr(str.length() - suffix.length()) == suffix;
}

void BPETokenizer::mergeTokens(std::vector<std::wstring>& tokens, int numMerges) {
    // Perform merging of tokens based on most frequent pairs
    for (int merge = 0; merge < numMerges; ++merge) {
        std::map<std::pair<std::wstring, std::wstring>, int> pairCounts;

        // Count the occurrences of token pairs
        for (size_t i = 0; i < tokens.size() - 1; ++i) {
            const std::wstring& token1 = tokens[i];
            const std::wstring& token2 = tokens[i + 1];

            // if first token is end of words, then ignore merging
            if (endsWith(tokens[i], TK_SPACE_)) {
              // noops
            } else {
               pairCounts[std::make_pair(token1, token2)]++;
            }
        }

        // Find the most frequent pair
        std::pair<std::wstring, std::wstring> maxPair;
        std::map<std::pair<std::wstring, std::wstring>, int> myPair;
        int maxCount = 0;
        for (const auto& entry : pairCounts) {
            // const std::pair<std::wstring, std::wstring>& pair = entry.first;
            int count = entry.second;
            log_wdetail( "Pairing: {} {} :  {:d}", 
                        wstringToUtf8(entry.first.first).c_str() , 
                        wstringToUtf8(entry.first.second).c_str() , count );
            if (count > maxCount || count > 2) {
                maxCount = count;
                myPair[std::make_pair(entry.first.first, entry.first.second)] = entry.second;
            }
        }

        log_wdetail( "Found pair: {} {:d}",  wstringToUtf8( maxPair.first + maxPair.second).c_str(), maxCount );

        // If no more frequent pair is found, exit the merge loop
        if (maxCount == 0) {
            break;
        }
 
        log_detail( "Max Count found: {:d}",  maxCount );

        for (const auto& pair : myPair) {
           const std::pair<std::wstring, std::wstring>& pair_ = pair.first;
           std::wstring mergedToken = pair_.first + pair_.second;
           for (size_t i = 0; i < tokens.size() - 1; ++i) {
               if (tokens[i] == pair_.first && tokens[i + 1] == pair_.second) {
                   tokens[i] = mergedToken;
                   tokens.erase(tokens.begin() + i + 1);
               }
           }
           this->vocab[mergedToken] += pair.second;
        }
    }
}


// Function to pretrain the tokenizer on a corpus. 
// This is where we construct the vocabulary.
void BPETokenizer::pretrain(const std::vector<std::wstring>& sentences, int numMerges,  int embeddingSize) {
    buildVocabulary(sentences, numMerges);
    if (embeddingSize > 0) {
        // Rebuild the Vector Metadata in DB.
        embeddings = new Embeddings();
        embeddings->initializeVectorandVocabMetadata(this->vocab, embeddingSize);
    } else {
        // error handling
    }
}

// Function to train the tokenizer on a corpus.
// This is where we fine-tune the vocabulary.
void BPETokenizer::train(const std::vector<std::wstring>& sentences, int numMerges) {

    log_info( "=======================================" );
    log_info( "Entering Training for BPETokenizer  ..." );

    buildVocabulary(sentences, numMerges);
    // No need to rebuild the Vector Metadata in DB.
    // We only need to sync new word from sentences to the vocabulary; sort of train new words.
    // For new words, the embeddings will be tuned during GloVe training.
    embeddings->crossReferenceVocabularyinDBandCache(this->vocab);
}

// Build / Reset Vocabulary
void BPETokenizer::buildVocabulary(const std::vector<std::wstring>& sentences, int numMerges) {
    std::vector<std::wstring> tokens = tokenizeCorpus(sentences);
    mergeTokens(tokens, numMerges);
    this->vocabSize = this->vocab.size();
}

std::vector<std::wstring> TokenModel::tokenize(const std::wstring& sentence) {
    const std::unordered_map<std::wstring, int> vocabulary = this->vocab;
    std::vector<std::wstring> tokens;
    std::wstring currentToken;

    log_info( "=================================" );
    log_info( "Tokenizing a sentence  ..." );

    std::vector<std::wstring> words;
    words = splitString(sentence);

    for (const auto& word : words) {
        bool found = false;
        std::wstring unit = word + TK_SPACE_;

        if (embeddings->isInVocabulary(unit)) { 
            found = true;
            tokens.push_back(unit);
        } else {
           size_t start = 0;
           size_t end = word.length();

           for (size_t i = start + 1; i <= end; ++i) {
               std::wstring token = unit.substr(start, end - start);
               if (embeddings->isInVocabulary(token)) {
                   found = true;
                   tokens.push_back(unit);
               }
           }
        }
        if (!found) {  // unknown word
            tokens.push_back(word + TK_UNK_);
        }
    }

    return tokens;

}

std::vector<std::vector<std::wstring>> TokenModel::tokenize(const std::vector<std::wstring>& sentences) {
    const std::unordered_map<std::wstring, int> vocabulary = this->vocab;
    std::vector<std::vector<std::wstring>> corpus;
    std::wstring currentToken;

    log_info( "=================================" );
    log_info( "Tokenizing Sentences  ..." );

    for (const auto& sentence : sentences) {

        std::vector<std::wstring> tokens;
        std::vector<std::wstring> words;
        words = splitString(sentence);

        for (const auto& word : words) {
            bool found = false;
            std::wstring unit = word + TK_SPACE_;
            if (embeddings->isInVocabulary(unit)) {
                found = true;
                tokens.push_back(unit);
            } else {
            size_t start = 0;
            size_t end = word.length();

            for (size_t i = start + 1; i <= end; ++i) {
                std::wstring token = unit.substr(start, end - start);
                if (embeddings->isInVocabulary(token)) {
                    found = true;
                    tokens.push_back(unit);
                }
            }
            }
            if (!found) {  // unknown word
                tokens.push_back(word + TK_UNK_);
            }
        }
        corpus.push_back(tokens);
    }

    return corpus;

}

void TokenModel::printVocabulary(int rows) { 
    int count = 0;
    std::cout << "Size of Vocabulary: " << this->vocab.size() << std::endl;
    for (const auto& entry : this->vocab) {
        count++;
        if (count >= rows) break;
        std::wcout << entry.first << L" : " << entry.second << std::endl;
    }
}

void TokenModel::printWordEmbeddings(int rows) { 
    int count = 0;
    // Print the learned word embeddings
    std::cout << "Size of Vocabulary: " << this->vocab.size() << std::endl;
    for (int i = 0; i < (int) this->vocab.size(); ++i) {
        count ++;
        if (count >= rows) break;
        const std::wstring& key = std::next(this->vocab.begin(), i)->first;
        std::wcout << "Word: " << key << ", Embedding: ";
        for (int j = 0; j < this->embeddingSize; ++j) {
            std::cout << this->wordEmbeddings(i, j) << " ";
        }
        std::cout << std::endl;
    }
}

// Function to train GloVe-like model. We tokenize a sentence such that the tokenize function
// requires already the existence of the constructed vocabulary.
void TokenModel::trainGloVe(std::vector<std::wstring>& sentences, int batchSize,
                            float learningRate, int maxIterations) {

    this->learningRate = learningRate;
    this->maxIterations = maxIterations;

    log_info( "=================================" );
    log_info( "Entering GloVe-like Training  ..." );

    std::vector<std::vector<std::wstring>> corpus = tokenize(sentences);

    // Fetch embeddings from the vector database specific to the current corpus
    // Also, create the token hash-to-index mapping and index-to-token mapping
    embeddings->prefetchVocabularyToCache(corpus);
    embeddings->prefetchEmbeddingsToCache();

    Eigen::MatrixXd& wordEmbeddings = embeddings->getWordEmbeddings();
    Eigen::VectorXd& wordBiases     = embeddings->getWordBiases();
    std::unordered_map<std::string, int>& tokenHashToIndex = embeddings->getTokenIndex();

    int vocabSize     = wordEmbeddings.rows();
    int embeddingSize = wordEmbeddings.cols();
    int tokenSize = tokenHashToIndex.size();

    log_detail( "Vocabulary Size: {:d}",  vocabSize );
    log_detail( "Embedding Size: {:d}",  embeddingSize );
    log_detail( "Token Size: {:d}", tokenSize );

    // Initialize Adagrad gradients
    Eigen::MatrixXd adagradGradients = Eigen::MatrixXd::Constant(vocabSize, embeddingSize, 1e-8);
    double totalLoss = 0.0;
    double totaliter = 0.0;

    // Compute the maximum sentence length in the current batch
    int maxBatchLength = 0;
    // Perform Right Padding to make the length of sentences consistent.
    for (int batchStart = 0; batchStart < (int) corpus.size(); batchStart += batchSize) {
        int batchEnd = std::min(batchStart + batchSize, static_cast<int>(corpus.size()));

        for (int i = batchStart; i < batchEnd; ++i) {
            int sentenceLength = corpus[i].size();
            maxBatchLength = std::max(maxBatchLength, sentenceLength);
        }

         // Perform right padding to make sentences equal in length
        for (int i = batchStart; i < batchEnd; ++i) {
            int sentenceLength = corpus[i].size();
            if (sentenceLength < maxBatchLength) {
                // Pad the sentence with a special padding token
                corpus[i].resize(maxBatchLength, TK_PAD_);
            }
        }
    }

    // Iterate over the corpus in batches
    for (int iteration = 0; iteration < this->maxIterations; ++iteration) {

        // Initialize the gradients for the batch
         Eigen::MatrixXd batchGradients = Eigen::MatrixXd::Zero(vocabSize, embeddingSize);
         Eigen::VectorXd batchBiasGradients = Eigen::VectorXd::Zero(vocabSize);

        // Iterate over each batch in the corpus
        // This, we use MP (multi-processing), splitting the workload per batch across processes
        for (int batchStart = 0; batchStart < (int) corpus.size(); batchStart += batchSize) {
            int batchEnd = std::min(batchStart + batchSize, static_cast<int>(corpus.size()));

            // Iterate over each sentence in the batch
            for (int i = batchStart; i < batchEnd; ++i) {
                const auto& sentence = corpus[i];

                // Iterate over each token in the sentence
                for (int j = 0; j < (int) sentence.size(); ++j) {
                    std::wstring targetWord = sentence[j];

                    // Get the context window for the target word
                    int start = std::max(0, j - 3);
                    int end = std::min(static_cast<int>(sentence.size()) - 1, j + 3);

                    // Iterate over the context token
                    for (int k = start; k <= end; ++k) {
                        // Skip the target token itself
                        if (k == j) {
                            continue;
                        }

                        std::wstring contextWord = sentence[k];

                        // Use the token-to-index mapping to get the indices
                        int targetIndex = tokenHashToIndex[sha256(targetWord)];
                        int contextIndex = tokenHashToIndex[sha256(contextWord)];

                        // Compute the co-occurrence weight
                        float weight = 1.0f / (std::abs(j - k) * 1.0f);

                        double dotProduct = (wordEmbeddings.row(targetIndex) * wordEmbeddings.row(contextIndex).transpose()).sum();
                        dotProduct += wordBiases(targetIndex) + wordBiases(contextIndex);

                        // Compute the loss and gradient (based on cross-entropy)
                        double loss = std::pow(dotProduct - std::log(weight), 2.0);
                        double gradient = 2.0 * (dotProduct - std::log(weight));

                        // Accumulate loss
                        totalLoss += loss;
                        totaliter += 1;

                        // Compute the gradients for embeddings and biases
                        batchGradients.row(targetIndex) += gradient * wordEmbeddings.row(contextIndex).cwiseAbs();
                        batchGradients.row(contextIndex) += gradient * wordEmbeddings.row(targetIndex).cwiseAbs();
                        batchBiasGradients(targetIndex) += gradient;
                        batchBiasGradients(contextIndex) += gradient;
                    }
                }
            }
        }

        // Compute the average loss
        double averageLoss = totalLoss / totaliter; 

        log_detail("Average Loss: {:8.10f}", averageLoss );
        std::cout << "Average Loss: " << averageLoss << std::endl;

        // Clip the gradients to a specified threshold
        // batchGradients = batchGradients.cwiseMax(-this->clipThreshold).cwiseMin(this->clipThreshold);

        // Update word embeddings and biases
        // adagradGradients = batchGradients;
        // Accumulate gradients for Adagrad
        adagradGradients += batchGradients.array().square().matrix();

        log_detail( "Calculation of Gradient (Iteration {}):",  iteration );
        log_matrix(  (this->learningRate * batchGradients.array() / (adagradGradients.array() + 1).sqrt().array())  );

        log_detail( "Updating Parameters (Before image) (Iteration {})...", iteration );
        log_matrix( wordEmbeddings );

        wordEmbeddings.array() -= this->learningRate * batchGradients.array() / (adagradGradients.array() + 1).sqrt();
        wordBiases.array() -= this->learningRate * batchBiasGradients.array();

        log_detail( "Updated Parameters (After image) (Iteration {})...", iteration );
        log_matrix( wordEmbeddings );

        // Apply regularization
        wordEmbeddings *= (1.0 - this->learningRate * this->regularization);

        //std::cout << "Updating parameters in Learning: \n" << wordEmbeddings << std::endl;

        // Checkpointing at every 10th iteration
        if (iteration % 10 == 0 ) {
            embeddings->updateEmbeddingsInDatabase(wordEmbeddings, wordBiases);
        }
    }
}






