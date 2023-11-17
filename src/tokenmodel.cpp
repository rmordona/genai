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
#include "tokenmodel.h"

namespace py = pybind11;
using namespace py::literals;

/************************************************************************************************
* BPETokenizer::tokenize_to_char
* Function tokenizes corpus one character at a time. This function is different from the
* BaseTokenModel::tokenize function which performs word-per-word validation of token in the
* vocabulary.
*************************************************************************************************/
template <class T>
std::vector<std::wstring> BPETokenizer<T>::tokenize_to_char(const std::vector<std::wstring>& corpus) {
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

template <class T>
bool BPETokenizer<T>::endsWith(const std::wstring& str, const std::wstring& suffix) {
    if (str.length() < suffix.length()) {
        return false;
    }
    return str.substr(str.length() - suffix.length()) == suffix;
}

/************************************************************************************************
* BPETokenizer::mergeTokens
* Function merges token pairs. The idea is to find the most frequent pairs and start
* merging them upto numMerges. The result is preserved in this->vocab.
*************************************************************************************************/
template <class T>
std::unordered_map<std::wstring, int> BPETokenizer<T>::mergeTokens(std::vector<std::wstring>& tokens, int numMerges) {
    // Perform merging of tokens based on most frequent pairs

    std::unordered_map<std::wstring, int> vocab;

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
        std::map<std::pair<std::wstring, std::wstring>, int> candidatePairs;
        int maxCount = 0;
        for (const auto& entry : pairCounts) {
            // const std::pair<std::wstring, std::wstring>& pair = entry.first;
            int count = entry.second;
            log_wdetail( "Pairing: {} {} :  {:d}", 
                        wstringToUtf8(entry.first.first).c_str() , 
                        wstringToUtf8(entry.first.second).c_str() , count );
            // Note here that we included an extra condition such that if the count is
            // greater than 2, continue to pair. This  exclude tokens with only 2 frequency and below.
            if (count > maxCount || count > 2) {
                maxCount = count;
                candidatePairs[std::make_pair(entry.first.first, entry.first.second)] = entry.second; // count frquency
            }
        }

        log_wdetail( "Found pair: {} {:d}",  wstringToUtf8( maxPair.first + maxPair.second).c_str(), maxCount );

        // If no more frequent pair is found, exit the merge loop
        if (maxCount == 0) {
            break;
        }
 
        log_detail( "Max Count found: {:d}",  maxCount );

        for (const auto& pair : candidatePairs) {
           const std::pair<std::wstring, std::wstring>& pair_ = pair.first;
           std::wstring mergedToken = pair_.first + pair_.second;
           for (size_t i = 0; i < tokens.size() - 1; ++i) {
               if (tokens[i] == pair_.first && tokens[i + 1] == pair_.second) {
                   tokens[i] = mergedToken;
                   tokens.erase(tokens.begin() + i + 1);
               }
           }
           vocab[mergedToken] += pair.second;
        }
    }
    return vocab;
}

/************************************************************************************************
* BPETokenizer::buildVocabulary
* Function to build the Vocabulary. This includes merging token pairs until certain threshold.
* See Function BPETokenizer::mergeToken for the algorithm.
*************************************************************************************************/
template <class T>
void BPETokenizer<T>::buildVocabulary(const std::vector<std::wstring>& sentences, int numMerges) {
    std::vector<std::wstring> tokens = tokenize_to_char(sentences);
    this->vocab = mergeTokens(tokens, numMerges);
    this->setVocabSize(this->vocab.size());
}

/************************************************************************************************
* BPETokenizer::merge
* Function to merge the tokenizer on a corpus.
* This is where we fine-tune the vocabulary.
*************************************************************************************************/
template <class T>
void BPETokenizer<T>::merge(const std::vector<std::wstring>& sentences, int numMerges) {

    log_info( "=======================================" );
    log_info( "Entering Merge for BPETokenizer  ..." );

    buildVocabulary(sentences, numMerges);

    log_detail("Size vocabulary: {0}", this->vocab.size());

    // No need to rebuild the Vector Metadata in DB.
    // We only need to sync new word from given sentences to the vocabulary; sort of train new words.
    // For new words, the embeddings will be tuned during GloVe training.
    this->getEmbeddings()->crossReferenceVocabularyinDBandCache(this->vocab);
}

/************************************************************************************************
* BPETokenizer::preload
* Function to preload the tokenizer on a corpus. 
* This is where we construct the initial vocabulary. After this, we can begin to use
* the BPETokenizer::merge function repeatedly for every new corpus.
* This stage does not require to build the embeddings. We are just tokenizing the corpus
* 
* The difference between preload function and merge function is that preload function
* assumes there is no vocabulary; therefore, it seeds the vocabulary first. No need to perform
* cross reference.
*
* Then after, this function also generates the initial embedding.
*************************************************************************************************/
template <class T>
void BPETokenizer<T>::preload(const std::vector<std::wstring>& sentences, int numMerges, int embeddingSize) {

    log_info( "=======================================" );
    log_info( "Entering Pre-load for BPETokenizer  ..." );

    this->merges = numMerges;

    buildVocabulary(sentences, numMerges);

    log_detail("Size vocabulary: {0}", this->vocab.size());

    // Initialize Embeddings Class  (e.g. this->getEmbeddings())
    this->initializeEmbeddings(embeddingSize);

    // Generate the initial embedding.
    this->getEmbeddings()->createInitialEmbeddings(embeddingSize, this->vocab);
}



/************************************************************************************************
* BaseTokenModel::splitString
* This Function splits  strings into words.
*************************************************************************************************/
template <class T>
std::vector<std::wstring> BaseTokenModel<T>::splitString(const std::wstring& str) {
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

/******************************************************************************************************
* BaseTokenModel::tokenize
* This Function splits sentences into words and searches the word from the vocabulary. 
* By using this function, one will get word-per-word token from the vocabulary.
* This is different from BPETokenizer::tokenize_to_char in that, that function is the actual tokenizer.
* It splits sentences into characters and the result is fed into a pair-merger.
*******************************************************************************************************/
template <class T>
std::vector<std::wstring> BaseTokenModel<T>::tokenize(const std::wstring& sentence) {

    log_info( "=================================" );
    log_info( "Tokenizing a sentence  ..." );

    const std::unordered_map<std::wstring, int> vocabulary = this->vocab;
    std::vector<std::wstring> tokens;
    std::wstring currentToken;

    std::vector<std::wstring> words;
    words = splitString(sentence);

    for (const auto& word : words) {
        bool found = false;
        std::wstring unit = word + TK_SPACE_;

        if (this->embeddings->isInVocabulary(unit)) { 
            found = true;
            tokens.push_back(unit);
        } else {
           size_t start = 0;
           size_t end = word.length();

           for (size_t i = start + 1; i <= end; ++i) {
               std::wstring token = unit.substr(start, end - start);
               if (this->embeddings->isInVocabulary(token)) {
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

/****************************************************************************************************************
* BaseTokenModel::tokenize
* This function accepts a vector of sentences to be tokenized.
****************************************************************************************************************/
template <class T>
std::vector<std::vector<std::wstring>> BaseTokenModel<T>::tokenize(const std::vector<std::wstring>& sentences) {

    log_info( "=================================" );
    log_info( "Tokenizing Sentences  ..." );


    const std::unordered_map<std::wstring, int> vocabulary = this->vocab;
    std::vector<std::vector<std::wstring>> corpus;
    std::wstring currentToken;

    for (const auto& sentence : sentences) {
        
        std::vector<std::wstring> tokens;
        std::vector<std::wstring> words;
        words = splitString(sentence);

        for (const auto& word : words) {
            std::wstring unit = word + TK_SPACE_;
            tokens.push_back(unit);
        }


        corpus.push_back(tokens);
    }

    return corpus;

}


/*******************************************************************************************************************
* BaseTokenModel::prefetchEmbeddings
* Prefetch the vocabulary and embeddings associated with the sentences.
********************************************************************************************************************/
template <class T>
void BaseTokenModel<T>::prefetchEmbeddings(const std::vector<std::wstring>& sentences, int token_limit, int batchSize) {

    // Generate the corpus.
    std::vector<std::vector<std::wstring>> corpus = this->tokenize(sentences);
    log_detail("Size of tokenized corpus: {0}", corpus.size());

    // First, let us fetch the voculabulary for the tokens in the generated corpus.
    // After which, we use the vocabulary to create or update our embeddings.
    this->embeddings->prefetchVocabularyToCache(corpus);

    // Next, let's build the co-occurrence matrix;
    this->embeddings->buildCoMatrix(corpus, batchSize);

    // Now, let us generate the Token Indices
    this->embeddings->generateTokenIndices();

    // Finally, Fetch embeddings from the vector database specific to the current corpus
    // Also, create the token hash-to-index mapping and index-to-token mapping
    // This is also where we perform the embedding if tokens do not have embeddings but
    // exists in the vocabulary. The mere fact that the token is used in the corpus
    // and is found in the vocabulary indidcates that we now have to start creating an embedding for the token.
    this->embeddings->prefetchEmbeddingsToCache();
 
}

/*******************************************************************************************************************
* BaseTokenModel::train
* Function to train GloVe-like model using GloVe algorithm. This is where we train the vectors, morphing them
* into an embedding. We tokenize a sentence such that the tokenize function
* requires already the existence of the constructed vocabulary via BPE (preload and merge functions).
********************************************************************************************************************/
template <class T>
void BaseTokenModel<T>::train(std::vector<std::wstring>& sentences, int batchSize, 
        const std::string& losstype, const std::string& optimizertype,
        T learningRate, int max_epoch, T clipThreshold,  T regularization) {

    log_info( "=================================" );
    log_info( "Entering GloVe-like Training  ..." );

    // Preserve hyperparameters.
    this->losstype = losstype;
    this->optimizertype  = optimizertype;
    this->learningRate   = learningRate;
    this->maxIteration  = maxIteration;
    this->clipThreshold  = clipThreshold;
    this->regularization = regularization;

    // Prefetch the Embeddings for given corpus
    prefetchEmbeddings(sentences, 4000, batchSize);

    std::unordered_map<std::wstring, int> comatrix = this->embeddings->getCoMatrix();
    std::unordered_map<std::wstring, std::vector<std::wstring>> tokens = this->embeddings->getTokens();

    // Now retrieve the generated embeddings
    aimatrix<T> weights_V    = this->embeddings->getWordEmbeddings();
    aimatrix<T> weights_U    = this->embeddings->getWordEmbeddings();
    aivector<T> biases_V     = this->embeddings->getWordBiases();
    aivector<T> biases_U     = this->embeddings->getWordBiases();

    // log_detail("weights:");
    // log_matrix(weights_V);

    log_detail("Size of tokens: {0}", tokens.size());
    log_detail("Size of weights: {0}x{1}", weights_V.rows(), weights_V.cols());
    log_detail("Size of biases: {0}", biases_V.size());

    aiscalar<T> totalloss = 0.0;
    aiscalar<T> totalcount = 0;
    aiscalar<T> alpha = 0.75f;
    aiscalar<T> Xmax = 5.0f; // Our threshold of dominating frequency is upto 5 only.

    // Gradients
    airowvector<T> gradient_U;
    airowvector<T> gradient_V;
    aiscalar<T> gradient_Bu;
    aiscalar<T> gradient_Bv;

    // Used for Adagrad (Accummulated sum of squares)
    airowvector<T> agradient_U = airowvector<T>::Constant(weights_V.cols(), 1.0);
    airowvector<T> agradient_V = airowvector<T>::Constant(weights_V.cols(), 1.0);
    aivector<T> agradient_Bu   = aivector<T>::Constant(weights_V.rows(), 1.0);
    aivector<T> agradient_Bv   = aivector<T>::Constant(weights_V.rows(), 1.0);

    airowvector<T> Vi;
    airowvector<T> Uj;
    aiscalar<T>    Bv;
    aiscalar<T>    Bu;

    std::wstring target;
    std::wstring context;

    aiscalar<T> weight;
    aiscalar<T> dotprod;

    int mod_epoch =  std::ceil(max_epoch * 0.10);

    log_detail("Starting Training ...");
    auto start_time        = std::chrono::system_clock::now();
    auto end_time          = std::chrono::system_clock::now();
    std::time_t next_time  = std::chrono::system_clock::to_time_t(end_time);
    double total_seconds = 0.0;
    int    total_iteration = 0;

    std::chrono::duration<double> elapsed_seconds;

    for (int iter = 0; iter < max_epoch; ++iter) {

        totalloss = 0.0;
        totalcount = 0;

        for (const auto& token : tokens) {

            // std::wcout << "iteration at " << iter << " token:   " << target << std::endl;

            target = token.first;

            // get the token index for the target word
            int i = this->embeddings->getTokenIndex(sha256(target));

            Vi = weights_V.row(i);  // center word
            Bv = biases_V(i);       // bias term for center word

            std::vector<std::wstring> contexts = token.second;
            for (int k = 0; k < (int) contexts.size(); k++) {

                context = contexts.at(k);

                // get the token index for the context word
                int j = this->embeddings->getTokenIndex(sha256(context));

                // std::wcout << "iteration at " << iteration << " contexts:   " << context << std::endl;
                // std::wcout << " contexts:   " << context << std::endl;

                Uj = weights_U.row(j);  // context word
                Bu = biases_U(j);       // bias term for context word

                int Xij = comatrix[ target + context ]; // co-occurrence frequency

                // implement smooth weighting  (capping co-occurrence frequency) = min(1, X/Xm^alpha)
                weight = (T) std::min((T) 1.0, (T) std::pow( ( Xij / Xmax ), alpha) );

                dotprod = Uj.dot(Vi) + Bv + Bu;

                totalloss+= weight * std::pow(dotprod - std::log(Xij), 2.0);

                totalcount ++;

                // Gradient with respect to Uj
                gradient_U = weight * (dotprod - std::log(Xij)) * Vi; // generate gradient for the entire embedding (row vector)
                // Gradient with respect to Vi
                gradient_V = weight * (dotprod - std::log(Xij)) * Uj; // generate gradient for the entire embedding (row vector)

                // Gradient with respect to Bias i
                gradient_Bu = weight * (dotprod - std::log(Xij));  
                // Gradient with respect to Bias j
                gradient_Bv = weight * (dotprod - std::log(Xij));  

                // Now update embedding Vi using Adagrad optimization
                Vi.array() -= (learningRate * gradient_V.array() / agradient_V.array().sqrt());
                // Now update embedding Uj using Adagrad optimization
                Uj.array() -= (learningRate * gradient_U.array() / agradient_U.array().sqrt());
                // Now update bias Bv using Adagrad optimization
                Bv -= (learningRate * gradient_Bv / sqrt(agradient_Bv[i]));
                // Now update bias Bu using Adagrad optimization
                Bu -= (learningRate * gradient_Bu / sqrt(agradient_Bu[j]));

                // Update actual embeddings
                weights_V.row(i) = Vi.array();
                weights_U.row(j) = Uj.array();
                biases_V[i]  = Bv;
                biases_U[j]  = Bu;

                // Update Adagrad parameters
                agradient_V.array() += gradient_V.array().square(); // std::pow((T) gradient_V, (T) 2.0);
                agradient_U.array() += gradient_U.array().square(); // std::pow((T) gradient_U, (T) 2.0);
                agradient_Bv[i] += std::pow((T) gradient_Bv, (T) 2.0);
                agradient_Bu[j] += std::pow((T) gradient_Bu, (T) 2.0);
           }
        }

        // log_detail("Total Loss: {:8.10f}", totalloss );

        totalloss = totalloss / totalcount;

        // Calculate Time, then display loss
        end_time = std::chrono::system_clock::now();
        elapsed_seconds = end_time - start_time;
        next_time = std::chrono::system_clock::to_time_t(end_time);
        start_time = end_time;

        total_seconds += elapsed_seconds.count();
        total_iteration++;


        // Print Progress
        if (iter == 1 || iter % mod_epoch == 0) {

            // Calculate Time, then display loss
            double avg_microseconds = (total_seconds / total_iteration) * 1000000;

            py_cout << "Epoch " << iter << "/" << max_epoch << " ... ";
            py_cout << "Loss: " << totalloss;
            py_cout << " ... elapsed " <<  avg_microseconds<< "us";
            py_cout << " at " << std::ctime(&next_time) << std::endl;

            // Also, log the result if Logging INFO is enabled
            log_detail( "Epoch {}/{} ... Loss: {:8.5f} ... Elapsed {}us at {}", iter, max_epoch, 
                totalloss,   avg_microseconds, std::ctime(&next_time) );

            total_seconds = 0.0;
            total_iteration = 0;

        }
 
        // Checkpointing at every 10th iteration
        if (iter % mod_epoch == 0 || iter == max_epoch - 1) {
            // Here, we add both V and U matrices and biases instead of averaging.
            aimatrix<T> weights = (weights_V + weights_U);
            aivector<T> biases  = (biases_V + biases_U);
            this->embeddings->updateEmbeddingsInDatabase(weights, biases);
        }
        agradient_U.setConstant(T(1.0));
        agradient_V.setConstant(T(1.0));
        agradient_Bu.setConstant(T(1.0));
        agradient_Bv.setConstant(T(1.0));
    }
}

/************************************************************************************************
* BaseTokenModel::list Tokens, Embeddings, and Sequences
* Helper function to list Tokens, Embeddings, and Sequences
*************************************************************************************************/
template <class T>
std::vector<std::wstring> BaseTokenModel<T>::listTokens() {
    return this->embeddings->listTokens();
}

template <class T>
aimatrix<T> BaseTokenModel<T>::listEmbeddings() {
    return this->embeddings->getWordEmbeddings();
}


template <class T>
aitensor<T> BaseTokenModel<T>::sequenceEmbeddings(const std::vector<std::wstring>& sentences) {

    typename Embeddings<T>::RecordStruct record;  

    aimatrix<T> sequence;

    aitensor<T> sequences;

    bool initialized = false;

    // Generate the corpus.
    std::vector<std::vector<std::wstring>> corpus = this->tokenize(sentences);

    int corpus_size = corpus.size();
    int sequence_size = 0;
    int embedding_size = 0;
     
    // Use the largest corpus size to build the tensor.  Shorter Sequences will be padded with zeroes.
    for (int i = 0; i < (int) corpus_size; i++) {
        std::wcout << " sentences: " << i << " : " << sentences[i] << std::endl;
        sequence_size = ((int) corpus[i].size() > sequence_size) ? (int) corpus[i].size() : sequence_size;
    }
    std::cout << " size corpus: " << corpus_size << std::endl;

    for (int i = 0; i < sequence_size; i++) {
        initialized = false;
        std::wcout << " next sequence: " << i << std::endl;
        for (int j = 0; j < corpus_size; j++) {
            if (i < (int) corpus[j].size()) {
                std::wstring token = corpus[j][i];

                std::wcout << " token: " << j << " : " << token << std::endl;

                bool result = this->embeddings->retrieveEmbeddings(sha256(token), record);

                if (result) {
                    if (!initialized) {    
                        embedding_size = record.embedding.size();
                        sequence = aimatrix<T>::Zero(corpus_size, embedding_size);
                        initialized = true;
                    }
                    sequence.row(j) = record.embedding.array();
                }
            }
        }
        std::cout << " sequence dim: " << sequence.rows() << "x" << sequence.cols() << std::endl;
        sequences.push_back(sequence);

    }

/*  Leave this here for now. This will be used to construct a tensor of BxNxW (Batch x Samples x Embeddings)
    for (int i = 0; i <   corpus_size; i++) {
        std::wcout << " corpus: " << i << " : " << sentences[i] << std::endl;
        initialized = false;
        for (int j = 0; j < (int) corpus[i].size(); j++) {
            std::wstring token = corpus[i][j];
            std::wcout << " token: " << j << " : " << token << std::endl;
            bool result = this->embeddings->retrieveEmbeddings(sha256(token), record);

            if (result) {
                if (!initialized) {    
                    embedding_size = record.embedding.size();
                    sequence = aimatrix<T>::Zero(sequence_size, embedding_size);
                    initialized = true;
                }
                sequence.row(j) = record.embedding.array();
            }
        }    
        std::cout << " sequence dim: " << sequence.rows() << "x" << sequence.cols() << std::endl;
        sequences.push_back(sequence);
    }
*/
    return sequences;
}

/************************************************************************************************
* BaseTokenModel::printVocabulary
* Helper function for printing Vocabulary
*************************************************************************************************/
template <class T>
void BaseTokenModel<T>::printVocabulary(int rows) { 

    int count = 0;
    std::cout << "Size of Vocabulary: " << this->vocab.size() << std::endl;
    for (const auto& entry : this->vocab) {
        count++;
        if (count >= rows) break;
        std::wcout << entry.first << L" : " << entry.second << std::endl;
    }

}

/************************************************************************************************
* BaseTokenModel::printWordEmbeddings
* Helper function for printing word embeddings
*************************************************************************************************/
template <class T>
void BaseTokenModel<T>::printWordEmbeddings(int rows) { 

    // Print the learned word embeddings

    int count = 0;
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


/************************************************************************************************
* Model::Model
* This is the Model Constructor Implementation - Note that this is only a meta model.
* The actual model is the BaseModel Class.
*************************************************************************************************/
TokenModel::TokenModel(const std::string& tokenizer, const std::string& datatype) {
    try {
        this->datatype = datatype;
        this->tokenizer = tokenizer;

        if (tokenizer == "bpetokenizer") {
            if (datatype == "float") {
                this->tokenizerf =  std::make_shared<BPETokenizer<float>>();
            } else
            if (datatype == "double") {
                this->tokenizerd =  std::make_shared<BPETokenizer<double>>();
            } else {
                throw std::invalid_argument("Unsupported datatype");            
            }
        } else {
            throw AIException("Unsupported Tokenizer ...");
        }

    } catch (const AIException& e) {
        std::cerr << "(TokenModel::TokenModel) Error: " << e.what() << std::endl;
    } catch (const std::exception& e) {
        // Catch standard exceptions
        std::cerr << "(TokenModel:TokenModel) Standard Error: " << e.what() << " at " << __LINE__ << std::endl;
    } catch (...) {
        // Catch all other exceptions
        std::cerr << "(TokenModel:TokenModel) Unknown Error:" << std::endl;
    }

    std::cout << "Got here :" << datatype << " learningRate " << learningRate << std::endl;
}

std::vector<std::wstring> TokenModel::tokenize(const std::wstring& sentence) {
    std::vector<std::wstring> empty_vector;
    if (this->datatype == "float") {
        return this->tokenizerf->tokenize(sentence);
    } else 
    if (this->datatype == "double") {
        return this->tokenizerd->tokenize(sentence);
    }
    return empty_vector;
}

std::vector<std::vector<std::wstring>> TokenModel::tokenize(const std::vector<std::wstring>& sentences) {
    std::vector<std::vector<std::wstring>> empty_vector;
    if (this->datatype == "float") {
        return this->tokenizerf->tokenize(sentences);
    } else 
    if (this->datatype == "double") {
        return this->tokenizerd->tokenize(sentences);
    }
    return empty_vector;
}

void TokenModel::preload(const std::vector<std::wstring>& sentences, int numMerges,  int embeddingSize) {
    if (this->datatype == "float") {
        this->tokenizerf->preload(sentences, numMerges, embeddingSize);
    } else 
    if (this->datatype == "double") {
        this->tokenizerd->preload(sentences, numMerges, embeddingSize);
    }
}

void TokenModel::merge(const std::vector<std::wstring>& sentences, int numMerges) {
    if (this->datatype == "float") {
        this->tokenizerf->merge(sentences, numMerges);
    } else 
    if (this->datatype == "double") {
        this->tokenizerd->merge(sentences, numMerges);
    }
}

/************************************************************************************************
* TokenModel::train
* Function to train (glove-like) a model
*************************************************************************************************/
void TokenModel::train(std::vector<std::wstring>& sentences, int batchSize, 
        const std::string& losstype , const std::string& optimizertype,
        double learningRate, int maxIteration, double clipThreshold ,double regularization ) {
    if (this->datatype == "float") {
        this->tokenizerf->train(sentences, batchSize, losstype, optimizertype, static_cast<float>(learningRate), maxIteration);
    } else 
    if (this->datatype == "double") {
        this->tokenizerd->train(sentences, batchSize, losstype, optimizertype, static_cast<double>(learningRate), maxIteration);
    }
}

/************************************************************************************************
* TokenModel::tokens
* Function to return tokens generated via the training.
*************************************************************************************************/
std::vector<std::wstring> TokenModel::tokens() {
    std::vector<std::wstring> tokens;
    if (this->datatype == "float") {
        tokens = this->tokenizerf->listTokens();
    } else 
    if (this->datatype == "double") {
        tokens = this->tokenizerd->listTokens();
    }
    return tokens;
}

/************************************************************************************************
* TokenModel::embeddingsDouble and embeddingsFloat
* Functions to return the embeddings constructed after the training.
*************************************************************************************************/
py::array_t<double> TokenModel::embeddingsDouble() {
    aimatrix<double> embeddingd;
    if (this->datatype == "double") {
        embeddingd = this->tokenizerd->listEmbeddings();

    } else 
    if (this->datatype == "float") {
        aimatrix<float> embeddingf = this->tokenizerf->listEmbeddings();
        embeddingd = embeddingf.cast<double>();
    }
    return ConvertData::topyarray(embeddingd);
}

py::array_t<float> TokenModel::embeddingsFloat() {
    aimatrix<float> embeddingf;
    if (this->datatype == "double") {
        aimatrix<double> embeddingd = this->tokenizerd->listEmbeddings();
        embeddingf = embeddingd.cast<float>();
    } else 
    if (this->datatype == "float") {
        embeddingf = this->tokenizerf->listEmbeddings();
    }
    return ConvertData::topyarray(embeddingf);
}

/************************************************************************************************
* TokenModel::sequenceDouble and sequenceFloat
* Functions to generate sequence of embeddings based on given list of sentences
*************************************************************************************************/
py::array_t<double> TokenModel::sequenceDouble(const std::vector<std::wstring>& sentences) {
    aitensor<double> embeddingd;
    if (this->datatype == "double") {
        embeddingd = this->tokenizerd->sequenceEmbeddings(sentences);

    } else 
    if (this->datatype == "float") {
        aitensor<float> embeddingf = this->tokenizerf->sequenceEmbeddings(sentences);
        for (int i = 0; i < (int) embeddingf.size(); i++) {
            embeddingd.push_back( embeddingf[i].cast<double>() );
        }
    }
    return ConvertData::topyarray(embeddingd);
}

py::array_t<float> TokenModel::sequenceFloat(const std::vector<std::wstring>& sentences) {
    aitensor<float> embeddingf;
    if (this->datatype == "double") {
        aitensor<double> embeddingd = this->tokenizerd->sequenceEmbeddings(sentences);
        for (int i = 0; i < (int) embeddingf.size(); i++) {
            embeddingf.push_back( embeddingd[i].cast<float>() );
        }
    } else 
    if (this->datatype == "float") {
        embeddingf = this->tokenizerf->sequenceEmbeddings(sentences);
    }
    return ConvertData::topyarray(embeddingf);
}

/************ Tokenizer / Embeddings initialize template ************/

template class BaseTokenModel<float>;  // Instantiate with float
template class BaseTokenModel<double>;  // Instantiate with double

template class BPETokenizer<float>;  // Instantiate with float
template class BPETokenizer<double>;  // Instantiate with double