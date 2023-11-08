import genai as ai
import numpy as np
from typing import List
# from memory_profiler import profile

import nltk
nltk.download('punkt')
from nltk import sent_tokenize

# Open the text file for reading
with open('dataset/inspirational.txt', 'r') as file:
    text = file.read()

# Tokenize the text into sentences
sentences = sent_tokenize(text)

# Print the sentences
#for sentence in sentences:
#    print(sentence)


dtype = "float"
tokenizer = ai.TokenModel(tokenizer="bpetokenizer", datatype=dtype);


tokenizer.preload(corpus = sentences, merges = 1000, size=5);


new_corpus = [
        "Got here :float learningRate 0.01",
        "SQL error: index vocab_index already exists" ];

# Use only if there are more corpus to merge. Otherwise, the preload already merges
#tokenizer.merge(corpus = sentences, merges = 1000);


# tokens = tokenizer.tokenize(sentences);

tokenizer.train(corpus=sentences, batchsize=2, losstype = "mse", optimizertype = "adagrad", learn_rate = 0.01, max_epoch = 2, clipthreshold = 5.0, regularization = 1.0);
