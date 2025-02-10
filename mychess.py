import spacy

# Load spaCy English language model
nlp = spacy.load('en_core_web_sm')

# Function to clean a single sentence
def clean_sentence(sentence):
    doc = nlp(sentence)
    cleaned_tokens = [token.text.lower() for token in doc if token.text != '.']
    return ' '.join(cleaned_tokens)


import nltk
nltk.download('punkt')
from nltk import sent_tokenize, word_tokenize
import string

# Open the text file for reading
with open('/Users/raymondordona/Workspace/genaiproj/dataset/chessmoves100.txt', 'r') as file:
    text = file.read()

# Tokenize the text into sentences
sentences = sent_tokenize(text)

sentences

# Clean each sentence in the list
cleaned_sentences = [clean_sentence(sentence) for sentence in sentences]
cleaned_sentences[1]


import genai as ai
random_seed  = 2024
tokenizer = ai.TokenModel(tokenizer="bpetokenizer", dtype=ai.DataType.float32, seed = random_seed);


# Size is the dimensionality of each token generated. Length of an embedding (Vector)
tokenizer.preload(corpus = cleaned_sentences, merges = 100, size=5);
tokenizer.train(corpus=cleaned_sentences, 
                batch_size=2, 
                losstype = "mse", 
                optimizertype = "adagrad", 
                learn_rate = 0.01, 
                max_epoch = 1000, 
                clipthreshold = 5.0, 
                regularization = 1.0);
