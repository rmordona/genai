import numpy as np
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
with open('/Users/raymondordona/Workspace/genaiproj/dataset/king_queen.txt', 'r') as file:
    text = file.read()

# Tokenize the text into sentences
sentences = sent_tokenize(text)

sentences

# Clean each sentence in the list
cleaned_sentences = [clean_sentence(sentence) for sentence in sentences]
cleaned_sentences


import genai as ai
dtype = "float"
random_seed  = 2023
tokenizer = ai.TokenModel(tokenizer="bpetokenizer", datatype=dtype, seed = random_seed);

tokenizer.preload(corpus = cleaned_sentences, merges = 100, size=80);
tokenizer.train(corpus=cleaned_sentences, 
                batch_size=2, 
                losstype = "mse", 
                optimizertype = "adagrad", 
                learn_rate = 0.01, 
                max_epoch = 1000, 
                clipthreshold = 5.0, 
                regularization = 1.0);


tokens = tokenizer.tokens()
for i, token in enumerate(tokens):
    #if token == "king<s>" or token == "queen<s>" or  token == "man<s>" or token == "woman<s>" or token == "love<s>" or token == "story<s>":
        print(i, token)

embeddings = tokenizer.embeddings()
rows, cols = embeddings.shape
# Print dimensions
print("Number of rows:", rows)
print("Number of columns:", cols)


import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Perform t-SNE
tsne = TSNE(n_components=2, random_state=42)
embedded_pcas = tsne.fit_transform(embeddings)

# Scatter plot the embedded tokens
plt.scatter(embedded_pcas[:, 0], embedded_pcas[:, 1])

# Annotate points with token labels (optional)
for i, token in enumerate(tokens):
    if (i == 71 or i == 72 or i == 77 or i == 96 or i == 107 or i == 109):
        plt.annotate(token, (embedded_pcas[i, 0], embedded_pcas[i, 1]))

# Show the plot
# plt.show()


# Fifty Sentences
sequences = tokenizer.sequence(corpus = cleaned_sentences, sample_size=10, chunk_size=10, sequence_type = "sentence", rowwise = False)
inp_sequences = sequences[0]
tgt_sequences = sequences[1]

dtype = "double"
random_seed  = 2024
modelgraph = ai.Model(datatype=dtype, seed = random_seed);
node1  = modelgraph.addNode("node1", ai.NodeType.Generic);
node2  = modelgraph.addNode("node2", ai.NodeType.Generic);

# Four Layered Encoder
node1.setOperations([ai.Encoder(heads=1, attention_size=40, feed_size=40, layers=1, bias=True, type="leakyrelu",  alpha=0.01 ),

                    ]
                 );

node2.setOperations([ai.Decoder(heads=2,   attention_size=40, feed_size=40, layers=1, bias=True, type="leakyrelu",  alpha=0.01 ),
                     ai.Dense(size=80, bias=True), ai.Activation(type="leakyrelu", alpha=0.01)
                    ]
                 );

node2.setOperations([
    ai.Dense(size=80, bias=True), ai.Activation(type="leakyrelu", alpha=0.01)]);

modelgraph.connect(node1, node2);

x1 = inp_sequences[:1]

x2 = inp_sequences[:1]

y = tgt_sequences[:1]

# Set the Data. Normalize if required. Apply Positional Encoding if required
node1.setData(data = x1, normalize=True, positional=False);

# Set the Decoder Data. Normalize if required. Apply Positional Encoding if required
node2.setDecoderData(data = x2, normalize=True, positional=False);

# Set The target
modelgraph.setTarget(data = y, normalize=True);

# Perform fitting
modelgraph.train(batch_size = 10, loss="mse", metrics=[], optimizer="nadam", max_epoch=10, learn_rate=0.001, use_step_decay = True, decay_rate = 0.90)
