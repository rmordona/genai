import genai as ai
import numpy as np
from typing import List

ai.print_string("Hello", True)

#class MyTokenModel(ai.TokenModel):
#   def __init__(self, tokenizer, datatype):
#     super().__init__(tokenizer, datatype);

dtype = "float"
tokenizer = ai.TokenModel(tokenizer="bpetokenizer", datatype=dtype);

#sentence = "I travel the world in search for the fountain of youth. Hello, 世界!"

# Assuming you have a UTF-32 encoded string
text1 = "Hello me, 世界";
text2 = "Hello you, 世界";

corpus = [ 
        "This is the first sentence.",
        "This is the first sentence.",
        "This is the first sentence.",
        "This is the first sentence.",
        "And this is the second sentence.",
        "And this is the second sentence.",
        "And this is the second sentence.",
        "And this is the second sentence.",
        "Finally, we have the third sentence."
       ]

new_corpus1 = [ 
        "This is the first sentence.",
        "This is the fourth sentence.",
        "This is the fourth sentence.",
        "This is the fourth sentence.",
        "This is the fourth sentence.",
        "This is the first sentence."];

new_corpus2 = [ 
        "This is the first sentence.",
        "This is the fourth sentence.",
        "This is the fourth sentence."];


# Create a list of UTF-32 strings

tokenizer.preload(corpus = corpus, merges = 10, size=5);

tokenizer.merge(corpus = new_corpus1, merges = 10);

tokenizer.merge(corpus = new_corpus2, merges = 10);


sentences = [
        "This is the second sentence.",
        "This is ray's first sentence."]

ai.print_string("Tokenize two sentences.", True)

tokens = tokenizer.tokenize(sentences);

print("Got the tokens ...");


#for x in tokens:
#   for y in x:
#      print(y)

tokenizer.train(corpus=sentences, batchsize=2, losstype = "mse", optimizertype = "adagrad", learningRate = 0.01, maxIteration = 100, clipTreshold = 5.0, regularization = 1.0);

ai.print_string("Done.", True)

