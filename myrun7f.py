import genai as ai
import numpy as np
import spacy

input_sequences =  [
                      [  
                         [0.10, 0.10, 0.10, 0.10],
                         [0.20, 0.20, 0.20, 0.20],
                         [0.30, 0.30, 0.30, 0.30],
                         [0.40, 0.40, 0.40, 0.40],
                      ]
                   ];
shifted_sequences =  [
                      [  
                         [0.20, 0.20, 0.20, 0.20],
                         [0.30, 0.30, 0.30, 0.30],
                         [0.40, 0.40, 0.40, 0.40],
                         [0.50, 0.50, 0.50, 0.50],
                      ]
                   ];
target_sequences =  [
                      [  
                         [1, 0, 0, 0],  
                         [0, 1, 0, 0], 
                         [0, 0, 1, 0],
                         [0, 0, 0, 1]  
                      ]
                   ];

dtype = "double"
random_seed  = 2024
modelgraph = ai.Model(datatype=dtype, seed = random_seed);
node1  = modelgraph.addNode("node1", ai.NodeType.Generic);
node2  = modelgraph.addNode("node2", ai.NodeType.Generic);

# Four Layered Encoder
node1.setOperations([
                       ai.Encoder(heads=1, attention_size=4, feed_size=4, layers=1, bias=True, activation_type="leakyrelu",  alpha=0.01 ),
                       #ai.Attention(attention_size=4),
                       #ai.MultiHeadAttention(heads=1, attention_size=4),
                       #ai.LayerNorm(),
                       #ai.FeedForward(feed_size = 4, bias = True, activation_type = "leakyrelu", alpha=0.01),
                       #ai.LayerNorm(),
                    ]
                 );

node2.setOperations([
                     # ai.Decoder(heads=8,   attention_size=80, feed_size=20, layers=1, bias=True, activation_type="leakyrelu",  alpha=0.01 ),
                     #ai.Dense(size=80, bias=True), ai.Activation(type="leakyrelu", alpha=0.01),
                     ai.Dense(size=4, bias=True), ai.Activation(type="softmax", alpha=0.01)
                    ]
                 );

modelgraph.connect(node1, node2);

encoder_input = input_sequences[:5]

decoder_input = shifted_sequences[:5]  # shifted

target  = target_sequences[:5]

# Set the Data. Normalize if required. Apply Positional Encoding if required
node1.setData(data = encoder_input, normalize=True, positional=True);

# Set the Decoder Data. Normalize if required. Apply Positional Encoding if required
# node2.setDecoderData(data = decoder_input, normalize=True, positional=True);

# Set The target
modelgraph.setTarget(data = target, normalize=True);

# Perform fitting
modelgraph.train(loss="cce", metrics=[], optimizer="nadam", batch_size = 10, max_epoch=1, learn_rate=0.001, use_step_decay = True, decay_rate = 0.90)


#yy_pred = modelgraph.predict();
#yyy_pred = yy_pred[0]
#yy_pred.shape

#print("Predicted YY dimension:");
#print(yy_pred.shape);
#pdecoded = tokenizer.decode(sequences = input_sequences);
#print("Decoded YY dimension:");
#pdecoded
