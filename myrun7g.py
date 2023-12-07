import genai as ai
import numpy as np
import spacy

input_sequences =  [
                      [  
                         [0.12, 0.78, 0.60],  
                         [0.50, 0.18, 0.40],  
                         [0.34, 0.32, 0.90]  
                      ]
                   ];
shifted_sequences =  [
                      [  
                         [0.50, 0.18, 0.40],  
                         [0.34, 0.32, 0.90], 
                         [0.41, 0.14, 0.89]  
                      ]
                   ];
target_sequences =  [
                      [  
                         [1, 0, 0],  
                         [0, 1, 0], 
                         [0, 0, 1]  
                      ]
                   ];

dtype = "double"
random_seed  = 2024
modelgraph = ai.Model(datatype=dtype, seed = random_seed);
node1  = modelgraph.addNode("node1", ai.NodeType.Generic);
node2  = modelgraph.addNode("node2", ai.NodeType.Generic);

# Four Layered Encoder
node1.setOperations([
                       ai.Encoder(heads=2, attention_size=2, feed_size=4, layers=1, bias=True, activation_type="leakyrelu",  alpha=0.01 ),
                       #ai.MultiHeadAttention(heads=1, attention_size=2),
                       #ai.LayerNorm(),
                       #ai.FeedForward(feed_size = 4, bias = True, activation_type = "leakyrelu", alpha=0.01),
                       #ai.LayerNorm(),
                    ]
                 );

node2.setOperations([
                     ai.Decoder(heads=2,   attention_size=2, feed_size=4, layers=1, bias=True, activation_type="leakyrelu",  alpha=0.01 ),
                     #ai.Dense(size=80, bias=True), ai.Activation(type="leakyrelu", alpha=0.01),
                     ai.Dense(size=3, bias=True), ai.Activation(type="softmax", alpha=0.01)
                    ]
                 );

modelgraph.connect(node1, node2);

encoder_input = input_sequences[:1]

decoder_input = shifted_sequences[:1]  # shifted

target  = target_sequences[:1]

# Set the Data. Normalize if required. Apply Positional Encoding if required
node1.setData(data = encoder_input, normalize=True, positional=True);

# Set the Decoder Data. Normalize if required. Apply Positional Encoding if required
node2.setDecoderData(data = decoder_input, normalize=True, positional=True);

# Set The target
modelgraph.setTarget(data = target, normalize=True);

# Perform fitting
modelgraph.train(loss="cce", metrics=[], optimizer="nadam", batch_size = 10, max_epoch=800, learn_rate=0.01, use_step_decay = False, decay_rate = 0.90)


#yy_pred = modelgraph.predict();
#yyy_pred = yy_pred[0]
#yy_pred.shape

#print("Predicted YY dimension:");
#print(yy_pred.shape);
#pdecoded = tokenizer.decode(sequences = input_sequences);
#print("Decoded YY dimension:");
#pdecoded
