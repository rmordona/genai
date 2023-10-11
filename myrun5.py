import genai as ai 
import numpy as np
ai.print_string("Hello", True)

class MyModel(ai.Model):

   def __init__(self, learningRate, datatype):
     super().__init__(learningRate, datatype);

sample = ai.SampleClass(0.01);

dtype = "float"
modelgraph = ai.Model(learningRate=0.02, datatype=dtype);

node1  = modelgraph.addNode("node1", ai.NodeType.Input);
#node1.setOperations([ ai.Dense(size=5), ai.Activation(type="leakyrelu", alpha=0.01) ]);
node1.setOperations([ai.Encoder(heads=2, size=6, bias=True, type="leakyrelu", alpha=0.01)]);
#node1.setOperations([ai.FeedForward(size=2, bias=True, type="leakyrelu", alpha=0.01)]);
#node1.setOperations([ai.Attention(size=5, bias=False), ai.Activation(type="leakyrelu", alpha=0.01)]);
#node1.setOperations([ai.Dense(size=2, bias=True), ai.LayerNorm(), ai.Activation(type="leakyrelu", alpha=0.01)]);

# MANY_TO_MANY
#node1.setOperations([ai.RNN(hidden_size=6, output_size=5,  num_layers=1, bidirectional=True, rnntype=ai.RNNtype.MANY_TO_MANY), ai.Activation(type="leakyrelu", alpha=0.01)]);
#node1.setOperations([ai.LSTM(hidden_size=6, output_size=5, num_layers=1, bidirectional=True, rnntype=ai.RNNtype.MANY_TO_MANY), ai.Activation(type="leakyrelu", alpha=0.01)]);
#node1.setOperations([ai.GRU(hidden_size=6, output_size=5, num_layers=1, bidirectional=True, rnntype=ai.RNNtype.MANY_TO_MANY), ai.Activation(type="leakyrelu", alpha=0.01)]);

# ONE_TO_MANY
#node1.setOperations([ai.RNN(hidden_size=6, output_size=5, output_sequence_length=3, num_layers=1, bidirectional=False, rnntype=ai.RNNtype.ONE_TO_MANY), ai.Activation(type="leakyrelu", alpha=0.01)]);
#node1.setOperations([ai.LSTM(hidden_size=6, output_size=5, output_sequence_length=3, num_layers=1, bidirectional=False, rnntype=ai.RNNtype.ONE_TO_MANY), ai.Activation(type="leakyrelu", alpha=0.01)]);
#node1.setOperations([ai.GRU(hidden_size=6, output_size=5, output_sequence_length=3, num_layers=1, bidirectional=False, rnntype=ai.RNNtype.ONE_TO_MANY), ai.Activation(type="leakyrelu", alpha=0.01)]);

node2  = modelgraph.addNode("node2", ai.NodeType.Input);
#node2.setOperations([ai.Dense(size=4, bias=True), ai.Activation(type="leakyrelu", alpha=0.01)]) 
node2.setOperations([ai.Decoder(heads=2, size=6, bias=True, type="leakyrelu", alpha=0.01), ai.Dense(size=4, bias=True), ai.Activation(type="leakyrelu", alpha=0.01)]);

embedding1 = [
               [  [1.11,1.12,1.13,1.14],  [1.21,1.22,1.23,1.24], [1.31,1.32,1.33,1.34]  ], # sequence 1 of batch 1,2,3
               [  [2.11,2.12,2.13,2.14],  [2.21,2.22,2.23,2.24], [2.31,2.32,2.33,2.34]  ], # sequence 2 of batch 1,2,3
               [  [3.11,3.12,3.13,3.14],  [3.21,3.22,3.23,3.24], [3.31,3.32,3.33,3.34]  ], # sequence 3 of batch 1,2,3
               [  [4.11,4.12,4.13,4.14],  [4.21,4.22,4.23,3.24], [4.31,4.32,4.33,5.34]  ], # sequence 4 of batch 1,2,3
               [  [5.11,5.12,5.13,5.14],  [5.21,5.22,5.23,3.24], [5.31,5.32,5.33,5.34]  ]  # sequence 5 of batch 1,2,3
             ];

embedding2 = [ 
               [  [1.11,1.12,1.13,1.14],  [1.21,1.22,1.23,1.24], [1.31,1.32,1.33,1.34]  ], # sequence 1 of batch 1,2,3
               [  [2.11,2.12,2.13,2.14],  [2.21,2.22,2.23,2.24], [2.31,2.32,2.33,2.34]  ], # sequence 2 of batch 1,2,3
               [  [3.11,3.12,3.13,3.14],  [3.21,3.22,3.23,3.24], [3.31,3.32,3.33,3.34]  ], # sequence 3 of batch 1,2,3
               [  [4.11,4.12,4.13,4.14],  [4.21,4.22,4.23,3.24], [4.31,4.32,4.33,5.34]  ], # sequence 4 of batch 1,2,3
               [  [5.11,5.12,5.13,5.14],  [5.21,5.22,5.23,3.24], [5.31,5.32,5.33,5.34]  ]  # sequence 5 of batch 1,2,3
             ];

node1.setData(data = np.array(embedding1, dtype=np.float32), normalize=False);
node2.setDecoderData(data = np.array(embedding2, dtype=np.float32), normalize=False);

modelgraph.connect(node1, node2);

target = [
           [  [2.11,2.12,2.13,2.14],  [2.21,2.22,2.23,2.24], [2.31,2.32,2.33,2.34]  ], # sequence 1 of batch 1,2,3
           [  [3.11,3.12,3.13,3.14],  [3.21,3.22,3.23,3.24], [3.31,3.32,3.33,3.34]  ], # sequence 2 of batch 1,2,3
           [  [4.11,4.12,4.13,4.14],  [4.21,4.22,4.23,3.24], [4.31,4.32,4.33,5.34]  ], # sequence 3 of batch 1,2,3
           [  [5.11,5.12,5.13,5.14],  [5.21,5.22,5.23,3.24], [5.31,5.32,5.33,5.34]  ], # sequence 4 of batch 1,2,3
           [  [6.11,6.12,6.13,6.14],  [6.21,6.22,6.23,6.24], [6.31,6.32,6.33,6.34]  ]  # sequence 5 of batch 1,2,3
         ];
modelgraph.setTarget(target);
modelgraph.train(loss="mse", optimizer="adam", learnrate=0.1, iter=1);

ai.print_string("Done.", True)
