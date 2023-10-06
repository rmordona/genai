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
#node1.setOperations([ai.Encoder(heads=2, size=5, bias=True, type="leakyrelu", alpha=0.01)]);
#node1.setOperations([ai.FeedForward(size=2, bias=True, type="leakyrelu", alpha=0.01)]);
#node1.setOperations([ai.Attention(size=5, bias=False), ai.Activation(type="leakyrelu", alpha=0.01)]);
#node1.setOperations([ai.Dense(size=2, bias=True), ai.LayerNorm(), ai.Activation(type="leakyrelu", alpha=0.01)]);

# MANY_TO_MANY
node1.setOperations([ai.RNN(hidden_size=6, output_size=5,  num_layers=1, bidirectional=True, rnntype=ai.RNNtype.MANY_TO_MANY), ai.Activation(type="leakyrelu", alpha=0.01)]);
#node1.setOperations([ai.LSTM(hidden_size=6, output_size=5, num_layers=1, bidirectional=True, rnntype=ai.RNNtype.MANY_TO_MANY), ai.Activation(type="leakyrelu", alpha=0.01)]);
#node1.setOperations([ai.GRU(hidden_size=6, output_size=5, num_layers=1, bidirectional=True, rnntype=ai.RNNtype.MANY_TO_MANY), ai.Activation(type="leakyrelu", alpha=0.01)]);

# ONE_TO_MANY
#node1.setOperations([ai.RNN(hidden_size=6, output_size=5, output_sequence_length=2, num_layers=1, bidirectional=False, rnntype=ai.RNNtype.ONE_TO_MANY), ai.Activation(type="leakyrelu", alpha=0.01)]);
#node1.setOperations([ai.LSTM(hidden_size=6, output_size=5, num_layers=1, bidirectional=True, rnntype=ai.RNNtype.MANY_TO_MANY), ai.Activation(type="leakyrelu", alpha=0.01)]);
#node1.setOperations([ai.GRU(hidden_size=6, output_size=5, num_layers=1, bidirectional=True, rnntype=ai.RNNtype.MANY_TO_MANY), ai.Activation(type="leakyrelu", alpha=0.01)]);


embedding1 = [[[1.0, 2.0, 3.0, 4.0], [3.0, 4.0, 5.0, 6.0],[7.0,8.0,9.0,10.0]],
              [[11.0, 10.0, 9.0, 8.0], [7.0, 6.0, 5.0, 4],[3.0,2.0,1.0,0.0]]];
node1.setData(data = np.array(embedding1, dtype=np.float32), normalize=False);

node2  = modelgraph.addNode("node2", ai.NodeType.Input);
node2.setOperations([ai.Dense(size=3, bias=True), ai.Activation(type="leakyrelu", alpha=0.01)]) 

modelgraph.connect(node1, node2);

target = [[[1.0, 2.0, 3.0], [3.0, 4.0, 5.0],[3.0,4.0,5.0]],
          [[5.0, 4.0, 3.0], [5.0, 4.0, 3.0],[5.0,4.0,3.0]]];
#target = [[1.0, 2.0, 3.0], [3.0, 4.0, 5.0],[3.0,4.0,5.0]];
modelgraph.setTarget(target);
modelgraph.train(loss="mse", optimizer="adam", learnrate=0.1, iter=200);

ai.print_string("Done.", True)
