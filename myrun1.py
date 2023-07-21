import genai as ai
import numpy as np
ai.print_string("Hello", True)

class MyModel(ai.Model):

   def __init__(self):
     super().__init__();

     print("Creating a Graph!")
     graph = ai.Graph();

     print("Add Node 1 ...")
     embedding1 = [[1.0, 2.0, 3.0], [3.0, 4.0, 6.0]]
     node1 = graph.addNode("Node 1", ai.NodeType.Input, embedding1)
     node1.setOperations([ai.Attention(heads=1, size=2), ai.Activation(type="leakyrelu", alpha=0.01)]);


     print("Add Node 3 ...")
     node3 = graph.addNode("Node 3", ai.NodeType.Output)
     node3.setOperations([ai.Linear(size=3, bias=False), ai.LayerNorm(), ai.Activation(type="leakyrelu", alpha=0.01)]) 

     # embedding = [[1.0, 2.0, 3.0], [3.0, 4.0, 6.0]]
     # graph.setData(embedding);

     # assign to the instance
     self.graph = graph;
     self.node1 = node1
     self.node3 = node3

     print("Connect nodes ...");
     self.graph.connect(self.node1, self.node3);

     self.setGraph(self.graph);

model = MyModel();

target = [[1.0, 2.0, 3.0], [3.0, 4.0, 6.0]];
model.setTarget(target);
model.train(loss="mse", optimizer="adam", learnrate=0.01, iter=200);

ai.print_string("Done.", True)
