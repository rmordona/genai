import genai as ai
import numpy as np
ai.print_string("Hello", True)

class MyModel(ai.Model):

   def __init__(self, learningRate, datatype):
     super().__init__(learningRate, datatype);


dtype = "double"
model = ai.Model(learningRate=0.02, datatype=dtype);
node1  = model.addNode("node1", ai.NodeType.Input);
node1.setOperations([
               ai.Linear(size=10),
               ai.Activation(type="relu")
           ]);
node2  = model.addNode("node2", ai.NodeType.Input);
model.connect(node1, node2);

target = [[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]];
#model.setTarget(target);
#model.train(loss="mse", optimizer="adam", learnrate=0.01, iter=500);

ai.print_string("Done.", True)
