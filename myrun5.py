import genai as ai
import numpy as np
ai.print_string("Hello", True)

class MyModel(ai.Model):

   def __init__(self, learningRate, datatype):
     super().__init__(learningRate, datatype);

sample = ai.SampleClass(0.01);

dtype = "float"
model = ai.Model(learningRate=0.02, datatype=dtype);

node1  = model.addNode("node1", ai.NodeType.Input);
node1.setOperations([
               ai.Linear(size=10),
               ai.Activation(type="relu")
           ]);

node1.setData(np.array([[1.0, 2.0, 3.0], [3.0, 4.0, 6.0]], dtype=np.float32));

node2  = model.addNode("node2", ai.NodeType.Input);

model.connect(node1, node2);

target = [[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]];
model.setTarget(target);
model.train(loss="mse", optimizer="adam", learnrate=0.01, iter=5);

ai.print_string("Done.", True)
