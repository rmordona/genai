import genai as ai
import numpy as np

import random
import numpy as np

seq_length = 10;

rand_x = [random.randint(100, 200) for _ in range(seq_length)]

print(rand_x)

# Shift the elements
shifted_y = rand_x[1:] + [rand_x[0]]

print(shifted_y)

# Convert the list to a NumPy array
tensor_x = np.array(rand_x)
tensor_y = np.array(shifted_y)


# Reshape the array to have dimensions SxIxE (seq_lengthx1x1)
x = tensor_x.reshape((seq_length, 1, 1))
y = tensor_y.reshape((seq_length, 1, 1))

print(x.shape)
print(y.shape)

dtype = "double"
modelgraph = ai.Model(datatype=dtype);
node1  = modelgraph.addNode("node1", ai.NodeType.Generic);
node2  = modelgraph.addNode("node2", ai.NodeType.Generic);


node1.setOperations([
                   ai.Dense(size=12, bias=True), 
                   ai.Activation(type="leakyrelu", alpha=0.01)
                 ]
                   );

# Set the Data
node1.setData(data = x, normalize=True);

node2.setOperations([ai.Dense(size=1, bias=True), ai.Activation(type="leakyrelu", alpha=0.01)])

modelgraph.connect(node1, node2);


# Set The target
modelgraph.setTarget(data = x, normalize=True);

# Perform fitting
modelgraph.train(loss="mse", metrics=[], optimizer="adam", max_epoch=100, learn_rate=0.001, use_step_decay = True, decay_rate = 0.90);
