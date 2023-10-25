import genai as ai 
import numpy as np
ai.print_string("Hello", True)

#class MyModel(ai.Model):
#
#   def __init__(self, learningRate, datatype):
#     super().__init__(learningRate, datatype);

sample = ai.SampleClass(0.01);

dtype = "float"
modelgraph = ai.Model(learningRate=0.02, datatype=dtype);

node1  = modelgraph.addNode("node1", ai.NodeType.Input);
node1.setOperations([ai.Convolution(kernel_size=2, stride=1, padding=1, dilation=1, bias=True)]);

node2  = modelgraph.addNode("node2", ai.NodeType.Input);
node2.setOperations([ai.Convolution(kernel_size=2, stride=1, padding=1, dilation=1, bias=True), 
                     ai.Flatten(), 
                     ai.Dropout(probability = 0.05),
                     ai.Dense(size=4, bias=True), 
                     ai.Activation(type="softmax", alpha=0.01)]);

embedding1 = [
               [  [1.11,8.12,1.13,5.14],  [1.21,1.22,1.23,13.24], [1.31,5.32,2.33,1.34]  ], # sequence 1 of batch 1,2,3
               [  [4.11,2.12,6.13,2.14],  [2.21,2.22,2.23,2.24], [9.31,3.32,8.33,0.34]  ], # sequence 2 of batch 1,2,3
               [  [3.11,3.12,3.13,3.14],  [3.21,5.22,4.23,3.24], [0.31,3.32,3.33,3.34]  ], # sequence 3 of batch 1,2,3
               [  [4.11,2.12,8.13,4.14],  [4.21,4.22,9.23,8.24], [8.31,6.32,4.33,1.34]  ], # sequence 4 of batch 1,2,3
               [  [5.11,9.12,5.13,5.14],  [6.21,2.22,5.23,13.24], [1.31,5.32,5.33,12.34]  ]  # sequence 5 of batch 1,2,3
             ];

node1.setData(data = np.array(embedding1, dtype=np.float32), normalize=True);

modelgraph.connect(node1, node2);

target = [
           [  [1.00, 0.00, 0.00, 0.00 ]  ],
           [  [0.00, 1.00, 0.00, 0.00 ]  ],
           [  [0.00, 0.00, 1.00, 0.00 ]  ],
           [  [0.00, 0.00, 0.00, 1.00 ]  ],
           [  [1.00, 0.00, 0.00, 0.00 ]  ]
         ];
modelgraph.setTarget(target);

modelgraph.train(loss="cce", optimizer="adam", learnrate=0.1, maxiteration=1);

ai.print_string("Show Graph ...", True);
p = modelgraph.generateDotFormat();
print(p);
ai.print_string("end show ...", True);


ai.print_string("Done.", True)
