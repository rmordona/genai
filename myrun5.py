import genai as ai
import numpy as np
ai.print_string("Hello", True)

class MyModel(ai.Model):

   def __init__(self, datatype = datatype):
     super().__init__(datatype = datatype);


model = MyModel(datatype = "double");

target = [[1.0, 2.0, 3.0], [3.0, 4.0, 5.0]];
#model.setTarget(target);
#model.train(loss="mse", optimizer="adam", learnrate=0.01, iter=500);

ai.print_string("Done.", True)
