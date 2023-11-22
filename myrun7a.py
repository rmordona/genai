import genai as ai
import numpy as np

dtype = "double"
modelgraph = ai.Model(datatype=dtype);
node1  = modelgraph.addNode("node1", ai.NodeType.Generic);
node2  = modelgraph.addNode("node2", ai.NodeType.Generic);

node1.setOperations([
                ai.LSTM(hidden_size=80, 
                        output_size=5, 
                        num_layers=1, 
                        bidirectional=False, 
                        rnntype=ai.RNNtype.MANY_TO_MANY),
                ai.Dense(size=5), ai.Activation(type="leakyrelu", alpha=0.01)
                ]
                   );

x1 = [[[ 0.34317529, -0.69063473,  0.65708005,  0.09877755,
         -0.05051189, -0.29432243,  0.10578649, -0.46993446,
         -0.72658622, -0.16194397,  0.13085125, -0.40490472,
         -0.88970828, -0.17892411, -0.60243028, -0.10152148,
         -0.26011023,  0.72278601,  0.01207577,  0.97832346,
         -1.08795989,  0.30683896,  1.02260303,  0.1672757 ,
          0.27075216,  0.20439036,  0.29486308, -0.03910881,
          1.14639497,  0.18527377,  0.04058237, -0.91703683,
         -0.49551144,  0.10791174,  0.6562736 , -0.98567939,
         -0.00632003,  0.3844589 ,  0.3790468 , -0.57600546,
          0.60687268,  0.63389015,  0.01343358, -0.69140279,
         -0.47948426,  0.28930652,  0.10598569, -0.1709933 ,
         -0.74618888,  0.64357972,  0.27579775,  0.80185759,
          0.15218598,  1.07663929,  0.31224859, -0.77955294,
          0.00339411,  0.19511873, -0.75116915, -0.30119729,
         -1.28724551, -1.09497666, -0.35066164,  0.19456808,
         -0.66860288, -0.1479075 ,  1.14903843, -0.58815849,
         -0.50004679, -0.69172359,  0.30856565,  0.44746417,
         -0.66354716, -1.02974474, -0.21128634,  1.04934049,
          0.26031271,  0.19317107, -0.59802079, -0.66017044]]];

x = [[
        [-0.62000191, -1.27273846,  0.64314526,  1.4581964 , -1.41556358],
        [-0.62000191, -1.27273846,  0.64314526,  1.4581964 , -1.41556358],
        [-0.62000191, -1.27273846,  0.64314526,  1.4581964 , -1.41556358],
        [-0.62000191, -1.27273846,  0.64314526,  1.4581964 , -1.41556358],
        [-0.62000191, -1.27273846,  0.64314526,  1.4581964 , -1.41556358]], 
     [  [ 3.6680336 , -4.72590256,  1.79263234,  2.09708142, 3.62206459],
        [ 3.6680336 , -4.72590256,  1.79263234,  2.09708142, 3.62206459],
        [ 3.6680336 , -4.72590256,  1.79263234,  2.09708142, 3.62206459],
        [ 3.6680336 , -4.72590256,  1.79263234,  2.09708142, 3.62206459],
        [ 3.6680336 , -4.72590256,  1.79263234,  2.09708142, 3.62206459]] ];

x2 = [[
        [-0.62000191, -1.27273846,  0.64314526,  1.4581964 , -1.41556358],
        [-0.62000191, -1.27273846,  0.64314526,  1.4581964 , -1.41556358],
        [-0.62000191, -1.27273846,  0.64314526,  1.4581964 , -1.41556358],
        [-0.62000191, -1.27273846,  0.64314526,  1.4581964 , -1.41556358],
        [-0.62000191, -1.27273846,  0.64314526,  1.4581964 , -1.41556358]  ],
       [[ 3.6680336 , -4.72590256,  1.79263234,  2.09708142, 3.62206459],
        [ 3.6680336 , -4.72590256,  1.79263234,  2.09708142, 3.62206459],
        [ 3.6680336 , -4.72590256,  1.79263234,  2.09708142, 3.62206459],
        [ 3.6680336 , -4.72590256,  1.79263234,  2.09708142, 3.62206459],
        [ 3.6680336 , -4.72590256,  1.79263234,  2.09708142, 3.62206459]],
       [[-0.62838537, -1.52371931,  0.60713536,  2.18721867, -1.07151377],
        [-0.62838537, -1.52371931,  0.60713536,  2.18721867, -1.07151377],
        [-0.62838537, -1.52371931,  0.60713536,  2.18721867, -1.07151377],
        [-0.62838537, -1.52371931,  0.60713536,  2.18721867, -1.07151377],
        [-0.62838537, -1.52371931,  0.60713536,  2.18721867, -1.07151377]],
       [[-0.64610696, -3.03179026,  0.65387177,  1.55924249, -1.79662549],
        [-0.64610696, -3.03179026,  0.65387177,  1.55924249, -1.79662549],
        [-0.64610696, -3.03179026,  0.65387177,  1.55924249, -1.79662549],
        [-0.64610696, -3.03179026,  0.65387177,  1.55924249, -1.79662549],
        [-0.64610696, -3.03179026,  0.65387177,  1.55924249, -1.79662549]],
       [[-1.4518218 , -2.77096462,  1.2759279 ,  2.18078423, -1.42411494],
        [-1.4518218 , -2.77096462,  1.2759279 ,  2.18078423, -1.42411494],
        [-1.4518218 , -2.77096462,  1.2759279 ,  2.18078423, -1.42411494],
        [-1.4518218 , -2.77096462,  1.2759279 ,  2.18078423, -1.42411494],
        [-1.4518218 , -2.77096462,  1.2759279 ,  2.18078423, -1.42411494]]];



node1.setData(data = x, normalize=True);

node2.setOperations([ai.Dense(size=5, bias=True), ai.Activation(type="softmax", alpha=0.01)])

modelgraph.connect(node1, node2);

y2 = [ [ [0,1,0,0,0] ] ];
y = [ 
      [ [0,1,0,0,0],
        [0,1,0,0,0],
        [0,1,0,0,0],
        [0,1,0,0,0],
        [0,1,0,0,0] ],
      [ [0,0,1,0,0],
        [0,0,1,0,0],
        [0,0,1,0,0],
        [0,0,1,0,0],
        [0,0,1,0,0] ] ];

y1 = [ 
      [ [0,1,0,0,0],
        [0,1,0,0,0],
        [0,1,0,0,0],
        [0,1,0,0,0],
        [0,1,0,0,0] ] ,
      [ [1,0,0,0,0],
        [1,0,0,0,0],
        [1,0,0,0,0],
        [1,0,0,0,0],
        [1,0,0,0,0] ] ,
      [ [0,0,1,0,0],
        [0,0,1,0,0],
        [0,0,1,0,0],
        [0,0,1,0,0],
        [0,0,1,0,0] ] ,
      [ [0,0,0,0,1],
        [0,0,0,0,1],
        [0,0,0,0,1],
        [0,0,0,0,1],
        [0,0,0,0,1] ] ,
      [ [0,0,0,1,0],
        [0,0,0,1,0],
        [0,0,0,1,0],
        [0,0,0,1,0],
        [0,0,0,1,0] ]];




# Set The target
modelgraph.setTarget(data = y);

# Perform fitting
modelgraph.train(loss="cce", metrics=[], optimizer="nadam", max_epoch=100, learn_rate=0.01, use_step_decay = False, decay_rate = 0.90);
