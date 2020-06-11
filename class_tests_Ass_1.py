# Updated 1/Feb/2019

import numpy as np 
import pandas as pd

######################################

def match_lists(x, y):
    return np.all([(np.isclose(a,b)).all() for a,b in zip(x, y)])

# Dataset
classification_dataset = pd.DataFrame({
    'var1':   [0, 0, 1, 1],
    'var2':   [0, 1, 0, 1],
    'output': [0, 1, 1, 0],
})
classification_dataset = pd.get_dummies(classification_dataset, columns=['output'])
classification_dataset['output'] = pd.Series([0, 1, 1, 0])
classification_x = classification_dataset[['var1','var2']].values
classification_y = classification_dataset[['output_0','output_1']].values

a = np.array([0.30592645, 0.06974693, 0.14607232, 0.18318092, 0.22803499,
       0.39258798, 0.09983689, 0.25711722, 0.29620728, 0.02322521,
       0.30377243, 0.08526206, 0.0325258 , 0.47444277, 0.48281602,
       0.40419867, 0.15230688, 0.04883606, 0.34211651, 0.22007625,
       0.06101912, 0.24758846, 0.01719426, 0.4546602 , 0.12938999,
       0.33126114, 0.15585554, 0.26003401, 0.27335514, 0.09242723,
       0.48479231, 0.38756641, 0.46974947, 0.44741368, 0.29894999,
       0.46093712, 0.04424625, 0.09799143, 0.02261364, 0.16266517,
       0.19433864, 0.13567452, 0.41436875, 0.17837666, 0.14046725,
       0.27134804, 0.07046211, 0.40109849, 0.03727532, 0.49344347])
b = np.array([0.38612238, 0.09935784, 0.00276106, 0.40773071, 0.35342867,
       0.36450358, 0.38563517, 0.03702233, 0.17923286, 0.05793453,
       0.43155171, 0.31164906, 0.16544901, 0.03177918, 0.15549116,
       0.16259166, 0.36480309, 0.31877874, 0.44360637, 0.23610746,
       0.05979712, 0.35662239, 0.38039252, 0.2806386 , 0.38548359,
       0.2468978 , 0.26136641, 0.21377051, 0.01270956, 0.05394571,
       0.01571459, 0.31820521, 0.15717799, 0.25428535, 0.45378324,
       0.12464611, 0.20519146, 0.37777557, 0.11439908, 0.03848995,
       0.14487573, 0.08061064, 0.46484883, 0.40406019, 0.31670188,
       0.4357303 , 0.40183604, 0.09328503, 0.4462795 , 0.26967112])
regression_dataset = pd.DataFrame({
    'var1':   a,
    'var2':   b,
    'output': a+b,
})
regression_x = regression_dataset[['var1','var2']].values
regression_y = regression_dataset[['output']].values

# Expected intermediate outputs 
######################################

initial_weights = [np.array([[ 0.49671415, -0.1382643 ,  0.64768854,  1.52302986, -0.23415337],
        [-0.23413696,  1.57921282,  0.76743473, -0.46947439,  0.54256004]]),
 np.array([[-0.46341769, -0.46572975],
        [ 0.24196227, -1.91328024],
        [-1.72491783, -0.56228753],
        [-1.01283112,  0.31424733],
        [-0.90802408, -1.4123037 ]])]
initial_biases = [np.array([0., 0., 0., 0., 0.]), np.array([0., 0.])]

initial_weights2 = [np.array([[ 0.49671415, -0.1382643 ,  0.64768854,  1.52302986, -0.23415337],
        [-0.23413696,  1.57921282,  0.76743473, -0.46947439,  0.54256004]]),
 np.array([[-0.46341769],
        [-0.46572975],
        [ 0.24196227],
        [-1.91328024],
        [-1.72491783]])]
initial_biases2 = [np.array([0., 0., 0., 0., 0.]), np.array([0.])]

######################################

task1_activations = [np.array([[0.5, 0.5, 0.5, 0.5, 0.5]]), np.array([[0.52150241, 0.47849759]])]
task1_deltas = [np.array([[-6.90168415e-05, -6.43356930e-02,  3.47054337e-02,
         3.96143410e-02, -1.50531455e-02]]), np.array([[-0.11940316,  0.11940316]])]

task1_weights = [np.array([[ 0.49671415, -0.1382643 ,  0.64768854,  1.52302986, -0.23415337],
       [-0.23413696,  1.57921282,  0.76743473, -0.46947439,  0.54256004]]), np.array([[-0.46282068, -0.46632677],
       [ 0.24255929, -1.91387726],
       [-1.72432082, -0.56288455],
       [-1.0122341 ,  0.31365032],
       [-0.90742706, -1.41290072]])]
task1_biases = [np.array([ 6.90168415e-07,  6.43356930e-04, -3.47054337e-04, -3.96143410e-04,
        1.50531455e-04]), np.array([ 0.00119403, -0.00119403])]

######################################

task2_activations = [np.array([[0.51538326, 0.63817935, 0.6211586 , 0.57068845, 0.53441073]]), np.array([[0.08321377]])]
task2_deltas = [np.array([[ 0.00537606,  0.00499497, -0.00264466,  0.02177275,  0.01993469]]), np.array([[-0.04644757]])]
task2_weights = [np.array([[ 0.49669771, -0.13827958,  0.64769663,  1.52296325, -0.23421436],
        [-0.23415772,  1.57919353,  0.76744494, -0.46955846,  0.54248307]]),
 np.array([[-0.46317831],
        [-0.46543333],
        [ 0.24225078],
        [-1.91301517],
        [-1.72466961]])]
task2_biases = [np.array([-5.37606218e-05, -4.99497172e-05,  2.64466424e-05, -2.17727455e-04,
        -1.99346905e-04]), np.array([0.00046448])]

######################################

task3_activations = [np.array([[0.5, 0.5, 0.5, 0.5, 0.5]]), np.array([[0.52150241, 0.47849759]])]
task3_deltas = [np.array([[-6.90168415e-05, -6.43356930e-02,  3.47054337e-02,
         3.96143410e-02, -1.50531455e-02]]), np.array([[-0.11940316,  0.11940316]])]
task3_weights = [np.array([[ 0.49671415, -0.1382643 ,  0.64768854,  1.52302986, -0.23415337],
       [-0.23413696,  1.57921282,  0.76743473, -0.46947439,  0.54256004]]), np.array([[-0.46282068, -0.46632677],
       [ 0.24255929, -1.91387726],
       [-1.72432082, -0.56288455],
       [-1.0122341 ,  0.31365032],
       [-0.90742706, -1.41290072]])]


task3_biases = [np.array([ 6.90168415e-07,  6.43356930e-04, -3.47054337e-04, -3.96143410e-04,
        1.50531455e-04]), np.array([ 0.00119403, -0.00119403])]

######################################

task5_activations = [np.array([[0.5       , 0.5       , 0.5       , 0.5       , 0.5       ],
       [0.44173171, 0.829093  , 0.68296571, 0.38474066, 0.63240775],
       [0.62168683, 0.46548889, 0.65648939, 0.82098421, 0.44172766],
       [0.56526972, 0.80860149, 0.80457276, 0.74145706, 0.5764963 ]]), np.array([[0.52150241, 0.47849759],
       [0.69044712, 0.30955288],
       [0.34856978, 0.65143022],
       [0.52880861, 0.47119139]])]
task5_deltas = [np.array([[-6.90168415e-05, -6.43356930e-02,  3.47054337e-02,
         3.96143410e-02, -1.50531455e-02],
       [ 8.41388306e-05,  4.50666140e-02, -3.71486013e-02,
        -4.63573488e-02,  1.72993843e-02],
       [ 4.30397497e-05,  4.24433417e-02, -2.07518526e-02,
        -1.54372562e-02,  9.84281893e-03],
       [-6.67064858e-05, -3.91617814e-02,  2.14627451e-02,
         2.98681614e-02, -1.44550088e-02]]), np.array([[-0.11940316,  0.11940316],
       [ 0.14756919, -0.14756919],
       [ 0.07914935, -0.07914935],
       [-0.11740679,  0.11740679]])]
task5_weights = [np.array([[ 0.49671439, -0.13829712,  0.64768143,  1.52288555, -0.23410725],
       [-0.23413713,  1.57915377,  0.76759159, -0.46930949,  0.5425316 ]]), np.array([[-0.46330093, -0.46584651],
       [ 0.24191672, -1.9132347 ],
       [-1.72490365, -0.56230171],
       [-1.01258115,  0.31399736],
       [-0.90803308, -1.4122947 ]])]


task5_biases = [np.array([ 8.54474697e-08,  1.59875187e-04,  1.73227504e-05, -7.68789733e-05,
        2.36595112e-05]), np.array([ 0.00010091, -0.00010091])]

lr = 0.01
def test_neural_network(cls, task):
    np.random.seed(42)
    args_dict = {
        'task1': {'input_size':2, 'hidden_nodes':5, 'output_size':2},
        'task2': {'input_size':2, 'hidden_nodes':5, 'output_size':1, 'mode':'regression'},
        'task3': {'nodes_per_layer':[2,5,2], 'mode':'classification'},
        'task4': {'nodes_per_layer':[2,5,2], 'mode':'classification'},
        'task5': {'nodes_per_layer':[2,5,2], 'mode':'classification'},
    }
    nn = cls(**args_dict[task])
    
#     print("Task is: ", task)
    
    if task == 'task1':
        assert(match_lists(nn.weights_, initial_weights))
        assert(match_lists(nn.biases_, initial_biases))
        print('Tests properly initialized')

        model_input  = classification_x[0,:].reshape((1,2))
        model_output = classification_y[0,:].reshape((1,2))
        
        activations = nn.forward_pass(model_input)
        assert(match_lists(activations, task1_activations))
        print('Forward pass is OK')

        deltas = nn.backward_pass(model_output, activations)

        assert(match_lists(deltas, task1_deltas))
        print('Backward pass is OK')

        layer_inputs = [model_input] + activations[:-1]
        nn.weight_update(deltas, layer_inputs, lr)
        assert(match_lists(nn.weights_, task1_weights))
        assert(match_lists(nn.biases_, task1_biases))
        print('Weight update is OK')

    elif task == 'task2':
        assert(match_lists(nn.weights_, initial_weights2))
        assert(match_lists(nn.biases_, initial_biases2))
        print('Tests properly initialized')
        
        model_input  = regression_x[0,:].reshape((1,2))
        model_output = regression_y[0,:].reshape((1,1))
        
        activations = nn.forward_pass(model_input)
        # print ("act -> ", activations)
        assert(match_lists(activations, task2_activations))
        print('Forward pass is OK')

        deltas = nn.backward_pass(model_output, activations)
        # print ("del ->", deltas)
        assert(match_lists(deltas, task2_deltas))
        print('Backward pass is OK')

        layer_inputs = [model_input] + activations[:-1]
        nn.weight_update(deltas, layer_inputs, lr)

        assert(match_lists(nn.weights_, task2_weights))
        assert(match_lists(nn.biases_, task2_biases))
        print('Weight update is OK')
    elif task == 'task3':
        assert(match_lists(nn.weights_, initial_weights))
        assert(match_lists(nn.biases_, initial_biases))
        print('Tests properly initialized')
        
        model_input  = classification_x[0,:].reshape((1,2))
        model_output = classification_y[0,:].reshape((1,2))
        
        activations = nn.forward_pass(model_input)
        assert(match_lists(activations, task3_activations))
        print('Forward pass is OK')

        deltas = nn.backward_pass(model_output, activations)
        assert(match_lists(deltas, task3_deltas))
        print('Backward pass is OK')

        layer_inputs = [model_input] + activations[:-1]
        nn.weight_update(deltas, layer_inputs, lr)
        assert(match_lists(nn.weights_, task3_weights))
        assert(match_lists(nn.biases_, task3_biases))
        print('Weight update is OK')
    
    elif task == 'task4':
        print('Task 4 does not have any tests')
        
    elif task == 'task5':
        assert(match_lists(nn.weights_, initial_weights))
        assert(match_lists(nn.biases_, initial_biases))
        print('Tests properly initialized')
        
        model_input  = classification_x
        model_output = classification_y
        
        activations = nn.forward_pass(model_input)
        assert(match_lists(activations, task5_activations))
        print('Forward pass is OK')

        deltas = nn.backward_pass(model_output, activations)
        assert(match_lists(deltas, task5_deltas))
        print('Backward pass is OK')

        layer_inputs = [model_input] + activations[:-1]
        nn.weight_update(deltas, layer_inputs, lr)
        assert(match_lists(nn.weights_, task5_weights))
        assert(match_lists(nn.biases_, task5_biases))
        print('Weight update is OK')
    else:
        print('Invalid task')
