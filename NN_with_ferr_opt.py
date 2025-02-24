##Neural Network test
import numpy as np

def create_net(input_size, output_size, hidden_layers_size):
    """
    input_size -> number of input neurons
    output_size -> number of output neurons
    hidden_layers_size -> list of integers, each integer represents the number of neurons in a hidden layer
    """
    net = []
    
    #input layer
    W = np.random.randn(input_size, hidden_layers_size[0]).astype(dtype=np.float32)
    b = np.random.randn(hidden_layers_size[0]).astype(dtype=np.float32)
    net.append((W, b))
    
    #hidden layers
    num_hidden_layers = len(hidden_layers_size)
    for i in range(1, num_hidden_layers):
        W = np.random.randn(hidden_layers_size[i-1], hidden_layers_size[i]).astype(dtype=np.float32)
        b = np.random.randn(hidden_layers_size[i]).astype(dtype=np.float32)
        net.append((W, b))

    W = np.random.randn(hidden_layers_size[-1], output_size).astype(dtype=np.float32)
    b = np.random.randn(output_size).astype(dtype=np.float32)
    net.append((W, b))

    return net


def sigmoid(A, compute_derivative=False):
    """Sigmoid activation function (element-wise), with the option of using the derivative wrt its argument"""
    if not compute_derivative:
        return 1. / (1. + np.exp(-A)) #element-wise
    else:
        return sigmoid(A) * (1. - sigmoid(A))
    

def ReLU(A, compute_derivative=False):
    """Rectified Linear Unit: can choose to compute the derivative wrt activation scores"""
    if not compute_derivative:
        return np.maximum(0.0, A)
    else:
        A[A<=0] = 0
        A[A>0] = 1
        return A
    
    
def leaky_ReLU(A, compute_derivative=False, sigma=0.01):
    """Leaky ReLU: will allow small outputs for x<0 to combat vanishing gradient"""
    if not compute_derivative:
        return np.maximum(sigma*A, A)
    else:
        A[A<=0] = sigma
        A[A>0] = 1
        return A
    
    
def mean_square_error(OL, y, compute_derivative=False, double_counter=0):
    """
    Mean Squared Error (MSE) loss function, or its derivative wrt the output layer
    OL -> output layer
    y -> labels
    compute_derivative -> boolean, if True computes the derivative of the loss function wrt the output layer
    double_counter -> integer, used to keep track of the number of times the loss function has been doubled
    """
    if not compute_derivative:
        return np.mean((OL - y)**2) * 2**double_counter
    else:
        return (2.0 / y.shape[0]) * (OL - y) * 2**double_counter


def forward(net, X, activation_function):
    """
    Computes activation scores and the outputs (using activation function) of each layer of the net
    net -> list of tuples (Wl, bl) containing weights and biases of the neural network
    X -> input data
    activation_function -> activation function passed as a callback
    """

    L = len(net) #number of layers
    O = [None] * L #empty list for output values
    A = [None] * L #empty list for activation values

    for l in range (0, L):
        Wl, bl = net[l]

        if l == 0:
            A[l] = np.matmul(X, Wl) + bl #first layer: the input is the actual data
        else:
            A[l] = np.matmul(O[l-1], Wl) + bl #following layers: the input is the previous output vector

        O[l] = activation_function(A[l], compute_derivative=False)
    return O, A



def backward (net, X, O, A, der_loss, activation_function):
    """
    Compute the gradients wrt weights and biases
    net -> list of tuples (Wl, bl) containing weights and biases of the neural network
    X -> list of input data
    O -> list of output values
    A -> list of activation values
    der_loss -> derivative of the loss function wrt the output layer
    """
    L = len(net) #number of layers
    G = [None] * L #list to collect gradients

    for l in range(L-1, -1, -1): #loop from layer L-1 to layer 0 (included)
        if l == L - 1:
            Delta = activation_function(A[l], compute_derivative=True) * der_loss #element-wise product
        else:
            W_l_plus_1, _ = net[l+1]
            Delta = activation_function(A[l], compute_derivative=True) * np.matmul(Delta, W_l_plus_1.T)

        if l > 0:
            W_l_grad = np.matmul(O[l-1].T, Delta)
        else:
            W_l_grad = np.matmul(X.T, Delta)
        
        b_l_grad = np.sum(Delta, axis=0) #to obtain the sum of the rows, is the same as multiplication by a vector of transposed ones
        G[l] = (W_l_grad, b_l_grad) #gradients for layer l

    return G


def update(net, G, lr):        
    """Update of weights and biases, given the gradient and the training step"""

    L = len(net)

    for l in range(0, L):
        Wl, bl = net[l] #weights and biases of layer l
        Wl_grad, bl_grad = G[l] #gradients wrt weights and biases of layer l

        Wl -= lr * Wl_grad
        bl -=  lr * bl_grad   
    
    
def ferrari_optimizer(lr, loss_grad):
    """
    Ferrari optimizer: given lr, computes the square norm of the loss, and returns a new learning rate
    """
    #square of the norm of the loss gradient
    loss_grad_norm_sq = np.sum(loss_grad**2)
    
    #new learing rate
    new_lr = lr / loss_grad_norm_sq
    
    return new_lr


    


def train(net, X, y, epochs=2000, lr=0.001, use_ferrari=True):
    """Train a neural network over multiple epochs"""
    lr0 = 0
    double_counter = 0
    stop_ferrari = not use_ferrari

    for e in range(0, epochs):        
        outputs, activation_scores = forward(net, X, sigmoid) #forward computation
        loss_value = mean_square_error(outputs[-1], y, compute_derivative=False, double_counter=double_counter)

        loss_derivative = mean_square_error(outputs[-1], y, compute_derivative=True, double_counter=double_counter) #backward
        gradients = backward(net, X, outputs, activation_scores, loss_derivative, sigmoid)
        
        true_loss_value = loss_value / 2**double_counter
        loss_grad = gradients[-1][0]

        update(net, gradients, lr) #weights & biases update
        
        #save starting conditions
        if (e == 0):
            starting_loss = loss_value
            lr0 = lr
        
        #Ferrari optimizer handling
        if not stop_ferrari:
            #apply Ferrari optimizer
            lr = ferrari_optimizer(lr, loss_grad)
            
            #if loss gets below 50%, double the MSE
            if (loss_value < starting_loss/2):
                double_counter +=1
            
            #Ferrari stopping condition:
            #if we are near a stationary point (both local minimum, saddle point and flat region)
            #the Ferrari will surely break since the loss gradient will be close to zero
            if (np.mean(loss_grad) < 0.005):
                lr = lr0
                double_counter = 0
                stop_ferrari = True
                print("Ferrari optimizer stopped.")
        else:
            #apply alternative optimizer (in this case a static learning rate) resuming from lr0
            pass
            
        #print loss value
        print("epoch: {}, true_loss: {}, loss_grad: {}, lr: {}".format(e+1, true_loss_value, np.mean(loss_grad), lr))

        #early stopping condition
        if (true_loss_value < 1e-5):
            print("Loss value is below threshold, training stopped.")
            break
    


if __name__ == "__main__":    
    ### Choose hyperparameters ###
    
    #set seed for reproducibility
    np.random.seed(420)
    
    #train & validation set split, between 0 and 1
    split_percentage = 0.8
    
    #network configuration
    input_size = 64
    output_size = 10
    neurons_per_hl = [30]
    
    #training parameters (optimal lr=0.08)
    epochs = 8000
    learning_rate = 3.0
    ferrari_opt = True #wheter to start the training with Ferrari optimizer or not
    
    ### End of hyperparameters config ###
    
    
    #NOTE: in the following code the difference between y and y_ is that y_ is the label in the form of a number (0-9)
    #while y is the label in the form of a 1-hot vector

    #training data load
    NPZfile = np.load("./train data.npz")
    X = np.empty([0, 64])
    y_ = np.empty([0, 1])
    X = NPZfile[NPZfile.files[0]] #load first array (X) from NPZfile
    y_ = NPZfile[NPZfile.files[1]] #then the second (y)
        
    #1hot encoding of labels
    y = np.zeros([y_.shape[0], 10], dtype=int)
    for i in range (0, y_.shape[0]):
        y[i, y_[i]] = 1
    
    #shuffle indexes
    n = X.shape[0]
    idx = np.arange(n)
    np.random.shuffle(idx)

    #datasplit
    l = int(n*split_percentage)
    train_idx = idx[:l]
    validate_idx = idx[l:]

    x_train = X[train_idx]
    y_train = y[train_idx]

    x_val = X[validate_idx]
    y_val = y[validate_idx]
    y_val_ = y_[validate_idx]

    #network creation
    my_net = create_net(input_size, output_size, neurons_per_hl)
    
    
    
    # TRANING LOOP
    print("Start training...")
    train(my_net, x_train, y_train, epochs=epochs, lr=learning_rate, use_ferrari=ferrari_opt)
    print("Training completed.")

    #prediction
    output_val, _ = forward(my_net, x_val, sigmoid)
    
    #building Confusion Matrix
    Confusion_Matrix = np.zeros((10, 10), dtype=int)
    firstPrint = True
    
    for i in range(0, x_val.shape[0]):
        prediction = np.argmax(output_val[-1][i], axis=0)
        label = y_val_[i]
        
        if (prediction != label):
            if (firstPrint == True):
                print("Printing all mismatches:")
                firstPrint = False
            print("Predicted output: {}, actual output: {}".format(prediction, label))
        
        Confusion_Matrix[label, prediction] += 1
        
    print("\nConfusion Matrix:")
    print(Confusion_Matrix)
    
    #checking that all validation data has been catalogued in Confusion_Matrix
    assert n-l == np.sum(Confusion_Matrix), "Confusion matrix doesn't add up correctly: {} vs {}".format(n-l, np.sum(Confusion_Matrix))

    
    confusion_metrics = np.zeros((10, 4), dtype=int)
    CM = {
        "TP": 0,
        "TN": 1,
        "FP": 2,
        "FN": 3
    }

    col_sum = Confusion_Matrix.sum(axis=0)
    row_sum = Confusion_Matrix.sum(axis=1)
    whole_sum = Confusion_Matrix.sum()

    for label in range(0, 10):
        # True positive for class Label
        TP = Confusion_Matrix[label, label]
        confusion_metrics[label, CM["TP"]] = TP

        # False positive for class Label
        FP = col_sum[label] - TP
        confusion_metrics[label, CM["FP"]] = FP

        # False negative for class Label
        FN = row_sum[label] - TP
        confusion_metrics[label, CM["FN"]] = FN

        # True negative for class Label
        TN = whole_sum - FN - FP - TP
        confusion_metrics[label, CM["TN"]] = TN

    print("\nTrue Positive / True Negative / False Positive / False Negative")
    print(confusion_metrics)
        
    """
    Accuracy = TP + TN / TP + TN + FP + FN
    Precision = TP / TP + FP
    Recall = TP / TP + FN
    F1 Score = 2*Precision*Recall / Precision + Recall
    """
    
    perf_metrics = np.zeros((10, 4)) #matrix containing performance matrix for each class
    PM = {
        "Accuracy": 0,
        "Precision": 1,
        "Recall": 2,
        "F1": 3
    }
    
    #to avoid division by zero
    epsilon = 1e-5
    confusion_metrics = np.clip(confusion_metrics, epsilon, 1e3)
    
    for index in range (0, 10):
        TP = confusion_metrics[index, CM["TP"]]
        TN = confusion_metrics[index, CM["TN"]]
        FP = confusion_metrics[index, CM["FP"]]
        FN = confusion_metrics[index, CM["FN"]]
        
        perf_metrics[index, PM["Accuracy"]] = round((TP + TN) / (TP + TN + FP + FN), 4)
        precision = perf_metrics[index, PM["Precision"]] = round(TP / (TP + FP), 4)
        recall = perf_metrics[index, PM["Recall"]] = round(TP / (TP + FN), 4)
        perf_metrics[index, PM["F1"]] = round(2 * precision * recall / (precision + recall + epsilon), 4)
        
    perf_metrics_average = np.mean(perf_metrics, axis=0)
    
    print("\nAccuracy / Precision / Recall / F1 score")
    print(perf_metrics)
    print("\nAverages")
    print(perf_metrics_average)