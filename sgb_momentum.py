import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoid_backward(x):
    out = np.multiply(sigmoid(x), (1 - sigmoid(x)))
    return out

def relu(x):
    return np.maximum(0, x)

def relu_backward(x):
    out = np.zeros(x.shape)
    out[x > 0] = 1.0
    return out

def foward_pass(input, weights, biases, activation_func):
    activations = []
    layer_out = []
    a = input
    activations.append(a)
    layer_out.append(a)
    for l in range(len(weights)):
        z = a @ weights[l] + biases[l]
        if l != len(weights) - 1:
            a = activation_func(z)
        else:
            a = z
        layer_out.append(z)
        activations.append(a)

    activations = activations[::-1]
    layer_out = layer_out[::-1]
    
    return activations, layer_out

def backward_pass(activations, layer_out, weights, y, activation_backward):
    dl_dz, dl_da = [], []
    grads_w, grads_b = [], []
    num_layers = len(weights) + 1
    weights = weights[::-1]

    for l in range(len(weights)):
        if l == 0:
            dlda = (activations[l] - y)
            dldz = dlda 
        else:
            dlda = dl_dz[l-1] @ weights[l-1].T
            dldz = dlda * activation_backward(activations[l])

        dl_dw = activations[l+1].T @ dldz
        dl_db = np.mean(dldz, axis = 0)
        
        dl_da.append(dlda)
        dl_dz.append(dldz)
        grads_w.append(dl_dw)  
        grads_b.append(dl_db)
    return grads_w[::-1], grads_b[::-1]

def main():
    # number of samples in the data set
    N_SAMPLES = 1000
    # ratio between training and test sets
    TEST_SIZE = 0.1

    X, y = make_moons(n_samples = N_SAMPLES, noise=0.2, random_state=100)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=42)

    y_train, y_test = np.expand_dims(y_train, axis = 1), np.expand_dims(y_test, axis = 1)

    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    activation = 'sigmoid'
    if activation == 'sigmoid':
        activation_func = sigmoid
        activation_backward = sigmoid_backward
    elif activation == 'relu':
        activation_func  = relu
        activation_backward = relu_backward

    # Number of layers and hidden_units, first dimension is input dimension
    num_layers = 4
    dims = [2, 4, 3, 1]
    lr = 0.0001
    momentum = 0.999
    
    # Initialise weights
    weights, biases = [], []

    '''Recap of SGD with momentum 
    veclocity - '''
    velocity_weights, velocity_biases = [], []
    for l in range(num_layers - 1):
        weights.append(np.random.randn(dims[l], dims[l+1]))
        biases.append(np.random.randn(dims[l+1]))
        velocity_weights.append(np.zeros((dims[l], dims[l+1])))
        velocity_biases.append(np.zeros((dims[l+1])))

    for epoch in range(100):
        activations, layer_out = foward_pass(X_train, weights, biases, activation_func=activation_func)
        loss = np.mean((activations[0] - y_train)**2) * 0.5
        grads_w, grads_b = backward_pass(activations, layer_out, weights, y_train, activation_backward=activation_backward)
        for i in range(len(weights)):
            velocity_weights[i] = momentum * velocity_weights[i] - lr * grads_w[i]
            velocity_biases[i] = momentum * velocity_biases[i] - lr * grads_b[i]
            weights[i] = weights[i] + velocity_weights[i]
            biases[i] = biases[i] + velocity_biases[i]
            
        print("Train loss for epoch ", epoch, loss)
        
    # Evaluate on test set
    activations, _ = foward_pass(X_test, weights, biases, activation_func=activation_func)
    loss = np.mean((activations[0] - y_test)**2) * 0.5
    print("Test loss", loss)


if __name__ == '__main__':
    main()