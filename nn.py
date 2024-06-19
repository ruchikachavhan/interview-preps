import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split



def sigmoid(x):
    return 1/(1 + np.exp(-x))

def sigmoid_backward(x):
    out = np.multiply(sigmoid(x), (1 - sigmoid(x)))
    return out

'''
dL_dw[l] = dL_dz[l] dz[l]_dw[l] = dL_dz[l] a_[l-1]
dL_dz[l] = dL_da[l] da[l]_dz[l] = dL_da[l] * g'(z[l])
dL_da[l] = dL_dz[l+1]  dz[l+1]_da[l] =  w[l+1]^T  dL_dz[l+1] 
'''

'''
generally we will start from the last layer, things to store
activations, weights, dL_dz[l], dL_da[l]
'''
def forward_pass(x, w1, b1, w2, b2):
    z_1 = x @ w1 + b1
    a_1 = sigmoid(z_1)
    z_2 = a_1 @ w2 + b2
    a_2 = sigmoid(z_2)
    return [z_1, a_1, z_2, a_2]

# def backward_pass(activations, weights, dz, da, num_layers = 2):
#     ''' weights are of the form [w1, b1, w2, b2]'''
#     for l in range(num_layers):
#         if 

def backward_pass(activations, x, y, w1, b1, w2, b2):
    z_1, a_1, z_2, a_2 = activations
    dl_dz2 = np.multiply((a_2 - y), sigmoid_backward(z_2)) # (900, 1)
    dl_dw2 = a_1.T @ dl_dz2
    dl_db2 = np.expand_dims(np.mean(dl_dz2, axis = 0), axis = 0)

    # dl_dw1 = (dl_da2 * da2_dz2) dz2_da1 (da1_dz1 * dz1_dw1) = ((a_2 - y) *  sigmoid_backward(z_2)) w2^T * (sigmoid_backward(z_1) x -> (900, 1), (4, 1), (900, 4)
    dl_da1 = dl_dz2 @ w2.T # (900, 4) 
    dl_dz1 = dl_da1 * sigmoid_backward(z_1) # (900, 4) 
    dl_dw1 = x.T @ dl_dz1
    dl_db1 = np.expand_dims(np.mean(dl_dz1, axis = 0), axis = 0)
    
    grads = [dl_dw1, dl_db1, dl_dw2, dl_db2]
    return grads

def main():
    # number of samples in the data set
    N_SAMPLES = 1000
    # ratio between training and test sets
    TEST_SIZE = 0.1

    X, y = make_moons(n_samples = N_SAMPLES, noise=0.2, random_state=100)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=42)

    y_train, y_test = np.expand_dims(y_train, axis = 1), np.expand_dims(y_test, axis = 1)

    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


    # SMALL 2 layer Neural network
    input_dim = 2
    hidden_dim = 4
    output_dim = 1

    lr = 0.01

    w1 = np.random.randn(input_dim, hidden_dim)
    b1 = np.zeros((1, hidden_dim))
    w2 = np.random.randn(hidden_dim, output_dim)
    b2 = np.zeros((1, output_dim))

    for epoch in range(0, 100):
        activations = forward_pass(X_train, w1, b1, w2, b2)
        loss = np.mean((activations[-1] - y_train)**2) * 0.5
        grads = backward_pass(activations, X_train, y_train, w1, b1, w2, b2)
        w1 = w1 - lr * grads[0]
        b1 = b1 - lr * grads[1]
        w2 = w2 - lr * grads[2]
        b2 = b2 - lr * grads[3]

        print("Loss", loss, w1, b1, w2, b2)

if __name__ == '__main__':
    main()