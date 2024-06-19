import numpy as np
import sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


'''KNN is a supervised learning method, which classifies testing samples using its distance to training samples
Step 1: Get distances of test samples from training samples
Step 2: Get closest neighbour
Step 3: Label of closest neighbour is the label of test sample'''

def main():
    data = load_iris()
    X, y = data['data'], data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    print(X_test.shape, X_train.shape)
    preds, gt = [], []
    num_neighbours = 4
    for i in range(X_test.shape[0]):
        dists = np.mean((X_train - X_test[i])**2, axis = 1)
        indx = np.argsort(dists, axis = 0)
        pred = y_train[indx][:num_neighbours]
        preds.append(pred)
        y_true = y_test[i]
        gt.append(y_true)

    # Calculate accuracy
    accuracy = 0.0
    preds = np.array(preds)
    for i in range(num_neighbours):
        accuracy += np.mean((preds[:, i] == gt))
    accuracy /= num_neighbours
    print("ACCURACY", accuracy)


if __name__ == '__main__':
    main()