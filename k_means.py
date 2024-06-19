import numpy as np
import sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def main():
    data = load_iris()
    X, y = data['data'], data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    print(X_test.shape, X_train.shape)

    K = 3
    max_iter = 1000
    # Randomly select K data points from data and use them as initial centroids
    idx = np.random.choice(X_train.shape[0], size = K, replace = False)
    centroids = X_train[idx]
    
    for iter in range(max_iter):
        clusters = [[] for i in range(K)]
        gt_clusters = [[] for i in range(K)]
        for i in range(X_train.shape[0]):
            x, y = X_train[i], y_train[i]
            dists = np.mean((x - centroids)**2, axis = 1)
            idx = np.argmin(dists, axis = 0)
            clusters[idx].append(x)
            gt_clusters[idx].append(y)

        centroids = [np.mean(np.array(clusters[i]), axis = 0) for i in range(K)]

    print(len(clusters))
    # Manually check labels and clusters
    for idx in range(len(clusters)):
        y_true = gt_clusters[idx]
        print(y_true, idx)

if __name__ == '__main__':
    main()