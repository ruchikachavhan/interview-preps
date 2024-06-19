import numpy as np
import sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import seaborn as sb
import matplotlib.pyplot as plt

def main():
    data = load_iris()
    X, y = data['data'], data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    print(X_test.shape, X_train.shape)

    n_components = 2
    # Standardise the matrix
    # X is of shape N x D
    mean = np.mean(X_train, axis = 0)
    X_norm = X_train - mean
    # Covariance matrix between features, D x D
    cov = np.cov(X_norm, rowvar=False)
    
    # Calculate eigenvectors and eigenvalues
    eigen_values, eigen_vectors = np.linalg.eigh(cov)

    # Sort the eigenvalues
    indxs = np.argsort(eigen_values)[::-1]
    sorted_eigen_vectors = eigen_vectors[indxs]
    sorted_eigen_vals = eigen_values[indxs]

    eigen_vectors_subset = eigen_vectors[:n_components]

    # Transform data 
    transformed_data = X_norm @ eigen_vectors_subset.T

    print(transformed_data.shape, y_train.shape)


    plt.figure(figsize = (6,6))
    labels = ['0', '1', '2']
    color = ['r', 'b', 'g']
    for i in range(transformed_data.shape[0]):
        plt.scatter(transformed_data[i][0], transformed_data[i][1], color = color[y_train[i]], label=labels[y_train[i]])
    plt.show()

if __name__ == '__main__':
    main()