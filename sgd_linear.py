import numpy as np

n = 20

# Scalar 
x = np.random.rand(n)
w_true = 2.4
b_true = 0.5
y_true = w_true * x + b_true

w = 1.0
b = 0.0
lr = 1.0
# Batch gradient descent
for epoch in range(100):
    # Loss = 1/n * 0.5 * sum[(wx + b - y)^2]
    loss = np.mean(0.5 * (w * x + b - y_true)**2)
    # gradients
    # grad(L)_w = 1/n * sum[(wx + b - y) * x]
    w = w - lr * np.mean((w * x + b  - y_true) * x)
    b = b - lr * np.mean((w * x + b  - y_true))


# multi-dimensinal input, scalar output
n = 20
d = 10
x = np.random.rand(n, d)
w_true = np.array([2.4, 1.5, 3.5, 4.5, 2.3, 1.2, 3.4, 2.3, 1.2, 3.4])
b_true = 0.5
y_true = np.dot(x, w_true) + b_true

w = np.random.rand(d)
b = 0.0
lr = 0.1
# Batch gradient descent
for epoch in range(100):
    y = np.dot(x, w) + b
    loss = np.mean(0.5 * (y - y_true)**2)
    # gradients
    # grad(L)_w = 1/n * sum[(wx + b - y) * x]
    w = w - lr * np.mean((y - y_true)[:, None] * x, axis = 0)
    b = b - lr * np.mean((y - y_true))

    # print("LOSS", loss, w, b)


# multi dimensional input and output
n = 20
d = 10
k = 5
x = np.random.rand(n, d)
w_true = np.random.rand(d, k)
b_true = np.random.rand(k)
y_true = x @ w_true + b_true

w = np.random.rand(d, k)
b = np.random.rand(k)
lr = 0.1
# Batch gradient descent
for epoch in range(0, 100):
    for i in range(0, n):
        y = x[i] @ w + b
        loss = 0.5 * np.mean((y - y_true[i])**2)
        # gradients
        w = w - lr * ((y - y_true[i])[:, None] * x[i]).T
        b = b - lr * (y - y_true[i])
        # print("LOSS", loss, w, b)


# vectorized
n = 20
d = 10
k = 5
x = np.random.rand(n, d)
w_true = np.random.rand(d, k)
b_true = np.random.rand(k)
y_true = x @ w_true + b_true

w = np.random.rand(d, k)
b = np.random.rand(k)
lr = 0.01
# Batch gradient descent
for epoch in range(0, 100):
    y = x @ w + b
    loss = 0.5 * np.mean((y - y_true)**2)
    # gradients
    w = w - lr * (x.T @ (y - y_true))
    b = b - lr * np.mean(y - y_true, axis = 0)
    print("LOSS", loss, w, b)