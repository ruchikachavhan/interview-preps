import numpy as np
import scipy

kernel_size = 2
input_size = 4
stride = 1
padding = 0

kernel = np.random.randn(kernel_size, kernel_size)
input = np.random.randn(input_size, input_size)

output_size = ((input_size - kernel_size + 2 * padding)//stride) + 1

# kernel as a weight matrix 
weights = np.zeros((output_size**2, input_size**2))
shift = 0
for i in range(output_size**2):  
    # Each loop fills one row -> one convolution operation
    for j in range(kernel_size):
        i1 = j * input_size + shift
        if weights[i, i1 : i1 + kernel_size].shape[0] == kernel_size:
            weights[i,  i1 : i1 + kernel_size] = kernel[j]
    if (i + 1) % output_size == 0:
        shift = 0
        shift += input_size * ((i + 1) // output_size)
    else:
        shift += stride

image = input.reshape(-1)
conv_out = weights @ image
conv_out = conv_out.reshape(output_size, output_size)

# # Check with scipy, this function does padding on the inpu, therefore lets only check if the unpadded output is same as our output
np_out = scipy.signal.correlate2d(input, kernel)

print("My convolution", conv_out)
print("Scipy convolution", np_out)