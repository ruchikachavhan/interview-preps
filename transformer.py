import numpy as np
import torch

class Attention(torch.nn.Module):
    def __init__(self, dim):
        super(Attention, self).__init__()
        # dim is the dimension of the input
        # Initialize the weights
        self.W_q = torch.nn.Parameter(torch.rand(dim, dim))
        self.W_k = torch.nn.Parameter(torch.rand(dim, dim))
        self.W_v = torch.nn.Parameter(torch.rand(dim, dim))

    def forward(self, x):
        # x is the input, n x d
        Q = x @ self.W_q
        K = x @ self.W_k
        V = x @ self.W_v

        # Calculate the attention weights
        attention_weights = torch.nn.functional.softmax(Q @ K.T, dim = 1)
        y = attention_weights @ V
        return y
    
# Multi head attention
    
class MultiHeadAttention(torch.nn.Module):
    def __init__(self, dim, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.n_heads = n_heads
        # Dimensions of the heads are smaller than dim
        self.head_dim = dim // n_heads
        self.W_q = torch.nn.Parameter(torch.rand(dim, dim))
        self.W_k = torch.nn.Parameter(torch.rand(dim, dim))
        self.W_v = torch.nn.Parameter(torch.rand(dim, dim))
        self.W_o = torch.nn.Parameter(torch.rand(dim, dim))
    
    def forward(self, x):
        Q = x @ self.W_q
        K = x @ self.W_k
        V = x @ self.W_v
        
        # Split the heads
        Q = Q.view(-1, self.n_heads, self.head_dim)
        K = K.view(-1, self.n_heads, self.head_dim)
        V = V.view(-1, self.n_heads, self.head_dim)
        
        # Calculate the attention weights
        attention_weights = torch.nn.functional.softmax(Q @ K.permute(0, 2, 1), dim = 2)
        y = attention_weights @ V
        y = y.view(-1, self.n_heads * self.head_dim) @ self.W_o
        return y

class TransformerBlock(torch.nn.Module):
    def __init__(self, dim, n_heads):
        super(TransformerBlock, self).__init__()
        self.attention = MultiHeadAttention(dim, n_heads)
        self.layer_norm1 = torch.nn.LayerNorm(dim)
        self.layer_norm2 = torch.nn.LayerNorm(dim)
        self.linear = torch.nn.Linear(dim, dim)
    
    def forward(self, x):
        y = self.attention(x)
        x = self.layer_norm1(x + y)
        y = self.linear(x)
        return self.layer_norm2(x + y)

def main():
    x = torch.rand(10, 5)
    attention = Attention(5)
    y = attention(x)
    

if __name__ == '__main__':
    main()