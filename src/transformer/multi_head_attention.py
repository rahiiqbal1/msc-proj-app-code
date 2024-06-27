import torch
from torch import Tensor
import torch.nn as nn
import math

class MultiHeadAttention(nn.Module):
    '''
    Multi-head attention component of transformer based NN.
    '''
    def __init__(self, model_dimension: int, num_heads: int):
        super(MultiHeadAttention, self).__init__()
        # Ensure that the model dimension is divisible by the number of heads:
        if model_dimension % num_heads != 0:
            raise ValueError("model_dimension must be divisible by num_heads")

        # Initialise dimensions:
        self.model_dimension: int = model_dimension
        # Number of attention heads:
        self.num_heads: int = num_heads
        # Dimension of each head's key, query, and value:
        self.key_query_val_dimension: int = model_dimension // num_heads

        # Linear Layers for transforming inputs:
        self.query_weights = nn.Linear(model_dimension, model_dimension)
        self.key_weights = nn.Linear(model_dimension, model_dimension)
        self.value_weights = nn.Linear(model_dimension, model_dimension)
        self.output_weights = nn.Linear(model_dimension, model_dimension)

    def scaled_dot_product_attention(
        self, 
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Tensor | None = None
        ) -> Tensor:
        '''
        Performs scaled dot product attention.
        '''
        # Calculate attention scores:
        attention_scores: Tensor = (
                torch.matmul(query, key.transpose(-2, -1)) /
                math.sqrt(self.key_query_val_dimension)
        )

        # Apply mask if provided (useful for preventing attention to certain
        # parts like padding):
        if mask is not None:
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

        # Apply softmax to obtain attention probabilities:
        attention_probs: Tensor = torch.softmax(
                attention_scores, dim = -1
        )

        # Multiply by values to obtain final output:
        output_tensor: Tensor = torch.matmul(attention_probs, value)
        return output_tensor

    def split_heads(self, input_to_reshape: Tensor) -> Tensor:
        '''
        Reshape the input to have num_heads for multi-head attention.
        Enables the model to process multiple attention heads concurrently.
        '''
        batch_size: int
        seq_length: int
        batch_size, seq_length, _ = input_to_reshape.size()

        return (
                input_to_reshape.view(
                batch_size,
                seq_length,
                self.num_heads,
                self.key_query_val_dimension
                ).
                transpose(1, 2)
               )
    
    def combine_heads(self, input_to_combine: Tensor) -> Tensor:
        '''
        To use after applying attention to each head separately. Combines
        results back into a single tensor of shape
        (batch_size, seq_length, model_dimension).
        '''
        batch_size: int
        seq_length: int
        batch_size, _, seq_length, _ = input_to_combine.size()

        return (
                input_to_combine.
                transpose(1, 2).
                contiguous().
                view(batch_size, seq_length, self.model_dimension)
               )
    
    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Tensor | None = None
        ) -> Tensor:
        '''
        Perform forward propagation.
        '''
        # Apply linear transformations and split heads:
        query = self.split_heads(self.query_weights(query))
        key = self.split_heads(self.key_weights(key))
        value = self.split_heads(self.value_weights(value))

        # Perform scaled dot-product attention:
        attention_output: Tensor = self.scaled_dot_product_attention(
            query,
            key,
            value,
            mask
        )

        # Combine heads and apply output transformation:
        return self.output_weights(self.combine_heads(attention_output))
