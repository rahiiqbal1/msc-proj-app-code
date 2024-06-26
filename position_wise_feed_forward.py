import torch
from torch import Tensor
import torch.nn as nn
import math

class PositionWiseFeedForward(nn.Module):
    '''
    Defines a position-wise feed forward network. For a transformer, this is
    applied to each position separately and identically. Helps in transforming
    the features learned by the attention mechanisms within the transformer,
    acting as an addition processing step for the attention outputs.
    '''
    def __init__(self, model_dimension: int, feed_forward_dimension: int):
        super(PositionWiseFeedForward, self).__init__()
        # Fully-connected layers:
        self.fc1 = nn.Linear(model_dimension, feed_forward_dimension)
        self.fc2 = nn.Linear(feed_forward_dimension, model_dimension)
        # ReLU:
        self.relu = nn.ReLU()

    def forward(self, input_tensor: Tensor) -> Tensor:
        '''
        Performs forward pass calculation on given tensor.
        '''
        return self.fc2(
                   self.relu(
                       self.fc1(
                           input_tensor
                           )
                       )
                   )
