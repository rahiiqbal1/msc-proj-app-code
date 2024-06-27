import torch
from torch import Tensor
import torch.nn as nn
import math
from multi_head_attention import MultiHeadAttention
from position_wise_feed_forward import PositionWiseFeedForward

class EncoderLayer(nn.Module):
    '''
    Defines a single layer of the transformer's encoder. Encapsulates a multi-
    head attention mechanism followed by a position-wise feed-forward block, 
    with residual connections, layer normalisation, and dropout used.
    '''
    def __init__(
        self,
        model_dimension: int,
        num_heads: int,
        feed_forward_dimension: int, 
        dropout: float
        ):
        # Inherit attributes from parent class:
        super(EncoderLayer, self).__init__()
        # Attention block:
        self.self_attention = MultiHeadAttention(model_dimension, num_heads)
        # Forward layer:
        self.feed_forward = PositionWiseFeedForward(
                model_dimension, feed_forward_dimension
        )
        # Normalisation layers:
        self.normalise1 = nn.LayerNorm(model_dimension)
        self.normalise2 = nn.LayerNorm(model_dimension)
        # Dropout:
        self.dropout = nn.Dropout(dropout)
