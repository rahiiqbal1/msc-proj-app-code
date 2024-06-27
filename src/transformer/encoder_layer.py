from torch import Tensor
import torch.nn as nn
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
        # Dropout layer, randomly sets some activations to 0:
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_tensor: Tensor, mask: Tensor) -> Tensor:
        '''
        Takes input_tensor as input to encoder layer and performs a forward
        pass through.
        '''
        # Pass through multi-head self-attention mechanism:
        attention_output: Tensor = self.self_attention(
            input_tensor,
            input_tensor,
            input_tensor,
            mask
        )

        # Attention output is added to the input of this stage (residual),
        # followed by dropout and normalisation:
        input_tensor = self.normalise1(
                input_tensor + self.dropout(attention_output)
        )

        feed_forward_output: Tensor = self.feed_forward(input_tensor)

        # Feed-forward output is added to the input of this stage (residual),
        # followed by dropout and normalisation:
        input_tensor = self.normalise2(
                input_tensor + self.dropout(feed_forward_output)
        )

        # Return output of encoder layer:
        return input_tensor
