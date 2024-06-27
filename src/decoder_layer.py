import torch
from torch import Tensor
import torch.nn as nn
import math
from multi_head_attention import MultiHeadAttention
from position_wise_feed_forward import PositionWiseFeedForward

class DecoderLayer(nn.Module):
    '''
    Defines a single layer of the transformer's decoder. Consists of a multi-
    head self-attention block, a multi-head cross-attention block (that attends
    to the encoder's output), a position-wise feed-forward layer, and the 
    corresponding residual connections, layer normalisation, and dropout 
    layers.
    '''
    def __init__(
        self,
        model_dimension: int,
        num_heads: int,
        feed_forward_dimension: int,
        dropout: float
        ):
        # Inherit attributes from parent class:
        super(DecoderLayer, self).__init__()
        # Self-attention block:
        self.self_attention = MultiHeadAttention(model_dimension, num_heads)
        # Cross-attention block:
        self.cross_attention = MultiHeadAttention(model_dimension, num_heads)
        # Feed-forward layer:
        self.feed_forward = PositionWiseFeedForward(
                model_dimension, feed_forward_dimension
        )
        # Normalisation layers:
        self.normalise1 = nn.LayerNorm(model_dimension)
        self.normalise2 = nn.LayerNorm(model_dimension)
        self.normalise3 = nn.LayerNorm(model_dimension)
        # Dropout layer for regularisation:
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        input_to_decoder: Tensor,
        encoder_output: Tensor, # Used for cross-attention.
        source_mask: Tensor, # Used to ignore certain parts of encoder's input.
        target_mask: Tensor # To ignore certain parts of decoder's input.
        ) -> Tensor:
        '''
        '''
        # Process input tensor through self-attention mechanism:
        attention_output: Tensor = self.self_attention(
            input_to_decoder,
            input_to_decoder,
            input_to_decoder,
            target_mask
        )
        
        # Output from self-attention is added to original input tensor,
        # followed by dropout and normalisation using normalise1:
        input_to_decoder = self.normalise1(
                input_to_decoder + self.dropout(attention_output)
        )

        # Process normalised input tensor through cross-attention layer that
        # attends to the encoder's output:
        attention_output = self.cross_attention(
            input_to_decoder,
            encoder_output,
            encoder_output,
            source_mask
        )

        # Output from cross-attention is added to the input as it is up to this
        # stage, followed by dropout and normalisation using normalise2:
        input_to_decoder = self.normalise2(
            input_to_decoder + self.dropout(attention_output)
        )

        # Output from 2nd normalisation etc. step is passed through feed-
        # forward layer:
        feed_forward_output: Tensor = self.feed_forward(input_to_decoder)

        # Feed-forward output is added to the input up to this stage, followed
        # by dropout and normalisation using normalise3:
        input_to_decoder = self.normalise3(
            input_to_decoder + self.dropout(feed_forward_output)
        )

        return input_to_decoder
