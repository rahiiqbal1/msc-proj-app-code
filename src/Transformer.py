import torch
from torch import Tensor
import torch.nn as nn
import math
from positional_encoding import PositionalEncoding
from encoder_layer import EncoderLayer
from decoder_layer import DecoderLayer

class Transformer(nn.Module):
    '''
    '''
    def __init__(
        self,
        source_vocab_size: int,
        target_vocab_size: int,
        model_dimension: int, # The dimensionality of the model's embeddings.
        num_heads: int,
        num_layers: int, # Number of layers for both encoder and decoder.
        feed_forward_dimension: int, # Dimensionality of inner layer in FF.
        max_seq_length: int, # Maximum sequence length for positional encoding.
        dropout: float # Dropout rate for regularisation.
        ):
        super(Transformer, self).__init__()

        # Embedding layer for the source sequence:
        self.encoder_embedding = nn.Embedding(
            source_vocab_size, model_dimension
        )

        # Embedding layer for the target sequence:
        self.decoder_embedding = nn.Embedding(
            target_vocab_size, model_dimension
        )

        # Positional encoding component:
        self.positional_encoding = PositionalEncoding(
            model_dimension, max_seq_length
        )

        # List of encoder layers:
        self.encoder_layers = nn.ModuleList(
            [
                EncoderLayer(
                    model_dimension,
                    num_heads,
                    feed_forward_dimension,
                    dropout
                    ) for _ in range(num_layers)
            ]
        )

        # List of decoder layers:
        self.decoder_layers = nn.ModuleList(
            [
                DecoderLayer(
                    model_dimension,
                    num_heads,
                    feed_forward_dimension,
                    dropout
                    ) for _ in range(num_layers)
            ]
        )

        # Final fully-connected layer mapping to target vocabulary size:
        self.fully_connected = nn.Linear(model_dimension, target_vocab_size)

        # Dropout layer:
        self.dropout = nn.Dropout(dropout)
