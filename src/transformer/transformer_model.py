import torch
from torch import Tensor
import torch.nn as nn
from positional_encoding import PositionalEncoding
from encoder_layer import EncoderLayer
from decoder_layer import DecoderLayer

class Transformer(nn.Module):
    '''
    Full transformer model following standard architecture.
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

    def generate_mask(
        self, source: Tensor, target: Tensor) -> tuple[Tensor, Tensor]:
        '''
        Used to create masks for the source and target sequences, ensuring that
        padding tokens are ignored and that future tokens are not visible 
        during training for the target sequence.
        '''
        # Source and target masks:
        source_mask: Tensor = (
            (source != 0).
            unsqueeze(1).
            unsqueeze(2)
        )
        target_mask: Tensor = (
            (target != 0).
            unsqueeze(1).
            unsqueeze(3)
        )

        # Length of sequence:
        seq_length: int = target.size(1)

        # No-peek mask prevents from attending to subsequent positions in 
        # sequence:
        nopeek_mask: Tensor = (
            (1 - 
             torch.triu(torch.ones(1, seq_length, seq_length), diagonal = 1)
            ).
            bool()
        )

        target_mask: Tensor = target_mask & nopeek_mask

        return source_mask, target_mask

    def forward(self, source: Tensor, target: Tensor) -> Tensor:
        '''
        Defines the forward pass for the transformer, taking source and target
        sequences and producing output predictions.
        '''
        # Getting masks:
        source_mask: Tensor
        target_mask: Tensor
        source_mask, target_mask = self.generate_mask(source, target)

        # Getting embeddings for source and target sequences and adding to 
        # their respective positional encodings:
        source_embedding: Tensor = self.dropout(
            self.positional_encoding(self.encoder_embedding(source))
        )
        target_embedding: Tensor = self.dropout(
            self.positional_encoding(self.encoder_embedding(target))
        )

        # Pass source sequence through encoder layers, with final encoder
        # output representing the processed source sequence:
        encoder_output: Tensor = source_embedding
        for encoder_layer in self.encoder_layers:
            encoder_output = encoder_layer(encoder_output, source_mask)

        # Pass target sequence and encoder's output through decoder layers,
        # giving decoder's output:
        decoder_output: Tensor = target_embedding
        for decoder_layer in self.decoder_layers:
            decoder_output = decoder_layer(
                decoder_output, encoder_output, source_mask, target_mask
            )

        # Decoder's output is mapped to the target vocabulary size using a 
        # fully-connected layer:
        return self.fully_connected(decoder_output)
