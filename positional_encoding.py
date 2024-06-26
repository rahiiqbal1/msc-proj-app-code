import torch
from torch import Tensor
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    '''
    '''
    def __init__(self, model_dimension: int, max_seq_length: int):
        super(PositionalEncoding, self).__init__()

        # Initialise positional encoding tensor:
        positional_encodings: Tensor = torch.zeros(
                max_seq_length, model_dimension
        )

        # Contains the position indices for each position in the sequence:
        position_indices: Tensor = (
            torch.arange(0, max_seq_length, dtype = torch.float).
            unsqueeze(1)
        )

        # used to scale the position indices:
        scale_terms: Tensor = torch.exp(
            torch.arange(0, model_dimension, 2).float() *
            -(math.log(10000.0) / model_dimension)
        )

        # Setting positional encodings:
        positional_encodings[:, 0::2] = torch.sin(
                position_indices * scale_terms
        )
        positional_encodings[:, 1::2] = torch.cos(
                position_indices * scale_terms
        )

        # Register positional_encodings as a buffer, so that is is part of the
        # module's state but not considered a trainable parameter:
        self.register_buffer(
                "positional_encodings",
                positional_encodings.unsqueeze(0)
        )

    def forward(self, input_tensor: Tensor) -> Tensor:
        return (
            input_tensor + self.positional_encodings[:, :input_tensor.size(1)]
        )
