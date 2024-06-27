import torch
from torch import Tensor
import torch.nn as nn
from transformer_model import Transformer

# Defining constants:
SOURCE_VOCAB_SIZE = 5000
TARGET_VOCAB_SIZE = 5000
MODEL_DIMENSION = 512
NUM_HEADS = 8
NUM_LAYERS = 6
FEED_FORWARD_DIMENSION = 2048
MAX_SEQ_LENGTH = 100
DROPOUT = 0.1

def main() -> None:
    # Initialise transformer:
    transformer = Transformer(
        SOURCE_VOCAB_SIZE,
        TARGET_VOCAB_SIZE,
        MODEL_DIMENSION,
        NUM_HEADS,
        NUM_LAYERS,
        FEED_FORWARD_DIMENSION,
        MAX_SEQ_LENGTH,
        DROPOUT
    )
    
    # Get random sample data:
    source_data: Tensor
    target_data: Tensor
    source_data, target_data = generate_random_data()

def generate_random_data(num_examples: int = 64) -> tuple[Tensor, Tensor]:
    '''
    Generates random sample data. Simulates a batch of data with num_examples
    examples (default 64) and sequences of length MAX_SEQ_LENGTH.
    '''
    # Generate random sample data:
    source_data: Tensor = torch.randint(
        1, SOURCE_VOCAB_SIZE, (num_examples, MAX_SEQ_LENGTH) 
    )
    target_data: Tensor = torch.randint(
        1, TARGET_VOCAB_SIZE, (num_examples, MAX_SEQ_LENGTH)
    )

    return source_data, target_data



if __name__ == "__main__":
    main()
