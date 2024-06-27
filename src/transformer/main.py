import torch
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
    
    # Generate random sample data:
    


if __name__ == "__main__":
    main()
