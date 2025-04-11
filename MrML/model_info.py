from MrML.types import *
from MrML.vocab import Vocab

class ModelInfo:
    def __init__(self, batch_size: int, seq_len: int, d_model: int, stride: int, vocab: Vocab, device, dtype: DType = torch.float32):
        self.shape = (batch_size, seq_len, d_model)
        
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.d_model = d_model
        
        self.stride = stride
        
        self.vocab = vocab
        self.vocab_len = len(vocab)
        
        self.device = device
        self.dtype = dtype