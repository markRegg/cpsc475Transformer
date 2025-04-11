import torch.nn.functional as F
from MrML.types import *
from MrML.model_info import ModelInfo

NEG_INF = float('-inf')

def softmax(logits: Tensor, dim: int = -1, epsilon: float = 1e-8) -> Tensor:
    # Exponentiate
    exp_logits = torch.exp(logits)
    
    # Compute the sum of exponentiated logits along the last axis (seq_len)
    exp_sum = exp_logits.sum(dim=dim, keepdim=True)  # Keep dim for broadcasting
    
    # Avoid division by zero by adding epsilon
    exp_sum = exp_sum + epsilon  # Adding epsilon to prevent division by zero
        
    # Apply softmax
    return exp_logits / exp_sum

def validate_stride(info: ModelInfo):
    if info.stride <= 0:
        raise ValueError(f"Invalid stride for chunking. Stride {info.stride} is not > 0.")
    
    if info.stride >= info.seq_len:
        raise ValueError(f"Invalid stride for chunking. Stride {info.stride} is not < seq_len for seq_len={info.seq_len}.")
    
def num_chunks_needed(elements: Tensor, info: ModelInfo) -> int:
    if elements.shape[0] <= info.seq_len:
        return 1
         
    stride = info.stride
    num_tokens = elements.shape[0] # Number of elements
    
    # Calculate number of chunks needed
    num_chunks = (num_tokens + stride - 1) // stride
    num_chunks = max(1, num_chunks) # Make sure num_chunks >= 1
    return int(num_chunks)
    
def chunk(elements: Tensor, info: ModelInfo, dim: int = 0) -> Tensor:
    """Chunks a padded sequence seq_len chunks with a stride

    Args:
        elements (Tensor): The sequence to chunk
        seq_len (int): sequence length of the model
        stride (int): The number of tokens to offset each chunk. 0 < stride < seq_len.
    Returns:
        Tensor: the elements chunked
    """
    num_chunks = num_chunks_needed(elements, info)
    stride = int(info.stride)
    seq_len = info.seq_len
        
    return torch.stack([
        elements[i * stride : i * stride + seq_len]
        for i in range(num_chunks)
        if (i * stride) + seq_len <= elements.shape[0]
    ], dim=dim)

def pad(tokens: Tensor, info: ModelInfo) -> Tuple[Tensor, Tensor]:
    """Pads a token sequence to allow it to have seq_len tokens in each window

    Args:
        tokens (Tensor): The token sequence, a tensor of torch.int with size (num_tokens,)
        seq_len (int): Model's seq_len
        stride (int): The number of tokens to offset each chunk. 0 < stride < seq_len.
        PAD (int): The token to use for padding
    Returns:
        Tuple[Tensor, Tensor]: tokens, mask
            tokens: Padded token sequence with shape (num_chunks, seq_len)
            mask: Mask parallel to tokens where PAD tokens are 1s and input tokens are 0
    """
    validate_stride(info)
    num_chunks = num_chunks_needed(tokens, info)
    stride = info.stride
    
    # Calculate numper of PAD tokens needed for last chunk
    num_pad = int(((num_chunks * stride) + info.seq_len - stride) - tokens.shape[0])
    num_pad = max(0, num_pad) # Make sure num_pad >= 0
    
    mask_token_part = torch.ones_like(tokens)
    mask_pad_part = torch.zeros(size=(num_pad,), dtype=torch.int)
    mask = torch.cat((mask_token_part, mask_pad_part), dim=0)
    
    # Pad the end of the token sequence so last chunk will have size (seq_len,)
    tokens = F.pad(tokens, (0, num_pad), value=info.vocab.pad)
    
    return tokens, mask

def make_mask(ones: torch.Size, shape: torch.Size, dtype: DType = torch.float32) -> Tensor:
    mask = torch.zeros(size=shape, dtype=dtype)
    one_part = torch.ones(size=ones, dtype=dtype)
    
    slices = tuple(slice(0, ones[i]) for i in range(len(ones)))
    mask[slices] = one_part
    
    return mask
