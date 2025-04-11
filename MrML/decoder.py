from MrML.types import *
from MrML.model_info import *
from MrML.attention import MultiHeadAttenionLayer
from MrML.feed_forward import FullyConnectedFeedForwardLayer
from MrML.linear import LinearLayer
from torch.nn import functional as F

DEFAULT_N_HEADS = 6
DEFAULT_N_LAYERS = 6

class DecoderLayer(nn.Module):
    def __init__(self, info: ModelInfo, n_heads: int = DEFAULT_N_HEADS):
        super().__init__()
        self.self_attention = MultiHeadAttenionLayer(info, n_heads)
        self.cross_attention = MultiHeadAttenionLayer(info, n_heads)
        self.feef_forward = FullyConnectedFeedForwardLayer(info)

    def forward(self, D: Tensor, E: Tensor, d_mask: Tensor, e_mask: Tensor) -> Tensor:
        result = self.self_attention(V=D, K=D, Q=D, mask=d_mask)
        D = D + result
        D = F.normalize(D)
        
        D = D + self.cross_attention(V=E, K=E, Q=D, mask=e_mask)
        D = F.normalize(D)
        
        D = D + self.feef_forward(D)
        D = F.normalize(D)
        
        return D

class Decoder(nn.Module):
    def __init__(self, info: ModelInfo, n_layers: int = DEFAULT_N_LAYERS, n_heads: int = DEFAULT_N_HEADS):
        super().__init__()
        self.layers = [DecoderLayer(info, n_heads) for _ in range(n_layers)]
        self.linear = LinearLayer(info, info.vocab_len)
    
    def forward(self, D: Tensor, E: Tensor, d_mask: Tensor, e_mask: Tensor) -> Tensor:
        for layer in self.layers:
            D = layer(D, E, d_mask, e_mask)
        
        return D
