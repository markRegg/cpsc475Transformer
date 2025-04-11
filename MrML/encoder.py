from MrML.types import *
from MrML.model_info import ModelInfo
from MrML.attention import MultiHeadAttenionLayer
from MrML.feed_forward import FullyConnectedFeedForwardLayer
from torch.nn import functional as F

DEFAULT_N_HEADS = 6
DEFAULT_N_LAYERS = 6

class EncoderLayer(nn.Module):
    def __init__(self, info: ModelInfo, n_heads: int = DEFAULT_N_HEADS):
        super().__init__()
        self.multi_head_attention = MultiHeadAttenionLayer(info, n_heads)
        self.feef_forward = FullyConnectedFeedForwardLayer(info)
    
    def forward(self, X: Tensor, mask: Tensor) -> Tensor:
        X = X + self.multi_head_attention(V=X, K=X, Q=X, mask=mask)
        X = F.normalize(X)
        
        X = X + self.feef_forward(X)
        X = F.normalize(X)
        
        return X

class Encoder(nn.Module):
    def __init__(self, info: ModelInfo, n_layers: int = DEFAULT_N_LAYERS, n_heads: int = DEFAULT_N_HEADS):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(info, n_heads) for _ in range(n_layers)])
    
    def forward(self, X: Tensor, mask: Tensor) -> Tensor:
        for layer in self.layers:
            X = layer(X, mask)
        
        return X