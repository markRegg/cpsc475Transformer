from MrML.types import *
from MrML.model_info import ModelInfo
from MrML.attention import MultiHeadAttenionLayer
from MrML.feed_forward import FullyConnectedFeedForwardLayer
from torch.nn import functional as F

DEFAULT_N_HEADS = 6
DEFAULT_N_LAYERS = 6

class EncoderLayer(nn.Module):
    def __init__(self, info: ModelInfo, n_heads: int = DEFAULT_N_HEADS, dropout: float = 0.1):
        super().__init__()
        self.multi_head_attention = MultiHeadAttenionLayer(info, n_heads)
        self.attn_dropout = nn.Dropout(dropout)
        self.attn_norm = nn.LayerNorm(info.d_model)
        
        self.feed_forward = FullyConnectedFeedForwardLayer(info)
        self.ff_dropout = nn.Dropout(dropout)
        self.ff_norm = nn.LayerNorm(info.d_model)
    
    def set_dropout_rate(self, p: int):
        self.attn_dropout.p = p
        self.ff_dropout.p = p
    
    def forward(self, X: Tensor, mask: Tensor) -> Tensor:
        attn_result = self.multi_head_attention(V=X, K=X, Q=X, mask=mask)
        droupout_result = self.attn_dropout(attn_result)
        X = self.attn_norm(X + droupout_result)
        
        ff_result = self.feed_forward(X)
        droupout_result = self.ff_dropout(ff_result)
        X = self.ff_norm(X + droupout_result)
        return X

class Encoder(nn.Module):
    def __init__(self, info: ModelInfo, n_layers: int = DEFAULT_N_LAYERS, n_heads: int = DEFAULT_N_HEADS, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(info, n_heads, dropout) for _ in range(n_layers)])
    
    def set_dropout_rate(self, p: int):
        for layer in self.layers:
            layer.set_dropout_rate(p)
    
    def forward(self, X: Tensor, mask: Tensor) -> Tensor:
        for layer in self.layers:
            X = layer(X, mask)
        
        return X