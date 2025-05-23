from MrML.types import *
from MrML.model_info import ModelInfo
from MrML.encoder import Encoder
from MrML.decoder import Decoder
from MrML.linear import LinearLayer
from MrML.tools import softmax
from MrML.embedding import Embedder

DEFAULT_N_HEADS = 6
DEFAULT_N_LAYERS = 6

class Transformer(nn.Module):
    def __init__(self, info: ModelInfo, n_layers: int = DEFAULT_N_LAYERS, n_heads: int = DEFAULT_N_HEADS):   
        super().__init__()
        self.info = info     
        self.encoder = Encoder(info, n_layers, n_heads)
        self.decoder = Decoder(info, n_layers, n_heads)
        self.linear = LinearLayer(info, info.vocab_len)
        
        dtype = info.dtype
        shape = info.shape
        probs_shape = (info.seq_len, info.vocab_len)
        
        self.E: Tensor = torch.empty(size=shape, dtype=dtype)
        self.D: Tensor = torch.empty(size=shape, dtype=dtype)
        self.logits: Tensor = torch.empty(size=probs_shape, dtype=dtype)
        self.probs: Tensor = torch.empty(size=probs_shape, dtype=dtype)
        
    def forward(self, X: Tensor, input_masks: Tensor, D: Tensor, d_masks: Tensor):
        self.E = self.encoder(X, input_masks)
        self.D = self.decoder(D, self.E, d_masks, input_masks)
        self.logits = self.linear(self.D)
        self.probs = softmax(self.logits)

# class LanguageAcceptanceClassifier(nn.Module):
#     def __init__(self, info: ModelInfo, n_layers: int = DEFAULT_N_LAYERS, n_heads: int = DEFAULT_N_HEADS, dropout: float = 0.15):   
#         super().__init__()
#         self.info = info   
#         self.embedder = Embedder(info)
#         self.encoder = Encoder(info, n_layers, n_heads)
#         self.linear = LinearLayer(info, output_shape=(1, info.d_model))
#         self.dropout = nn.Dropout(p=dropout)
                
#     def forward(self, X: Tensor, input_masks: Tensor):
#         embeddings = self.embedder(X) 
               
#         X = self.encoder(embeddings, input_masks)
        
#         X = X.mean(dim=1)
#         X = self.linear(X)
#         X = X.squeeze(0).squeeze(-1)
                                
#         X = self.dropout(X)
#         return X

class LanguageAcceptanceClassifier(nn.Module):
    def __init__(self, info: ModelInfo, n_layers: int = DEFAULT_N_LAYERS, n_heads: int = DEFAULT_N_HEADS, dropout: float = 0.1):   
        super().__init__()
        self.info = info   
        self.embedder = Embedder(info)
        self.encoder = Encoder(info, n_layers, n_heads, dropout)
        self.classifier = nn.Linear(info.d_model, 1)
    
    def set_dropout_rate(self, p: float):
        self.encoder.set_dropout_rate(p)
                
    def forward(self, X: Tensor, input_masks: Tensor):
        embeddings = self.embedder(X) 
        X = self.encoder(embeddings, input_masks)
        X = self.classifier(X)
        X = torch.sum(X * input_masks.unsqueeze(-1), dim=1) / torch.sum(input_masks, dim=1, keepdim=True)
        X = X.squeeze(-1)
        return X
