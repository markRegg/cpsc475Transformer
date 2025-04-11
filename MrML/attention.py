import math
from MrML.types import *
from MrML.tools import softmax, NEG_INF
from MrML.linear import LinearLayer
from MrML.model_info import ModelInfo

def scaled_dot_product_attention(V: Tensor, K: Tensor, Q: Tensor, mask: Tensor) -> Tensor:
    # Find the similarities between the query Q and the keys K
    similarities = Q @ K.transpose(-2, -1)
    d_k = K.shape[-1]
    # Scale to smaller numbers to prevent diminishing softmax gradients
    scaled = similarities / math.sqrt(d_k)
        
    if mask is not None:
        mask = mask[:, :, None, None] 
        
        # Apply the mask
        masked = scaled.masked_fill(mask == 1.0, float('-inf'))
    else:
        masked = scaled
        
    # Calculate attention using softmax on the last dimension
    attn = softmax(masked, dim=-1)
    
    # Apply attention to values to gather contextual info
    contextual_info = attn @ V
    return contextual_info

class MultiHeadAttenionLayer(nn.Module):
    def __init__(self, info: ModelInfo, num_heads: int):
        super().__init__()
        self.info = info
        d_model = info.d_model
        
        self.n_heads = num_heads
        self.d_v = self.d_k = d_model // num_heads
                
        self.W_v = LinearLayer(info, output_shape=(d_model, d_model))
        self.W_q = LinearLayer(info, output_shape=(d_model, d_model))
        self.W_k = LinearLayer(info, output_shape=(d_model, d_model))
        self.W_o = LinearLayer(info, output_shape=(d_model, d_model))
    
    def forward(self, V: Tensor, K: Tensor, Q: Tensor, mask: Tensor) -> Tensor:
        batch_size, seq_len, d_model = self.info.shape
        
        V = self.W_v(V).view(batch_size, seq_len, self.n_heads, self.d_k)
        K = self.W_k(K).view(batch_size, seq_len, self.n_heads, self.d_k)
        Q = self.W_q(Q).view(batch_size, seq_len, self.n_heads, self.d_k)
                
        output = scaled_dot_product_attention(V=V, K=K, Q=Q, mask=mask)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        output = self.W_o(output)
        return output
