from MrML.types import *
from MrML.model_info import ModelInfo
from torch.nn import functional as F

class FullyConnectedFeedForwardLayer(nn.Module):
    def __init__(
        self,
        info: ModelInfo,
        d_ff: Optional[int] = None, 
    ):
        super().__init__()
        self.info = info
        dtype = info.dtype
        d_model = info.d_model
        
        if d_ff is None:
            d_ff = d_model * 4
                
        self.W1 = nn.Parameter(torch.empty(size=(d_model, d_ff), dtype=dtype, device=info.device))
        nn.init.kaiming_normal_(self.W1, nonlinearity='relu')
        
        self.b1 = nn.Parameter(torch.zeros(size=(d_ff,), dtype=dtype, device=info.device))
        
        self.W2 = nn.Parameter(torch.empty(size=(d_ff, d_model), dtype=dtype, device=info.device))
        nn.init.kaiming_normal_(self.W2, nonlinearity='relu')
        
        self.b2 = nn.Parameter(torch.zeros(size=(d_model,), dtype=dtype, device=info.device))
        
    def forward(self, X: Tensor) -> Tensor:
        X = X
        X = (X @ self.W1) + self.b1                 # expansion
        X = F.leaky_relu(X, negative_slope=0.02)    # Apply the activation function
        X = (X @ self.W2) + self.b2                 # Projection back to model dimensions
        return X