from MrML.types import *
from MrML.model_info import ModelInfo
from MrML.tools import leaky_relu, weight_tensor

class FullyConnectedFeedForwardLayer(nn.Module):
    def __init__(
        self,
        info: ModelInfo,
        d_ff: Optional[int] = None, 
        activation_fn: Callable[[DType], DType] = leaky_relu
    ):
        super().__init__()
        dtype = info.dtype
        d_model = info.d_model
        
        if d_ff is None:
            d_ff = d_model * 4
        
        self.activation_fn = activation_fn
        self.W1 = weight_tensor(size=(d_model, d_ff), dtype=dtype)
        self.b1 = weight_tensor(size=(d_ff,), dtype=dtype)
        self.W2 = weight_tensor(size=(d_ff, d_model), dtype=dtype)
        self.b2 = weight_tensor(size=(d_model,), dtype=dtype)
        
    def forward(self, X: Tensor) -> Tensor:
        X = (X @ self.W1) + self.b1         # expansion
        X = X.apply_(self.activation_fn)    # Apply the activation function
        X = (X @ self.W2) + self.b2         # Projection back to model dimensions
        return X