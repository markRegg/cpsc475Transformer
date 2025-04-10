from MrML.types import *
from MrML.model_info import ModelInfo
from MrML.tools import weight_tensor

class LinearLayer(nn.Module):
    def __init__(self, info: ModelInfo, output_size: int, bias: bool = True):
        super().__init__()
        self.W = weight_tensor(size=(output_size, info.d_model), dtype=info.dtype)
        self.b = weight_tensor(size=(output_size,), dtype=info.dtype) if bias else None
        
    def forward(self, X: Tensor) -> Tensor:
        # Matrix multiplication: (batch_size, seq_len, d_model) @ (d_model, output_size)
        output = X @ self.W.T

        # Add bias if present (broadcast over batch and seq_len)
        if self.b is not None:
            output += self.b.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, output_size)

        return output