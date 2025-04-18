from MrML.types import *
from MrML.model_info import ModelInfo

class LinearLayer(nn.Module):
    def __init__(self, info: ModelInfo, output_shape: Tuple[int], bias: bool = True):
        super().__init__()
        self.info = info
        
        self.W = nn.Parameter(
            torch.empty(size=output_shape, dtype=info.dtype, device=info.device)
        )
        nn.init.xavier_uniform_(self.W)

        if bias:
            self.b = nn.Parameter(
                torch.empty(size=output_shape[:-1], dtype=info.dtype, device=info.device)
            )
            nn.init.uniform_(self.b, -0.01, 0.01)
        else:
            self.b = None
        
    def forward(self, X: Tensor) -> Tensor:
        # Matrix multiplication: (batch_size, seq_len, d_model) @ (d_model, output_size)
        output = X @ self.W.T

        # Add bias if present (broadcast over batch and seq_len)
        if self.b is not None:
            bias = self.b.unsqueeze(0).unsqueeze(0)  # Shape: (1, 1, output_size)
            output = output + bias

        return output