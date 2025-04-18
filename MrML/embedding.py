import math
from MrML.types import *
from MrML.model_info import ModelInfo

class Embedding(nn.Module):
    def __init__(self, info: ModelInfo):
        super().__init__()
        self.embeddings = nn.Parameter(
            torch.empty(size=(info.vocab_len, info.d_model), dtype=info.dtype, device=info.device)
        )
        nn.init.xavier_uniform_(self.embeddings)

    def forward(self, tokens: Tensor) -> Tensor:
        return self.embeddings[tokens]

class Embedder(nn.Module):
    """Handles token and positional embedding"""
    
    def __init__(self, info: ModelInfo):
        """Creates an Embedder

        Args:
            seq_len (int): The model's sequence length
            d_model (int): The model's dimensionality
            vocab_len (int): The number of tokens in the vocabulary
        """
        super().__init__()
        self.info = info

        # Initalize token embeddings to random values
        self.token_embedder = Embedding(info)

        # Calculate and cache positional embeddings for seq_len tokens
        self._pos_embeds = tensor([], dtype=info.dtype, device=info.device)
        self._get_pos_embeds(count=info.seq_len)
    
    def _get_pos_embeds(self, count: int) -> Tensor:
        """Gets positional embeddings for tokens in positions 0 to count using
        sinusoidal positional embedding

        Args:
            count (int): The number of tokens

        Returns:
            Tensor: The positional embeddings with shape (count, d_model)
        """        
        num_cached = self._pos_embeds.shape[0]  # Num pos embeddings already calculated
        num_new = count - num_cached            # Num pos embeddings that need to be calculated
                
        if num_new > 0: # We need to calculate some new pos embeddings
            # Create space for new pos embeddings
            new_space = torch.zeros(size=(num_new, self.info.d_model), dtype=self.info.dtype, device=self.info.device)
            self._pos_embeds = torch.cat((self._pos_embeds, new_space), dim=0)
            
            # Calculate new pos embeddings
            for pos in range(num_cached, count):
                for i in range(self.info.d_model // 2):
                    angle = pos / (10000 ** ((2 * i) / self.info.d_model))
                    self._pos_embeds[pos, 2 * i] = math.sin(angle)
                    self._pos_embeds[pos, (2 * i) + 1] = math.cos(angle)
        
        return self._pos_embeds[:count] # Return the positional embeddings for each token in 0 to count

    def forward(self, tokens: Tensor) -> Tensor:
        """The vectorized embeddings for each token in a sequence as a combination of
        the token embeddings and positional embeddings

        Args:
            tokens (Tensor): The token sequence with shape (num_tokens,)

        Returns:
            Tensor: The embeddings for each token with shape (num_tokens, d_model)
        """

        # Get the token embeddings for each token in the sequence
        token_embeds = self.token_embedder(tokens)

        # Get the positional embeddings for each token in the sequence
        pos_embeds = self._get_pos_embeds(count=self.info.seq_len)
        pos_embeds = pos_embeds.unsqueeze(0).expand(self.info.batch_size, -1, -1)
        
        # Add the token and positional embeddings element-wise
        embeddings = token_embeds + pos_embeds

        return embeddings
