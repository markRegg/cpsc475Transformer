import torch
from torch import Tensor, tensor
import math

class Embedder:
    """Handles token and positional embedding"""
    
    def __init__(self, seq_len: int, d_model: int, vocab_len: int):
        """Creates an Embedder

        Args:
            seq_len (int): The model's sequence length
            d_model (int): The model's dimensionality
            vocab_len (int): The number of tokens in the vocabulary
        """
        self.seq_len = seq_len
        self.d_model = d_model

        # Initalize token embeddings to random values
        self._token_embeds = torch.randn(size=(vocab_len, d_model), dtype=torch.float32)

        # Calculate and cache positional embeddings for seq_len tokens
        self._pos_embeds = tensor([], dtype=torch.float32)
        _ = self.pos_embeds(count=seq_len)
    
    def pos_embeds(self, count: int) -> Tensor:
        """Gets positional embeddings for tokens in positions 0 to count using
        sinusoidal positional embedding

        Args:
            count (int): The number of tokens

        Returns:
            Tensor: The positional embeddings for count tokens as a
                tensor of torch.float32 with size (count, d_model)
        """
        # Calculate number of not cached embeddings to calculate
        num_cached_pos = self._pos_embeds.size(0)
        num_new_pos = count - num_cached_pos

        if num_new_pos > 0:
            # Create space in the cache for the new positional embeddings
            new_pos = torch.zeros(size=(num_new_pos, self.d_model), dtype=torch.float32)
            self._pos_embeds = torch.cat((self._pos_embeds, new_pos))

            # Generate the new positional embeddings
            for pos in range(num_cached_pos, count):
                for i in range(self.d_model // 2):
                    angle = pos / (10000 ** ((2 * i) / self.d_model))
                    self._pos_embeds[pos, 2 * i] = math.sin(angle)
                    self._pos_embeds[pos, (2 * i) + 1] = math.cos(angle)
        
        # Return the positional embeddings for each token in 0 to count
        return self._pos_embeds[:count]
    
    def token_embeds(self, tokens: Tensor) -> Tensor:
        """Gets the token embedings for tokens in a sequence

        Args:
            tokens (Tensor): The token sequence, a tensor of torch.int with size (tokens.size(0),)

        Returns:
            Tensor: The token embeddings for each token as a tensor of torch.float32
                with size (num_tokens, d_type)
        """
        # Create tensor to store token embeddings in
        output = torch.zeros(size=(tokens.size(0), self.d_model), dtype=torch.float32)

        # Get the cached token embedding for each token in the sequence
        for i, token in enumerate(tokens):
            output[i] = self._token_embeds[token]
        
        return output
    
    def pad(self, tokens: Tensor, PAD: int, window_size: int) -> Tensor:
        """Pads a token sequence to allow it to have seq_len tokens in each window

        Args:
            tokens (Tensor): The token sequence, a tensor of torch.int with size (tokens.size(0),)
            PAD (int): The int value of the padding token
            window_size (int): The number of tokens in a window for a sliding window
                if there are more tokens than seq_len

        Returns:
            Tensor: Padded token sequence as tensor of torch.int with size (tokens.size(0) + padding,)
        """
        # Find number of tokens needed to pad till seq_len
        num_pad_tokens = self.seq_len - tokens.size(0)

        if num_pad_tokens < 0:
            # If there are more tokens than seq_len, we need to pad with enough 
            # tokens to make sure that the last window will have seq_len tokens
            num_pad_tokens = tokens.size(0) % window_size

        if num_pad_tokens > 0:
            # Add padding tokens if needed
            padding = torch.full(fill_value=PAD, size=(num_pad_tokens,), dtype=torch.int)
            tokens = torch.cat((tokens, padding))

        return tokens

    def embed(self, tokens: Tensor, PAD: int, window_size: int) -> Tensor:
        """The vectorized embeddings for each token in a sequence as a combination of
        the token embeddings and positional embeddings

        Args:
            tokens (Tensor): The token sequence, a tensor of torch.int with size (tokens.size(0),)
            PAD (int): The int value of the padding token
            window_size (int): The number of tokens in a window for a sliding window
                if there are more tokens than seq_len

        Returns:
            Tensor: The embeddings for each token as a tensor of torch.float32
                with size (num_tokens, d_type)
        """
        # Pad tokens to ensure each window has seq_len tokens
        tokens = self.pad(tokens=tokens, PAD=PAD, window_size=window_size)

        # Get the token embeddings for each token in the sequence
        token_embeds = self.token_embeds(tokens)

        # Get the positional embeddings for each token in the sequence
        token_count = tokens.size(0)
        pos_embeds = self.pos_embeds(count=token_count)

        # Add the token and positional embeddings element-wise
        embeddings = token_embeds + pos_embeds
        return embeddings
