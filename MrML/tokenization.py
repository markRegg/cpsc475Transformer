import torch
from torch import Tensor, tensor

class Tokenizer:
    """Tokenizes text and converts tokens to text"""
    
    def __init__(self, vocab: list[str], PAD: int, SOS: int, EOS: int):
        """Creates a new Tokenizer

        Args:
            vocab (list[str]): A list of all the tokens in the vocabulary
            PAD (int): The int value of the padding token
            SOS (int): The int value of the start-of-stream token
            EOS (int): The int value of the end-of-stream token
        """
        # Save vocab info
        self.vocab = vocab
        self.vocab_len = len(vocab)
        
        # Save special token int values
        self.PAD = PAD
        self.SOS = SOS
        self.EOS = EOS
        
        # Save a tenor of SOS and PAD tokens so we don't have to remake it 
        # every time we format output
        self._SOS_PAD = tensor([self.SOS, self.PAD])
        
        # Create a mapping of token lexems to their int values
        self.tokens = { token : i for i, token in enumerate(vocab) }
    
    def tokenize(self, string: str) -> Tensor:
        """Converts a string into a sequence of token int values

        Args:
            string (str): The string to tokenize

        Returns:
            Tensor: The token int values as a tensor of torch.int with size (num_tokens,)
        """
        return tensor([self.tokens[c] for c in string], dtype=torch.int)

    def stringify(self, token_seq: Tensor) -> str:
        """Represents all of the token int values in a sequence as a string, including 
        PAD, SOS, and EOS tokens and any tokens that occur after the first EOS token

        Args:
            token_seq (Tensor): The token int values as a tensor of torch.int 
                with size (num_tokens,)

        Returns:
            str: The concatenation of each token's lexeme
        """
        char_seq = [self.vocab[i] for i in token_seq]
        return "".join(char_seq)
    
    def format_output(self, token_seq: Tensor) -> str:
        """Converts a sequence of token int values to its output text, ignoring all tokens
        after the first EOS token and removing all instances of SOS and PAD tokens.

        Args:
            token_seq (Tensor): The token int values as a tensor of torch.int 
                with size (num_tokens,)

        Returns:
            str: The concatenation of each non-special token's lexeme
        """
        # Trim input to only tokens before the EOS token
        eos_index = (tensor == self.EOS).nonzero(as_tuple=True)[0]
        token_seq = token_seq[:eos_index]

        # Remove SOS and PAD tokens from token sequence
        remove_sos_pad_mask = ~torch.isin(token_seq, self._SOS_PAD)
        token_seq = token_seq[remove_sos_pad_mask]

        # Return tokens as a string
        return self.stringify(token_seq)