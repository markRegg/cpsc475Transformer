from MrML.types import *
from random import randint, sample, seed

class Vocab:
    def __init__(self, tokens: List[str]):
        count = len(tokens)
        self.lexemes = tokens + ["<SOS>", "<EOS>", "<PAD>"]
        self.ids = { lexeme : i for i, lexeme in enumerate(self.lexemes) }
                
        self.sos = count
        self.eos = count + 1
        self.pad = count + 2
        
    def __len__(self) -> int:
        return len(self.lexemes)
    
    def __getitem__(self, key: Union[int, str]) -> Union[int, str]:
        if isinstance(key, int):
            return self.lexemes[key]
        elif isinstance(key, str):
            return self.ids[key]
        else:
            raise TypeError(f"Vocab index must be int or str, received {type(key)} instead.")
    
    def tokenize(self, prompt: str) -> Tensor:
        return tensor([self.ids[c] for c in list(prompt)], dtype=torch.int)
    
    def stringify(self, prediction: Tensor, format: bool = True) -> List[str]:
        prediction = prediction.tolist()

        response = ""   
             
        for token in prediction:
            if format:
                if token == self.eos:
                    break
                if token == self.sos or token == self.pad:
                    continue
                
            response += self.lexemes[token]

        return response
    
    def random_token(self):
        """Returns a pseudorandomly generated token from the vocabulary"""
        return self[randint(0, len(self) - 3)]