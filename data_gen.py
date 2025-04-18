from __future__ import annotations
import os
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from random import randint, sample, seed
from MrML import *

SentenceDataDict = Dict[str, Union[List[str], Tensor]]

class SentenceDataset(Dataset):
    def __init__(self, info: ModelInfo, data: Union[SentenceDataDict, SentenceDataset], device: Optional[str] = None):
        super().__init__()
        self.info = info
        self.sentences = data["sentences"]
        self.tokens = data["tokens"].to(device)
        self.masks = data["masks"].to(device)
        self.labels = data["labels"].to(device)

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, i):
        return self.sentences[i], self.tokens[i], self.masks[i], self.labels[i]
    
    def subset(self, indices: Tensor) -> SentenceDataset:
        return SentenceDataset(self.info, {
            "sentences": [self.sentences[i] for i in indices.tolist()],
            "tokens": self.tokens[indices], 
            "masks": self.masks[indices], 
            "labels": self.labels[indices]
        }, device=self.tokens.device)
    
    def epoch(self):
        batch_size = self.info.batch_size
        num_elements = len(self)
        order = torch.randperm(num_elements, dtype=torch.long)
        
        for batch_start in range(0, num_elements, batch_size):
            indices = order[batch_start : batch_start + batch_size]
            
            sentences = [self.sentences[i] for i in indices.tolist()]
            tokens = self.tokens[indices]
            masks = self.masks[indices]
            labels = self.labels[indices]
            
            yield sentences, tokens, masks, labels
    
    def save(self, filepath: str = "data/data.pt"):
        torch.save({
            "sentences": self.sentences,
            "tokens": self.tokens, 
            "masks": self.masks, 
            "labels": self.labels,
        }, "data/data.pt")
            
        print(f"Saved dataset of {len(self)} entries to {filepath}")
    
    @staticmethod
    def load(filepath: str, info: ModelInfo, device: Optional[str] = None):
        data = torch.load(filepath, map_location=device, weights_only=False)
        return SentenceDataset(info, data, device)
    
    def class_distribution(self):
        trues = 0
        falses = 0

        # Loop through the data and count the labels
        for label in self.labels:
            if label >= 0.5:
                trues += 1
            else:
                falses += 1
        
        total = trues + falses
        true_pct = trues / total
        false_pct = falses / total
        print(f"Trues: {trues} ({true_pct:.2%}), Falses: {falses} ({false_pct:.2%})")
        
        return (trues, falses), (true_pct, false_pct)



def generate_valid_sentence(vocab: Vocab, min_n: int = 0, max_n: int = 100) -> str:
    token_a = vocab.random_token()
    token_b = vocab.random_token()
    
    while token_a == token_b:
        token_b = vocab.random_token()
    
    n = range(randint(min_n, max_n))
    return "".join([token_a for _ in n] + [token_b for _ in n] + [token_a for _ in n])

def generate_valid_sentences(size: int, vocab: Vocab, min_n: int = 0, max_n: int = 100):
    sentences = set()
    
    while len(sentences) < size:
        sentences.add(generate_valid_sentence(vocab, min_n, max_n))
    
    return list(sentences)

def add_errors(string: str) -> str:
    token_a, token_b = set(string)
    num_errs = randint(1, len(string) // 3) # Number of errors to add to the output

    # Generate num_errs unique indexes in the output to flip 'a' <-> 'b'
    err_indexes = sample(range(len(string)), num_errs)

    # Flip a_token <-> b_token for each err_index
    for i in err_indexes:
        new_letter = token_b if string[i] == token_a else token_b   # flipped letter
        string = string[:i] + new_letter + string[i+1:] # replace in output

    return string

def generate_dataset(info: ModelInfo, size: int, pct_valid: float = 0.5, rand_seed: int = 0, device: Optional[str] = None) -> SentenceDataset:
    print("Generating Dataset...")
    seed(rand_seed)
    torch.manual_seed(rand_seed)
    
    sentences = list(generate_valid_sentences(size, info.vocab, min_n=1, max_n=info.seq_len // 3))
    labels = torch.ones(size=(size,), dtype=info.dtype)
        
    num_invalid = int(size * (1 - pct_valid))
    indexes_to_bork = torch.randperm(size)[:num_invalid]
    labels[indexes_to_bork] = 0.0
    
    for i in indexes_to_bork.tolist():
        sentence = sentences[i]
        sentences[i] = add_errors(sentence)
    
    tokens = []
    masks = []
    
    for sentence in sentences:
        t = info.vocab.tokenize(sentence, device=info.device)
        t, m = pad(t, info)
        
        tokens.append(t)
        masks.append(m)
    
    tokens = torch.stack(tokens, dim=0)
    masks = torch.stack(masks, dim=0)
    
    dataset = SentenceDataset(info, {
        "sentences": sentences,
        "tokens": tokens, 
        "masks": masks, 
        "labels": labels,
    })
    
    print("Finished Generating Dataset, saving...")
    dataset.save()
    print("Saved Dataset")
    return dataset

def make_train_test_split(dataset: SentenceDataset, test_pct: float = 0.3, rand_seed: int = 0) -> Tuple[SentenceDataset, SentenceDataset]:
    print("Making train/test split...")
    train_indices, test_indices = train_test_split(
        range(len(dataset)),
        test_size=test_pct,
        random_state=rand_seed,
        stratify=dataset.labels.cpu().numpy()
    )
    
    train_indices = tensor(train_indices, dtype=torch.long)
    test_indices = tensor(test_indices, dtype=torch.long)
    
    train = dataset.subset(train_indices)
    test = dataset.subset(test_indices)
    
    print("Finished Generating train/test split, saving...")
    torch.save(train, "data/train.pt")
    torch.save(test, "data/test.pt")
    
    print("Saved train/test split")
    return train, test

def load_train_set(info: ModelInfo, device: Optional[str] = None):
    return SentenceDataset.load("data/train.pt", info=info, device=device)

def load_test_set(info: ModelInfo, device: Optional[str] = None):
    return torch.load("data/test.pt", info=info, device=device)

def load_train_test_split(info: ModelInfo, total_size: int, make_new: bool = False, device: Optional[str] = None):
    print("Loading train/test split...")
    if make_new or not os.path.exists("data/data.pt") or not os.path.exists("data/test.pt") or not os.path.exists("data/train.pt"):
        if make_new or not os.path.exists("data/data.pt"):
            dataset = generate_dataset(info, total_size)
        else:
            dataset = SentenceDataset.load("data/data.pt", info=info, device=device)
        
        return make_train_test_split(dataset)
    
    train, test = load_train_set(info, device), load_test_set(info, device)
    print("Loaded train/test split")
    return train, test
