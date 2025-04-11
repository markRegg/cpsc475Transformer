import json
from random import randint, sample, seed
from MrML import *
from info import *

MIN_N = 1
MAX_N = SEQ_LEN // 3
seed(9)                 # Seed pseudo-RNG for reproducibility

def generate_valid_string() -> str:
    """Generates a valid or invalid string of pseudorandom length in language L

    Args:
        valid (bool): Whether or not to generate a valid string

    Returns:
        string: A valid or invalid string of pseudorandom length in langauge L
    """
    
    token_a = vocab.random_token()
    token_b = vocab.random_token()
    
    while token_a == token_b:
        token_b = vocab.random_token()
    
    n = range(randint(MIN_N, MAX_N))
    return "".join([token_a for _ in n] + [token_b for _ in n] + [token_a for _ in n])

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

def save(file_path: str, tokens: List, masks: List, labels: List, strings: List):
    token_lists = [t.tolist() for t in tokens]
    masks_lists = [m.tolist() for m in masks]
    
    file_output = {
        "tokens": token_lists, 
        "masks": masks_lists, 
        "labels": labels, 
        "strings": strings
    }
    
    with open(file_path, "w") as file:
        json.dump(file_output, file)
    
    print(f"Saved dataset of {len(labels)} entries to {file_path}")

def generate_dataset(file_path: str, count: int = BATCH_SIZE * 20, pct_valid: float = 0.5) -> List[Tuple[Tensor, Tensor, bool, str]]:
    strings = set()
    
    while len(strings) < count:
        strings.add(generate_valid_string())
    
    strings = list(strings)
    labels = [1.0 for _ in range(count)]
        
    num_invalid = int(count * (1 - pct_valid))
    indexes_to_bork = sample(range(count), num_invalid)
    
    for i in indexes_to_bork:
        string = strings[i]
        strings[i] = add_errors(string)
        labels[i] = 0.0
    
    tokens = []
    masks = []
    
    for string in strings:
        t = vocab.tokenize(string, device=info.device)
        t, m = pad(t, info)
        
        tokens.append(t)
        masks.append(m)
        
    save(file_path, tokens, masks, labels, strings)
    return tokens, masks, labels, strings
