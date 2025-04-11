import json
from random import randint, sample, seed
from MrML import *
from info import *

MIN_N = 1
MAX_N = SEQ_LEN // 3
MAX_ERRS_IN_STRING = 3  # Max number of errors to put in a string
seed(0)                 # Seed pseudo-RNG for reproducibility

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
    
    n = randint(MIN_N, MAX_N)
    return ("a" * n) + ("b" * n) + ("a" * n)

def add_errors(string: str) -> str:
    token_a, token_b = set(string)
    num_errs = randint(1, MAX_ERRS_IN_STRING) # Number of errors to add to the output

    # Generate num_errs unique indexes in the output to flip 'a' <-> 'b'
    err_indexes = sample(range(len(string)), num_errs)

    # Flip a_token <-> b_token for each err_index
    for i in err_indexes:
        new_letter = token_b if string[i] == token_a else token_b   # flipped letter
        string = string[:i] + new_letter + string[i+1:] # replace in output

    return string

def generate_dataset(file_path: str, count: int = BATCH_SIZE * 20, pct_valid: float = 0.5) -> List[Tuple[Tensor, Tensor, bool, str]]:
    strings = [{ "input": generate_valid_string(), "is_valid": True } for _ in range(count) ]
    
    num_invalid = int(count * (1 - pct_valid))
    indexes_to_bork = sample(range(count), num_invalid)
    
    for i in indexes_to_bork:
        string = strings[i]["input"]
        strings[i]["input"] = add_errors(string)
        strings[i]["is_valid"] = False
    
    ret_tokens = []
    ret_masks = []
    ret_labels = []
    ret_strings = []
    
    for i in range(len(strings)):
        string = strings[i]["input"]
        tokens = vocab.tokenize(string)
        tokens, mask = pad(tokens, info)
        
        ret_tokens.append(tokens)
        ret_masks.append(mask)
        ret_labels.append(1.0 if strings[i]["is_valid"] else 0.0)
        ret_strings.append(string)
        
    token_lists = [t.tolist() for t in ret_tokens]
    masks_lists = [m.tolist() for m in ret_masks]
    file_output = {
        "tokens": token_lists, 
        "masks": masks_lists, 
        "labels": ret_labels, 
        "strings": ret_strings
    }
    
    with open(file_path, "w") as file:
        json.dump(file_output, file)
    
    print(f"Saved dataset of {count} entries with {num_invalid} invalid entries to {file_path}")
    return ret_tokens, ret_masks, ret_labels, ret_strings
    
if __name__ == "__main__":
    generate_dataset("data/train.json", BATCH_SIZE * 2000)
    generate_dataset("data/test.json", BATCH_SIZE * 20)
    generate_dataset("data/example.json", BATCH_SIZE * 5)