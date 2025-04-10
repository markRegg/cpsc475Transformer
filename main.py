from MrML import *
    
if __name__ == "__main__":
    nums = list("0123456789")
    lowercase = list("abcdefghijklmnopqrstuvwxyz")
    uppercase = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
    symbols = list(",./;'[]\`-=<>?:\"{\}|~_+!@#$%^&*() ")
    vocab = Vocab(nums + lowercase + uppercase + symbols)
    
    BATCH_SIZE = 4
    SEQ_LEN = 64
    D_MODEL = 32
    STRIDE = SEQ_LEN // 2
    N_LAYERS = 1
    N_HEADS = 4
    
    info = ModelInfo(BATCH_SIZE, SEQ_LEN, D_MODEL, STRIDE, vocab)
    model = Transformer(info, N_LAYERS, N_HEADS)
    
    def embed_prompt(prompt: str) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        tokens = vocab.tokenize(prompt)
        tokens, mask = pad(tokens, info)
            
        output = torch.full(fill_value=info.vocab.pad, size=tokens.size())
        output[0] = info.vocab.sos
            
        embedder = Embedder(info)
        embeddings = embedder(tokens)
        output = embedder(output)
        
        chunks = chunk(embeddings, info)
        masks = chunk(mask, info)
        
        D = chunk(output, info)
        d_mask = make_mask(ones=(1,), shape=(D.shape[1],), dtype=torch.int)
        d_mask = chunk(d_mask, info)
        
        return chunks, masks, D, d_mask
    
    prompts = [
        ("a" * 20) + ("b" * 20) + ("a" * 20),
        ("7" * 20) + ("x" * 20) + ("7" * 20),
        ("!" * 20) + ("-" * 20) + ("!" * 20),
        ("." * 20) + ("?" * 20) + ("." * 20),
    ]
    
    input_info = [embed_prompt(prompt) for prompt in prompts]

    E = torch.stack([info[0][0] for info in input_info], dim=0)
    masks = torch.stack([info[1][0] for info in input_info], dim=0)
    D = torch.stack([info[2][0] for info in input_info], dim=0)
    d_masks = torch.stack([info[3][0] for info in input_info], dim=0)
    
    print(f"""
Model Info: 
    batch_size: {BATCH_SIZE},
    seq_len:    {SEQ_LEN},
    d_model:    {D_MODEL}, 
    stride:     {STRIDE}, 
    n_heads:    {N_HEADS},
    n_layers:   {N_LAYERS}
""")

    print(f"""
Model Inputs: 
    Number of Prompts:       {len(prompts)},
    Embeddings Shape:        {E.shape},
    Input Masks Shape:       {masks.shape}, 
    Output Embeddings Shape: {D.shape}, 
    Output Masks Shape:      {d_masks.shape}
""")
    
    model.predict(E, masks, D, d_masks)
    
    predictions = torch.argmax(model.probs, dim=-1)
    pred_text = "\n\n".join([f"\"{vocab.stringify(p, False)}\"" for p in predictions])
    pretty_pred_text = "\n\n".join([f"\"{vocab.stringify(p, True)}\"" for p in predictions])
    
    print(f"""
Model Predictions:

{pred_text}

Pretty Predictions:

{pretty_pred_text}
    
Predicted Tokens:

{predictions}
    """)
    