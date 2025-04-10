# from MrML.types import *
# from MrML.model_info import ModelInfo
# from MrML.embedding import Embedder
# from MrML.tools import pad, chunk, make_mask

# class TransformerBatch:
#     def __init__(self, model_info: ModelInfo, prompt: str):
#         self.id = get_id()
#         self.prompt = prompt
#         self.info = model_info
        
#         self.tokens = self.info.vocab.tokenize(prompt)
#         self.tokens, mask = pad(self.tokens, self.info)
            
#         output = torch.full(fill_value=self.info.vocab.pad, size=self.tokens.size())
#         self.output[0] = self.info.vocab.sos
            
#         self.embedder = Embedder(self.info)
#         embeddings = self.embedder(self.tokens)
#         self.E = chunk(embeddings, self.info) # (num_chunks, seq_len, d_model)
#         self.e_masks = chunk(mask, self.info) # (num_chunks, seq_len)
        
#         self.D = self.embedder(output) # (seq_len, d_model)
#         self.d_masks = make_mask(ones=(1,), shape=(self.D.shape[1],), dtype=torch.int) # (seq_len,)
        
#         self.logits = torch.empty(size=(self.info.seq_len, self.info.d_model))  # (seq_len, d_model)
#         self.probs = torch.empty(size=(self.info.seq_len, self.info.vocab_len)) # (seq_len, vocab_len)
#         self.predictions = []
        
#         self.pos = 0
#         self.chunk_count = 0
#         self.is_finished = False








# next_id = 0

# def get_id() -> int:
#     output = next_id
#     next_id += 1
#     return output

# class TransformerJob:
#     def __init__(self, model_info: ModelInfo, prompt: str):
#         self.id = get_id()
#         self.prompt = prompt
#         self.info = model_info
        
#         self.tokens = self.info.vocab.tokenize(prompt)
#         self.tokens, mask = pad(self.tokens, self.info)
            
#         output = torch.full(fill_value=self.info.vocab.pad, size=self.tokens.size())
#         self.output[0] = self.info.vocab.sos
            
#         self.embedder = Embedder(self.info)
#         embeddings = self.embedder(self.tokens)
#         self.E = chunk(embeddings, self.info) # (num_chunks, seq_len, d_model)
#         self.e_masks = chunk(mask, self.info) # (num_chunks, seq_len)
        
#         self.D = self.embedder(output) # (seq_len, d_model)
#         self.d_masks = make_mask(ones=(1,), shape=(self.D.shape[1],), dtype=torch.int) # (seq_len,)
        
#         self.logits = torch.empty(size=(self.info.seq_len, self.info.d_model))  # (seq_len, d_model)
#         self.probs = torch.empty(size=(self.info.seq_len, self.info.vocab_len)) # (seq_len, vocab_len)
#         self.predictions = []
        
#         self.pos = 0
#         self.chunk_count = 0
#         self.is_finished = False
    
#     def output(self, format: bool = True) -> str:
#         return self.info.vocab.stringify(self.predictions, format)
    
#     def __str__(self) -> str:
#         return f"{self.id}: {self.is_finished}, {self.output(False)}, {self.predictions}"
    
#     def update(self, D: Tensor, logits: Tensor,) -> Tuple[Tensor, Tensor]:
        

#     def next_chunk(self) -> bool:
#         if self.chunk_count < self.E.size[0]:
#             self.pos = self.info.stride - 1
#             self.chunk_count += 1
            
    
    





# def inc_chunk(self) -> bool:
#         if self.masks.numel() > 0:
#             return True
#         else:
#             self.chunks = self.chunks[1:]
#             self.masks = self.masks[1:]
#             self.D = self.D[1:]
#             self.d_mask = make_mask(ones=(self.info.stride,), shape=(self.info.seq_len,), dtype=torch.int)
#             return False





# class PreppedTransformerJob:
#     def __init__(self, job: TransformerJob, info: ModelInfo):
#         self.id = job.id
#         self.info = info
        
#         tokens = self.info.vocab.tokenize(job.prompt)
#         tokens, mask = pad(tokens, info)
        
#         output = torch.full(fill_value=info.vocab.pad, size=tokens.size())
#         output[0] = info.vocab.sos
        
#         self.embedder = Embedder(info)
#         embeddings = self.embedder(tokens)
#         output = self.embedder(output)
        
#         self.chunks = chunk(embeddings, info)
#         self.masks = chunk(mask, info)
        
#         self.D = chunk(output, info)
#         d_mask = make_mask(ones=(1,), shape=(D.shape[1],), dtype=torch.int)
#         d_mask = chunk(d_mask, info)
        
#         self.current_chunk = 0
        
#         self.is_finished = False
#         self.probs: Tensor = tensor([], dtype=torch.float32)
#         self.predictions: Tensor = tensor([], dtype=torch.int)
    
#     def update(self, output: Tensor, probs: Tensor, i: int):
#         self.probs = probs
        
#         if not self.is_finished and i < self.D.shape[0]:
#             self.D[i] = output[i]
#             self.predictions = torch.argmax(probs, dim=-1)
            
#             if (self.predictions == self.info.vocab.eos).any():
#                 self.d_mask[:] = 1
#                 self.is_finished = True
#             elif i + 1 < self.d_mask.shape[0]:
#                 self.d_mask[i + 1] = 1
    
#     def inc_chunk(self) -> bool:
#         if self.masks.numel() > 0:
#             return True
#         else:
#             self.chunks = self.chunks[1:]
#             self.masks = self.masks[1:]
#             self.D = self.D[1:]
#             self.d_mask = make_mask(ones=(self.info.stride,), shape=(self.info.seq_len,), dtype=torch.int)
#             return False

# class TransformerJobResult:
#     def __init__(self, job: PreppedTransformerJob):
#         self.id = job.id
#         self.info = job.info
#         self.predictions = job.predictions
#         self.is_finished = job.is_finished
    
#     def output(self, format: bool = True) -> str:
#         return self.info.vocab.stringify(self.predictions, format)
    
#     def __str__(self) -> str:
#         return f"{self.id}: {self.is_finished}, {self.output(False)}, {self.predictions}"
