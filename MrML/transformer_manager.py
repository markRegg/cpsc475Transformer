from typing import Generator
from MrML import *

next_id = 0

def get_id() -> Generator[int]:
    yield next_id
    next_id += 1

class TransformerJob:
    def __init__(self, prompt: str):
        self.id = get_id()
        self.prompt = prompt
        self.predictions = []

class TransformerManager:
    def __init__(self, model: Transformer):
        self.info = model.info
        self.model = model
        self.should_run = False
        self.thread: Optional[Thread] = None
        self.jobs: Queue[TransformerJob] = Queue()
        
        self.embeddings: List[Tensor] = []
        self.e_masks: List[Tensor] = []
        self.outputs: List[Tensor] = []
        
        self.embedder = Embedder(self.info)

    def run(self):
        self.should_run = True
        self.thread = Thread(target=self.handle_jobs, daemon=True)
        self.thread.run()
    
    def stop(self):
        self.should_run = False
    
    def predict(self, job: TransformerJob):
        self.jobs.put(job)
    
    def handle_jobs(self):
        while self.should_run:
            job = self.jobs.get()
    
            if job is None: continue
            self.prepare_job(job)
            
            if len(self.e_masks) == self.model.info.batch_size:
                self.run_batch()
    
        
        del self.thread
        self.thread = None
    
    def prepare_job(self, job: TransformerJob):
        tokens = self.info.vocab.tokenize(job.prompt)
        tokens, e_mask = pad(tokens, self.info)
        embeddings = self.embedder(tokens)
                
        output = torch.full(
            fill_value=self.info.vocab.pad, 
            size=(self.info.seq_len,)
        )
        output[0] = self.info.vocab.sos
        output = self.embedder(output)
        
        self.embeddings.append(embeddings)
        self.e_masks.append(e_mask)
        self.outputs.append(output)
    
    def run_batch(self):
