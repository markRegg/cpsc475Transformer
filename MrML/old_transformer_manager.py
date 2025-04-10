# from MrML.types import *
# from MrML.jobs import *
# from MrML.transformer import *

# class TransformerManager:
#     def __init__(self, info: ModelInfo, n_layers: int = DEFAULT_N_LAYERS, n_heads: int = DEFAULT_N_HEADS):
#         self.info = info
#         self.model = Transformer(info, n_layers, n_heads)
#         self.model_thread: Optional[Thread] = None 
        
#         self.jobs: List[PreppedTransformerJob] = []
#         self.new_jobs: Queue[List[PreppedTransformerJob]] = Queue()
#         self.results_queue: Queue[TransformerJobResult] = Queue()
    
#     def predict(self, jobs: List[TransformerJob]):
#         self.new_jobs.put([PreppedTransformerJob(job, self.info) for job in jobs])
        
#         if self.model_thread is None:
#             self.model_thread = Thread(target=self.handle_jobs, daemon=True)
#             self.model_thread.start()
    
#     def results(self):
#         while True:
#             update = self.results_queue.get()
            
#             if update is None:
#                 break
            
#             yield update
        
#     def handle_jobs(self):
#         while True:
#             new_jobs = self.new_jobs.get()
            
#             if new_jobs is None:
#                 break
            
#             self.jobs += new_jobs
#             self.prepare_batch()
#             self.predict_batch()
        
#     def prepare_batch(self):
#         batch_size = self.info.batch_size
#         num_jobs = len(self.jobs)
                
#         if num_jobs < batch_size:
#             filler_jobs = self.create_filler_jobs(batch_size - num_jobs)
#             self.jobs += filler_jobs
        
#     def predict_batch(self):
#         batch_size, seq_len = self.info.shape[:2]
#         batch = self.jobs[:batch_size]
        
#         for i in range(1, seq_len):
#             X = torch.stack([job.chunks[0] for job in batch], dim=0)
#             input_masks = torch.stack([job.masks[0] for job in batch], dim=0)
#             D = torch.cat([job.D for job in batch], dim=0)
#             d_masks = torch.stack([job.d_mask for job in batch])
                        
#             self.model.predict(X, input_masks, D, d_masks)
            
#             can_stop = True
            
#             for i, element in enumerate(zip(batch, self.model.D, self.model.probs)):
#                 job, output, probs = element
#                 print(f"JOB {i} UPDATE: {probs.shape}")
#                 job.update(output, probs, i)
                
#                 if not job.is_finished:
#                     can_stop = False
            
#             result_jobs = [TransformerJobResult(job) for job in batch]
#             self.results_queue.put(result_jobs)
            
#             if can_stop:
#                 break
        
#         for i, job in enumerate(reversed(batch)):
#             if job.inc_chunk():
#                 self.jobs.pop(i)
        
#     def create_filler_jobs(self, num_filler_jobs: int) -> Tensor:
#         filler_jobs = [TransformerJob(id=-1, prompt="") for _ in range(num_filler_jobs)]
#         return [PreppedTransformerJob(job, self.info) for job in filler_jobs]
