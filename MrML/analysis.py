from typing import Tuple
import torch
from torchmetrics.classification import BinaryPrecision, BinaryRecall, BinaryAccuracy
from MrML.model_info import ModelInfo

class ModelAnalysis:
    def __init__(self, info: ModelInfo):
        self.total_loss = 0.0
        self.batches = 0
        self.accuracy = BinaryAccuracy().to(info.device)
        self.precision = BinaryPrecision().to(info.device)
        self.recall = BinaryRecall().to(info.device)
    
    def results(self) -> Tuple[float, float, float, float]:
        return (
            self.loss(),
            self.accuracy.compute(),
            self.precision.compute(),
            self.recall.compute()
        )
    
    def loss(self) -> float:
        return self.total_loss / max(1.0, self.batches)
    
    def update(self, outputs, labels, loss):
        probabilities = torch.sigmoid(outputs.detach())
        predictions = (probabilities >= 0.5).float()
        
        self.total_loss += loss
        self.batches += 1
        self.accuracy.update(predictions, labels)
        self.precision.update(predictions, labels)
        self.recall.update(predictions, labels)
    
    def reset(self):
        self.total_loss = 0
        self.batches = 0
        self.accuracy.reset()
        self.precision.reset()
        self.recall.reset()
    
    def report(self) -> str:
        loss, accuracy, precision, recall = self.results()
        return f"Loss: {loss:.4f}, Accuracy: {accuracy:.4%}, Precision: {precision:.4%}, Recall: {recall:.4%}"