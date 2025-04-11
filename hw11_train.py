import os
from torchmetrics.classification import BinaryPrecision, BinaryRecall, BinaryAccuracy
from MrML import *
from info import *
from generate_data import *

class ModelAnalysis:
    def __init__(self, info: ModelInfo, criterion):
        self.criterion = criterion
        self.total_loss = 0.0
        self.num_labels = 0
        self.accuracy = BinaryAccuracy().to(info.device)
        self.precision = BinaryPrecision().to(info.device)
        self.recall = BinaryRecall().to(info.device)
    
    def results(self) -> Tuple[float, float, float, float]:
        return (
            self.total_loss / self.num_labels,
            self.accuracy.compute(),
            self.precision.compute(),
            self.recall.compute()
        )
    
    def update(self, outputs, labels, backpropogate: bool = False):
        loss = self.criterion(outputs, labels)
        
        if backpropogate:
            loss.backward()
        
        probabilities = torch.sigmoid(outputs)
        predictions = (probabilities >= 0.5).float()
        
        self.total_loss += loss.item()
        self.num_labels += labels.shape[0]
        self.accuracy.update(predictions, labels)
        self.precision.update(predictions, labels)
        self.recall.update(predictions, labels)
    
    def reset(self):
        self.total_loss = 0
        self.num_labels = 0
        self.accuracy.reset()
        self.precision.reset()
        self.recall.reset()
    
    def report(self) -> str:
        loss, accuracy, precision, recall = self.results()
        return f"Loss: {loss:.4f}, Accuracy: {accuracy:.4%}, Precision: {precision:.4%}, Recall: {recall:.4%}"
        
def shuffle(datasets: Tuple[List]) -> Tuple[List]:
    if len(datasets) == 0:
        return []
    
    indices = torch.randperm(len(datasets[0]))
    return [[d[i] for i in indices] for d in datasets]

def batch(data: List) -> List[Tensor]:
    batch_range = range(0, len(data), BATCH_SIZE)
    batches = [data[i: i + BATCH_SIZE] for i in batch_range]
    return [torch.stack(batch, dim=0).to(device) for batch in batches]

def batch_labels(labels: List) -> List[Tensor]:
    batch_range = range(0, len(labels), BATCH_SIZE)
    return [tensor(labels[i: i + BATCH_SIZE], dtype=info.dtype).to(device) for i in batch_range]

def train(model: LanguageAcceptanceClassifier, data: Tuple[List, List, List, List], criterion, optimizer, num_epochs: int = 10):
    tokens, masks, labels, _ = shuffle(data)
    
    tokens = batch(tokens)
    masks = batch(masks)
    labels = batch_labels(labels)
        
    analysis = ModelAnalysis(info, criterion)

    print(f"Training: Each epoch will contain {len(labels)} batches\n")
    
    model.train()
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        
        for inputs, mask_tensor, expected in zip(tokens, masks, labels):
            outputs = model(inputs, mask_tensor)
            analysis.update(outputs, expected, backpropogate=True)
            
            optimizer.step()
            optimizer.zero_grad()
        
        print(f"Epoch {epoch + 1} - {analysis.report()}")
        analysis.reset()
        
        if epoch % 5 == 0:
            torch.save(model, 'model.pth')
            print("Saved model\n")
            
    torch.save(model, 'model.pth')
    print("Saved model\n")



if __name__ == "__main__":
    model = LanguageAcceptanceClassifier(info, N_LAYERS, N_HEADS, dropout=0.1)
    model = model.to(device)
    
    train_set = generate_dataset("data/train.json", BATCH_SIZE * 1000)
    test_set = generate_dataset("data/test.json", BATCH_SIZE * 1000)
    example_set = generate_dataset("data/example.json", BATCH_SIZE * 1)
            
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    print()
    
    train(model, train_set, criterion, optimizer, num_epochs=30)
