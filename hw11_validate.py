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

def eval(model: LanguageAcceptanceClassifier, data: List[Tuple[List, List, List]], criterion):
    tokens, masks, labels, _ = data
    
    tokens = batch(tokens)
    masks = batch(masks)
    labels = batch_labels(labels)
        
    analysis = ModelAnalysis(info, criterion)
        
    model.eval()
    
    with torch.no_grad():
        for inputs, mask_tensor, expected in zip(tokens, masks, labels):
            outputs = model(inputs, mask_tensor)
            analysis.update(outputs, expected, backpropogate=False)
            
    return analysis

def example(model: LanguageAcceptanceClassifier, data: List[Tuple[Tensor, Tensor, bool, str]], criterion):
    tokens, masks, labels, sentences = data
    
    tokens = batch(tokens)
    masks = batch(masks)
    labels = batch_labels(labels)
    
    analysis = ModelAnalysis(info, criterion)
    
    print("Examples:\n")
    
    counter = 0
    model.eval()
    
    with torch.no_grad():
        for inputs, mask_tensor, expected in zip(tokens, masks, labels): 
            outputs = model(inputs, mask_tensor)            
            analysis.update(outputs, expected, backpropogate=False)
            probabilities = torch.sigmoid(outputs)            

            for probability, label in zip(probabilities, expected):
                counter += 1
                prediction = "True" if probability >= 0.5 else "False"
                actual = "True" if label >= 0.5 else "False"
                match = "Correct" if prediction == actual else "Incorrect"
                print(f"{counter}. \"{sentences[counter - 1]}\"")
                print(f"\tPrediction: {prediction}, Actual: {actual} ({match})")
    
    print(f"Results - {analysis.report()}")

def load(filename):
     with open(filename, "r") as file:
        data = json.load(file)
        return (
            [tensor(series, dtype=torch.int) for series in data["tokens"]],
            [tensor(series, dtype=torch.int) for series in data["masks"]],
            data["labels"],
            data["strings"]
        )

if __name__ == "__main__":
    model = torch.load("model.pth", weights_only=False)
    model.to(device)
    
    test_set = load("data/test.json")
    example_set = load("data/example.json")
            
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    print()
    example(model, example_set, criterion)
    
    print()
    print(f"Validation: {eval(model, test_set, criterion).report()}")