from MrML import *
from info import *
from data_gen import *

def eval(model: LanguageAcceptanceClassifier, data: List[Tuple[List, List, List]], criterion):
    tokens, masks, labels, _ = data
    
    tokens = batch(info, tokens)
    masks = batch(info, masks)
    labels = batch_labels(info, labels)
    analysis = ModelAnalysis(info)
        
    model.eval()
    
    with torch.no_grad():
        for inputs, mask_tensor, expected in zip(tokens, masks, labels):
            outputs = model(inputs, mask_tensor)
            loss = criterion(outputs, expected) # validation loss
            analysis.update(outputs, expected, loss.item())
            
    return analysis

def example(model: LanguageAcceptanceClassifier, data: List[Tuple[Tensor, Tensor, bool, str]], criterion):
    tokens, masks, labels, sentences = data
    
    tokens = batch(info, tokens)
    masks = batch(info, masks)
    labels = batch_labels(info, labels)
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
    model = torch.load("model.pth", weights_only=False).to(device)
    
    test_set = load("data/test.json")
    example_set = load("data/example.json")
            
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    print()
    example(model, example_set, criterion)
    
    print()
    print(f"Validation: {eval(model, test_set, criterion).report()}")