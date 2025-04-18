import os
import argparse
from MrML import *
from info import *
from data_gen import *

def eval(model: LanguageAcceptanceClassifier, test_set: SentenceDataDict, criterion):
    analysis = ModelAnalysis(info)
    model.eval()
    
    with torch.no_grad():            
        for _, tokens, masks, labels in test_set.epoch():
            outputs = model(tokens, masks)
            loss = criterion(outputs, labels)  # Validation loss
            analysis.update(outputs, labels, loss.item())
            
    return analysis

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training Script")
    parser.add_argument('-p', '--model-path', type=str, default=None, required=False, help="Path to load saved model from (don't use this to start from new model)")
    args = parser.parse_args()

    if args.model_path is not None and os.path.exists(args.model_path):
        model = torch.load(args.model_path, weights_only=False).to(device)
    else:
        model = LanguageAcceptanceClassifier(info, N_LAYERS, N_HEADS, dropout=0).to(device)
        torch.save(model, "saved_models/model.pth")
                
    print()
    eval(model=model, test_set=load_test_set(device), criterion=nn.BCEWithLogitsLoss())