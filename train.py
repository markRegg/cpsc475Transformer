import os
import argparse
from MrML import *
from info import *
from data_gen import *
from eval import eval

epoch_count = 0
num_epochs = 100
min_epochs = 10
dropout = 0.1
dropout_start = 2
dropout_end = 20
early_stop_allowed = True

def train(
    model: LanguageAcceptanceClassifier, 
    train_set: SentenceDataDict, 
    test_set: SentenceDataDict, 
    criterion, 
    optimizer, 
    scheduler, 
    early_stopping
):
    global epoch_count, num_epochs, min_epochs, dropout, dropout_start, dropout_end, early_stop_allowed
    analysis = ModelAnalysis(info)
        
    for epoch in range(num_epochs):
        if epoch == dropout_start:
            model.set_dropout_rate(dropout)
        if epoch == dropout_end:
            model.set_dropout_rate(0.0)
            
        epoch_count = epoch
        model.train()
        analysis.reset()
        
        for _, tokens, masks, labels in train_set.epoch():
            optimizer.zero_grad()
            outputs = model(tokens, masks)
            loss = criterion(outputs, labels)  # Training loss
            analysis.update(outputs, labels, loss.item())
            loss.backward()  # Backpropagate training loss
            optimizer.step()  # Update model parameters
            
        scheduler.step()
        
        print(f"Epoch {epoch + 1:>4} - {analysis.report()}")
        
        val_loss = eval(model, test_set, criterion)
        print(f"Validation - {val_loss.report()}\n")
        
        if early_stop_allowed and epoch > min_epochs and early_stopping(val_loss.loss()):
            print(f"Early Stop (Loss: {loss})")
            torch.save(model, 'saved_models/model-saved-at-exit.pth')
            print("Saved model\n")
            break
        elif epoch % 2 == 0:
            filename = f"saved_models/model-epoch-{epoch}.pth"
            torch.save(model, filename)
            print(f"Saved model at epoch {epoch} to {filename}\n")
    
    filename = f"saved_models/model-epoch-{epoch}.pth"
    torch.save(model, filename)
    print(f"Saved finished model at epoch {epoch} to {filename}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Training Script")

    parser.add_argument('-d', '--dropout', type=float, default=0.1, required=False, help='Dropout rate')
    parser.add_argument('-S', '--dropout-start', type=int, default=2, required=False, help='The first batch to use a dropout on')
    parser.add_argument('-E', '--dropout-end', type=int, default=20, required=False, help='Last epoch number to use dropout on')
    parser.add_argument('-l', '--learn-rate', type=float, default=0.05, required=False, help='Initial learning rate')
    parser.add_argument('-m', '--min-epochs', type=int, default=20, required=False, help='Min. number of epochs to train the model for')
    parser.add_argument('-e', '--epochs', type=int, default=100, required=False, help='Max. number of epochs to train the model for')
    parser.add_argument('-n', '--new-data', action='store_true', required=False, help='Creates new train/test sets and a new model')
    parser.add_argument('-x', '--no-early-stop', action='store_true', required=False, help="Won't stop model training before max. epochs reached")
    parser.add_argument('-p', '--model-path', type=str, default=None, required=False, help="Path to load saved model from (don't use this to start from new model)")

    args = parser.parse_args()
    
    if args.new_data or args.model_path is None or not os.path.exists(args.model_path):
        model = LanguageAcceptanceClassifier(info, N_LAYERS, N_HEADS, dropout=0).to(device)
        torch.save(model, "saved_models/model-start.pth")
    else:
        model = torch.load(args.model_path, weights_only=False).to(device)
    
    dropout = args.dropout
    dropout_start = args.dropout_start
    dropout_end = args.dropout_end
    min_epochs = args.min_epochs
    max_epochs = args.epochs
    early_stop_allowed = not args.no_early_stop
    
    train_set, test_set = load_train_test_split(info, BATCH_SIZE * 1500, make_new=args.new_data, device=device)
    
    print("\nTrain set class distribution:")
    train_set.class_distribution()
    
    print("\nTest set class distribution:")
    train_set.class_distribution()
    
    optimizer = torch.optim.Adam(model.parameters(), args.learn_rate)
            
    print()
    
    train(
        model           = model,
        train_set       = train_set,
        test_set        = test_set,
        criterion       = nn.BCEWithLogitsLoss(),
        optimizer       = optimizer,
        scheduler       = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_epochs, eta_min=1e-4),
        early_stopping  = EarlyStopping(patience=args.min_epochs, delta=1e-5)
    )