from MrML import *
from info import *
from data_gen import *
from validate import eval

def train(model: LanguageAcceptanceClassifier, data: Tuple[List, List, List, List], criterion, optimizer, scheduler, early_stopping, test_set, max_num_epochs: int = 10):
    tokens, masks, labels, _ = shuffle(data)
    
    tokens = batch(info, tokens)
    masks = batch(info, masks)
    labels = batch_labels(info, labels)
    analysis = ModelAnalysis(info)

    print(f"Training: Each epoch will contain {len(labels)} batches\n")
        
    for epoch in range(max_num_epochs):
        model.train()
        analysis.reset()
        
        for inputs, mask_tensor, expected in zip(tokens, masks, labels):
            optimizer.zero_grad()
            outputs = model(inputs, mask_tensor)
            loss = criterion(outputs, expected)  # Training loss
            loss.backward()  # Backpropagate training loss
            optimizer.step()  # Update model parameters
    
            analysis.update(outputs, expected, loss.item())
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 100.0)
        
        if early_stopping(analysis.loss()):
            print(f"Early Stop (Loss: {loss})")
            torch.save(model, 'model.pth')
            print("Saved model\n")
            break
            
        scheduler.step()
        
        print(f"Epoch {epoch + 1:>4} - {analysis.report()}")
        
        val_loss = eval(model, test_set, criterion)
        print(f"Validation - {val_loss.report()}")
        
        print()
        # for param_group in optimizer.param_groups:
        #     print(f"Current LR: {param_group['lr']}")
                    
        # for name, param in model.named_parameters():
        #     if param.grad is not None:
        #         print(f"{name} grad norm: {param.grad.norm().item()}")
                
        # print()
        
        if epoch % 5 == 0:
            torch.save(model, 'model.pth')
            print("Saved model\n")
            
    torch.save(model, 'model.pth')
    print("Saved model\n")
    
def print_grad(grad):
    print("Gradient received:", grad.norm())

if __name__ == "__main__":
    model = LanguageAcceptanceClassifier(info, N_LAYERS, N_HEADS, dropout=0.1).to(device)
    
    train_set = generate_dataset("data/train.json", BATCH_SIZE * 1500)
    test_set = generate_dataset("data/test.json", BATCH_SIZE * 500)
    example_set = generate_dataset("data/example.json", BATCH_SIZE * 1)
            
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.003)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)
    early_stopping = EarlyStopping(patience=10, delta=1e-5)
    
    print()
    train(model, train_set, criterion, optimizer, scheduler, early_stopping, test_set, max_num_epochs=400)
