import torch
import torch.nn as nn
import torch.optim as optim
from dataloader_rgbd import *
from temporal_model_1 import *
from torch.utils.data.dataset import random_split
#from zoedepth.utils.misc import compute_metrics
from loss import *

def train_model(model, train_loader, val_loader, epochs, criterion, optimizer, device):
    model.to(device)
    best_val_loss = float('inf')
    best_model_path = 'best_model2.pth' 

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for i, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(inputs)

            # Compute loss
            loss = criterion(outputs, targets)

            
            
            # Zero the parameter gradients
            optimizer.zero_grad()

            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            # Print statistics
            running_loss += loss.item()
            #print(compute_errors(targets.cpu().detach().numpy(), outputs.cpu().detach().numpy()))
            if i % 10 == 9:  # Print every 10 mini-batches
                print(f'Epoch {epoch + 1}, Batch {i + 1}, Training Loss: {running_loss / 10}')
                running_loss = 0.0

        # Validation phase
        val_loss = 0.0
        model.eval()  # Set model to evaluate mode
        with torch.no_grad():
            for i, (inputs, targets) in enumerate(val_loader):
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                val_loss += loss.item()

        avg_val_loss = val_loss / i
        print(f'Epoch {epoch + 1}, Validation Loss: {avg_val_loss}')
        
        # Save the best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), best_model_path)
            print(f'Saved new best model at Epoch {epoch + 1}')

    print('Finished Training')
    return best_model_path  # Returns the path to the best model for later use

if __name__ == "__main__":
    # Load your dataset
    full_dataset = KittiTemporalRGBDDataset('../train_test_inputs/train_files_custom.txt', n_frames=7)
    
    # Split dataset into train and validation sets
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Create dataloaders for train and validation sets
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8, shuffle=True)
    
    # Initialize model, loss function, optimizer
    model = Simplified3DCNN()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-5)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f'Training on {device}')

    # Train the model and save the best performing model
    epochs = 10
    best_model_path = train_model(model, train_loader, val_loader, epochs, criterion, optimizer, device)
    
    # Load the best model for evaluation or further training later
    # best_model = Simplified3DCNN()
    # best_model.load_state_dict(torch.load(best_model_path))
    # best_model.to(device)