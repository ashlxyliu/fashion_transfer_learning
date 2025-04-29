import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=10, scheduler=None, 
                device='cuda', model_path='model.pth', early_stopping_patience=5):
    since = time.time()
    model = model.to(device)
    
    # Initialize best model weights and best accuracy
    best_model_wts = model.state_dict()
    best_acc = 0.0
    
    # Initialize early stopping counter
    early_stopping_counter = 0
    
    # Lists to track metrics
    train_loss_history = []
    train_acc_history = []
    val_loss_history = []
    val_acc_history = []
    
    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        # Training phase
        model.train()
        running_loss = 0.0
        running_corrects = 0
        
        # Iterate over data
        for inputs, labels in tqdm(train_loader, desc='Training'):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Zero the parameter gradients
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        
        epoch_train_loss = running_loss / len(train_loader.dataset)
        epoch_train_acc = running_corrects.double() / len(train_loader.dataset)
        
        train_loss_history.append(epoch_train_loss)
        train_acc_history.append(epoch_train_acc.item())
        
        print(f'Train Loss: {epoch_train_loss:.4f} Acc: {epoch_train_acc:.4f}')
        
        # Validation phase
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        
        # No gradient calculation needed for validation
        with torch.no_grad():
            for inputs, labels in tqdm(val_loader, desc='Validation'):
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # Forward pass
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
        
        epoch_val_loss = running_loss / len(val_loader.dataset)
        epoch_val_acc = running_corrects.double() / len(val_loader.dataset)
        
        val_loss_history.append(epoch_val_loss)
        val_acc_history.append(epoch_val_acc.item())
        
        print(f'Val Loss: {epoch_val_loss:.4f} Acc: {epoch_val_acc:.4f}')
        
        # Step the scheduler after validation (if it's a ReduceLROnPlateau)
        if scheduler:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(epoch_val_loss)  # Use validation loss
            else:
                scheduler.step()  # For other schedulers
        
        # Check if this is the best model so far
        if epoch_val_acc > best_acc:
            best_acc = epoch_val_acc
            best_model_wts = model.state_dict().copy()
            # Save the model
            torch.save(best_model_wts, model_path)
            print(f'Saved new best model with accuracy: {best_acc:.4f}')
            # Reset early stopping counter
            early_stopping_counter = 0
        else:
            # Increment early stopping counter
            early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_patience:
                print(f'Early stopping triggered after {epoch+1} epochs')
                break
        
        print()
    
    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:.4f}')
    
    # Load best model weights
    model.load_state_dict(best_model_wts)
    
    # Return model and history
    history = {
        'train_loss': train_loss_history,
        'val_loss': val_loss_history,
        'train_acc': train_acc_history,
        'val_acc': val_acc_history
    }
    
    return model, history

# Function to evaluate a model
def evaluate_model(model, test_loader, criterion, device='cuda', class_names=None):
    model = model.to(device)
    model.eval()
    
    running_loss = 0.0
    running_corrects = 0
    
    # Lists to store predictions and true labels
    all_preds = []
    all_labels = []
    
    # No gradient calculation needed for evaluation
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Evaluation'):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            
            # Save predictions and labels
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    test_loss = running_loss / len(test_loader.dataset)
    test_acc = running_corrects.double() / len(test_loader.dataset)
    
    print(f'Test Loss: {test_loss:.4f} Acc: {test_acc:.4f}')
    
    # Generate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    if class_names:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    else:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix - Test Accuracy: {test_acc:.4f}')
    plt.tight_layout()
    
    # Print classification report
    if class_names:
        print(classification_report(all_labels, all_preds, target_names=class_names))
    else:
        print(classification_report(all_labels, all_preds))
    
    # Return metrics
    results = {
        'test_loss': test_loss,
        'test_acc': test_acc.item(),
        'confusion_matrix': cm,
        'predictions': all_preds,
        'true_labels': all_labels
    }
    
    return results

# Function to plot training history
def plot_training_history(history, title='Training History'):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot loss
    ax1.plot(history['train_loss'], label='Train Loss')
    ax1.plot(history['val_loss'], label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot accuracy
    ax2.plot(history['train_acc'], label='Train Accuracy')
    ax2.plot(history['val_acc'], label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(f"results/{title.replace(' ', '_')}.png")
    plt.show()

# Function to visualize predictions
def visualize_predictions(model, dataloader, class_names, device='cuda', num_images=25, normalize=True):
    model = model.to(device)
    model.eval()
    
    data_iter = iter(dataloader)
    images, labels = next(data_iter)
    
    with torch.no_grad():
        images = images.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
    
    # Move tensors to CPU for plotting
    images = images.cpu()
    preds = preds.cpu().numpy()
    labels = labels.numpy()
    
    # Plot images with predictions
    fig = plt.figure(figsize=(15, 15))
    
    for i in range(min(num_images, len(images))):
        ax = fig.add_subplot(5, 5, i + 1, xticks=[], yticks=[])
        
        # Convert tensor to image
        if normalize:
            img = images[i].permute(1, 2, 0)
            img = img * 0.5 + 0.5  # Unnormalize
        else:
            img = images[i].permute(1, 2, 0)
        
        # Handle grayscale images
        if img.shape[2] == 1:
            img = img.squeeze()
            plt.gray()
            
        plt.imshow(img)
        
        # Set color based on correctness
        if preds[i] == labels[i]:
            color = 'green'
        else:
            color = 'red'
            
        ax.set_title(f'P: {class_names[preds[i]]}\nT: {class_names[labels[i]]}', 
                    color=color)
    
    plt.tight_layout()
    plt.savefig(f"results/predictions_{model.__class__.__name__}.png")
    plt.show()