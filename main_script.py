import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split

# Import our custom modules
from model_architecture import SimpleFashionCNN, DeepFashionModel, FashionAdapter
from training_functions import train_model, evaluate_model, plot_training_history, visualize_predictions

# Set random seeds for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create directories for saving models and results
os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

# Fashion-MNIST class names
fashion_mnist_classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                         'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Add this class implementation
class DeepFashionDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, partition, transform=None):
        """
        Args:
            root_dir (string): Directory with the DeepFashion dataset.
            partition (string): 'train', 'val', or 'test'.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform
        
        # Load list file
        list_file = os.path.join(root_dir, 'Eval', 'list_eval_partition.txt')
        with open(list_file, 'r') as f:
            lines = f.readlines()[2:]  # Skip header lines
        
        self.image_paths = []
        self.labels = []
        
        # Parse list file
        for line in lines:
            image_path, label_idx, part = line.strip().split()
            if part == partition:
                self.image_paths.append(os.path.join(root_dir, 'Img', image_path))
                self.labels.append(int(label_idx) - 1)  # Convert to 0-indexed
                
        self.num_classes = max(self.labels) + 1
        print(f"Loaded {len(self.image_paths)} images for {partition} partition with {self.num_classes} classes")
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = plt.imread(img_path)
        label = self.labels[idx]
        
        # Convert to PIL Image for transformations
        image = transforms.ToPILImage()(image)
        
        if self.transform:
            image = self.transform(image)
            
        return image, label
    
def run_baseline_fashion_mnist():
    """Train and evaluate baseline model on Fashion-MNIST"""
    print("\n=== Training baseline model on Fashion-MNIST ===\n")
    
    # Define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    # Load datasets
    train_dataset = datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
    
    # Split training set into train and validation
    val_size = 10000
    train_size = len(train_dataset) - val_size
    train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
    
    # Create data loaders
    batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Create model
    model = SimpleFashionCNN()
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5, verbose=True)
    
    # Train model
    model_path = "models/fashion_mnist_baseline.pth"
    model, history = train_model(
        model, train_loader, val_loader, criterion, optimizer,
        num_epochs=20, scheduler=scheduler, device=device, model_path=model_path
    )
    
    # Plot training history
    plot_training_history(history, title="Fashion-MNIST Baseline Training")
    
    # Evaluate model
    results = evaluate_model(model, test_loader, criterion, device=device, class_names=fashion_mnist_classes)
    
    # Visualize predictions
    visualize_predictions(model, test_loader, fashion_mnist_classes, device=device)
    
    return model, results

def prepare_deepfashion():
    """Prepare DeepFashion dataset"""
    print("\n=== Preparing DeepFashion dataset ===\n")
    
    # Define path to DeepFashion dataset
    DEEPFASHION_ROOT = './data/DeepFashion/Category_and_Attribute_Prediction_Benchmark'
    
    # Define transforms for DeepFashion
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create DeepFashion datasets
    train_dataset = DeepFashionDataset(
        root_dir=DEEPFASHION_ROOT,
        partition='train',
        transform=transform
    )
    
    val_dataset = DeepFashionDataset(
        root_dir=DEEPFASHION_ROOT,
        partition='val',
        transform=transform
    )
    
    test_dataset = DeepFashionDataset(
        root_dir=DEEPFASHION_ROOT,
        partition='test',
        transform=transform
    )
    
    # Create data loaders
    batch_size = 32  # Smaller batch size for larger images
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Define class names that correspond to Fashion-MNIST
    # Note: You might need to adjust these based on the actual DeepFashion classes
    deepfashion_classes = [
        'Top', 'Trouser', 'T-shirt', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Boot'
    ]
    
    num_classes = train_dataset.num_classes
    
    return train_loader, val_loader, test_loader, deepfashion_classes, num_classes

def train_deepfashion_model(train_loader, val_loader, num_classes, pretrained=True):
    """Train model on DeepFashion dataset"""
    print(f"\n=== Training DeepFashion model (pretrained={pretrained}) ===\n")
    
    # Create model
    model = DeepFashionModel(num_classes, pretrained=pretrained)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Lower learning rate for pre-trained model
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5, verbose=True)
    
    # Train model
    model_path = f"models/deepfashion_{'pretrained' if pretrained else 'scratch'}.pth"
    model, history = train_model(
        model, train_loader, val_loader, criterion, optimizer,
        num_epochs=20, scheduler=scheduler, device=device, model_path=model_path
    )
    
    # Plot training history
    plot_training_history(history, title=f"DeepFashion {'Pretrained' if pretrained else 'From Scratch'} Training")
    
    return model, history

def train_adapter_model(train_loader, val_loader, num_classes, pretrained_path=None, unfreeze_after=5):
    """Train adapter model to transfer from Fashion-MNIST to DeepFashion"""
    print(f"\n=== Training Adapter model (pretrained_path={pretrained_path}) ===\n")
    
    # Create model
    model = FashionAdapter(num_classes, pretrained_path=pretrained_path)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    
    # Initially only train the adapter layers
    initial_optimizer = optim.Adam(model.adapter.parameters(), lr=0.001)
    
    # Train model's adapter layers first
    print("Training adapter layers only...")
    model_path = f"models/fashion_adapter_{'with_pretrained' if pretrained_path else 'scratch'}.pth"
    model, initial_history = train_model(
        model, train_loader, val_loader, criterion, initial_optimizer,
        num_epochs=unfreeze_after, device=device, model_path=model_path
    )
    
    # Unfreeze backbone and continue training with a lower learning rate
    print("Unfreezing backbone and continuing training...")
    model.unfreeze_backbone()
    full_optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(full_optimizer, 'min', patience=2, factor=0.5, verbose=True)
    
    model, full_history = train_model(
        model, train_loader, val_loader, criterion, full_optimizer,
        num_epochs=15, scheduler=scheduler, device=device, model_path=model_path
    )
    
    # Combine histories
    combined_history = {
        'train_loss': initial_history['train_loss'] + full_history['train_loss'],
        'val_loss': initial_history['val_loss'] + full_history['val_loss'],
        'train_acc': initial_history['train_acc'] + full_history['train_acc'],
        'val_acc': initial_history['val_acc'] + full_history['val_acc']
    }
    
    # Plot training history
    plot_training_history(combined_history, 
                         title=f"Fashion-MNIST to DeepFashion Adapter {'With Pretrained' if pretrained_path else 'From Scratch'}")
    
    return model, combined_history

def evaluate_all_models(deepfashion_test_loader, deepfashion_classes, num_classes):
    """Evaluate all models on DeepFashion test set and compare results"""
    print("\n=== Evaluating all models on DeepFashion test set ===\n")
    
    criterion = nn.CrossEntropyLoss()
    results = {}
    
    # Load models
    model_fashion_mnist = SimpleFashionCNN()
    model_fashion_mnist = model_fashion_mnist.to(device)
    model_fashion_mnist.load_state_dict(torch.load("models/fashion_mnist_baseline.pth"))
    
    model_deepfashion_scratch = DeepFashionModel(num_classes, pretrained=False)
    model_deepfashion_scratch = model_deepfashion_scratch.to(device)
    model_deepfashion_scratch.load_state_dict(torch.load("models/deepfashion_scratch.pth"))
    
    model_deepfashion_pretrained = DeepFashionModel(num_classes, pretrained=True)
    model_deepfashion_pretrained = model_deepfashion_pretrained.to(device)
    model_deepfashion_pretrained.load_state_dict(torch.load("models/deepfashion_pretrained.pth"))
    
    model_adapter_scratch = FashionAdapter(num_classes)
    model_adapter_scratch = model_adapter_scratch.to(device)
    model_adapter_scratch.load_state_dict(torch.load("models/fashion_adapter_scratch.pth"))
    
    model_adapter_pretrained = FashionAdapter(num_classes)
    model_adapter_pretrained = model_adapter_pretrained.to(device)
    model_adapter_pretrained.load_state_dict(torch.load("models/fashion_adapter_with_pretrained.pth"))
    
    # Can't directly evaluate Fashion-MNIST model on DeepFashion test set due to different input sizes
    # and channels, so we'll skip that comparison
    
    # Evaluate DeepFashion from scratch model
    print("Evaluating DeepFashion from scratch model...")
    results['deepfashion_scratch'] = evaluate_model(
        model_deepfashion_scratch, deepfashion_test_loader, criterion, 
        device=device, class_names=deepfashion_classes
    )
    
    # Evaluate DeepFashion pretrained model
    print("Evaluating DeepFashion pretrained model...")
    results['deepfashion_pretrained'] = evaluate_model(
        model_deepfashion_pretrained, deepfashion_test_loader, criterion,
        device=device, class_names=deepfashion_classes
    )
    
    # Evaluate Adapter from scratch model
    print("Evaluating Adapter from scratch model...")
    results['adapter_scratch'] = evaluate_model(
        model_adapter_scratch, deepfashion_test_loader, criterion,
        device=device, class_names=deepfashion_classes
    )
    
    # Evaluate Adapter with pretrained model
    print("Evaluating Adapter with pretrained model...")
    results['adapter_pretrained'] = evaluate_model(
        model_adapter_pretrained, deepfashion_test_loader, criterion,
        device=device, class_names=deepfashion_classes
    )
    
    # Visualize predictions for all models
    visualize_predictions(model_deepfashion_scratch, deepfashion_test_loader, 
                         deepfashion_classes, device=device)
    visualize_predictions(model_deepfashion_pretrained, deepfashion_test_loader,
                         deepfashion_classes, device=device)
    visualize_predictions(model_adapter_scratch, deepfashion_test_loader,
                         deepfashion_classes, device=device)
    visualize_predictions(model_adapter_pretrained, deepfashion_test_loader,
                         deepfashion_classes, device=device)
    
    return results

def compare_results(results):
    """Compare and visualize results from different models"""
    print("\n=== Comparing results from different models ===\n")
    
    # Extract accuracies
    models = list(results.keys())
    accuracies = [results[model]['test_acc'] for model in models]
    
    # Bar plot
    plt.figure(figsize=(12, 6))
    sns.barplot(x=models, y=accuracies)
    plt.title('Test Accuracy Comparison')
    plt.ylabel('Accuracy')
    plt.xlabel('Model')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("results/model_comparison.png")
    plt.show()
    
    # Create comparison table
    comparison_df = pd.DataFrame({
        'Model': models,
        'Accuracy': accuracies
    })
    
    print(comparison_df.sort_values('Accuracy', ascending=False))
    
    # Save comparison table
    comparison_df.to_csv("results/model_comparison.csv", index=False)
    
    return comparison_df

def main():
    """Run the full experiment pipeline"""
    print("Starting experiment: From Fashion-MNIST to DeepFashion")
    
    # Train baseline Fashion-MNIST model
    model_fashion_mnist, fashion_mnist_results = run_baseline_fashion_mnist()
    
    # Prepare DeepFashion dataset
    train_loader, val_loader, test_loader, deepfashion_classes, num_classes = prepare_deepfashion()
    
    # Train DeepFashion models
    model_deepfashion_scratch, history_deepfashion_scratch = train_deepfashion_model(
        train_loader, val_loader, num_classes, pretrained=False
    )
    
    model_deepfashion_pretrained, history_deepfashion_pretrained = train_deepfashion_model(
        train_loader, val_loader, num_classes, pretrained=True
    )
    
    # Train adapter models
    model_adapter_scratch, history_adapter_scratch = train_adapter_model(
        train_loader, val_loader, num_classes, pretrained_path=None
    )
    
    model_adapter_pretrained, history_adapter_pretrained = train_adapter_model(
        train_loader, val_loader, num_classes, pretrained_path="models/deepfashion_pretrained.pth"
    )
    
    # Evaluate all models
    results = evaluate_all_models(test_loader, deepfashion_classes, num_classes)
    
    # Compare results
    comparison = compare_results(results)
    
    print("Experiment completed!")
    
if __name__ == "__main__":
    main()