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
import argparse

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

# Configuration for lighter training (adjust these to reduce resource usage)
REDUCED_EPOCHS = 5  # Reduced number of epochs for faster training
SMALLER_BATCH_SIZE = 16  # Smaller batch size to reduce memory usage
REDUCED_WORKERS = 2  # Fewer worker threads

def run_baseline_fashion_mnist(epochs=20, batch_size=64, workers=4):
    """Train and evaluate baseline model on Fashion-MNIST"""
    print("\n=== Training baseline model on Fashion-MNIST ===\n")
    
    # Check if model already exists
    model_path = "models/fashion_mnist_baseline.pth"
    if os.path.exists(model_path):
        print(f"Model already exists at {model_path}. Loading pre-trained model...")
        model = SimpleFashionCNN()
        model = model.to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        
        # Load datasets for evaluation
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        test_dataset = datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
        
        # Evaluate model
        criterion = nn.CrossEntropyLoss()
        results = evaluate_model(model, test_loader, criterion, device=device, class_names=fashion_mnist_classes)
        
        # Create dummy history for return consistency
        history = {
            'train_loss': [0],
            'val_loss': [0],
            'train_acc': [0],
            'val_acc': [0]
        }
        
        return model, results
    
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
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    
    # Create model
    model = SimpleFashionCNN()
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5, verbose=True)
    
    # Train model
    model, history = train_model(
        model, train_loader, val_loader, criterion, optimizer,
        num_epochs=epochs, scheduler=scheduler, device=device, model_path=model_path
    )
    
    # Plot training history
    plot_training_history(history, title="Fashion-MNIST Baseline Training")
    
    # Evaluate model
    results = evaluate_model(model, test_loader, criterion, device=device, class_names=fashion_mnist_classes)
    
    # Visualize predictions
    visualize_predictions(model, test_loader, fashion_mnist_classes, device=device)
    
    return model, results

def prepare_deepfashion(batch_size=32, workers=4):
    """Prepare DeepFashion dataset"""
    print("\n=== Preparing DeepFashion dataset ===\n")
    
    # Define transforms for DeepFashion (or CIFAR-10 substitute)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Check if we should use the actual DeepFashion dataset
    deepfashion_path = './data/DeepFashion/Category_and_Attribute_Prediction_Benchmark'
    use_actual_deepfashion = os.path.exists(deepfashion_path)
    
    if use_actual_deepfashion:
        print("Using actual DeepFashion dataset...")
        
        # Define DeepFashionDataset class
        class DeepFashionDataset(torch.utils.data.Dataset):
            def __init__(self, root_dir, partition, transform=None):
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
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        image_path, label_idx, part = parts[0], parts[1], parts[2]
                        if part == partition:
                            self.image_paths.append(os.path.join(root_dir, 'Img', image_path))
                            self.labels.append(int(label_idx) - 1)  # Convert to 0-indexed
                
                self.num_classes = max(self.labels) + 1 if self.labels else 0
                print(f"Loaded {len(self.image_paths)} images for {partition} partition with {self.num_classes} classes")
            
            def __len__(self):
                return len(self.image_paths)
            
            def __getitem__(self, idx):
                img_path = self.image_paths[idx]
                try:
                    from PIL import Image
                    image = Image.open(img_path).convert('RGB')
                    label = self.labels[idx]
                    
                    if self.transform:
                        image = self.transform(image)
                    
                    return image, label
                except Exception as e:
                    print(f"Error loading image {img_path}: {e}")
                    # Return a placeholder image and the label
                    placeholder = torch.zeros((3, 224, 224))
                    return placeholder, self.labels[idx]
        
        # Create DeepFashion datasets
        train_dataset = DeepFashionDataset(
            root_dir=deepfashion_path,
            partition='train',
            transform=transform
        )
        
        val_dataset = DeepFashionDataset(
            root_dir=deepfashion_path,
            partition='val',
            transform=transform
        )
        
        test_dataset = DeepFashionDataset(
            root_dir=deepfashion_path,
            partition='test',
            transform=transform
        )
        
        num_classes = train_dataset.num_classes
        
        # Mapping actual DeepFashion classes to more fashion-like names
        # Note: You might need to adjust these based on the actual classes
        deepfashion_classes = [
            'Top', 'Trouser', 'T-shirt', 'Dress', 'Coat',
            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Boot'
        ][:num_classes]  # Trim to actual number of classes
    else:
        print("DeepFashion dataset not found. Using CIFAR-10 as a simulated substitute...")
        
        # Load CIFAR-10 as a stand-in
        train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
        
        # Use a subset of CIFAR-10 to simulate DeepFashion
        val_size = 5000
        train_size = len(train_dataset) - val_size
        train_dataset, val_dataset = random_split(train_dataset, [train_size, val_size])
        
        # Mapping CIFAR-10 classes to more fashion-like names for our simulation
        deepfashion_classes = [
            'Top', 'Trouser', 'T-shirt', 'Dress', 'Coat',
            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Boot'
        ]
        
        num_classes = 10  # CIFAR-10 has 10 classes
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=workers)
    
    return train_loader, val_loader, test_loader, deepfashion_classes, num_classes

def train_deepfashion_model(train_loader, val_loader, num_classes, pretrained=True, epochs=20):
    """Train model on DeepFashion dataset"""
    print(f"\n=== Training DeepFashion model (pretrained={pretrained}) ===\n")
    
    # Check if model already exists
    model_path = f"models/deepfashion_{'pretrained' if pretrained else 'scratch'}.pth"
    if os.path.exists(model_path):
        print(f"Model already exists at {model_path}. Loading pre-trained model...")
        model = DeepFashionModel(num_classes, pretrained=False)  # Initialize without pre-trained weights
        model = model.to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        
        # Create dummy history for return consistency
        history = {
            'train_loss': [0],
            'val_loss': [0],
            'train_acc': [0],
            'val_acc': [0]
        }
        
        return model, history
    
    # Create model
    model = DeepFashionModel(num_classes, pretrained=pretrained)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)  # Lower learning rate for pre-trained model
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, factor=0.5, verbose=True)
    
    # Train model
    model, history = train_model(
        model, train_loader, val_loader, criterion, optimizer,
        num_epochs=epochs, scheduler=scheduler, device=device, model_path=model_path
    )
    
    # Plot training history
    plot_training_history(history, title=f"DeepFashion {'Pretrained' if pretrained else 'From Scratch'} Training")
    
    return model, history

def train_adapter_model(train_loader, val_loader, num_classes, pretrained_path=None, unfreeze_after=3, epochs=10):
    """Train adapter model to transfer from Fashion-MNIST to DeepFashion"""
    print(f"\n=== Training Adapter model (pretrained_path={pretrained_path}) ===\n")
    
    # Check if model already exists
    model_path = f"models/fashion_adapter_{'with_pretrained' if pretrained_path else 'scratch'}.pth"
    if os.path.exists(model_path):
        print(f"Model already exists at {model_path}. Loading pre-trained model...")
        model = FashionAdapter(num_classes)  # Initialize without pre-trained weights
        model = model.to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        
        # Create dummy history for return consistency
        history = {
            'train_loss': [0],
            'val_loss': [0],
            'train_acc': [0],
            'val_acc': [0]
        }
        
        return model, history
    
    # Create model
    model = FashionAdapter(num_classes, pretrained_path=pretrained_path)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    
    # Initially only train the adapter layers
    initial_optimizer = optim.Adam(model.adapter.parameters(), lr=0.001)
    
    # Train model's adapter layers first
    print("Training adapter layers only...")
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
        num_epochs=epochs-unfreeze_after, scheduler=scheduler, device=device, model_path=model_path
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
    
    # Check if model files exist
    model_files = {
        'fashion_mnist_baseline': "models/fashion_mnist_baseline.pth",
        'deepfashion_scratch': "models/deepfashion_scratch.pth",
        'deepfashion_pretrained': "models/deepfashion_pretrained.pth",
        'adapter_scratch': "models/fashion_adapter_scratch.pth",
        'adapter_pretrained': "models/fashion_adapter_with_pretrained.pth"
    }
    
    missing_models = [name for name, path in model_files.items() if not os.path.exists(path)]
    if missing_models:
        print(f"Warning: The following models are missing and will be skipped in evaluation: {', '.join(missing_models)}")
    
    # Load models that exist
    if os.path.exists(model_files['deepfashion_scratch']):
        model_deepfashion_scratch = DeepFashionModel(num_classes, pretrained=False)
        model_deepfashion_scratch = model_deepfashion_scratch.to(device)
        model_deepfashion_scratch.load_state_dict(torch.load(model_files['deepfashion_scratch'], map_location=device))
        
        # Evaluate DeepFashion from scratch model
        print("Evaluating DeepFashion from scratch model...")
        results['deepfashion_scratch'] = evaluate_model(
            model_deepfashion_scratch, deepfashion_test_loader, criterion, 
            device=device, class_names=deepfashion_classes
        )
        
        # Visualize predictions
        visualize_predictions(model_deepfashion_scratch, deepfashion_test_loader, 
                             deepfashion_classes, device=device)
    
    if os.path.exists(model_files['deepfashion_pretrained']):
        model_deepfashion_pretrained = DeepFashionModel(num_classes, pretrained=False)
        model_deepfashion_pretrained = model_deepfashion_pretrained.to(device)
        model_deepfashion_pretrained.load_state_dict(torch.load(model_files['deepfashion_pretrained'], map_location=device))
        
        # Evaluate DeepFashion pretrained model
        print("Evaluating DeepFashion pretrained model...")
        results['deepfashion_pretrained'] = evaluate_model(
            model_deepfashion_pretrained, deepfashion_test_loader, criterion,
            device=device, class_names=deepfashion_classes
        )
        
        # Visualize predictions
        visualize_predictions(model_deepfashion_pretrained, deepfashion_test_loader,
                             deepfashion_classes, device=device)
    
    if os.path.exists(model_files['adapter_scratch']):
        model_adapter_scratch = FashionAdapter(num_classes)
        model_adapter_scratch = model_adapter_scratch.to(device)
        model_adapter_scratch.load_state_dict(torch.load(model_files['adapter_scratch'], map_location=device))
        
        # Evaluate Adapter from scratch model
        print("Evaluating Adapter from scratch model...")
        results['adapter_scratch'] = evaluate_model(
            model_adapter_scratch, deepfashion_test_loader, criterion,
            device=device, class_names=deepfashion_classes
        )
        
        # Visualize predictions
        visualize_predictions(model_adapter_scratch, deepfashion_test_loader,
                             deepfashion_classes, device=device)
    
    if os.path.exists(model_files['adapter_pretrained']):
        model_adapter_pretrained = FashionAdapter(num_classes)
        model_adapter_pretrained = model_adapter_pretrained.to(device)
        model_adapter_pretrained.load_state_dict(torch.load(model_files['adapter_pretrained'], map_location=device))
        
        # Evaluate Adapter with pretrained model
        print("Evaluating Adapter with pretrained model...")
        results['adapter_pretrained'] = evaluate_model(
            model_adapter_pretrained, deepfashion_test_loader, criterion,
            device=device, class_names=deepfashion_classes
        )
        
        # Visualize predictions
        visualize_predictions(model_adapter_pretrained, deepfashion_test_loader,
                             deepfashion_classes, device=device)
    
    return results

def compare_results(results):
    """Compare and visualize results from different models"""
    print("\n=== Comparing results from different models ===\n")
    
    if not results:
        print("No results to compare. Please train and evaluate models first.")
        return None
    
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

def main(args):
    """Run the selected parts of the experiment pipeline"""
    print("Starting experiment: From Fashion-MNIST to DeepFashion")
    print(f"Running with settings: {args}")
    
    # Train baseline Fashion-MNIST model if selected
    if args.run_fashion_mnist:
        model_fashion_mnist, fashion_mnist_results = run_baseline_fashion_mnist(
            epochs=args.epochs, batch_size=args.batch_size, workers=args.workers
        )
    
    # Prepare DeepFashion dataset
    if args.run_deepfashion or args.run_adapter or args.run_evaluation:
        train_loader, val_loader, test_loader, deepfashion_classes, num_classes = prepare_deepfashion(
            batch_size=args.batch_size, workers=args.workers
        )
    
    # Train DeepFashion models if selected
    if args.run_deepfashion:
        if args.train_scratch:
            model_deepfashion_scratch, history_deepfashion_scratch = train_deepfashion_model(
                train_loader, val_loader, num_classes, pretrained=False, epochs=args.epochs
            )
        
        if args.train_pretrained:
            model_deepfashion_pretrained, history_deepfashion_pretrained = train_deepfashion_model(
                train_loader, val_loader, num_classes, pretrained=True, epochs=args.epochs
            )
    
    # Train adapter models if selected
    if args.run_adapter:
        if args.train_scratch:
            model_adapter_scratch, history_adapter_scratch = train_adapter_model(
                train_loader, val_loader, num_classes, pretrained_path=None, 
                unfreeze_after=args.unfreeze_after, epochs=args.epochs
            )
        
        if args.train_pretrained and os.path.exists("models/deepfashion_pretrained.pth"):
            model_adapter_pretrained, history_adapter_pretrained = train_adapter_model(
                train_loader, val_loader, num_classes, pretrained_path="models/deepfashion_pretrained.pth",
                unfreeze_after=args.unfreeze_after, epochs=args.epochs
            )
        elif args.train_pretrained:
            print("Warning: Could not find pretrained DeepFashion model. Skipping pretrained adapter training.")
    
    # Evaluate all models if selected
    if args.run_evaluation:
        results = evaluate_all_models(test_loader, deepfashion_classes, num_classes)
        
        # Compare results
        comparison = compare_results(results)
    
    print("Selected experiment parts completed!")

if __name__ == "__main__":
    # Create argument parser
    parser = argparse.ArgumentParser(description='Run Fashion-MNIST to DeepFashion experiment')
    
    # Add arguments for selecting which parts to run
    parser.add_argument('--run_fashion_mnist', action='store_true', help='Run Fashion-MNIST baseline training')
    parser.add_argument('--run_deepfashion', action='store_true', help='Run DeepFashion model training')
    parser.add_argument('--run_adapter', action='store_true', help='Run adapter model training')
    parser.add_argument('--run_evaluation', action='store_true', help='Run model evaluation and comparison')
    parser.add_argument('--run_all', action='store_true', help='Run all parts of the experiment')
    
    # Add arguments for training configuration
    parser.add_argument('--train_scratch', action='store_true', help='Train models from scratch')
    parser.add_argument('--train_pretrained', action='store_true', help='Train models with pretraining')
    parser.add_argument('--train_all', action='store_true', help='Train both scratch and pretrained models')
    
    # Add arguments for training parameters
    parser.add_argument('--epochs', type=int, default=REDUCED_EPOCHS, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=SMALLER_BATCH_SIZE, help='Batch size for training')
    parser.add_argument('--workers', type=int, default=REDUCED_WORKERS, help='Number of worker threads for data loading')
    parser.add_argument('--unfreeze_after', type=int, default=3, help='Epochs before unfreezing backbone in adapter training')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Set defaults if run_all is selected
    if args.run_all:
        args.run_fashion_mnist = True
        args.run_deepfashion = True
        args.run_adapter = True
        args.run_evaluation = True
    
    # Set defaults if train_all is selected
    if args.train_all:
        args.train_scratch = True
        args.train_pretrained = True
    
    # If no parts are selected, run evaluation only
    if not any([args.run_fashion_mnist, args.run_deepfashion, args.run_adapter, args.run_evaluation]):
        print("No specific parts selected. Running model evaluation only.")
        args.run_evaluation = True
    
    # If no training types are selected but training is needed, train both
    if not any([args.train_scratch, args.train_pretrained]) and any([args.run_deepfashion, args.run_adapter]):
        print("No training types selected. Training both scratch and pretrained models.")
        args.train_scratch = True
        args.train_pretrained = True
    
    # Run the main function with parsed arguments
    main(args)