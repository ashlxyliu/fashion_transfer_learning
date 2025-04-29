import torch
import torch.nn as nn
import torch.nn.functional as F

# Simple CNN model for Fashion-MNIST (grayscale, 28x28)
class SimpleFashionCNN(nn.Module):
    def __init__(self):
        super(SimpleFashionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, 10)  # 10 classes in Fashion-MNIST
        self.dropout = nn.Dropout(0.25)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  # -> 32x14x14
        x = self.pool(F.relu(self.conv2(x)))  # -> 64x7x7
        x = self.pool(F.relu(self.conv3(x)))  # -> 128x3x3
        x = x.view(-1, 128 * 3 * 3)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# Transfer model for DeepFashion (RGB, 224x224)
class DeepFashionModel(nn.Module):
    def __init__(self, num_classes, pretrained=True, model_type="resnet18"):
        super(DeepFashionModel, self).__init__()
        
        # Model options
        if model_type == "resnet18":
            self.backbone = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=pretrained)
            feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()  # Remove final FC layer
        elif model_type == "resnet50":
            self.backbone = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=pretrained)
            feature_dim = self.backbone.fc.in_features
            self.backbone.fc = nn.Identity()  # Remove final FC layer
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
            
        # Custom classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features)
        return output

# Adapter model for Fashion-MNIST to DeepFashion transfer
class FashionAdapter(nn.Module):
    def __init__(self, num_classes, pretrained_path=None):
        super(FashionAdapter, self).__init__()
        
        # CNN for preprocessing grayscale to RGB
        self.adapter = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Upsample(size=(224, 224), mode='bilinear', align_corners=True)
        )
        
        # DeepFashion model
        self.deepfashion_model = DeepFashionModel(num_classes, pretrained=False)
        
        # Load pre-trained weights if provided
        if pretrained_path:
            self.deepfashion_model.load_state_dict(torch.load(pretrained_path))
            
        # Freeze DeepFashion model parameters for initial training
        for param in self.deepfashion_model.parameters():
            param.requires_grad = False
            
    def forward(self, x):
        # Convert grayscale to RGB and resize
        x = self.adapter(x)
        # Pass through DeepFashion model
        x = self.deepfashion_model(x)
        return x
    
    def unfreeze_backbone(self):
        # Unfreeze DeepFashion model parameters for fine-tuning
        for param in self.deepfashion_model.parameters():
            param.requires_grad = True