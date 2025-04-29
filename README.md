# From Fashion-MNIST to DeepFashion

A machine learning project that quantitatively and visually demonstrates how starting with a simple but off domain dataset (Fashion-MNIST) impacts performance on real world fashion classification tasks, and how different pre-training strategies can help close this gap. This project shows the transfer learning gap between the simplified Fashion-MNIST dataset and more complex DeepFashion dataset. It implements and compares several model architectures and training strategies:

1. **Fashion-MNIST Baseline**: A simple CNN trained on Fashion-MNIST
2. **DeepFashion From Scratch**: A ResNet-based model trained from scratch on DeepFashion
3. **DeepFashion with ImageNet**: A ResNet-based model pre-trained on ImageNet, fine-tuned on DeepFashion
4. **Adapter From Scratch**: A custom model that adapts Fashion-MNIST features to DeepFashion, trained from scratch
5. **Adapter with Pre-training**: A custom model that adapts Fashion-MNIST features to DeepFashion, using pre-trained weights


## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/fashion-transfer-learning.git
cd fashion-transfer-learning
```

2. Install the required dependencies:
```bash
pip install torch torchvision tqdm matplotlib seaborn pandas scikit-learn numpy
```

3. Download the DeepFashion dataset:
   - Go to [DeepFashion](http://mmlab.ie.cuhk.edu.hk/projects/DeepFashion.html)
   - Download the "Category and Attribute Prediction Benchmark" subset
   - Extract the files to `./data/DeepFashion/Category_and_Attribute_Prediction_Benchmark/`

## How to Run

### Using the Standard Script

The `main_script.py` file runs the entire experiment pipeline sequentially:

```bash
python main_script.py
```

This will:
1. Train the Fashion-MNIST baseline model
2. Prepare the DeepFashion dataset
3. Train DeepFashion models (from scratch and with pre-training)
4. Train adapter models (from scratch and with pre-training)
5. Evaluate all models and compare results

### Using the Optimized Script

The `optimized_main_script.py` file allows more control over which parts of the experiment to run:

```bash
# To run only the Fashion-MNIST baseline model
python optimized_main_script.py --run_fashion_mnist

# To run only the DeepFashion models
python optimized_main_script.py --run_deepfashion --train_all

# To run only the adapter models
python optimized_main_script.py --run_adapter --train_all

# To evaluate all models
python optimized_main_script.py --run_evaluation

# To run everything with optimized settings for slower machines
python optimized_main_script.py --run_all --epochs 5 --batch_size 16 --workers 2
```

### Command-line Arguments

The optimized script supports the following arguments:

| Argument | Description |
|----------|-------------|
| `--run_fashion_mnist` | Run Fashion-MNIST baseline training |
| `--run_deepfashion` | Run DeepFashion model training |
| `--run_adapter` | Run adapter model training |
| `--run_evaluation` | Run model evaluation and comparison |
| `--run_all` | Run all parts of the experiment |
| `--train_scratch` | Train models from scratch |
| `--train_pretrained` | Train models with pre-training |
| `--train_all` | Train both scratch and pre-trained models |
| `--epochs` | Number of training epochs (default: 5) |
| `--batch_size` | Batch size for training (default: 16) |
| `--workers` | Number of worker threads for data loading (default: 2) |
| `--unfreeze_after` | Epochs before unfreezing backbone in adapter training (default: 3) |

## Model Architectures

### SimpleFashionCNN
A basic CNN designed for Fashion-MNIST classification with three convolutional layers followed by fully connected layers.

### DeepFashionModel
A model for DeepFashion classification using a pre-trained ResNet backbone with a custom classifier head.

### FashionAdapter
An adapter architecture that bridges the domain gap between Fashion-MNIST and DeepFashion with:
- A preprocessing module that converts grayscale 28×28 images to RGB 224×224
- A feature extraction backbone (which can be pre-trained)
- A classification head

## Acknowledgments

- Fashion-MNIST dataset by Zalando Research
- DeepFashion dataset by Multimedia Laboratory, The Chinese University of Hong Kong
- PyTorch and torchvision libraries