Here's a simple, straightforward README for your project based on the code you uploaded. 

# Cat vs Dog Image Classifier

A convolutional neural network (CNN) built using PyTorch for classifying images as either a cat, a dog, or none of the two classes.

## Features

* Custom CNN architecture
* GPU support using CUDA
* Data augmentation during training
* Batch Normalization
* GELU activation function
* Dropout regularization
* AdamW optimizer
* Train/Test dataset split
* Model checkpoint saving

## Dataset

The project expects the following directory structure:

```text
archive/
├── cats_set/
│   ├── cat1.jpg
│   ├── cat2.jpg
│   └── ...
│
└── dogs_set/
    ├── dog1.jpg
    ├── dog2.jpg
    └── ...
```

Images are automatically labeled based on their filenames.

## Model Architecture

The network consists of:

* 5 Convolutional Layers
* Batch Normalization after each convolution
* GELU activation
* Max Pooling layers
* Dropout layer
* Fully Connected output layer

Output classes:

```text
Cat
Dog
None
```

## Data Augmentation

The following augmentations are applied during training:

* Random Resized Crop
* Random Horizontal Flip
* Random Rotation
* Random Affine Transform
* Random Perspective Transform
* Normalization

## Training

The dataset is shuffled and split into:

```text
85% Training
15% Testing
```

Training uses:

```text
Optimizer: AdamW
Learning Rate: 0.002
Weight Decay: 1e-4
Epochs: 20
```

## Running

Install dependencies:

```bash
pip install torch torchvision matplotlib onnx onnxscript
```

Run training:

```bash
python CNN.py
```

Example output:

```text
Training Accuracy 0.98
Epoch : 15 Loss : 1.27

Testing Accuracy : 80.6
Average Test Loss : 0.38
```

## Saved Model

After training, the model weights are saved as:

```text
CNN.pth
```

## Future Improvements

Possible extensions:

* Knowledge Distillation
* Ensemble Models
* ResNet / EfficientNet Teachers
* Biomedical Image Classification
* ONNX Export
* Gradio Web Interface
* Multi-label Classification
* Self-Supervised Pretraining (SimCLR)

## Frameworks Used

* PyTorch
* Torchvision
* ONNX
* CUDA (optional)

## Author

Adithya Krishna

This project was developed as a learning exercise for convolutional neural networks, data augmentation, and image classification using PyTorch.
