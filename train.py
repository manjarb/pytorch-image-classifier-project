import argparse
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

import utility
import model_classifier

parser = argparse.ArgumentParser()
parser.add_argument('data_dir', action='store', help='directory containing images')
parser.add_argument('--save_dir', action='store', help='save trained checkpoint to this directory' )
parser.add_argument('--arch', action='store', help='Choose architecture', default='vgg16')
parser.add_argument('--learning_rate', action='store', default=0.01)
parser.add_argument('--hidden_units', action='store', default=512)
parser.add_argument('--output_size', action='store', type=float, default=102)
parser.add_argument('--epochs', action='store', default=5)
parser.add_argument('--gpu', action='store_true')


args = parser.parse_args()
data_dir = args.data_dir
print(data_dir)

# Get datasets
train_datasets, valid_datasets, test_datasets, train_loaders, valid_loaders, test_loaders = utility.load_data(data_dir, args.gpu)

# Create model
model = utility.get_pretrained_model(args.arch)
for param in model.parameters():
    # Freeze pretrained_model'param to avoid backpropagation 
    param.requires_grad = False
    
#calculate input size into the network classifier
input_size = utility.get_input_size(model, args.arch)
    
# Classify network
model.classifier = model_classifier.Classifier(input_size, args.output_size, [args.hidden_units, int(args.hidden_units / 2)], 0.2)

#define the loss function and the optimization parameters
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

current_device = model_classifier.get_current_device(args.gpu)

# Train Model
model_classifier.train_classifier(
    model, 
    args.epochs, 
    criterion, 
    optimizer, 
    train_loaders, 
    valid_loaders, 
    current_device
)

# Test Model
test_loss, accuracy = model_classifier.get_accuracy_loss(model, criterion, test_loaders, current_device)

print("Test Accuracy: {:.3f}".format(accuracy/len(test_loaders)))
print("Test Loss: {:.3f}".format(test_loss/len(test_loaders)))

file_name = utility.save_checkpoint(model, train_datasets)
print("file_name: {}".format(file_name))