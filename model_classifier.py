import matplotlib.pyplot as plt
import datetime
import numpy as np

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils import data
from torchvision import datasets, transforms, models
from torch.autograd import Variable
import time
import random
import os
from workspace_utils import active_session
from PIL import Image


class Classifier(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, drop_out=0.2):
        super().__init__()
        
        # Input Layer
        self.hidden_layers = nn.ModuleList([nn.Linear(input_size, hidden_layers[0])])
        
        # Hidden Layers
        h_layers = zip(hidden_layers[:-1], hidden_layers[1:])
        self.hidden_layers.extend([nn.Linear(h_input, h_output) for h_input, h_output in h_layers])
        
        # Output Layer
        self.output_layer = nn.Linear(hidden_layers[-1], output_size)
        
        # Dropout module with drop_out drop probability
        self.dropout = nn.Dropout(p=drop_out)
        
    def forward(self, x):
        # make sure input tensor is flattened
        x = x.view(x.shape[0], -1)
        
        # Add Dropout to hidden layers
        for layer in self.hidden_layers:
            x = self.dropout(F.relu(layer(x)))
            
        # Output
        x = F.log_softmax(self.output_layer(x), dim=1)
        
        return x
    
def get_current_device(gpu):
    return torch.device("cuda:0" if gpu else "cpu")
    
    
def train_classifier(model, epochs_no, criterion, optimizer, trainloader, validloader, current_device):
    # Move the network and data to current hardware config (GPU or CPU)
    model.to(current_device)
    
    epochs = epochs_no
    steps = 0
    print_every = 1
    running_loss = 0

    # Looping through epochs, each epoch is a full pass through the network
    for epoch in range(epochs):
        
        # Switch to the train mode
        print("model.train()")
        model.train()

        # Looping through images, get a batch size of images on each loop
        for inputs, labels in trainloader:

            # print("steps: {}".format(steps))
            steps += 1
            
            # Move input and label tensors to the default device
            # print("2")
            inputs, labels = inputs.to(current_device), labels.to(current_device)


            # Clear the gradients, so they do not accumulate
            # print("3")
            optimizer.zero_grad()

            # Forward pass, then backward pass, then update weights
            # print("4")
            # print("start: {}".format(datetime.datetime.now().time()))
            log_ps = model.forward(inputs)
            # print("end: {}".format(datetime.datetime.now().time()))
            
            # print("5")
            # print("start: {}".format(datetime.datetime.now().time()))
            loss = criterion(log_ps, labels)
            # print("end: {}".format(datetime.datetime.now().time()))
            # print("6")
            # print("start: {}".format(datetime.datetime.now().time()))
            loss.backward()
            # print("end: {}".format(datetime.datetime.now().time()))
            # print("7")
            optimizer.step()
            
            # print("8")
            running_loss += loss.item()
            # print("last running_lossrunning_loss")

        # Track the loss and accuracy on the validation set to determine the best hyperparameters
        # print("steps: {}".format(steps))
        # print("print_every: {}".format(print_every))
        # print("steps % print_every: {}".format(steps % print_every))
        if steps % print_every == 0:

            # Put in evaluation mode
            model.eval()

            # Turn off gradients for validation, save memory and computations
            with torch.no_grad():

                # Validate model
                test_loss, accuracy = find_test_lossess(model, criterion, validloader, current_device)
                
            train_loss = running_loss/print_every
            valid_loss = test_loss/len(validloader)
            valid_accuracy = accuracy/len(validloader)

            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {train_loss:.3f}.. "
                  f"Test loss: {valid_loss:.3f}.. "
                  f"Test accuracy: {valid_accuracy:.3f}")

            running_loss = 0
            
            # Switch back to the train mode
            model.train()
                
    # Return last metrics
    return train_loss, valid_loss, valid_accuracy

def get_accuracy_loss(model, criterion, test_loaders, current_device):
    # Do validation on the test set
    model.eval()

    # Turn off gradient for testing
    with torch.no_grad():
        test_loss, accuracy = find_test_lossess(model, criterion, test_loaders, current_device)
        
    return test_loss, accuracy


def rebuild_model(filepath):
    
    # Load model metadata
    # Loading weights for CPU model whoch were trained on GPU
    # https://discuss.pytorch.org/t/loading-weights-for-cpu-model-while-trained-on-gpu/1032
    checkpoint = torch.load(filepath, map_location=lambda storage, loc: storage)
    
    # Recreate the pretrained base model
    #model = models.vgg16(pretrained=True)
    model = getattr(models, checkpoint['name'])(pretrained=True)
    
    # Replace the classifier part of the model
    model.classifier = checkpoint['classifier']
    
    # Rebuild saved state dict
    model.load_state_dict(checkpoint['model_state_dict'])

    # Load class_to_idx
    model.class_to_idx = checkpoint['class_to_idx']

    return model


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    # Resize the images where the shortest side is 256 pixels, keeping the aspect ratio
    if image.width > image.height:
        image.thumbnail((1000000, 256))
    else:
        image.thumbnail((256, 1000000))
        
    # Center crop the image
    crop_size = 224
    left_margin = (image.width - crop_size) / 2
    bottom_margin = (image.height - crop_size) / 2
    right_margin = left_margin + crop_size
    top_margin = bottom_margin + crop_size  
    image = image.crop((left_margin, bottom_margin, right_margin, top_margin))
    
    # Convert values to range of 0 to 1 instead of 0-255
    image = np.array(image)
    image = image / 255
    
    # Standardize values
    means = np.array([0.485, 0.456, 0.406])
    stds = np.array([0.229, 0.224, 0.225])
    image = (image - means) / stds
    
    # Move color channels to first dimension as expected by PyTorch
    image = image.transpose(2, 0, 1)
    
    return image


def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    
    # Move model into evaluation mode and to CPU
    model.eval()
    model.cpu()
   
    # Open image
    image = Image.open(image_path)
    
    # Process image
    image = process_image(image) 
    
    # Change numpy array type to a PyTorch tensor
    image = torch.from_numpy(image).type(torch.FloatTensor) 
    
    # Format tensor for input into model
    # (add batch of size 1 to image)
    # https://pytorch.org/docs/stable/torch.html#torch.unsqueeze
    image = image.unsqueeze(0)
    
    # Predict top K probabilities
    # Reverse the log conversion
    probs = torch.exp(model.forward(image))
    top_probs, top_labs = probs.topk(topk)
    #print(top_probs)
    #print(top_labs)
    
    # Convert from Tesors to Numpy arrays
    top_probs = top_probs.detach().numpy().tolist()[0]
    idx_to_class = {model.class_to_idx[k]: k for k in model.class_to_idx}
    
    # Map tensor indexes to classes
    labs = []
    for label in top_labs.numpy()[0]:
        labs.append(idx_to_class[label])

    return top_probs, labs


def find_test_lossess(model, criterion, testloader, current_device):
    
    # Move the network and data to current hardware config (GPU or CPU)
    model.to(current_device)
    
    test_loss = 0
    accuracy = 0
        
    # Looping through images, get a batch size of images on each loop
    for inputs, labels in testloader:

        # Move input and label tensors to the default device
        inputs, labels = inputs.to(current_device), labels.to(current_device)

        # Forward pass, then backward pass, then update weights
        log_ps = model.forward(inputs)
        batch_loss = criterion(log_ps, labels)
        test_loss += batch_loss.item()

        # Convert to softmax distribution
        ps = torch.exp(log_ps)
        
        # Compare highest prob predicted class with labels
        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        
        # Calculate accuracy
        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
    
    return test_loss, accuracy