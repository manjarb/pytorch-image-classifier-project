import datetime

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils import data
from torchvision import datasets, transforms, models
import time
from workspace_utils import active_session

def load_data(data_dir, is_gpu):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # Environment controls
    # Choose if run in CPU or GPU enabled hardware
    # devices = ['cpu', 'cuda']
    # current_device = devices[0]
    # or use CUDA if available
    current_device = torch.device("cuda:0" if torch.cuda.is_available() and is_gpu else "cpu")
    current_device
    
    image_size = 224
    image_resize = 255
    rotation = 30
    means = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    batch_size = 64

    train_transforms = transforms.Compose([transforms.RandomRotation(rotation),
                                            transforms.RandomResizedCrop(image_size),
                                            transforms.RandomHorizontalFlip(),
                                            transforms.ToTensor(),
                                            transforms.Normalize(means, std)]) 

    valid_transforms = transforms.Compose([transforms.Resize(image_resize),
                                           transforms.CenterCrop(image_size),
                                           transforms.ToTensor(),
                                           transforms.Normalize(means, std)]) 

    test_transforms = transforms.Compose([transforms.Resize(image_resize),
                                               transforms.CenterCrop(image_size),
                                               transforms.ToTensor(),
                                               transforms.Normalize(means, std)]) 

    # TODO: Load the datasets with ImageFolder
    train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_datasets = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    test_datasets = datasets.ImageFolder(test_dir, transform=test_transforms)

    # TODO: Using the image datasets and the trainforms, define the dataloaders
    train_loaders = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size, shuffle=True)
    valid_loaders = torch.utils.data.DataLoader(valid_datasets, batch_size=batch_size)
    test_loaders = torch.utils.data.DataLoader(test_datasets, batch_size=batch_size)
    
    return train_datasets, valid_datasets, test_datasets, train_loaders, valid_loaders, test_loaders

def get_pretrained_model(arch):
    model = getattr(models, arch)(pretrained=True)
    model.name = arch
    return model

def get_input_size(model, arch):
    input_size = 0
    
    if('vgg' in arch): return model.classifier[0].in_features
    elif('densenet' in arch): return model.classifier.in_features
    elif('squeezenet' in arch): return model.classifier[1].in_channels
    elif(('resnet' in arch) or ('inception'in arch) ): return model.fc.in_features
    elif('alexnet' in arch): return model.classifier[1].in_features
        
    if(input_size == 0): raise Error    
    return input_size


def save_checkpoint(model, train_datasets):
    
    # Mapping of classes to indices
    model.class_to_idx = train_datasets.class_to_idx
    
    # Create model metadata dictionary
    checkpoint = {
        'name': model.name,
        'class_to_idx': model.class_to_idx,
        'classifier': model.classifier,
        'model_state_dict': model.state_dict()
    }

    # Save to a file
    timestr = time.strftime("%Y%m%d_%H%M%S")
    file_name = 'model_' + timestr + '.pth'
    torch.save(checkpoint, file_name)
    
    return file_name
