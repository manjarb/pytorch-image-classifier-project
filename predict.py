import json
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
parser.add_argument('input', action='store', help='path to image to be classified')
parser.add_argument('checkpoint', action='store', help='path to stored model')
parser.add_argument('--top_k', action='store', type=int, default=3, help='Top K most likely class')
parser.add_argument('--category_names', action='store', help='Mapping of categories to real names')
parser.add_argument('--gpu', action='store_true')
args=parser.parse_args()
    
current_device = model_classifier.get_current_device(args.gpu)

model = model_classifier.rebuild_model(args.checkpoint)
top_probs, classes = model_classifier.predict(args.input, model, args.top_k)

cat_to_name = {}
with open(args.category_names, 'r') as f:
    cat_to_name = json.load(f)
    
print(top_probs)
print(classes)
print([cat_to_name[x] for x in classes])