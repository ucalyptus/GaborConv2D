from torchsummary import summary
from gabor import *
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model  = GaborNN(200,16).to(device)


summary(model,input_size=(7,7,200),batch_size=16))
