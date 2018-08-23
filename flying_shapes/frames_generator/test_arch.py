from __future__ import print_function
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
from train_loop import TrainLoop
import torch.optim as optim
import torch.utils.data
from model import *

# Training settings
parser = argparse.ArgumentParser(description='Test new architectures')
parser.add_argument('--latent-size', type=int, default=100, metavar='S', help='latent layer dimension (default: 100)')
parser.add_argument('--im-res', type=int, default=64, metavar='S', help='Image resolution (default: 64)')
parser.add_argument('--im-channels', type=int, default=3, metavar='S', help='input channels (default: 3)')
args = parser.parse_args()

generator = Generator().eval()
disc = Discriminator(optim.Adam, 'adam', 0.1, (0.1, 0.1)).train()

z =  torch.rand(10, args.latent_size, 1, 1)
im = torch.rand(10, args.im_channels, args.im_res, args.im_res)

im_g = generator(z)

out_d = disc.forward(im)

print(im_g.size(), out_d.size())

