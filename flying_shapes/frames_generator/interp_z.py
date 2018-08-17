from __future__ import print_function

import argparse
import os
import matplotlib.pyplot as plt
import torch.utils.data
import numpy as np
import torch
import torch.utils.data
from torchvision.transforms import transforms
from model import *
from PIL import ImageEnhance

def denorm(unorm):
	norm = (unorm + 1) / 2

	return norm.clamp(0, 1)


def interpolate(model, n_between, cuda_mode):

	n_cols, n_rows = (1, n_between)
	fig, axes = plt.subplots(n_cols, n_rows, figsize=(n_rows, n_cols))

	z12 = torch.randn(2, 100).view(-1, 100, 1, 1)

	if cuda_mode:
		z12 = z12.cuda()
	
	out12 = model.forward(z12)

	alpha_list = list(np.arange(1, n_between)/n_between)

	z = []
	for alpha in alpha_list:
		z.append((1-alpha)*z12[0] + alpha*z12[-1])
		
	z = torch.stack(z)
	print(z.size())

	if cuda_mode:
		z = z.cuda()

	out = model.forward(z)

	out_interp = [denorm(x.detach().cpu()) for x in out]

	out_list = [denorm(out12[0].detach().cpu())] + out_interp + [denorm(out12[-1].detach().cpu())]


	for ax, img in zip(axes.flatten(), out_list):
		ax.axis('off')
		ax.set_adjustable('box-forced')

		img = (((img - img.min()) * 255) / (img.max() - img.min())).numpy().transpose(1, 2, 0).astype(np.uint8).squeeze()

		ax.imshow(img)

	plt.subplots_adjust(wspace=0, hspace=0)

	save_fn = 'z_interpolation.pdf'
	plt.savefig(save_fn)

	plt.close()

if __name__ == '__main__':

	# Testing settings
	parser = argparse.ArgumentParser(description='Testing GANs under max hyper volume training')
	parser.add_argument('--cp-path', type=str, default=None, metavar='Path', help='Checkpoint/model path')
	parser.add_argument('--n-between', type=int, default=40, metavar='N', help='number of samples to generate (default: 4)')
	parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables GPU use')
	args = parser.parse_args()
	args.cuda = True if not args.no_cuda and torch.cuda.is_available() else False
	print('Cuda Mode is: {}'.format(args.cuda))

	if args.cp_path is None:
		raise ValueError('There is no checkpoint/model path. Use arg --cp-path to indicate the path!')

	model = Generator()

	ckpt = torch.load(args.cp_path, map_location=lambda storage, loc: storage)
	model.load_state_dict(ckpt['model_state'])

	if args.cuda:
		model = model.cuda()

	interpolate(model, args.n_between, args.cuda)
