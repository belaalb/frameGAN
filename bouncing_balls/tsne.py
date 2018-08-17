from __future__ import print_function
import argparse
import torch
import models_zoo
from data_load import Loader
import subprocess
from sklearn import manifold

import matplotlib.pyplot as plt
import numpy as np


def tsne_model(generator, n_z, cuda_mode):
	generator.eval()


	colors = ['lightblue', 'lavender', 'lightgreen', 'pink', 'coral']

	z_ = torch.randn(n_z, 100).view(-1, 100, 1)

	if args.cuda:
		z_ = z_.cuda()

	out = generator.forward(z_).squeeze()
	out = out.detach().cpu().numpy()

	out_reshape = np.zeros((out.shape[0]*out.shape[1], out.shape[2]))

	for i in range(out.shape[0]):
		out_reshape[i*30:(i+1)*30, :] = out[i, :, :]	

	many_z = np.random.randn(200, 100)

	X = np.concatenate((out_reshape, many_z))

	tsne = manifold.TSNE(n_components=2, init='pca')
	X_tsne = tsne.fit_transform(X)

	plt.plot(X_tsne[0:30, 0], X_tsne[0:30, 1], 'x', c = 'lightblue')
	plt.plot(X_tsne[30:60, 0], X_tsne[30:60, 1], 'x', c = 'pink')
	plt.plot(X_tsne[60:90, 0], X_tsne[60:90, 1], 'x', c = 'lightgreen')
	plt.plot(X_tsne[90:120, 0], X_tsne[90:120, 1], 'x', c = 'coral')
	plt.plot(X_tsne[120:150, 0], X_tsne[120:150, 1], 'x', c = 'lavender')
	plt.plot(X_tsne[150:, 0], X_tsne[150:, 1], 'o', c = 'magenta', alpha = 0.2)

	save_fn = 'tsne.pdf'
	plt.savefig(save_fn)

	plt.close()


if __name__ == '__main__':

	# Testing settings
	parser = argparse.ArgumentParser(description='Testing online transfer learning for emotion recognition tasks')
	parser.add_argument('--cp-path', type=str, default=None, metavar='Path', help='Checkpoint/model path')
	parser.add_argument('--n-z', type=int, default=5, metavar='N', help='number of tests  (default: 4)')
	parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables GPU use')

	args = parser.parse_args()
	args.cuda = True if not args.no_cuda and torch.cuda.is_available() else False

	ckpt = torch.load(args.cp_path, map_location = lambda storage, loc: storage)

	generator = models_zoo.Generator_linear(args.cuda)

	generator.load_state_dict(ckpt['generator_state'])

	if args.cuda:
		generator = generator.cuda()

	tsne_model(generator, args.n_z, args.cuda)

