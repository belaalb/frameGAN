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

	many_z = np.random.randn(1000, 100)

	z12 = np.random.randn(2, 100)

	z12_ = np.random.randn(2, 100)
	
	alpha_list = list(np.arange(1, 29)/28)

	z_interp1 = []
	for alpha in alpha_list:
		z_interp1.append((1-alpha)*z12[0] + alpha*z12[-1])

	z_interp1 = np.asarray(z_interp1)
	
	z_interp2 = []
	for alpha in alpha_list:
		z_interp2.append((1-alpha)*z12_[0] + alpha*z12_[-1])

	z_interp2 = np.asarray(z_interp2)
	
	X = np.concatenate((out_reshape, many_z, z12, z_interp1, z12_, z_interp2))

	tsne = manifold.TSNE(n_components=2, init='pca')
	X_tsne = tsne.fit_transform(X)

	plt.plot(X_tsne[1210:, 0], X_tsne[1210:, 1], '^', c = 'black')
	plt.plot(X_tsne[210:1210, 0], X_tsne[210:1210, 1], 'o', c = 'lavender', alpha = 0.6)
	plt.plot(X_tsne[0:30, 0], X_tsne[0:30, 1], 'x', c = 'blue')
	plt.plot(X_tsne[30:60, 0], X_tsne[30:60, 1], 'x', c = 'pink')
	plt.plot(X_tsne[60:90, 0], X_tsne[60:90, 1], 'x', c = 'green')
	plt.plot(X_tsne[90:120, 0], X_tsne[90:120, 1], 'x', c = 'coral')
	plt.plot(X_tsne[120:150, 0], X_tsne[120:150, 1], 'x', c = 'purple')
	plt.plot(X_tsne[150:180, 0], X_tsne[150:180, 1], 'x', c = 'orange')
	plt.plot(X_tsne[180:210, 0], X_tsne[180:210, 1], 'x', c = 'red')

	save_fn = 'tsne.pdf'
	plt.savefig(save_fn)
	plt.show()
	plt.close()

	iso = manifold.Isomap(n_neighbors=6, n_components=2)
	iso.fit(X)
	X_isomap = iso.transform(X)

	plt.plot(X_isomap[1210:, 0], X_isomap[1210:, 1], '^', c = 'black')
	plt.plot(X_isomap[210:1210, 0], X_isomap[210:1210, 1], 'o', c = 'lavender', alpha = 0.6)
	plt.plot(X_isomap[0:30, 0], X_isomap[0:30, 1], 'x', c = 'blue')
	plt.plot(X_isomap[30:60, 0], X_isomap[30:60, 1], 'x', c = 'pink')
	plt.plot(X_isomap[60:90, 0], X_isomap[60:90, 1], 'x', c = 'green')
	plt.plot(X_isomap[90:120, 0], X_isomap[90:120, 1], 'x', c = 'coral')
	plt.plot(X_isomap[120:150, 0], X_isomap[120:150, 1], 'x', c = 'purple')
	plt.plot(X_isomap[150:180, 0], X_isomap[150:180, 1], 'x', c = 'orange')
	plt.plot(X_isomap[180:210, 0], X_isomap[180:210, 1], 'x', c = 'red')

	save_fn = 'isomap.pdf'
	plt.savefig(save_fn)
	plt.show()
	plt.close()



if __name__ == '__main__':

	# Testing settings
	parser = argparse.ArgumentParser(description='Testing online transfer learning for emotion recognition tasks')
	parser.add_argument('--cp-path', type=str, default=None, metavar='Path', help='Checkpoint/model path')
	parser.add_argument('--n-z', type=int, default=7, metavar='N', help='number of tests  (default: 4)')
	parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables GPU use')

	args = parser.parse_args()
	args.cuda = True if not args.no_cuda and torch.cuda.is_available() else False

	ckpt = torch.load(args.cp_path, map_location = lambda storage, loc: storage)

	generator = models_zoo.Generator_linear(args.cuda)

	generator.load_state_dict(ckpt['generator_state'])

	if args.cuda:
		generator = generator.cuda()

	tsne_model(generator, args.n_z, args.cuda)

