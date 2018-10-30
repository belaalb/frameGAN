from __future__ import print_function
import argparse
import torch
import models_zoo
from data_load import Loader
import subprocess
from sklearn import manifold
from sklearn import decomposition

import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple
import numpy as np

from pylab import rcParams
rcParams['figure.figsize'] = 15, 10

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

	many_z = np.random.randn(10000, 100)

	z12 = np.random.randn(2, 100)

	z12_ = np.random.randn(2, 100)
	
	alpha_list = list(np.arange(1, 29)/30)

	z_interp1 = []
	for alpha in alpha_list:
		z_interp1.append((1-alpha)*z12[0] + alpha*z12[-1])

	z_interp1 = np.asarray(z_interp1)
	
	z_interp2 = []
	for alpha in alpha_list:
		z_interp2.append((1-alpha)*z12_[0] + alpha*z12_[-1])

	z_interp2 = np.asarray(z_interp2)
	
	X = np.concatenate((out_reshape, many_z, z12, z_interp1, z12_, z_interp2))

	plt.cla()
	pca = decomposition.PCA(n_components=2)
	pca.fit(many_z)
	X_pca = pca.transform(X)

	p1 = plt.plot(X_pca[10211:, 0], X_pca[10211:, 1], '^', c = 'black', label = 'Linear interpolation')
	p2 = plt.plot(X_pca[210:10210, 0], X_pca[210:10210, 1], 'x', c = 'lightgreen', alpha = 0.5, label = '$\mathbf{z}\sim \mathcal{N}(0,I_{100})$')
	p3 = plt.plot(X_pca[0:30, 0], X_pca[0:30, 1], 'o', c = 'blue', label = '$\mathbf{z}_{F}$')
	p4 = plt.plot(X_pca[30:60, 0], X_pca[30:60, 1], 'o', c = 'pink')#, label = '$\mathbf{z}_{F}$')
	p5 = plt.plot(X_pca[60:90, 0], X_pca[60:90, 1], 'o', c = 'green')#, label = '$\mathbf{z}_{F}$')
	p6 = plt.plot(X_pca[90:120, 0], X_pca[90:120, 1], 'o', c = 'magenta')#, label = '$\mathbf{z}_{F}$')
	p7 = plt.plot(X_pca[120:150, 0], X_pca[120:150, 1], 'o', c = 'purple')#, label = '$\mathbf{z}_{F}$')
	p8 = plt.plot(X_pca[150:180, 0], X_pca[150:180, 1], 'o', c = 'orange')#, label = '$\mathbf{z}_{F}$')
	p9 = plt.plot(X_pca[180:210, 0], X_pca[180:210, 1], 'o', c = 'red')#, label = '$\mathbf{z}_{F}$')

	#plt.legend([(p3, p4, p5, p6, p7, p8, p9), p1, p2], ['$\mathbf{z}_{F}$', 'Linear interpolation', '$\mathbf{z}\sim \mathcal{N}(0,I_{100})$'])

	#plt.legend([(p3, p4, p5, p6, p7, p8, p9)], ['$\mathbf{z}_{F}$'], numpoints=1, handler_map={tuple: HandlerTuple(ndivide=None)})#, 'Linear interpolation', '$\mathbf{z}\sim \mathcal{N}(0,I_{100})$'])
	save_fn = 'pca.png'
	plt.legend()
	plt.savefig(save_fn)
	plt.show()
	plt.close()


	tsne = manifold.TSNE(n_components=2, init='pca')
	X_tsne = tsne.fit_transform(X)

	p1 = plt.plot(X_tsne[10211:, 0], X_tsne[10211:, 1], '^', c = 'black', label = 'Linear interpolation')
	p2 = plt.plot(X_tsne[210:10210, 0], X_tsne[210:10210, 1], 'x', c = 'lightgreen', alpha = 0.5, label = '$\mathbf{z}\sim \mathcal{N}(0,I_{100})$')
	p3 = plt.plot(X_tsne[0:30, 0], X_tsne[0:30, 1], 'o', c = 'blue', label = '$\mathbf{z}_{F}$')
	p4 = plt.plot(X_tsne[30:60, 0], X_tsne[30:60, 1], 'o', c = 'pink')#, label = '$\mathbf{z}_{F}$')
	p5 = plt.plot(X_tsne[60:90, 0], X_tsne[60:90, 1], 'o', c = 'green')#, label = '$\mathbf{z}_{F}$')
	p6 = plt.plot(X_tsne[90:120, 0], X_tsne[90:120, 1], 'o', c = 'magenta')#, label = '$\mathbf{z}_{F}$')
	p7 = plt.plot(X_tsne[120:150, 0], X_tsne[120:150, 1], 'o', c = 'purple')#, label = '$\mathbf{z}_{F}$')
	p8 = plt.plot(X_tsne[150:180, 0], X_tsne[150:180, 1], 'o', c = 'orange')#, label = '$\mathbf{z}_{F}$')
	p9 = plt.plot(X_tsne[180:210, 0], X_tsne[180:210, 1], 'o', c = 'red')#, label = '$\mathbf{z}_{F}$')

	save_fn = 'tsne.png'
	plt.legend()
	plt.savefig(save_fn)
	plt.show()
	plt.close()
	

	iso = manifold.Isomap(n_neighbors=6, n_components=2)
	iso.fit(X)
	X_isomap = iso.transform(X)

	p1 = plt.plot(X_isomap[10211:, 0], X_isomap[10211:, 1], '^', c = 'black', label = 'Linear interpolation')
	p2 = plt.plot(X_isomap[210:10210, 0], X_isomap[210:10210, 1], 'x', c = 'lightgreen', alpha = 0.5, label = '$\mathbf{z}\sim \mathcal{N}(0,I_{100})$')
	p3 = plt.plot(X_isomap[0:30, 0], X_isomap[0:30, 1], 'o', c = 'blue', label = '$\mathbf{z}_{F}$')
	p4 = plt.plot(X_isomap[30:60, 0], X_isomap[30:60, 1], 'o', c = 'pink')#, label = '$\mathbf{z}_{F}$')
	p5 = plt.plot(X_isomap[60:90, 0], X_isomap[60:90, 1], 'o', c = 'green')#, label = '$\mathbf{z}_{F}$')
	p6 = plt.plot(X_isomap[90:120, 0], X_isomap[90:120, 1], 'o', c = 'magenta')#, label = '$\mathbf{z}_{F}$')
	p7 = plt.plot(X_isomap[120:150, 0], X_isomap[120:150, 1], 'o', c = 'purple')#, label = '$\mathbf{z}_{F}$')
	p8 = plt.plot(X_isomap[150:180, 0], X_isomap[150:180, 1], 'o', c = 'orange')#, label = '$\mathbf{z}_{F}$')
	p9 = plt.plot(X_isomap[180:210, 0], X_isomap[180:210, 1], 'o', c = 'red')#, label = '$\mathbf{z}_{F}$')

	#plt.legend([(p3, p4, p5, p6, p7, p8, p9), p1, p2], ['$\mathbf{z}_{F}$', 'Linear interpolation', '$\mathbf{z}\sim \mathcal{N}(0,I_{100})$'])

	#plt.legend([(p3, p4, p5, p6, p7, p8, p9)], ['$\mathbf{z}_{F}$'], numpoints=1, handler_map={tuple: HandlerTuple(ndivide=None)})#, 'Linear interpolation', '$\mathbf{z}\sim \mathcal{N}(0,I_{100})$'])
	save_fn = 'isomap.png'
	plt.legend()
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

