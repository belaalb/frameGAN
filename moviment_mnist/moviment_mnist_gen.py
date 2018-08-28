from __future__ import print_function
import os
import numpy as np
from tqdm import trange
import cv2
import glob


def data_gen(n_samples=10000,im_size=64, n_frames=20,debug=False,debug_opencv=False):
	
	mov_mnist = np.load("mnist_test_seq.npy")

	# Change shape (20, 10000, 64, 64) to (10000, 20, 64, 64)
	mov_mnist = np.rollaxis(np.asarray(mov_mnist), 1, 0)
	
	## Limit n_samples
	if(n_samples != 10000):
		mov_mnist = mov_mnist[0:n_samples]

	
	## Limit img_size
	#aux = []
	#if(im_size != 64):
	#	for i in range(0,len(mov_mnist)):
	#		for j in range(0,n_frames):
	#			teste = cv2.resize(mov_mnist[i][j], dsize=(32,32))
	#			aux.append(teste)
	#			print(np.asarray(aux[i]).shape)
	#			cv2.imshow("Testando", np.asarray(aux[i]))
	#			cv2.waitKey(0)
	if(debug):
		print(mov_mnist.shape)
	
	# Reshape (10000, 20, 64, 64) to (10000, 1, 20, 64, 64)
	mov_mnist = np.reshape(mov_mnist, (mov_mnist.shape[0],1,mov_mnist.shape[1],mov_mnist.shape[2],mov_mnist.shape[3]))
	if(debug):	
		print(mov_mnist.shape)
	
	if(debug_opencv):
		visualize_moviment_mnist(mov_mnist)

	return mov_mnist

def visualize_moviment_mnist(mov_mnist):
	for i in range(0,len(mov_mnist)):
		for k in range(0, mov_mnist.shape[2]):
			cv2.imshow("Teste", mov_mnist[i][0][k][:][:])
			cv2.waitKey(0)


if __name__ == '__main__':
	
	import argparse
	import pickle
	import h5py
	import numpy as np

	# Data settings
	parser = argparse.ArgumentParser(description='Generate Flying Shapes Dataset')
	parser.add_argument('--im-size', type=int, default=64, metavar='N', help='H and W of frames (default: 30)')
	parser.add_argument('--n-frames', type=int, default=20, metavar='N', help='Number of frames per sample (default: 128)')
	parser.add_argument('--n-samples', type=int, default=10000, metavar='N', help='Number of output samples (default: 500)')
	parser.add_argument('--output-path', type=str, default='./', metavar='Path', help='Path for output')
	parser.add_argument('--file-name', type=str, default='train.hdf', metavar='Path', help='Output file name')
	parser.add_argument('--debug', type=int, default=0, metavar='N', help='Debug Flag')
	parser.add_argument('--opencv', type=int, default=0, metavar='N', help='Debug Opencv Flag')
	args = parser.parse_args()
	

	dat = data_gen(n_samples=args.n_samples, im_size=args.im_size, n_frames=args.n_frames,debug=args.debug,debug_opencv=args.opencv)

	### DEBUG ##
	if(args.debug):
		print(dat.shape)
	
	### SAVE DATASET ##
	hdf5 = h5py.File(args.output_path+args.file_name, 'w')
	dataset = hdf5.create_dataset('data', data=dat)
	hdf5.close()