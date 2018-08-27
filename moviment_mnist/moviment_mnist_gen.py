from __future__ import print_function
import os
import numpy as np
from tqdm import trange
import cv2
import glob


def data_gen(n_samples=500,im_size=30, n_frames=128,batch_size=5,debug=False,debug_opencv=False):
	
	mov_mnist = np.load("mnist_test_seq.npy")
	#a = data['a']
	print(mov_mnist.shape)


	mov_mnist = np.rollaxis(np.asarray(mov_mnist), 1, 0)
	mov_mnist = np.rollaxis(np.asarray(mov_mnist), 3,4)
	mov_mnist = np.rollaxis(np.asarray(mov_mnist), 2,3)
	print(mov_mnist.shape)

	#(10000, 64, 64, 20)
	#exit(0)
	for i in range(0,len(mov_mnist)):
		print(len(mov_mnist))
		for k in range(0, mov_mnist.shape[1]):
			#print(mov_mnist.shape[-1]-1)
			cv2.imshow("Teste",mov_mnist[i][k][:][:])
			cv2.waitKey(0)



'''
	exit(0)
	data_source = dataset.FlyingShapesDataHandler(batch_size=batch_size,seq_len=n_frames,im_size=im_size)
	dat = []
	for n in trange(0,n_samples):

		np_batch = data_source.GetUnlabelledBatch()
		batch = {
		'image': np_batch['image'],
		'bbox': np_batch['bbox']
		}

		sample = []
		for i in range(n_frames):
			sample.append(batch['image'][0][i])
		
			if(debug):
				print(batch['image'][0][i].shape)
				print("sample: ", len(sample))
				print(np.asarray(sample).shape)
			if(debug_opencv):
				print(debug_opencv)
				import cv2
				cv2.imshow("debug",np.asarray(batch['image'][0][i]))
				cv2.waitKey(0)

		dat.append(np.moveaxis(sample, 0, -1))
	dat = np.rollaxis(np.asarray(dat), 4, 3)
	return dat
'''

if __name__ == '__main__':
	
	#davis_count_frames_per_class()
	
	#import argparse
	#
	## Data settings
	#parser = argparse.ArgumentParser(description='Generate Flying Shapes Dataset')
	#parser.add_argument('--im-size', type=int, default=64, metavar='N', help='H and W of frames (default: 64)')
	#parser.add_argument('--output-path', type=str, default='./results/', metavar='Path', help='Path for output')
	#args = parser.parse_args()
	#
	#dat = davis_gen(im_size=args.im_size, output_path=args.output_path)

	import argparse
	import pickle
	import h5py
	import numpy as np

	# Data settings
	'''
	parser = argparse.ArgumentParser(description='Generate Flying Shapes Dataset')
	parser.add_argument('--im-size', type=int, default=30, metavar='N', help='H and W of frames (default: 30)')
	parser.add_argument('--n-frames', type=int, default=128, metavar='N', help='Number of frames per sample (default: 128)')
	parser.add_argument('--n-samples', type=int, default=500, metavar='N', help='Number of output samples (default: 500)')
	parser.add_argument('--output-path', type=str, default='./', metavar='Path', help='Path for output')
	parser.add_argument('--file-name', type=str, default='train.hdf', metavar='Path', help='Output file name')
	parser.add_argument('--debug', type=int, default=0, metavar='N', help='Debug Flag')
	parser.add_argument('--opencv', type=int, default=0, metavar='N', help='Debug Opencv Flag')
	args = parser.parse_args()
	'''
	#cv2.imshow()
	#cv2.waitKey(0)
	dat = data_gen()
	##dat = data_gen(n_samples=args.n_samples, im_size=args.im_size, n_frames=args.n_frames,debug=args.debug,debug_opencv=args.opencv)
	#
	#dat = np.asarray(dat)
	#
	### DEBUG ##
	#if(args.debug):
	#	print(dat.shape)
	#
	### SAVE DATASET ##
	#hdf5 = h5py.File(args.output_path+args.file_name, 'w')
	#dataset = hdf5.create_dataset('data', data=dat)
	#hdf5.close()