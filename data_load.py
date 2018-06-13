import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
import os


class Loader(Dataset):

	def __init__(self, hdf5_name):
		super(Loader, self).__init__()
		self.hdf5_name = hdf5_name

		open_file = h5py.File(self.hdf5_name, 'r')
		self.length = len(open_file['data'])
		open_file.close()

		self.open_file = None

	def __getitem__(self, index):

		if not self.open_file: self.open_file = h5py.File(self.hdf5_name, 'r')

		scene = torch.from_numpy(self.open_file['data'][index]).float().view([40, 30, 30])
		idx = np.random.randint(10)
		img = scene[idx:(idx+30)]
		img = (img - 0.5) / 0.5

		return img.unsqueeze(0)

	def __len__(self):
		return self.length
