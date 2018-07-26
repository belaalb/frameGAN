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

		scene_1 = self.open_file['data'][index]
		scene_2 = np.moveaxis(scene_1, -1, 0)
		scene_3 = np.moveaxis(scene_2, -1, 1)
		scene_4 = torch.from_numpy(scene_3).float()

		idx = np.random.randint(scene_4.size(1))
		img = scene_4[:, idx, :, :]

		return img

	def __len__(self):
		return self.length
