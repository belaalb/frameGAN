import torch
from torch.autograd import Variable
import torch.nn.init as init
import torch.nn.functional as F

import numpy as np
import pickle

import os
from glob import glob
from tqdm import tqdm

class TrainLoop(object):

	def __init__(self, gen, f_gen, disc_list, optimizer, train_loader, nadir_slack=1.1, checkpoint_path=None, checkpoint_epoch=None, hyper=True, cuda=True):
		if checkpoint_path is None:
			# Save to current directory
			self.checkpoint_path = os.getcwd()
		else:
			self.checkpoint_path = checkpoint_path
			if not os.path.isdir(self.checkpoint_path):
				os.mkdir(self.checkpoint_path)

		self.save_epoch_fmt_generator = os.path.join(self.checkpoint_path, 'checkpoint_{}ep.pt')
		self.save_epoch_fmt_disc = os.path.join(self.checkpoint_path, 'D{}_checkpoint.pt')
		self.cuda_mode = cuda
		self.generator = gen
		self.f_generator = f_gen
		self.disc_list = disc_list
		self.optimizer = optimizer
		self.train_loader = train_loader
		#self.valid_loader = valid_loader
		self.history = {'hv': [], 'disc': []}
		self.total_iters = 0
		self.cur_epoch = 0
		self.nadir_slack = nadir_slack
		self.hyper_mode = hyper

		if checkpoint_epoch is not None:
			self.load_checkpoint(self.save_epoch_fmt_generator.format(checkpoint_epoch))

	def train(self, n_epochs=1, patience = 5, save_every=10):

		while self.cur_epoch < n_epochs:
			print('Epoch {}/{}'.format(self.cur_epoch+1, n_epochs))
			train_iter = tqdm(enumerate(self.train_loader))

			hv_epoch=0.0
			disc_epoch=0.0

			# Train step

			for t,batch in train_iter:
				hv, disc = self.train_step(batch)
				self.total_iters += 1
				hv_epoch+=hv
				disc_epoch+=disc

			self.history['hv'].append(hv_epoch/(t+1))
			self.history['disc'].append(disc_epoch/(t+1))

			print('NLH and Discriminators loss : {:0.4f}, {:0.4f}'.format(self.history['hv'][-1], self.history['disc'][-1]))

			self.cur_epoch += 1

			if self.cur_epoch % save_every == 0:
				self.checkpointing()


		# saving final models
		print('Saving final models...')
		self.checkpointing()

	def train_step(self, batch):

		self.generator.train()

		x = batch
		z_ = torch.randn(x.size(0), 100).view(-1, 100)
		y_real_ = torch.ones(x.size(0))
		y_fake_ = torch.zeros(x.size(0))

		if self.cuda_mode:
			x = x.cuda()
			z_ = z_.cuda()
			y_real_ = y_real_.cuda()
			y_fake_ = y_fake_.cuda()

		x = Variable(x)
		z_ = Variable(z_)
		y_real_ = Variable(y_real_)
		y_fake_ = Variable(y_fake_)

		out = self.generator.forward(z_)

		frames_list = []

		for i in range(out.size(1)):
			gen_frame = self.f_generator(out[:,i,:].squeeze().contiguous())
			frames_list.append(gen_frame.unsqueeze(2))

		out = torch.cat(frames_list, 2)

		out_d = out.detach()

		loss_d = 0

		for disc in self.disc_list:
			d_real = disc.forward(x).squeeze()
			d_fake = disc.forward(out_d).squeeze()
			loss_disc = F.binary_cross_entropy(d_real, y_real_) + F.binary_cross_entropy(d_fake, y_fake_)
			disc.optimizer.zero_grad()
			loss_disc.backward()
			disc.optimizer.step()

			loss_d += loss_disc.data[0]

		loss_d /= len(self.disc_list)

		## Train G

		loss_G = 0

		if self.hyper_mode:

			losses_list_float = []
			losses_list_var = []

			for disc in self.disc_list:
				losses_list_var.append(F.binary_cross_entropy(disc.forward(out).squeeze(), y_real_))
				losses_list_float.append(losses_list_var[-1].data[0])

			self.update_nadir_point(losses_list_float)

			for i, loss in enumerate(losses_list_var):
				loss_G -= torch.log(self.nadir - loss)

		else:

			for disc in self.disc_list:
				loss_G += F.binary_cross_entropy(disc.forward(out).squeeze(), y_real_)
			self.proba = np.ones(len(self.disc_list)) * 1 / len(self.disc_list)

		self.optimizer.zero_grad()
		loss_G.backward()
		self.optimizer.step()

		return loss_G.data[0], loss_d

	def checkpointing(self):

		# Checkpointing
		print('Checkpointing...')
		ckpt = {'generator_state': self.generator.state_dict(),
		'optimizer_state': self.optimizer.state_dict(),
		'history': self.history,
		'total_iters': self.total_iters,
		'cur_epoch': self.cur_epoch}
		torch.save(ckpt, self.save_epoch_fmt_generator.format(self.cur_epoch))

		for i, disc in enumerate(self.disc_list):
			ckpt = {'generator_state': disc.state_dict(),
				'optimizer_state': disc.optimizer.state_dict()}
			torch.save(ckpt, self.save_epoch_fmt_disc.format(i + 1))

	def load_checkpoint(self, ckpt):

		if os.path.isfile(ckpt):

			ckpt = torch.load(ckpt)
			# Load generator state
			self.generator.load_state_dict(ckpt['generator_state'])
			# Load optimizer state
			self.optimizer.load_state_dict(ckpt['optimizer_state'])
			# Load history
			self.history = ckpt['history']
			self.total_iters = ckpt['total_iters']
			self.cur_epoch = ckpt['cur_epoch']

			for i, disc in enumerate(self.disc_list):
				ckpt = torch.load(self.save_epoch_fmt_disc.format(i + 1))
				disc.load_state_dict(ckpt['generator_state'])
				disc.optimizer.load_state_dict(ckpt['optimizer_state'])

		else:
			print('No checkpoint found at: {}'.format(ckpt))

	def update_nadir_point(self, losses_list):
		self.nadir = float(np.max(losses_list) * self.nadir_slack + 1e-8)

	def print_params_norms(self):
		norm = 0.0
		for params in list(self.generator.parameters()):
			norm+=params.norm(2).data[0]
		print('Sum of weights norms: {}'.format(norm))

	def print_grad_norms(self):
		norm = 0.0
		for params in list(self.generator.parameters()):
			norm+=params.grad.norm(2).data[0]
		print('Sum of grads norms: {}'.format(norm))
