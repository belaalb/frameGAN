import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable as Variable

class Generator(nn.Module):
	def __init__(self, cuda_mode):
		super(Generator, self).__init__()

		self.cuda_mode = cuda_mode

		## Considering (30, 90) inputs

		self.features = nn.Sequential(
			nn.Linear(100, 512),
			nn.BatchNorm1d(512),
			nn.ReLU(),
			nn.Linear(512, 1024),
			nn.BatchNorm1d(1024),
			nn.ReLU(),
			nn.Linear(1024, 2048),
			nn.BatchNorm1d(2048),
			nn.ReLU(),
			nn.Linear(2048, 3840),
			nn.BatchNorm1d(3840),
			nn.ReLU() )

		self.lstm = nn.LSTM(128, 256, 2, bidirectional=True, batch_first=False)

		self.fc = nn.Linear(256*2, 100)

	def forward(self, x):

		x = self.features(x)

		x = x.view(30, x.size(0), -1)

		batch_size = x.size(1)
		seq_size = x.size(0)

		h0 = Variable(torch.zeros(4, batch_size, 256))
		c0 = Variable(torch.zeros(4, batch_size, 256))

		if self.cuda_mode:
			h0 = h0.cuda()
			c0 = c0.cuda()

		
		x, h_c = self.lstm(x, (h0, c0))
		
		x = F.tanh( self.fc( x.view(batch_size*seq_size, -1) ) )

		return x.view(batch_size, seq_size, -1)

class frames_generator(torch.nn.Module):
	def __init__(self):
		super(frames_generator, self).__init__()

		#linear layer
		self.linear = torch.nn.Sequential()

		linear = nn.Linear(100, 2*2*1024)

		self.linear.add_module('linear', linear)

		# Initializer
		nn.init.normal(linear.weight, mean=0.0, std=0.02)
		nn.init.constant(linear.bias, 0.0)

		# Batch normalization
		bn_name = 'bn0'
		self.linear.add_module(bn_name, torch.nn.BatchNorm1d(2*2*1024))

		# Activation
		act_name = 'act0'
		self.linear.add_module(act_name, torch.nn.ReLU())

		# Hidden layers
		num_filters = [1024, 512, 256]
		self.hidden_layer = torch.nn.Sequential()
		for i in range(3):
			# Deconvolutional layer
			if i == 0:
				deconv = nn.ConvTranspose2d(1024, num_filters[i], kernel_size=4, stride=2, padding=1)
			else:
				deconv = nn.ConvTranspose2d(num_filters[i - 1], num_filters[i], kernel_size=4, stride=2, padding=1)

			deconv_name = 'deconv' + str(i + 1)
			self.hidden_layer.add_module(deconv_name, deconv)

			# Initializer
			nn.init.normal(deconv.weight, mean=0.0, std=0.02)
			nn.init.constant(deconv.bias, 0.0)

			# Batch normalization
			bn_name = 'bn' + str(i + 1)
			self.hidden_layer.add_module(bn_name, torch.nn.BatchNorm2d(num_filters[i]))

			# Activation
			act_name = 'act' + str(i + 1)
			self.hidden_layer.add_module(act_name, torch.nn.ReLU())

		# Output layer
		self.output_layer = torch.nn.Sequential()
		# Deconvolutional layer
		out = torch.nn.ConvTranspose2d(256, 1, kernel_size=4, stride=2, padding=2)
		self.output_layer.add_module('out', out)
		# Initializer
		nn.init.normal(out.weight, mean=0.0, std=0.02)
		nn.init.constant(out.bias, 0.0)
		# Activation
		self.output_layer.add_module('act', torch.nn.Tanh())

	def forward(self, x):

		x = x.view(x.size(0), -1)
		x = self.linear(x)

		h = self.hidden_layer(x.view(x.size(0), 1024, 2, 2))
		out = self.output_layer(h)
		return out

class Discriminator(torch.nn.Module):
	def __init__(self, optimizer, lr, betas, batch_norm=False):
		super(Discriminator, self).__init__()

		self.projection = nn.utils.weight_norm(nn.Conv3d(1, 1, kernel_size=8, stride=3, padding=2, bias=False), name="weight")
		self.projection.weight_g.data.fill_(1)

		# Hidden layers
		self.hidden_layer = torch.nn.Sequential()
		num_filters = [256, 512, 1024]
		for i in range(len(num_filters)):
			# Convolutional layer
			if i == 0:
				conv = nn.Conv3d(1, num_filters[i], kernel_size=4, stride=2, padding=1)
			else:
				conv = nn.Conv3d(num_filters[i-1], num_filters[i], kernel_size=4, stride=1, padding=1)

			conv_name = 'conv' + str(i + 1)
			self.hidden_layer.add_module(conv_name, conv)

			# Initializer
			nn.init.normal(conv.weight, mean=0.0, std=0.02)
			nn.init.constant(conv.bias, 0.0)

			# Batch normalization
			if i != 0 and batch_norm:
				bn_name = 'bn' + str(i + 1)
				self.hidden_layer.add_module(bn_name, torch.nn.BatchNorm3d(num_filters[i]))

			# Activation
			act_name = 'act' + str(i + 1)
			self.hidden_layer.add_module(act_name, torch.nn.LeakyReLU(0.2))

		# Output layer
		self.output_layer = torch.nn.Sequential()
		# Convolutional layer
		out = nn.Conv3d(num_filters[i], 1, kernel_size=4, stride=1, padding=1)
		self.output_layer.add_module('out', out)
		# Initializer
		nn.init.normal(out.weight, mean=0.0, std=0.02)
		nn.init.constant(out.bias, 0.0)
		# Activation
		self.output_layer.add_module('act', nn.Sigmoid())

		self.optimizer = optimizer(list(self.hidden_layer.parameters()) + list(self.output_layer.parameters()), lr=lr, betas=betas)

	def forward(self, x):
		p_x = self.projection(x)
		h = self.hidden_layer(p_x)
		out = self.output_layer(h)
		return out