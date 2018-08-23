import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(torch.nn.Module):
	def __init__(self):
		super(Generator, self).__init__()

		num_filters = [1024, 512, 256]
		input_dim = 3
		output_dim = 3

		# Hidden layers
		self.hidden_layer = torch.nn.Sequential()
		for i in range(len(num_filters)):
			# Deconvolutional layer
			if i == 0:
				deconv = nn.ConvTranspose2d(input_dim, num_filters[i], kernel_size=4, stride=1, padding=0)
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
		out = torch.nn.ConvTranspose2d(num_filters[i], output_dim, kernel_size=4, stride=2, padding=1)
		self.output_layer.add_module('out', out)
		# Initializer
		nn.init.normal(out.weight, mean=0.0, std=0.02)
		nn.init.constant(out.bias, 0.0)
		# Activation
		self.output_layer.add_module('act', torch.nn.Tanh())

	def forward(self, x):
		h = self.hidden_layer(x)
		out = self.output_layer(h)
		return out

class Discriminator(torch.nn.Module):
	def __init__(self, optimizer, optimizer_name, lr, betas, batch_norm=False):
		super(Discriminator, self).__init__()

		self.projection = nn.utils.weight_norm(nn.Conv2d(3, 1, kernel_size=8, stride=2, padding=3, bias=False), name="weight")
		#self.projection.weight_g.fill_(1)
		nn.init.constant_(self.projection.weight_g, 1)

		# Hidden layers
		self.hidden_layer = torch.nn.Sequential()
		num_filters = [256, 512, 1024]
		for i in range(3):
			# Convolutional layer
			if i == 0:
				conv = nn.Conv2d(1, num_filters[i], kernel_size=4, stride=2, padding=2)
			else:
				conv = nn.Conv2d(num_filters[i - 1], num_filters[i], kernel_size=4, stride=2, padding=1)

			conv_name = 'conv' + str(i + 1)
			self.hidden_layer.add_module(conv_name, conv)

			# Initializer
			nn.init.normal_(conv.weight, mean=0.0, std=0.02)
			nn.init.constant_(conv.bias, 0.0)

			# Batch normalization
			if i != 0 and batch_norm:
				bn_name = 'bn' + str(i + 1)
				self.hidden_layer.add_module(bn_name, torch.nn.BatchNorm2d(num_filters[i]))

			# Activation
			act_name = 'act' + str(i + 1)
			self.hidden_layer.add_module(act_name, torch.nn.LeakyReLU(0.2))

		# Output layer
		self.output_layer = torch.nn.Sequential()
		# Convolutional layer
		out = nn.Conv2d(num_filters[i], 1, kernel_size=4, stride=1, padding=1)
		self.output_layer.add_module('out', out)
		# Initializer
		nn.init.normal_(out.weight, mean=0.0, std=0.02)
		nn.init.constant_(out.bias, 0.0)
		# Activation
		self.output_layer.add_module('act', nn.Sigmoid())

		if optimizer_name == 'adam':
			self.optimizer = optimizer(list(self.hidden_layer.parameters()) + list(self.output_layer.parameters()), lr=lr, betas=betas)
		elif optimizer_name == 'amsgrad':
			self.optimizer = optimizer(list(self.hidden_layer.parameters()) + list(self.output_layer.parameters()), lr=lr, betas=betas, amsgrad = True)
		elif optimizer_name == 'rmsprop':
			self.optimizer = optimizer(list(self.hidden_layer.parameters()) + list(self.output_layer.parameters()), lr=lr)

	def forward(self, x):
		p_x = self.projection(x)
		h = self.hidden_layer(p_x)
		out = self.output_layer(h)
		return out

class Discriminator(torch.nn.Module):
	def __init__(self, optimizer_name, lr, betas, batch_norm=False):
		super(Discriminator, self).__init__()

		num_filters = [128, 256, 512, 1024]
		input_dim = 3
		output_dim = 1

		self.projection = nn.utils.weight_norm(nn.Conv2d(input_dim, 1, kernel_size=8, stride=2, padding=3, bias=False), name="weight")
		self.projection.weight_g.data.fill_(1)

		# Hidden layers
		self.hidden_layer = torch.nn.Sequential()
		for i in range(len(num_filters)):
			# Convolutional layer
			if i == 0:
				conv = nn.Conv2d(1, num_filters[i], kernel_size=4, stride=2, padding=1)
			else:
				conv = nn.Conv2d(num_filters[i - 1], num_filters[i], kernel_size=4, stride=2, padding=1)

			conv_name = 'conv' + str(i + 1)
			self.hidden_layer.add_module(conv_name, conv)

			# Initializer
			nn.init.normal(conv.weight, mean=0.0, std=0.02)
			nn.init.constant(conv.bias, 0.0)

			# Batch normalization
			if i != 0 and batch_norm:
				bn_name = 'bn' + str(i + 1)
				self.hidden_layer.add_module(bn_name, torch.nn.BatchNorm2d(num_filters[i]))

			# Activation
			act_name = 'act' + str(i + 1)
			self.hidden_layer.add_module(act_name, torch.nn.LeakyReLU(0.2))

		# Output layer
		self.output_layer = torch.nn.Sequential()
		# Convolutional layer
		out = nn.Conv2d(num_filters[i], output_dim, kernel_size=4, stride=1, padding=1)
		self.output_layer.add_module('out', out)
		# Initializer
		nn.init.normal(out.weight, mean=0.0, std=0.02)
		nn.init.constant(out.bias, 0.0)
		# Activation
		self.output_layer.add_module('act', nn.Sigmoid())

		if optimizer_name == 'adam':
			self.optimizer = optimizer(list(self.hidden_layer.parameters()) + list(self.output_layer.parameters()), lr=lr, betas=betas)
		elif optimizer_name == 'amsgrad':
			self.optimizer = optimizer(list(self.hidden_layer.parameters()) + list(self.output_layer.parameters()), lr=lr, betas=betas, amsgrad = True)
		elif optimizer_name == 'rmsprop':
			self.optimizer = optimizer(list(self.hidden_layer.parameters()) + list(self.output_layer.parameters()), lr=lr)

	def forward(self, x):
		p_x = self.projection(x)
		h = self.hidden_layer(p_x)
		out = self.output_layer(h)
		return out
