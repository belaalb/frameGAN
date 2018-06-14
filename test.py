from __future__ import print_function
import argparse
import torch
from torch.autograd import Variable
import models_zoo
from cup_generator.model import Generator
from data_load import Loader
from train_loop import TrainLoop

import matplotlib.pyplot as plt
import numpy as np

from PIL import Image, ImageEnhance

import torchvision.transforms as transforms

def test_model(generator, f_generator, n_tests, cuda_mode, enhancement):

	model.eval()

	to_pil = transforms.ToPILImage()

	for i in range(n_tests):

		z_ = torch.randn(1, 100).view(-1, 100)

		if cuda_mode:
			sample_in = sample_in.cuda()

		z_ = Variable(z_)
		out = self.generator.forward(z_)

		frames_list = []

		for j in range(out.size(1)):
			gen_frame = f_generator(out[:,j,:].squeeze(1).contiguous())
			frames_list.append(gen_frame.squeeze().unsqueeze(2))

		sample_rec = torch.cat(frames_list, 0)

		save_gif(sample_rec, str(i+1)+'_rec.gif', enhance=enhancement)

def save_gif(data, file_name, enhance):

	data = data.view([40, 30, 30])

	to_pil = transforms.ToPILImage()

	if enhance:
		frames = [ImageEnhance.Sharpness( to_pil(frame.unsqueeze(0)) ).enhance(10.0) for frame in data]
	else:
		frames = [to_pil(frame.unsqueeze(0)) for frame in data]

	frames[0].save(file_name, save_all=True, append_images=frames[1:])


def plot_learningcurves(history, keys):

	for i, key in enumerate(keys):
		plt.figure(i+1)
		plt.plot(history[key])
		plt.title(key)
	
	plt.show()

if __name__ == '__main__':

	# Testing settings
	parser = argparse.ArgumentParser(description='Testing online transfer learning for emotion recognition tasks')
	parser.add_argument('--cp-path', type=str, default=None, metavar='Path', help='Checkpoint/model path')
	parser.add_argument('--generator-path', type=str, default=None, metavar='Path', help='Path for generator params')
	parser.add_argument('--input-data-path', type=str, default='./data/input/', metavar='Path', help='Path to data input data')
	parser.add_argument('--targets-data-path', type=str, default='./data/targets/', metavar='Path', help='Path to output data')
	parser.add_argument('--n-tests', type=int, default=4, metavar='N', help='number of samples to  (default: 64)')
	parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables GPU use')
	parser.add_argument('--no-plots', action='store_true', default=False, help='Disables plot of train/test losses')
	parser.add_argument('--enhance', action='store_true', default=True, help='Enables enhancement')
	parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
	args = parser.parse_args()
	args.cuda = True if not args.no_cuda and torch.cuda.is_available() else False

	torch.manual_seed(args.seed)
	if args.cuda:
		torch.cuda.manual_seed(args.seed)

	generator = models_zoo.Generator(args.cuda)
	frames_generator = models_zoo.frames_generator().eval()

	ckpt = torch.load(args.cp_path, map_location = lambda storage, loc: storage)

	history = ckpt['history']

	generator.load_state_dict(ckpt['model_state'])

	f_gen_state = torch.load(args.generator_path, map_location=lambda storage, loc: storage)
	frames_generator.load_state_dict(gen_state['model_state'])

	if args.cuda:
		generator = generator.cuda()
		frames_generator = frames_generator.cuda()

	test_model(generator=generator, f_generator=frames_generator, n_tests=args.n_tests, cuda_mode=args.cuda, enhancement=args.enhance)

	if not args.no_plots:
		plot_learningcurves(history, list(history.keys()))
