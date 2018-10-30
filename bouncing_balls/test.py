from __future__ import print_function
import argparse
import torch
import models_zoo
from data_load import Loader
import subprocess

import matplotlib.pyplot as plt
import numpy as np

from PIL import Image, ImageEnhance

import torchvision.transforms as transforms


from sklearn.metrics import mean_squared_error


def test_lin_interp(f_generator, n_tests, cuda_mode):

	intra_mse = []

	for test in range(n_tests):

		z12 = np.random.randn(30, 100)

		#z12_ = np.random.randn(2, 100)
	
		#alpha_list = list(np.arange(1, 31)/30)

		#z_interp = []
		#for alpha in alpha_list:
		#	z_interp.append((1-alpha)*z12[0] + alpha*z12[-1])

		#z_interp = torch.Tensor(np.asarray(z_interp))
		z_interp = torch.Tensor(np.asarray(z12))

		f_generator.eval()
		generator.eval()

		to_pil = transforms.ToPILImage()

		if args.cuda:
			z_interp = z_interp.cuda()

		frames_list = []

		for j in range(z_interp.size(0)):
			gen_frame = f_generator(z_interp[j,:].unsqueeze(0).contiguous())
			frames_list.append(gen_frame.squeeze().unsqueeze(2))
		mse_sk = 0.0
		#mse_sk = []
		count = 0
		for j in range(1, z_interp.size(0)):
			if j%1==0:
				a = frames_list[j].detach().cpu().numpy().squeeze()
				a[a > 0] = 1.
				a[a < 0] = 0.
				b = frames_list[j-1].detach().cpu().numpy().squeeze()
				b[b > 0] = 1.
				b[b < 0] = 0.
				count += 1
				mse_sk += mean_squared_error(a, b)
		
		mse_sk /= count

		intra_mse.append(np.asarray(mse_sk))

	return intra_mse


def test_model(generator, f_generator, n_tests, cuda_mode, enhancement, delay):

	f_generator.eval()
	generator.eval()

	to_pil = transforms.ToPILImage()

	n_cols, n_rows = (n_tests, 30)
	fig, axes = plt.subplots(n_cols, n_rows, figsize=(n_rows, n_cols))

	for i in range(n_tests):

		z_ = torch.randn(1, 100).view(-1, 100, 1)

		if args.cuda:
			z_ = z_.cuda()

		out = generator.forward(z_)

		frames_list = []

		for j in range(out.size(1)):
			gen_frame = f_generator(out[:,j,:].squeeze(1).contiguous())
			frames_list.append(gen_frame.squeeze().unsqueeze(2))

		sample_rec = torch.cat(frames_list, 0)
		save_gif(sample_rec, str(i+1)+'_rec.gif', enhance=enhancement, delay = delay)

		data = sample_rec.view([30, 30, 30]).cpu().detach()

		for ax, img in zip(axes[i, :].flatten(), data):
			ax.axis('off')
			ax.set_adjustable('box-forced')

			ax.imshow(img, cmap="gray", aspect='equal')
		
		plt.subplots_adjust(wspace=0, hspace=0)

	save_fn = 'all.pdf'
	plt.savefig(save_fn)

	plt.close()

def save_separate(generator, f_generator, n_tests, cuda_mode, enhancement, delay):

	f_generator.eval()
	generator.eval()

	to_pil = transforms.ToPILImage()

	intra_mse = []

	for i in range(n_tests):

		z_ = torch.randn(1, 100).view(-1, 100, 1)

		if args.cuda:
			z_ = z_.cuda()

		out = generator.forward(z_)

		frames_list = []

		for j in range(out.size(1)):
			gen_frame = f_generator(out[:,j,:].squeeze(1).contiguous())
			frames_list.append(gen_frame.squeeze().unsqueeze(2))

		mse_sk = 0.0
		#mse_sk = []
		count = 0
		for j in range(1, out.size(1)):
			if j%1==0:
				a = frames_list[j].detach().cpu().numpy().squeeze()
				a[a > 0] = 1.
				a[a < 0] = 0.
				b = frames_list[j-1].detach().cpu().numpy().squeeze()
				b[b > 0] = 1.
				b[b < 0] = 0.
				count += 1
				mse_sk += mean_squared_error(a, b)
			
			#mse += torch.nn.functional.mse_loss(frames_list[j], frames_list[j-1]).item()

		#mse /= (j+1)

		mse_sk /= count

		intra_mse.append(np.asarray(mse_sk))

		sample_rec = torch.cat(frames_list, 0)
		data = sample_rec.view([30, 30, 30]).cpu().detach()

		n_cols, n_rows = (1, 30)
		fig, axes = plt.subplots(n_cols, n_rows, figsize=(n_rows, n_cols))

		for ax, img in zip(axes.flatten(), data):
			ax.axis('off')
			ax.set_adjustable('box-forced')

			ax.imshow(img, cmap="gray", aspect='equal')
		
		plt.subplots_adjust(wspace=0, hspace=0)

		save_fn = 'video'+ str(i) +'.png'
		plt.savefig(save_fn)

		plt.close()

	return intra_mse

def save_gif(data, file_name, enhance, delay):

	data = data.view([30, 30, 30]).detach().cpu()

	to_pil = transforms.ToPILImage()

	if enhance:
		frames = [ImageEnhance.Sharpness( to_pil(frame.unsqueeze(0)) ).enhance(10.0) for frame in data]
	else:
		frames = [to_pil(frame.unsqueeze(0)) for frame in data]

	frames[0].save(file_name, save_all=True, append_images=frames[1:])

	subprocess.call("gifsicle --delay " + str(delay) + " " + file_name + " > " + "s" + file_name, shell = True)

def plot_real(n_tests, data_path):

	real_loader = Loader(hdf5_name = data_path)

	n_cols, n_rows = (n_tests, 30)
	fig, axes = plt.subplots(n_cols, n_rows, figsize=(n_rows, n_cols))

	intra_mse = []

	for i in range(n_tests):

		img_idx = np.random.randint(len(real_loader))
		real_sample = real_loader[img_idx].squeeze()
	
		for ax, img in zip(axes[i, :].flatten(), real_sample):
			ax.axis('off')
			ax.set_adjustable('box-forced')

			ax.imshow(img, cmap="gray", aspect='equal')
		
		plt.subplots_adjust(wspace=0, hspace=0)

		mse = 0.0
		#mse = []		
		count = 0
		for j in range(1, real_sample.size(0)):
			if j%1==0:
				a = real_sample[j].numpy().squeeze()
				a[a > 0] = 1.
				a[a < 0] = 0.
				b = real_sample[j-1].numpy().squeeze()
				b[b > 0] = 1.
				b[b < 0] = 0.
				count += 1
				mse += mean_squared_error(a, b)
				#mse+=torch.nn.functional.mse_loss(real_sample[j,:,:], real_sample[j-1,:,:]).item()

		mse /= count

		intra_mse.append(np.asarray(mse))

	save_fn = 'real.pdf'
	plt.savefig(save_fn)

	plt.close()

	return intra_mse

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
	parser.add_argument('--gen-arch', choices=['linear', 'conv'], default='linear', help='Linear or convolutional generator')
	parser.add_argument('--generator-path', type=str, default=None, metavar='Path', help='Path for generator params')
	parser.add_argument('--n-tests', type=int, default=4, metavar='N', help='number of tests  (default: 4)')
	parser.add_argument('--delay', type=int, default=20, metavar='N', help='Delay between frames  (default: 20)')
	parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables GPU use')
	parser.add_argument('--no-plots', action='store_true', default=False, help='Disables plot of train/test losses')
	parser.add_argument('--plot-real', action='store_true', default=False, help='Disables plot of real data')
	parser.add_argument('--realdata-path', type=str, default=None, metavar='Path', help='Dataset path')
	parser.add_argument('--enhance', action='store_true', default=True, help='Enables enhancement')
	parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
	args = parser.parse_args()
	args.cuda = True if not args.no_cuda and torch.cuda.is_available() else False

	torch.manual_seed(args.seed)
	if args.cuda:
		torch.cuda.manual_seed(args.seed)

	if args.gen_arch == 'conv':
		generator = models_zoo.Generator_conv(args.cuda)
	elif args.gen_arch == 'linear':
		generator = models_zoo.Generator_linear(args.cuda)

	frames_generator = models_zoo.frames_generator().eval()

	ckpt = torch.load(args.cp_path, map_location = lambda storage, loc: storage)

	history = ckpt['history']

	generator.load_state_dict(ckpt['generator_state'])

	f_gen_state = torch.load(args.generator_path, map_location=lambda storage, loc: storage)
	frames_generator.load_state_dict(f_gen_state['model_state'])

	if args.cuda:
		generator = generator.cuda()
		frames_generator = frames_generator.cuda()

	if args.plot_real:
		real_mse = plot_real(args.n_tests, args.realdata_path)
		print(np.mean(real_mse), np.std(real_mse))
		#real_mse = np.asarray(real_mse)

	fake_mse = save_separate(generator=generator, f_generator=frames_generator, n_tests=args.n_tests, cuda_mode=args.cuda, enhancement=args.enhance, delay=args.delay)
	print(np.mean(fake_mse), np.std(fake_mse))
	#test_model(generator=generator, f_generator=frames_generator, n_tests=args.n_tests, cuda_mode=args.cuda, enhancement=args.enhance, delay=args.delay)

	lin_interp_mse = test_lin_interp(f_generator=frames_generator, n_tests=args.n_tests, cuda_mode=args.cuda)
	print(np.mean(lin_interp_mse), np.std(lin_interp_mse))

	if not args.no_plots:
		plot_learningcurves(history, list(history.keys()))

	plt.hist(real_mse)
	plt.hist(fake_mse)
	plt.hist(lin_interp_mse)
	plt.savefig('histogram.png')
	plt.show()
	plt.close()

	plt.boxplot([real_mse, fake_mse, lin_interp_mse])
	plt.savefig('boxplot.png')
	plt.show()


