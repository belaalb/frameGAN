from __future__ import print_function
import argparse
import torch
import models_zoo
from data_load import Loader
from train_loop import TrainLoop
from torch.utils.data.dataloader import DataLoader
import torch.optim as optim

# Training settings
parser = argparse.ArgumentParser(description='frame GAN')
parser.add_argument('--batch-size', type=int, default=128, metavar='N', help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=200, metavar='N', help='number of epochs to train (default: 200)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='LR', help='learning rate (default: 0.0002)')
parser.add_argument('--beta1', type=float, default=0.5, metavar='beta1', help='Adam beta 1 (default: 0.5)')
parser.add_argument('--beta2', type=float, default=0.999, metavar='beta2', help='Adam beta 2 (default: 0.99)')
parser.add_argument('--targets-data-path', type=str, default='./data/targets/', metavar='Path', help='Path to output data')
parser.add_argument('--checkpoint-epoch', type=int, default=None, metavar='N', help='epoch to load for checkpointing. If None, training starts from scratch')
parser.add_argument('--checkpoint-path', type=str, default=None, metavar='Path', help='Path for checkpointing')
parser.add_argument('--generator-path', type=str, default=None, metavar='Path', help='Path for frames generator params')
parser.add_argument('--ndiscriminators', type=int, default=8, help='Number of discriminators. Default=8')
parser.add_argument('--nadir-slack', type=float, default=1.5, metavar='nadir', help='factor for nadir-point update. Only used in hyper mode (default: 1.5)')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--save-every', type=int, default=5, metavar='N', help='how many batches to wait before logging training status. (default: 5)')
parser.add_argument('--n-workers', type=int, default=4)
parser.add_argument('--gen-arch', choices=['linear', 'conv'], default='linear', help='Linear or convolutional generator')
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables GPU use')
parser.add_argument('--average-mode', action='store_true', default=False, help='Disables hypervolume maximization and uses average loss instead')
args = parser.parse_args()
args.cuda = True if not args.no_cuda and torch.cuda.is_available() else False

train_data_set = Loader(hdf5_name=args.targets_data_path)
train_loader = DataLoader(train_data_set, batch_size=args.batch_size, shuffle=False, num_workers=args.n_workers)

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

if args.gen_arch == 'conv':
	generator = models_zoo.Generator_conv(args.cuda)
elif args.gen_arch == 'linear':
	generator = models_zoo.Generator_linear(args.cuda)

frames_generator = models_zoo.frames_generator().eval()

gen_state = torch.load(args.generator_path, map_location=lambda storage, loc: storage)
frames_generator.load_state_dict(gen_state['model_state'])

disc_list = []
for i in range(args.ndiscriminators):
	disc = models_zoo.Discriminator(optim.Adam, args.lr, (args.beta1, args.beta2)).train()
	disc_list.append(disc)

if args.cuda:
	generator = generator.cuda()
	frames_generator = frames_generator.cuda()
	for disc in disc_list:
		disc = disc.cuda()
	torch.backends.cudnn.benchmark=True

optimizer_g = optim.Adam(generator.parameters(), lr=args.lr, betas=(args.beta1, args.beta2))

trainer = TrainLoop(generator, frames_generator, disc_list, optimizer_g, train_loader, nadir_slack=args.nadir_slack, checkpoint_path=args.checkpoint_path, checkpoint_epoch=args.checkpoint_epoch, hyper=not args.average_mode, cuda=args.cuda)

print('Cuda Mode is: {}'.format(args.cuda))

trainer.train(n_epochs=args.epochs, save_every = args.save_every)
