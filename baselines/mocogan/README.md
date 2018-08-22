# MoCoGAN Baseline

Hello there, this repository is a copy of the original MoCoGAN repository used to generate our baseline test. Here, we had to make changes in arg parser and part of imports like said in original repository issues to make MoCoGAN available for our comparison. This repo fix issues like [#3](https://github.com/sergeytulyakov/mocogan/issues/3) and [#11](https://github.com/sergeytulyakov/mocogan/issues/11). The argument parser is static at the moment.

## Working Tree

```
.
└── mocogan (This Folder)
    ├── data  (Folder datasets)
    ├── originalRepFiles (Folder with original content)
    ├── *.py (files to run MoCoGAN)
```

## Install Dependencies

To run this code you need to install the packages below:
	
	
	# You can create a conda env. To help your setup
	conda create -n mocogan python=2.7
	source activate mocogan

### Python 2.7

### Numpy 1.14.1 (or 1.15.0)
	conda install numpy

### Docopt 0.6.2
	conda install docopt

### PIL 5.2.0
	
	# Try both, but I think that the second is the correct one.
	/home/$USER/anaconda2/envs/mocogan/bin/pip uninstall PIL
	/home/$USER/anaconda2/envs/mocogan/bin/pip uninstall Pillow
	# if you have problems with Pillow, uninstansll and install again.

### Tensorflow-gpu 1.4

	/home/$USER/anaconda2/envs/mocogan/bin/pip install tensorflow-gpu==1.4

### Pytorch 0.2.0

	conda install pytorch=0.1.12 torchvision cuda80 -c soumith` 

### Scipy 1.1.0
	conda install scipy
### TQDM 4.25.0
	conda install tqdm
## How to run

### Train MoCoGAN

	python train.py data/actions/ output
	# To follow MoCoGAN training process (Saving every 100k ite.)
	cd output
	tensorboard --logdir=./ --port=8890


### Generate Videos
	
	# '*' is the number of iterations
	python generate_videos.py output/generator_*.pytorch output/
	
ps: The format working is .mp4

## Acknowledgments (or GitHub References)

[MoCoGAN Official](https://github.com/sergeytulyakov/mocogan) - **Sergey Tulyakov, Ming-Yu Liu, Xiaodong Yang, Jan Kautz, "MoCoGAN: Decomposing Motion and Content for Video Generation"**
