

# Flying Shape Dataset Gen

Hello there, with [flying_shapes_gen.py](https://github.com/belaalb/frameGAN/blob/master/flying_shapes/flying_shapes_gen.py "flying_shapes_gen.py"), you can create your flying shape dataset to use in your project.

[![FlyingShapesGif](https://github.com/belaalb/frameGAN/blob/master/flying_shapes/flying_shapes.gif)](https://github.com/belaalb/frameGAN/blob/master/flying_shapes/flying_shapes.gif "FlyingShapesGif")

## Install Dependencies

To run this code you need to install the packages below:

### Python 3.6 

The original code was done to python 2.7, so we upgraded to 3.6

### Numpy 1.14.5
	conda install numpy

### Matplotlib 2.2.2 (Used in original code, so still need it in dataset.py file)
	conda install matplotlib

### TQDM 4.23.4 (Progress Bar)
	conda install tqdm

### H5PY 2.8.0 (Save dataset to hdf5 format)
	conda install h5py

### OpenCV 3.1.0 (Debug mode)

If you don't want to see samples generator, you don't need to install (you can see in matplotlib too, but you need to adapt the code)

	conda install -c menpo opencv3

## How to run

	python flying_shapes_gen.py --im-size 10 --n-frames 10 --n-samples 5 --file-name rapela.hdf --debug 1 --opencv 1

### Parameters

--im-size: Height and Width of frames (default: 30)

--n-frames: Number of frames per sample (default: 128)

--n-samples: Number of output samples (default: 500)

--output-path: Path for output

--file-name: Output file name

--debug: Debug Flag 	1/0

--opencv:  Debug Opencv Flag 1/0

* Debug flag is used to print in console the dataset and samples tensors
* Opencv flag is used to display each sample generated

## Acknowledgments (GitHub References)

The flying-shapes dataset was created with https://github.com/brain-research/flying-shapes
