# Video Object Segmentation (DAVIS) Dataset Gen

Hello there, with [davis_gen.py](https://github.com/belaalb/frameGAN/blob/master/davis/davis_gen.py "davis_gen.py"), you can create your davis dataset to use in your gan project, it removes the foreground of images using the annotation.

[![DavisGif](https://github.com/belaalb/frameGAN/blob/master/davis/davis.gif)](https://github.com/belaalb/frameGAN/blob/master/davis/davis.gif "DavisGif")

## Download Video Object Segmentation (DAVIS) Dataset 2016

Download DAVIS dataset and unfolder it near this README.md:

  wget https://graphics.ethz.ch/Downloads/Data/Davis/DAVIS-data.zip

Working Tree

```
.
└── davis (This Folder)
    ├── Annotations  (Folder with segmented images)
    ├── JPEGImages (Folder with original images)
    ├── ImageSets (Separate Sets / We don't use it)
    ├── davis_gen.py (File to generate input for your project)
    ├── gif_gen.py (Generate frames to create a gif of the dataset)
  
```

## Install Dependencies

To run this code you need to install the packages below:

### Python 3.6 

### Numpy 1.14.5
  conda install numpy


### TQDM 4.23.4 (Progress Bar)
  conda install tqdm

### H5PY 2.8.0 (Save dataset to hdf5 format)
  conda install h5py

### OpenCV 3.1.0 (Debug mode)

If you don't want to see samples generator, you don't need to install (you can see in matplotlib too, but you need to adapt the code)

  conda install -c menpo opencv3

## How to run

  python davis_gen.py

### Parameters

--im-size: Height and Width of frames (default: 64)

--output-path: Path for output


## Acknowledgments (or GitHub References)

Thanks to **A Benchmark Dataset and Evaluation Methodology for Video Object Segmentation** ([F. Perazzi](http://graphics.ethz.ch/~perazzif), [J. Pont-Tuset](http://jponttuset.github.io), [B. McWilliams](https://www.inf.ethz.ch/personal/mcbrian/), [L. Van Gool](https://www.vision.ee.ethz.ch/en/members/get_member.cgi?id=1), [M. Gross](http://www.disneyresearch.com/people/markus-gross), and [A. Sorkine-Hornung](http://www.ahornung.net) _Computer Vision and Pattern Recognition (CVPR) 2016_) **to make the DAVIS dataset available for use**.
