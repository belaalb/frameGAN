# Copyright 2018 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import pdb
import os
import sys
import cv2
import numpy as np
from third_party import dataset


def data_gen():
  
  #################
  # CONFIG
  #################

  experiment_path = 'rapela'#None
  n_samples = 2 # num of samples
  num_frames = 30 # num of frames per sample (The image size is changed inside dataset.py)
  batch_size = 5 # default of original file
  config = locals() #dict(locals(), **FLAGS) #update locals with any flags passed by cmdln

  i = 0
  while os.path.exists(experiment_path + "_" + str(i)):
    i += 1
  experiment_path = experiment_path + "_" + str(i)
  os.mkdir(experiment_path)
  config['experiment_path'] = experiment_path
  print('Saving to ' + str(experiment_path))

  # write config file
  #with open(os.path.join(experiment_path, 'exp_config.txt'), 'w') as f:
  #  for key in sorted(config):
  #    f.write(key + '\t' + str(config[key]) + '\n')

  data_source = dataset.FlyingShapesDataHandler(batch_size=batch_size,seq_len=num_frames)
  
  for n in range(n_samples):
    np_batch = data_source.GetUnlabelledBatch()
    batch = {
    'image': np_batch['image'],
    'bbox': np_batch['bbox']
    }
    print ('Sample ' + str(n+1) )
    for i in range(num_frames):
      print(batch['image'][0][i].shape)
      cv2.imshow("teste",batch['image'][0][i])
      cv2.waitKey(0)
  exit(0)


        
#import argparse
#import pickle
#import h5py
#import numpy as np
#
#import matplotlib
#matplotlib.use('Agg')
#import matplotlib.pyplot as plt
#

if __name__ == '__main__':
  

  # Data settings
  #parser = argparse.ArgumentParser(description='Generate Bouncing balls dataset')
  #parser.add_argument('--im-size', type=int, default=32, metavar='N', help='H and W of frames (default: 32)')
  #parser.add_argument('--n-balls', type=int, default=3, metavar='N', help='Number of bouncing balls (default: 3)')
  #parser.add_argument('--n-frames', type=int, default=128, metavar='N', help='Number of frames per sample (default: 128)')
  #parser.add_argument('--n-samples', type=int, default=500, metavar='N', help='Number of output samples (default: 500)')
  #parser.add_argument('--output-path', type=str, default='./', metavar='Path', help='Path for output')
  #parser.add_argument('--file-name', type=str, default='train.hdf', metavar='Path', help='Output file name')
  #args = parser.parse_args()

  dat = data_gen()

  for i in range(args.n_samples):
    sample=bounce_mat(res=args.im_size, n=args.n_balls, T=args.n_frames)
    dat.append( np.moveaxis(sample[i], 0, -1) )
    print(dat[i].shape)
    print(i)

  hdf5 = h5py.File(args.output_path+args.file_name, 'w')
  dataset = hdf5.create_dataset('data', data=np.asarray(dat))
  hdf5.close()