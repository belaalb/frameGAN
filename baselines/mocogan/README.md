## Train Mocogan

python train.py ../data/actions output

cd output
tensorboard --logdir=./ --port=8890


## Generate Videos

python generate_videos.py output/generator_*.pytorch output/
