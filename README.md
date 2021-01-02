# Music Generation with LSTM-RBM

This repository contains our final project for CMU's [10-701 Introduction to Machine Learning](https://www.cs.cmu.edu/~epxing/Class/10701-20/).
Our idea came from the RNN-RBM model in this ICML paper [(Boulanger-Lewandowski et al., 2012)](https://arxiv.org/pdf/1206.6392.pdf).
We implemented the LSTM-RBM model in PyTorch for chord pitch generation and additional LSTM layer for chord duration generation.

## Dependencies
The list of libraries and their versions that we are using.
```
torch 1.6.0
numpy 1.18.5
music21 6.3.0
```
You can install or upgrade these libraries using `pip`.

## Instructions

### Training
To train the model, first run the following command to initialize the model and the weights.
The first argument `num_hidden` is the size of hidden layer of RBM, and the second and third arguments `num_hidden_v` and `num_hidden_u` are the sizes of hidden layers of the two LSTMs respectively.
The number of epochs is 50, learning rate is 0.001, and k is 25<sup>[1](#k)</sup> by default.
```commandline
python3 initialize.py
python3 initialize.py -n [num_hidden] -v [num_hidden_v] -u [num_hidden_u] -e [num_epoch] -r [learning_rate] -k [sample_step]
```

Then, run the following command to train the model.
The number of epochs is 100, learning rate is 0.001, and k is 25 by default.
This file can be run repeatedly, the output model will be trained based on the latest model.
To train a new network you need to repeat the initialization step.
```commandline
python3 train.py
python3 train.py -e [num_epoch] -r [learning_rate] -k [sample_step]
```

Finally generate music after the loss converges. 
The tempo is 75.0, length is 100, and k is 10 by default.
```commandline
python3 generate.py
python3 generate.py -k [sample_step] -l [length] -t [tempo] -f [output_filename]
```

### Select Your Own Dataset
Put any `.midi` files<sup>[2](#midi)</sup> in the `data/midi/` directory and run the following command.
```commandline
python3 process.py
```
Then the default dataset will be replaced by your selected music in the directory.
Currently the data has only one music, so ideally you should select some of your own music.

----------
<a name="k">1.</a> RBM generates data by performing a Gibbs sampling with k steps, and you need to specify this constant k here.

<a name="midi">2.</a> The midi-version musics of some popular classical music composers can be downloaded from [Classical Piano Midi Page](http://www.piano-midi.de/midicoll.htm). We would suggest using songs of the same instrument, and if possible the same composer, in order to get good results.*
