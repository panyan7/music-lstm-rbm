# Music Generation with LSTM-RBM

This repository contains our final project for CMU's 10-701 Introduction to Machine Learning.
We implemented the LSTM-RBM model in PyTorch for chord pitch generation and additional LSTM layer for chord duration generation.

## Instructions

### Training
To train the model, first run the following command to initialize the model and the weights.
```commandline
python3 initialize.py
```
Then run the following command to train the model with 100 epochs (default).
```commandline
python3 train.py
```
Or add a commandline argument to define the number of epochs.
```commandline
python3 train.py -e 100
```
Finally generate music after the loss converges.
```commandline
python3 generate.py
```

### Select Your Own Dataset
Put your `.midi` files in the `data/midi` directory and run the following command.
```commandline
python3 read_midi.py
```
Then the default dataset will be replaced by your music.