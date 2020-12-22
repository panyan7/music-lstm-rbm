# Music Generation with LSTM-RBM

This repository contains our final project for CMU's [10-701 Introduction to Machine Learning](https://www.cs.cmu.edu/~epxing/Class/10701-20/).
We implemented the LSTM-RBM model in PyTorch for chord pitch generation and additional LSTM layer for chord duration generation.

*Note: We are still wrapping up the final code. The full features are not supported yet.*

## Instructions

### Training
To train the model, first run the following command to initialize the model and the weights.
The first argument `num_hidden` is the size of hidden layer of RBM, and the second and third arguments `num_hidden_v` and `num_hidden_u` are the sizes of hidden layers of the two LSTMs respectively.
```commandline
python3 initialize.py
python3 initialize.py -h [num_hidden] -v [num_hidden_v] -u [num_hidden_u] -e [num_epochs] -r [learning_rate]
```

Then, run the following command to train the model with 100 epochs (default).
```commandline
python3 train.py
python3 train.py -e [num_epochs] -r [learning_rate]
```

Finally generate music after the loss converges.
```commandline
python3 generate.py
python3 generate.py -l [length] -t [tempo]
```

### Select Your Own Dataset
Put your `.midi` files in the `data/midi` directory and run the following command.
```commandline
python3 read_midi.py
```
Then the default dataset will be replaced by your music.
