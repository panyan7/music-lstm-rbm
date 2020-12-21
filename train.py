import time
import pickle
import argparse
import torch
import torch.optim as optim
from model import Model


# Helper functions
def get_accuracy(x, x_o):
    x = x.clone().detach()
    x_o = x_o.clone().detach()
    res = 0
    for t in range(1, x.shape[0]):
        v = x[t].int()
        v_o = x_o[t].int()
        try:
            res += int(torch.bitwise_and(v, v_o).sum()) / int(torch.bitwise_or(v, v_o).sum())
        except ZeroDivisionError:
            res += 1
    res = res / x.shape[0]
    return res


def cross_entropy_loss(d, d_o):
    loss = 0
    for t in range(1, d.shape[0]):
        u = d[t].float()
        u_o = d_o[t].float()
        loss -= torch.matmul(u, torch.log(u_o).t())
    loss = loss / d.shape[0]
    return loss


# The training begins here
starttime = time.time()

# Load dataset
with open('data/pitch.pkl', 'rb') as f:
    pitch = pickle.load(f)
with open('data/duration.pkl', 'rb') as f:
    duration = pickle.load(f)

# Load model
with open('data/model.pkl', 'rb') as f:
    model = pickle.load(f)


# The constants
num_pitch = pitch[0].shape[1]
num_duration = duration[0].shape[1]
num_hidden = model.nh
num_hidden_v = model.nhv
num_hidden_u = model.nhu
learning_rate = 1e-3
epochs = 100
K = 25
num_data = len(pitch)
save_for_every = 5


# Print the constants
print("Parameters:")
print("Number of songs in dataset {}".format(num_data))
print("Number of pitch: {}".format(num_pitch))
print("Number of duration: {}".format(num_duration))
print("RBM hidden states: {}".format(num_hidden))
print("LSTM hidden states (pitch): {}".format(num_hidden_v))
print("LSTM hidden states (duration): {}".format(num_hidden_u))
print("Learning rate:{}".format(learning_rate))
print("Current epoch:{}".format(model.num_epoch))
print("Size of data set:{}".format(num_data))


# Initialize the optimizer
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)


# Training
for e in range(epochs):
    loss_epoch_v = 0
    loss_epoch_u = 0
    accuracy_epoch = 0
    for i in range(num_data):
        optimizer.zero_grad()
        x = pitch[i].float().clone().detach()
        d = duration[i].float().clone().detach()
        x_o, d_o, loss_v = model.forward(x, d, K)
        loss_u = cross_entropy_loss(d, d_o)
        loss_epoch_u += loss_u.float()
        loss_epoch_v += loss_v.float()
        accuracy_epoch += get_accuracy(x, x_o)
        loss_v.backward()
        loss_u.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)  # gradient clipping
        optimizer.step()

    loss_epoch_v /= num_data
    loss_epoch_u /= num_data
    accuracy_epoch /= num_data

    model.num_epoch += 1
    if model.num_epoch % save_for_every == 0:  # Save the model after some epochs
        with open('../data/model/epochs/model_{}.pkl'.format(model.num_epoch), 'wb') as f:
            pickle.dump(model, f)
        with open('../data/model/model.pkl', 'wb') as f:
            pickle.dump(model, f)

    print("Finished epoch {}, free energy difference {}, cross entropy loss {}, accuracy {}".format(model.num_epoch, loss_epoch_v, loss_epoch_u, accuracy_epoch))

print("Finished in {} seconds".format(int(time.time() - starttime)))
