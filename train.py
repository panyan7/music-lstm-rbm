import os
import time
import pickle
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from model import MusicGenerationModel


# Helper functions
def get_accuracy(x, x_o):
    res = 0
    for t in range(1, x.shape[0]):
        v, v_o = x[t].int(), x_o[t].int()
        try:
            res += int(torch.bitwise_and(v, v_o).sum()) / int(torch.bitwise_or(v, v_o).sum())
        except ZeroDivisionError:
            res += 1
    return res / x.shape[0]


def cross_entropy_loss(d, d_o):
    loss = 0
    for t in range(1, d.shape[0]):
        u, u_o = d[t].float(), d_o[t].float()
        loss -= torch.matmul(u, torch.log(u_o).t())
    loss = loss / d.shape[0]
    return loss


def main():
    starttime = time.time()
    # Load dataset
    with open('data/data.pkl', 'rb') as f:
        (pitch, duration) = pickle.load(f)

    # Load model
    with open('data/model.pkl', 'rb') as f:
        model = pickle.load(f)

    # The constants
    num_data = len(pitch)
    num_pitch = pitch[0].shape[1]
    num_duration = duration[0].shape[1]
    num_hidden = model.nh
    num_hidden_v = model.nhv
    num_hidden_u = model.nhu

    print("num_data: {}".format(num_data))
    print("num_pitch: {}".format(num_pitch))
    print("num_duration: {}".format(num_duration))
    print("num_hidden: {}".format(num_hidden))
    print("num_hidden_v: {}".format(num_hidden_v))
    print("num_hidden_u: {}".format(num_hidden_u))

    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--num_epoch", type=int, default=100,
                        help="Number of epochs")
    parser.add_argument("-r", "--learning_rate", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("-k", "--sample_step", type=int, default=25,
                        help="Number of rounds for Gibbs sampling")
    parser.add_argument("-s", "--save_for_every", type=int, default=5,
                        help="Number of epochs to save the model")
    args = parser.parse_args()

    for arg in vars(args):
        print("{}: {}".format(arg, getattr(args, arg)))

    epochs = args.num_epoch
    learning_rate = args.learning_rate
    K = args.sample_step
    save_for_every = args.save_for_every

    print("current_epoch:{}".format(model.num_epoch))

    # Initialize the optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

    # Training
    for e in range(epochs):
        loss_epoch_v, loss_epoch_u, accuracy_epoch = 0, 0, 0
        for i in range(num_data):
            x = pitch[i].float().clone().detach()
            d = duration[i].float().clone().detach()

            # Feed forward the current sequence
            optimizer.zero_grad()  # Set the gradients to 0
            x_o, d_o, loss_v = model.forward(x, d, K)
            loss_u = cross_entropy_loss(d, d_o)

            # Loss and accuracy to be printed
            loss_epoch_u += loss_u.float() / num_data
            loss_epoch_v += loss_v.float() / num_data
            accuracy_epoch += get_accuracy(x, x_o) / num_data

            # Back-propagation through time
            loss_v.backward()
            loss_u.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10)  # gradient clipping
            optimizer.step()

        model.num_epoch += 1  # Update the number of epochs
        if model.num_epoch % save_for_every == 0:  # Save the model for each certain number of epochs
            with open('data/model.pkl', 'wb') as f:
                pickle.dump(model, f)

        print("Finished epoch {}, free energy difference {:.5f}, cross entropy loss {:.5f}, accuracy {:.5f}".format(model.num_epoch, loss_epoch_v, loss_epoch_u, accuracy_epoch))

    print("Finished in {:.5f} seconds".format(time.time() - starttime))


if __name__ == '__main__':
    main()

