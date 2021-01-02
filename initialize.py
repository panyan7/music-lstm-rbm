import time
import pickle
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from model import Model


class RBM(nn.Module):
    def __init__(self, visible_dim, hidden_dim):
        super(RBM, self).__init__()
        self.nv = visible_dim
        self.nh = hidden_dim
        self.w = nn.Parameter(torch.zeros(self.nh, self.nv))
        self.bh = nn.Parameter(torch.zeros(self.nh))
        self.bv = nn.Parameter(torch.zeros(self.nv))

    def sample_h(self, v):
        activation = self.bh + torch.matmul(self.w, v)
        p_h_given_v = torch.sigmoid(activation)
        return p_h_given_v, torch.bernoulli(p_h_given_v)

    def sample_v(self, h):
        activation = self.bv + torch.matmul(self.w.t(), h)
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)

    def gibbs_sampling(self, v, k):
        vs = v.clone().detach()
        for i in range(k):
            hp, hs = self.sample_h(vs)
            vp, vs = self.sample_v(hs)
        res = vs.clone().detach()
        return vp, res

    def free_energy_cost(self, v, k):
        def F(v_):
            return -torch.log(1 + torch.exp(torch.matmul(v_, self.w.t()) + self.bh)).sum() - torch.matmul(self.bv.t(), v_)

        _, v_o = self.gibbs_sampling(v, k)
        cost = torch.sub(F(v), F(v_o))
        return v_o, cost

    def forward(self, x, K):
        T = x.shape[0]
        cost = 0
        for t, v in enumerate(x):
            v_o, c = self.free_energy_cost(v, K)
            cost += c
        return (cost / T)


starttime = time.time()
with open('data/pitch.pkl', 'rb') as f:
    pitch = pickle.load(f)
with open('data/duration.pkl', 'rb') as f:
    duration = pickle.load(f)
with open('data/data.pkl', 'rb') as f:
    (pitch, duration) = pickle.load(f)

assert len(pitch) == len(duration)
num_data = len(pitch)
num_pitch = pitch[0].shape[1]
num_duration = duration[0].shape[1]

print("num_data: {}".format(num_data))
print("num_pitch: {}".format(num_pitch))
print("num_duration: {}".format(num_duration))

parser = argparse.ArgumentParser()
parser.add_argument("-n", "--num_hidden", type=int, default=int(num_pitch * 5),
                    help="Number of hidden layers for RBM")
parser.add_argument("-v", "--num_hidden_v", type=int, default=int(num_pitch * 10),
                    help="Number of hidden layers for the LSTM that generates pitch")
parser.add_argument("-u", "--num_hidden_u", type=int, default=int(num_duration*2),
                    help="Number of hidden layers for the LSTM that generates duration")
parser.add_argument("-e", "--num_epoch", type=int, default=50,
                    help="Number of epochs")
parser.add_argument("-r", "--learning_rate", type=float, default=1e-3,
                    help="Learning rate")
parser.add_argument("-k", "--sample_step", type=int, default=25,
                    help="Number of rounds for Gibbs sampling")
args = parser.parse_args()

num_hidden = args.num_hidden
num_hidden_v = args.num_hidden_v
num_hidden_u = args.num_hidden_u
epoch = args.num_epoch
learning_rate = args.learning_rate
K = args.sample_step
for arg in vars(args):
    print("{}: {}".format(arg, getattr(args, arg)))

rbm = RBM(num_pitch, num_hidden)
optimizer = optim.AdamW(rbm.parameters(), lr=learning_rate)


# Pre-training the RBM
for e in range(epoch):
    loss_epoch = 0
    for song in pitch:
        optimizer.zero_grad()
        x = song.clone().detach().float()
        loss = rbm.forward(x, K)
        loss.backward()
        optimizer.step()
        loss_epoch += loss.float() / num_data
    print("Finished epoch {}, loss {}".format(e+1, loss_epoch))

model = Model(num_pitch, num_duration, num_hidden, num_hidden_v, num_hidden_u)

model.w = nn.Parameter(rbm.w.clone().detach().float())
model.bh = nn.Parameter(rbm.bh.clone().detach().float())
model.bv = nn.Parameter(rbm.bv.clone().detach().float())

with open('data/model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Finished in {:.5f} seconds".format(time.time() - starttime))

