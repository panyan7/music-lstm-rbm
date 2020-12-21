import torch
import torch.nn as nn
import pickle


class Model(nn.Module):
    def __init__(self, num_notes, num_duration, num_hidden, num_hidden_v, num_hidden_u):
        super(Model, self).__init__()
        self.nv = num_notes
        self.nu = num_duration
        self.nh = num_hidden
        self.nhv = num_hidden_v
        self.nhu = num_hidden_u
        self.nvu = self.nv + self.nu

        # trainable variables
        self.w = nn.Parameter(torch.zeros(self.nh, self.nv))
        self.wh = nn.Parameter(torch.zeros(self.nh, self.nhv))
        self.wv = nn.Parameter(torch.zeros(self.nv, self.nhv))
        self.bv = nn.Parameter(torch.zeros(self.nv))
        self.bh = nn.Parameter(torch.zeros(self.nh))
        self.lstmv = nn.LSTMCell(self.nvu, self.nhv)
        self.lstmu = nn.LSTMCell(self.nvu, self.nhu)
        self.whz = nn.Parameter(torch.zeros(self.nu, self.nhu))
        self.wvz = nn.Parameter(torch.zeros(self.nu, self.nv))
        self.bz = nn.Parameter(torch.zeros(self.nu))
        self.softmax = nn.Softmax(dim=0)

        self.hv0 = nn.Parameter(torch.zeros(self.nhv))  # Initial state of LSTM
        self.cv0 = nn.Parameter(torch.zeros(self.nhv))  # Initial cell of LSTM
        self.hu0 = nn.Parameter(torch.zeros(self.nhu))
        self.cu0 = nn.Parameter(torch.zeros(self.nhu))

        self.num_epoch = 0  # to memorize the number of epochs trained

    def sample_h(self, v, bh):
        activation = bh + torch.matmul(self.w, v)
        p_h_given_v = torch.sigmoid(activation)
        return p_h_given_v, torch.bernoulli(p_h_given_v)

    def sample_v(self, h, bv):
        activation = bv + torch.matmul(self.w.t(), h)
        p_v_given_h = torch.sigmoid(activation)
        return p_v_given_h, torch.bernoulli(p_v_given_h)

    def gibbs_sampling(self, v, bh, bv, k):
        vs = v.clone().detach()
        for i in range(k):
            hp, hs = self.sample_h(vs, bh)
            vp, vs = self.sample_v(hs, bv)
        res = vs.clone().detach()
        return vp, res

    def free_energy_cost(self, v, bh, bv, k):
        def F(v_):
            return -torch.log(1 + torch.exp(torch.matmul(v_, self.w.t()) + bh)).sum() - torch.matmul(bv.t(), v_)

        _, v_o = self.gibbs_sampling(v, bh, bv, k)
        cost = torch.sub(F(v), F(v_o))  # .mean()
        return v_o, cost

    def forward(self, x, d, k):
        x_o = [x[0]]
        d_o = [d[0]]
        hv = self.hv0
        cv = self.cv0
        hu = self.hu0
        cu = self.cu0
        cost = 0
        for t in range(1, x.shape[0]):
            v = x[t-1].clone().detach()
            u = d[t-1].clone().detach()
            vu = torch.cat((v, u), 0)
            bh_t = self.bh + torch.matmul(self.wh, hv)
            bv_t = self.bv + torch.matmul(self.wv, hv)
            hv, cv = self.lstmv(vu.view(1, self.nvu), (hv.view(1, self.nhv), cv.view(1, self.nhv)))
            hu, cu = self.lstmu(vu.view(1, self.nvu), (hu.view(1, self.nhu), cu.view(1, self.nhu)))
            hv = hv.view(self.nhv)
            cv = cv.view(self.nhv)
            hu = hu.view(self.nhu)
            cu = cu.view(self.nhu)
            v_o, cst = self.free_energy_cost(v, bh_t, bv_t, k)
            # u_o = self.softmax(self.linear(hu))
            u_o = self.softmax(torch.matmul(self.whz, hu.t()) +
                               torch.matmul(self.wvz, x[t].t().clone().detach()) + self.bz)
            cost += cst
            x_o.append(v_o)
            d_o.append(u_o)
        return torch.stack(x_o), torch.stack(d_o), (cost / x.shape[0])

    def generate(self, v0, d0, max_time, k):
        music = []
        duration = []
        prob = []
        v = v0.clone().detach().float()
        u = d0.clone().detach().float()
        hv = self.hv0.clone().detach()
        cv = self.cv0.clone().detach()
        hu = self.hu0.clone().detach()
        cu = self.cu0.clone().detach()
        for t in range(max_time):
            vu = torch.cat((v, u), 0)
            bh_t = self.bh + torch.matmul(self.wh, hv)
            bv_t = self.bv + torch.matmul(self.wv, hv)
            hv, cv = self.lstmv(vu.view(1, self.nvu), (hv.view(1, self.nhv), cv.view(1, self.nhv)))
            hu, cu = self.lstmu(vu.view(1, self.nvu), (hu.view(1, self.nhu), cu.view(1, self.nhu)))
            hv = hv.view(self.nhv)
            cv = cv.view(self.nhv)
            hu = hu.view(self.nhu)
            cu = cu.view(self.nhu)
            vp, v = self.gibbs_sampling(v, bh_t, bv_t, k)
            u = self.softmax(torch.matmul(self.whz, hu.t()) +
                             torch.matmul(self.wvz, v.t()) + self.bz)
            music.append(v)
            duration.append(u)
            prob.append(vp)
        duration = torch.stack(duration)
        am = torch.argmax(duration, dim=1)
        duration = torch.zeros_like(duration)
        for i, j in enumerate(am):
            duration[i][j] = 1
        music = torch.stack(music).clone().detach()
        return music, duration, torch.stack(prob).clone().detach()
