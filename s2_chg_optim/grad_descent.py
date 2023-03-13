import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable


class TVLoss(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(TVLoss, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        t_x = x.size()[1]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_t = self._tensor_size(x[:, 1:, :, :])
        count_h = self._tensor_size(x[:, :, 1:, :])
        count_w = self._tensor_size(x[:, :, :, 1:])
        t_tv = torch.pow((x[:, 1:, :, :] - x[:, :t_x - 1, :, :]), 2).sum()
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * 3 * (h_tv / count_h + w_tv / count_w + t_tv / count_t) / batch_size

    def _tensor_size(self, t):
        return t.size()[1] * t.size()[2] * t.size()[3]


def grad_descent_ON_OFF(L, inter_on, inter_off, optim_iter=100):

    rangeT, rangeY, rangeX = inter_on.shape
    W_ON = torch.ones(rangeT, rangeY, rangeX) * (L / 2 - inter_on.sum(0)) / rangeT
    W_ON = Variable(W_ON, requires_grad=True)
    W_OFF = torch.ones(rangeT - 1, rangeY, rangeX) * (L / 2 - inter_off.sum(0)) / rangeT
    W_OFF = Variable(W_OFF, requires_grad=True)

    if torch.is_tensor(L) == False:
        L = torch.from_numpy(L)
    if torch.is_tensor(inter_on) == False:
        inter_on = torch.from_numpy(inter_on)
    if torch.is_tensor(inter_off) == False:
        inter_off = torch.from_numpy(inter_off)

    L_TV = TVLoss()

    optimizer = optim.SGD([W_ON, W_OFF], lr=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, optim_iter)
    for i in range(optim_iter):
        scheduler.step()
        with torch.autograd.set_detect_anomaly(True):

            W_OFF_last = L - inter_off.sum(0) - inter_on.sum(0) - W_ON.sum(0) - W_OFF.sum(0)

            M_ON = W_ON
            M_OFF = torch.cat((W_OFF, W_OFF_last.unsqueeze(0)), 0)

            M_ON_L1 = torch.norm(M_ON, p=1)
            M_OFF_L1 = torch.norm(M_OFF, p=1)
            loss_L1 = torch.sum(M_ON_L1 + M_OFF_L1)

            NM_ON = inter_on + M_ON
            NM_OFF = inter_off + M_OFF

            loss_TV_ON = L_TV(NM_ON.unsqueeze(0))
            loss_TV_OFF = L_TV(NM_OFF.unsqueeze(0))
            loss_TV = torch.sum(loss_TV_ON + loss_TV_OFF)

            if i == 0:
                lamb_1 = 1
                lamb_2 = 150000
                lamb = [lamb_1, lamb_2]

            optimizer.zero_grad()
            if i % 2 == 0:
                loss = lamb[0] * loss_L1
            if i % 2 == 1:
                loss = lamb[1] * loss_TV

            loss.backward(retain_graph=True)
            optimizer.step()

            # all_loss = lamb[0] * loss_L1.item()+ lamb[1] * loss_TV.item()
            # if i % 10 == 0:
            #     print(all_loss, loss_L1.item(), loss_TV.item())

    return (inter_on + M_ON).detach().numpy(), (inter_off + M_OFF).detach().numpy(), (M_ON).detach().numpy(), (M_OFF).detach().numpy()


if __name__ == '__main__':
    import numpy as np
    L = [[3, 0, 0], [0, 3, 0], [0, 0, 3]]
    E_ON = [[[2, 0, 0], [0, 0, 0], [0, 0, 2]], [[0, 0, 0], [0, 2, 0], [0, 0, 2]], [[2, 0, 0], [0, 2, 0], [0, 0, 0]]]
    E_OFF = [[[-1, 0, 0], [0, -1, 0], [0, 0, 0]], [[-1, 0, 0], [0, -1, 0], [0, 0, -1]], [[0, 0, 0], [0, -1, 0], [0, 0, -1]]]
    grad_descent_ON_OFF(np.array(L), np.array(E_ON), np.array(E_OFF))