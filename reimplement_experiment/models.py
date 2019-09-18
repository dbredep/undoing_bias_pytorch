import torch
import torch.nn as nn
import pdb


class LinearSVM(nn.Module):
    """Support Vector Machine"""

    def __init__(self, in_feature):
        super(LinearSVM, self).__init__()
        self.w = torch.rand((in_feature, 1), requires_grad=True, device="cuda")
        self.b = torch.rand(1, requires_grad=True, device="cuda")

    def forward(self, x):
        h = torch.mm(x, self.w) + self.b
        return h


class UndoingSVM(nn.Module):
    """Undoing Support Vector Machine"""

    def __init__(self, in_feature, dataset_num):
        super(UndoingSVM, self).__init__()
        self.w = torch.rand((in_feature, 1), requires_grad=True, device="cuda")
        self.b = torch.rand(1, requires_grad=True, device="cuda")
        self.delta_w = {}
        self.delta_b = {}
        for i in range(dataset_num):
            self.delta_w[i] = torch.rand((in_feature, 1), requires_grad=True, device="cuda")
            self.delta_b[i] = torch.rand(1, requires_grad=True, device="cuda")

    def forward(self, x, dataset_id):
        #pdb.set_trace()
        d = int(dataset_id.item())
        h = torch.mm(x, self.w) + self.b
        h_bias = torch.mm(x, self.w + self.delta_w[d]) + self.b + self.delta_b[d]
        return h, h_bias