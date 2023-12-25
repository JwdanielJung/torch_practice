import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable
from torch.nn import Parameter
from torch import Tensor
import torch.nn.functional as F
from math import sqrt

# you can refer to the implementation provided by PyTorch for more information
# https://pytorch.org/docs/stable/generated/torch.nn.GRUCell.html
"""
Parameters
input_size – The number of expected features in the input x
hidden_size – The number of features in the hidden state h
bias – If False, then the layer does not use bias weights b_ih and b_hh. Default: True

Inputs: input, hidden
input of shape (batch, input_size): tensor containing input features
hidden of shape (batch, hidden_size): tensor containing the initial hidden state for each element in the batch. Defaults to zero if not provided.

Outputs: h'
h’ of shape (batch, hidden_size): tensor containing the next hidden state for each element in the batch
"""
class GRUCell_assignment(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(GRUCell_assignment, self).__init__()
        # hyper-parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

        self.Wir = nn.Linear(input_size, hidden_size, bias)
        self.Whr = nn.Linear(hidden_size, hidden_size, bias)
        self.Wiz = nn.Linear(input_size, hidden_size, bias)
        self.Whz = nn.Linear(hidden_size, hidden_size, bias)
        self.Win = nn.Linear(input_size, hidden_size, bias)
        self.Whn = nn.Linear(hidden_size, hidden_size, bias)
        nn.init.uniform_(self.Wir.weight.data, -1/sqrt(hidden_size), 1/sqrt(hidden_size))
        nn.init.uniform_(self.Wiz.weight.data, -1/sqrt(hidden_size), 1/sqrt(hidden_size))
        nn.init.uniform_(self.Win.weight.data, -1/sqrt(hidden_size), 1/sqrt(hidden_size))
    def forward(self, inputs, hidden=False):
        if hidden is False: hidden = inputs.new_zeros(inputs.shape[0], self.hidden_size)
        x, h = inputs, hidden

        r = torch.sigmoid(self.Wir(x) + self.Whr(h))
        z = torch.sigmoid(self.Wiz(x) + self.Whz(h))
        n = torch.tanh(self.Win(x) + r * self.Whn(h))
        h = (1-z) * n + z * h

        return h


# you can refer to the implementation provided by PyTorch for more information
# https://pytorch.org/docs/stable/generated/torch.nn.LSTMCell.html

class LSTMCell_assignment(nn.Module):
    def __init__(self, input_size, hidden_size, bias=True):
        super(LSTMCell_assignment, self).__init__()
        # hyper-parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        ### YOUR CODE HERE (~12 Lines)
        ### TODO - Initialize each gate in LSTM cell. Be aware every gate is initialized along the distribution w.r.t '''hidden_size'''
        ###Parameters
        ### input_size – The number of expected features in the input x
        ### hidden_size – The number of features in the hidden state h
        ### bias – If False, then the layer does not use bias weights b_ih and b_hh. Default: True
        
        #  define weight
        
        # forget gate 
        # f = sigmoid(wif*x + whf*h +bias)
        self.Wif = nn.Linear(input_size, hidden_size, bias = bias)
        self.Whf = nn.Linear(hidden_size, hidden_size, bias = bias)
        # input gate
        # g = sigmoid(wig*x + whg*h + bias)
        self.Wig = nn.Linear(input_size, hidden_size, bias = bias)
        self.Whg = nn.Linear(hidden_size, hidden_size, bias = bias)
        # i = tanh(wii*x + whi*h + bias)
        self.Wii = nn.Linear(input_size, hidden_size, bias = bias)
        self.Whi = nn.Linear(hidden_size, hidden_size, bias = bias)
        # output gate o = sigmoid(wio*x + who*h + bias)
        self.Wio = nn.Linear(input_size, hidden_size, bias = bias)
        self.Who = nn.Linear(hidden_size, hidden_size, bias = bias)

        # initialize weight matrix
        nn.init.uniform_(self.Wif.weight.data, -1/sqrt(hidden_size), 1/sqrt(hidden_size))
        nn.init.uniform_(self.Wig.weight.data, -1/sqrt(hidden_size), 1/sqrt(hidden_size))
        nn.init.uniform_(self.Wii.weight.data, -1/sqrt(hidden_size), 1/sqrt(hidden_size))
        nn.init.uniform_(self.Wio.weight.data, -1/sqrt(hidden_size), 1/sqrt(hidden_size))

        if bias:
            nn.init.uniform_(self.Wif.bias, -1/sqrt(hidden_size), 1/sqrt(hidden_size))
            nn.init.uniform_(self.Wig.bias, -1/sqrt(hidden_size), 1/sqrt(hidden_size))
            nn.init.uniform_(self.Wii.bias, -1/sqrt(hidden_size), 1/sqrt(hidden_size))
            nn.init.uniform_(self.Wio.bias, -1/sqrt(hidden_size), 1/sqrt(hidden_size))

        

    def forward(self, inputs, dec_state):
        x, h, c = inputs, dec_state[0], dec_state[1]
        ### Inputs: input, (h_0, c_0)
        ### input of shape (batch, input_size): tensor containing input features
        ### h_0 of shape (batch, hidden_size): tensor containing the initial hidden state for each element in the batch.
        ### c_0 of shape (batch, hidden_size): tensor containing the initial cell state for each element in the batch.
        ### If (h_0, c_0) is not provided, both h_0 and c_0 default to zero.
        

        ### Outputs: (h_1, c_1)
        ### h_1 of shape (batch, hidden_size): tensor containing the next hidden state for each element in the batch
        ### c_1 of shape (batch, hidden_size): tensor containing the next cell state for each element in the batch
        ### YOUR CODE HERE (~6 Lines)
        ### TODO - Implement forward prop in LSTM cell. 

        # dec_state not provided
        # if len(dec_state) == 0:
        #     h, c = 0, 0

        # forget gate: sigmoid(wif*x + whf*h +bias)
        f = torch.sigmoid(self.Wif(x) + self.Whf(h))
        # input gate
        # g = sigmoid(wig*x + whg*h + bias)
        g = torch.sigmoid(self.Wig(x) + self.Whg(h)) 
        # i = tanh(wii*x + whi*h + bias)
        i = torch.tanh(self.Wii(x) + self.Whi(h))

        # cell state update (elementwise dot product)
        c = f*c + i*g

        # output gate o = sigmoid(wio*x + who*h + bias)
        o = torch.sigmoid(self.Wio(x) + self.Who(h))
        h = o*torch.tanh(c)

        return (h, c)