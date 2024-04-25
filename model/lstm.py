import torch.nn as nn
import torch
import torch.nn.functional as F

from model.norm import ChannelwiseLayerNorm, GlobalLayerNorm
from .cnns import Conv1D


class ConvLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        input_dim : int
            Number of channels in input tensor
        hidden_dim : int
            Number of channels in hidden state
        kernel_size : (int, int)
            Size of the conv kernel
        bias : bool
            Bias :)
        """
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.bias = bias

        # self.padding = (kernel_size[0] // 2, kernel_size[1] // 2)
        self.padding = kernel_size // 2  # 1d?
        # self.stride = (1, 1)

        self.conv = Conv1D(
            in_channels=input_dim + hidden_dim,
            out_channels=hidden_dim * 4,  # split on 4 parts
            kernel_size=kernel_size,
            stride=1,
            padding=self.padding,
            bias=bias,
        )

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state  # short long

        # len of the time axis of the state must be the same as tensor
        if h_cur.shape[-1] < input_tensor.shape[-1]:
            h_cur = F.pad(
                h_cur, (0, input_tensor.shape[-1] - h_cur.shape[-1]), "constant", 0
            )
            c_cur = F.pad(
                c_cur, (0, input_tensor.shape[-1] - c_cur.shape[-1]), "constant", 0
            )
        elif h_cur.shape[-1] > input_tensor.shape[-1]:
            h_cur = h_cur[:, :, : input_tensor.shape[-1]]
            c_cur = c_cur[:, :, : input_tensor.shape[-1]]

        
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined = self.conv(combined)

        # separate inputs of i, f, o, g
        x_i, x_f, x_o, x_g = torch.split(combined, self.hidden_dim, dim=1)

        i = torch.sigmoid(x_i)  # input gate
        f = torch.sigmoid(x_f)  # forget gate
        o = torch.sigmoid(x_o)  # output gate
        g = torch.tanh(x_g)  # input modulation gate

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next  # h_next = y_out

    def init_hidden_state(self, batch_size, tensor_size):
        return torch.zeros(
            batch_size, self.hidden_dim, tensor_size, device=self.conv.weight.device
        ), torch.zeros(
            batch_size, self.hidden_dim, tensor_size, device=self.conv.weight.device
        )


class CLSTM_TCN_Spk(nn.Module):
    """
    Temporal convolutional network block with ConvLSTM Cell,
        ConvLSTM - 1x1Conv - PReLU - Norm - DConv - PReLU - Norm - SConv
        The first tcn block takes additional speaker embedding as inputs
    Input: 3D tensor with [N, C_in, L_in]
    Input Speaker Embedding: 2D tensor with [N, D]
    Output: 3D tensor with [N, C_out, L_out]
    """

    def __init__(
        self,
        in_channels=256,
        spk_embed_dim=100,
        conv_channels=512,
        kernel_size=3,
        dilation=1,
        causal=False,
        bias=True,
    ):
        super().__init__()

        self.lstm = ConvLSTMCell(
            input_dim=in_channels + spk_embed_dim,
            hidden_dim=conv_channels,
            kernel_size=kernel_size,
            bias=bias,
        )
        self.state = None
        self.conv1x1 = Conv1D(in_channels + spk_embed_dim, conv_channels, 1)
        self.prelu1 = nn.PReLU()
        self.norm1 = (
            GlobalLayerNorm(conv_channels, elementwise_affine=True)
            if not causal
            else (ChannelwiseLayerNorm(conv_channels, elementwise_affine=True))
        )
        dconv_pad = (
            (dilation * (kernel_size - 1)) // 2
            if not causal
            else (dilation * (kernel_size - 1))
        )
        self.dconv = nn.Conv1d(
            conv_channels,
            conv_channels,
            kernel_size,
            groups=conv_channels,
            padding=dconv_pad,
            dilation=dilation,
            bias=True,
        )
        self.prelu2 = nn.PReLU()
        self.norm2 = (
            GlobalLayerNorm(conv_channels, elementwise_affine=True)
            if not causal
            else (ChannelwiseLayerNorm(conv_channels, elementwise_affine=True))
        )
        self.sconv = nn.Conv1d(conv_channels, in_channels, 1, bias=True)
        self.causal = causal
        self.dconv_pad = dconv_pad
        self.dilation = dilation

    def forward(self, x, aux):
        # Repeatedly concated speaker embedding aux to each frame of the representation x
        T = x.shape[-1]
        batch_size = x.shape[0]  # batch_first!
        aux = torch.unsqueeze(aux, -1)
        aux = aux.repeat(1, 1, T)
        y = torch.cat([x, aux], 1)

        if self.state is None:
            self.state = self.lstm.init_hidden_state(batch_size, T)

        self.state = self.lstm(y, self.state)
        y = self.state[0]

        y = self.conv1x1(y)
        y = self.norm1(self.prelu1(y))
        y = self.dconv(y)
        if self.causal:
            y = y[:, :, : -self.dconv_pad]
        y = self.norm2(self.prelu2(y))
        y = self.sconv(y)
        y += x
        return y

    def clear_state(self):
        self.state = None
