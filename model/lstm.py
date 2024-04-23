from re import X
from turtle import forward
import torch.nn as nn
import torch


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
        self.stride = (1, 1)

        self.conv = nn.Conv1d(
            in_channels=input_dim + hidden_dim,
            out_channels=hidden_dim * 4,
            kernel_size=kernel_size,
            stride=1,
            padding=self.padding,
            bias=bias,
        )

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state  # short long
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


# class ConvLSTM(nn.Module):

#     def __init__(
#         self,
#         input_dim,
#         hidden_dim,
#         kernel_size,
#         bias=True,
#         num_layers=1,
#         batch_first=True,
#         # return_all_layers=False,
#     ):
#         """
#         input_dim : int
#             Number of channels in input tensor
#         hidden_dim : int
#             Number of channels in hidden state
#         kernel_size : (int, int)
#             Size of the conv kernel
#         bias : bool
#             Bias param in conv
#         num_layers : int
#             Numvber of stacked ConvLSTMCell layers
#         batch_first : bool
#             Wether first dim is a batch
#         return_all_layers : bool
#             TODO
#         """
#         super().__init__()
#         kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
#         hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)

#         self.input_dim = input_dim
#         self.hidden_dim = hidden_dim
#         self.kernel_size = kernel_size
#         self.bias = bias
#         self.batch_first = batch_first

#         blocks = [
#             ConvLSTMCell(
#                 input_dim=self.input_dim if i == 0 else self.hidden_dim[i - 1],
#                 hidden_dim=self.hidden_dim[i],
#                 kernel_size=self.kernel_size[i],
#                 bias=self.bias,
#             )
#             for i in range(self.num_layers)
#         ]

#         self.cell_list = nn.ModuleList(blocks)
    
#     def forward(self, input_batch):
#         pass


#     @staticmethod
#     def _extend_for_multilayer(param, num_layers: int):
#         if isinstance(param, int):
#             return (param,) * num_layers
#         elif isinstance(param, list):
#             return tuple(param)
#         else:
#             return param
