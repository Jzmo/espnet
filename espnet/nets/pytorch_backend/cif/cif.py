import numpy
import torch
from torch import nn
import torch.nn.functional as F

from espnet.nets.pytorch_backend.cif.convolution import ConvolutionModule

MIN_VALUE = float(numpy.finfo(numpy.float32).min)


class Cif(nn.Module):
    """Lightweight Convolution layer.

    This implementation is based on
    https://github.com/pytorch/fairseq/tree/master/fairseq

    Args:
        wshare (int): the number of kernel of convolution
        n_feat (int): the number of features
        dropout_rate (float): dropout_rate
        kernel_size (int): kernel size (length)
        use_kernel_mask (bool): Use causal mask or not for convolution kernel
        use_bias (bool): Use bias term or not.

    """

    def __init__(
        self,
        channels,
        th=1.0,
        bias=True,
    ):
        """Construct Lightweight Convolution layer."""
        super(Cif, self).__init__()

        self.conv = nn.Conv1d(
            channels,
            channels,
            kernel_size=5,
            stride=1,
            padding=2,
            bias=bias,
        )
        self.norm = nn.BatchNorm1d(channels)
        self.activation = nn.ReLU()
        #self.conv = ConvolutionModule(channels, 5)
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(channels, 1),
            nn.Sigmoid(),
        )
        self.th = th

    def forward(self, hs_pad, hs_mask, ys_pad=None, tf=False):
        """Forward of 'Lightweight Convolution'.

        This function takes query, key and value but uses only query.
        This is just for compatibility with self-attention layer (attention.py)

        Args:
            tf: teacher-forcing

        Return:
            x (torch.Tensor): (batch, time1, d_model) ouput

        """
        # hs mask
        B, T, C = hs_pad.size()

        # lightweight conv layer and linear projection
        xs_batch = self.conv(hs_pad.transpose(1, 2))
        
        # maybe add conv layer and relu activataion fn?
        xs_batch = self.activation(xs_batch)
        xs_batch = self.linear(xs_batch.transpose(1, 2))

        loss_pen = xs_batch.sum(1)
        T_y = 0
        if ys_pad is not None:
            T_y = ys_pad.size(1)
        loss_pen = abs(loss_pen - T_y)
        # cif
        #with torch.no_grad():
        cs = []
        xs_batch = xs_batch.squeeze()

        hs_mask = hs_mask.squeeze().eq(0)  # (batch, 1, T)
        min_value = float(
            numpy.finfo(torch.tensor(0, dtype=xs_batch.dtype).numpy().dtype).min
        )
        xs_batch = xs_batch.masked_fill(hs_mask, 0.0)

        for x, h in zip(xs_batch, hs_pad):
            if tf:
                x = x * T_y / sum(x)
            c = self.integrate_and_fire(x, h, T_y)
            cs.append(c)
        cs = torch.nn.utils.rnn.pad_sequence(cs, batch_first=True)
        cs_mask = cs.ne(0.0).any(-1).unsqueeze(1)

        return cs, cs_mask, loss_pen.sum()

    def integrate_and_fire(self, alpha, h_pad, T_y):
        assert alpha.size(0) == h_pad.size(0)
        c = []
        p_start = 0
        alpha_accum = 0
        h_accum = 0
        for u in range(alpha.size(0)):
            alpha_accum = alpha_accum + alpha[u]
            if alpha_accum >= self.th:
                a1 = self.th - (alpha_accum - alpha[u])
                h_accum = h_accum + torch.matmul(
                    alpha[p_start:u].unsqueeze(0),
                    h_pad[p_start:u]
                ) + a1 * h_pad[u] 
                c.append(h_accum)
                alpha_accum = alpha[u] - a1
                h_accum = alpha_accum * h_pad[u]
                p_start = u+1
        if alpha_accum >= self.th / 2:
            c.append(
                h_accum + torch.matmul(
                    alpha[p_start:].unsqueeze(0),
                    h_pad[p_start:]
                )
            )
        c = torch.stack(c).squeeze()
        return c
