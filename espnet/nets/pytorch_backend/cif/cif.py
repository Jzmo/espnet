import numpy
import torch
from torch import nn
import torch.nn.functional as F

from espnet.nets.pytorch_backend.cif.convolution import ConvolutionModule
from espnet.nets.pytorch_backend.nets_utils import pad_list

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

        #self.conv = nn.Conv1d(
        #    channels,
        #    channels,
        #    kernel_size=5,
        #    stride=1,
        #    padding=2,
        #    bias=bias,
        #)
        self.norm = nn.BatchNorm1d(channels)
        self.activation = nn.ReLU()
        self.conv = ConvolutionModule(channels, 5)
        self.linear = torch.nn.Sequential(
            torch.nn.Linear(channels, 1),
            nn.Sigmoid(),
        )
        self.th = th

    def forward(self, hs_pad, hs_mask, ys_list=None, tf=False, pad_value=-1, T_max=None):
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
        xs_batch = self.conv(hs_pad)

        # maybe add conv layer and relu activataion fn?
        xs_batch = self.activation(self.norm(xs_batch.transpose(1, 2))) #(B, C, T)
        xs_batch = self.linear(xs_batch.transpose(1, 2)).squeeze() #(B, T, 1)

        hs_mask = hs_mask.squeeze().eq(0)  # (batch, 1, T)
        min_value = float(
            numpy.finfo(torch.tensor(0, dtype=xs_batch.dtype).numpy().dtype).min
        )
        xs_batch = xs_batch.masked_fill(hs_mask, 0.0)
        
        loss_pen = xs_batch.sum(1)
        T_y = torch.tensor([0 for _ in range(B)])
        if ys_list is not None:
            T_y = torch.tensor([ys.size(0) for ys in ys_list]).to(xs_batch.device)
            loss_pen = abs(loss_pen - T_y).sum()
        # cif
        #with torch.no_grad()
        cs = []

        for x, h, t_y in zip(xs_batch, hs_pad, T_y):
            if tf:
                x = x * t_y / sum(x)
            c = self.integrate_and_fire(x, h)
            cs.append(c)
        
        cs = pad_list(cs, pad_value)
        #cs = torch.nn.utils.rnn.pad_sequence(cs, batch_first=True)
        if self.training and cs.size(1) < T_max:
            cs = torch.nn.functional.pad(cs, pad=(0,0,0,T_max-cs.size(1),0,0), mode="constant",value=pad_value)

        # maske square mask to avoid attend padded tokens
        cmax = cs.size(1)
        cs_mask = cs.ne(pad_value).any(-1).unsqueeze(1)
        cs_mask_tmp = cs_mask.transpose(1, 2).repeat(1, 1, cmax)
        cs_mask = cs_mask.repeat(1, cmax, 1) & cs_mask_tmp

        return cs, cs_mask, loss_pen.sum()

    def integrate_and_fire(self, alpha, h_pad):
        assert alpha.size(0) == h_pad.size(0)
        c = []
        p_start = 0
        alpha_accum = 0
        h_accum = 0
        #print(alpha)
        for u in range(alpha.size(0)):
            alpha_accum = alpha_accum + alpha[u]
            #print(alpha_accum.item(), alpha[u].item())
            if alpha_accum >= self.th - 0.1:
                #print("cut:", alpha_accum.item())
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
        #import pdb
        #pdb.set_trace()
        return c
