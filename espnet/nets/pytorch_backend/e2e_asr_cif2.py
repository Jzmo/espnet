# Copyright 2020 Johns Hopkins University (Shinji Watanabe)
#                Waseda University (Yosuke Higuchi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""
Mask CTC based non-autoregressive speech recognition model (pytorch).

See https://arxiv.org/abs/2005.08700 for the detail.

"""

from itertools import groupby
import logging
import math

from distutils.util import strtobool
import numpy
import torch

from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.transformer.mask import target_mask
from espnet.nets.pytorch_backend.transformer.dp import dynamic_matching

from espnet.nets.pytorch_backend.conformer.encoder import Encoder
from espnet.nets.pytorch_backend.conformer.argument import (
    add_arguments_conformer_common,  # noqa: H301
    verify_rel_pos_type,  # noqa: H301
)
from espnet.nets.pytorch_backend.e2e_asr import CTC_LOSS_THRESHOLD
from espnet.nets.pytorch_backend.e2e_asr_transformer import E2E as E2ETransformer
from espnet.nets.pytorch_backend.cif.cif import Cif2
from espnet.nets.pytorch_backend.cif.decoder import Decoder

from espnet.nets.pytorch_backend.nets_utils import make_non_pad_mask
from espnet.nets.pytorch_backend.nets_utils import th_accuracy


class E2E(E2ETransformer):
    """E2E module.

    :param int idim: dimension of inputs
    :param int odim: dimension of outputs
    :param Namespace args: argument Namespace containing options

    """

    @staticmethod
    def add_arguments(parser):
        """Add arguments."""
        E2ETransformer.add_arguments(parser)
        E2E.add_cif_arguments(parser)

        return parser

    @staticmethod
    def add_cif_arguments(parser):
        """Add arguments for maskctc model."""
        group = parser.add_argument_group("maskctc specific setting")

        group.add_argument(
            "--cif-quantity-loss-weight",
            default=1.0,
            type=float,
            help="the quantity of integrated embeddings closer to the quantity of targeted labels."
        )
        group.add_argument(
            "--cif-nat-decoder",
            default=False,
            type=bool,
        )
        group.add_argument(
            "--cif-threshold",
            default=1.0,
            type=float,
        )

        group = add_arguments_conformer_common(group)

        return parser

    def __init__(self, idim, odim, args, ignore_id=-1):
        """Construct an E2E object.

        :param int idim: dimension of inputs
        :param int odim: dimension of outputs
        :param Namespace args: argument Namespace containing options
        """
        super().__init__(idim, odim, args, ignore_id)
                
        assert 0.0 <= self.mtlalpha < 1.0, "mtlalpha should be [0.0, 1.0)"
        self.cif_quantity_loss_weight = args.cif_quantity_loss_weight
        self.cif_nat_decoder = args.cif_nat_decoder
        if args.transformer_attn_dropout_rate is None:
            args.transformer_attn_dropout_rate = args.dropout_rate

        # Check the relative positional encoding type
        args = verify_rel_pos_type(args)
        self.encoder = Encoder(
            idim=idim,
            attention_dim=args.adim,
            attention_heads=args.aheads,
            linear_units=args.eunits,
            num_blocks=args.elayers,
            input_layer=args.transformer_input_layer,
            dropout_rate=args.dropout_rate,
            positional_dropout_rate=args.dropout_rate,
            attention_dropout_rate=args.transformer_attn_dropout_rate,
            pos_enc_layer_type=args.transformer_encoder_pos_enc_layer_type,
            selfattention_layer_type=args.transformer_encoder_selfattn_layer_type,
            activation_type=args.transformer_encoder_activation_type,
            macaron_style=args.macaron_style,
            use_cnn_module=args.use_cnn_module,
            cnn_module_kernel=args.cnn_module_kernel,
        )
        if False: #self.cif_nat_decoder:
            self.decoder = Decoder(
                odim=odim,
                attention_dim=args.adim,
                attention_heads=args.aheads,
                linear_units=args.dunits,
                num_blocks=args.dlayers,
                dropout_rate=args.dropout_rate,
                positional_dropout_rate=args.dropout_rate,
                attention_dropout_rate=args.transformer_attn_dropout_rate,
                selfattention_layer_type=args.transformer_decoder_selfattn_layer_type,
            )

        self.cif = Cif2(channels=args.adim, th=args.cif_threshold)
        self.reset_parameters(args)

    def forward(self, xs_pad, ilens, ys_pad):
        """E2E forward.

        :param torch.Tensor xs_pad: batch of padded source sequences (B, Tmax, idim)
        :param torch.Tensor ilens: batch of lengths of source sequences (B)
        :param torch.Tensor ys_pad: batch of padded target sequences (B, Lmax)
        :return: ctc loss value
        :rtype: torch.Tensor
        :return: attention loss value
        :rtype: torch.Tensor
        :return: accuracy in attention decoder
        :rtype: float
        """
        # 1. forward encoder
        xs_pad = xs_pad[:, : max(ilens)]  # for data parallel
        src_mask = make_non_pad_mask(ilens.tolist()).to(xs_pad.device).unsqueeze(-2)

        hs_pad, hs_mask = self.encoder(xs_pad, src_mask)
        self.hs_pad = hs_pad

        # forward cif and compute quantity loss
        ys_in_pad, ys_out_pad = add_sos_eos(
                ys_pad, self.sos, self.eos, self.ignore_id
        )
        #parse padded ys
        ys_list = [y[y != self.ignore_id] for y in ys_pad]
        cs_pad, cs_mask, loss_qua = self.cif(hs_pad,
                                             hs_mask,
                                             ys_list,
                                             tf=self.training,
                                             pad_value=self.eos,
                                             T_max=ys_out_pad.size(1)
        )

        # 2. forward decoder
        if self.decoder is not None:
            # non autoregressive decoder
            # jzmo: optinally use autoregressive to help training here
            ys_mask = None
            if not self.cif_nat_decoder:
                ys_mask = target_mask(ys_in_pad, self.ignore_id)
                #sos_in = torch.full(cs_pad.size()[:-1], self.sos, device=ys_in_pad.device)
                #sos_mask = torch.full(ys_mask.size(), False, device=ys_mask.device)
                pred_pad, pred_mask = self.decoder(ys_in_pad, ys_mask, cs_pad, cs_mask)
                #pred_pad, pred_mask = self.decoder(sos_in, None, cs_pad, cs_mask)
            else:
                #cs_mask = target_mask(cs_pad.sum(-1), 0.0)
                #ys_mask = torch.full(ys_mask.size(), False, device=ys_mask.device)
                pred_pad, pred_mask = self.decoder(cs_pad, cs_mask, hs_pad, hs_mask)
                #pred_pad, pred_mask = self.decoder(cs_pad, cs_mask)
            self.pred_pad = pred_pad

        # 3. compute attention loss
        loss_att, self.acc = 0.0, 0.0
        if self.training:
            loss_att = self.criterion(pred_pad, ys_out_pad)

            self.acc = th_accuracy(
                pred_pad.view(-1, self.odim), ys_out_pad, ignore_label=self.ignore_id
            )

        # 4. compute ctc loss
        loss_ctc, cer_ctc = None, None
        if self.mtlalpha > 0:
            batch_size = xs_pad.size(0)
            hs_len = hs_mask.view(batch_size, -1).sum(1)
            loss_ctc = self.ctc(hs_pad.view(batch_size, -1, self.adim), hs_len, ys_pad)
            if self.error_calculator is not None:
                ys_hat = self.ctc.argmax(hs_pad.view(batch_size, -1, self.adim)).data
                cer_ctc = self.error_calculator(ys_hat.cpu(), ys_pad.cpu(), is_ctc=True)
            # for visualization
            if not self.training:
                self.ctc.softmax(hs_pad)

        # 5. compute cer/wer
        if self.error_calculator is None or self.decoder is None:
            cer, wer = None, None
        else:
            ys_hat = pred_pad.argmax(dim=-1)
            cer, wer = self.error_calculator(ys_hat.cpu(), ys_pad.cpu())
        if not self.training:
            print("===========================")
            print(pred_pad.argmax(dim=-1)[0])
            print(ys_out_pad.cpu()[0])
            print("===========================")
            #import pdb
            #pdb.set_trace()
        alpha = self.mtlalpha
        weight_qua = self.cif_quantity_loss_weight
        if alpha == 0:
            self.loss = loss_att
            loss_att_data = float(loss_att)
            loss_ctc_data = None
        else:
            self.loss = (1 - alpha) * loss_att + weight_qua * loss_qua + alpha * loss_ctc  
            loss_att_data = float(loss_att)
            loss_ctc_data = float(loss_ctc)
        loss_data = float(self.loss)
        #print(f"loss_ctc:{loss_ctc.item()}, loss_att:{loss_att.item()}, loss_qua:{loss_qua.item()}")
        if loss_data - weight_qua * loss_qua < CTC_LOSS_THRESHOLD and not math.isnan(loss_data):
            self.reporter.report(
                loss_ctc_data, loss_att_data, self.acc, cer_ctc, cer, wer, loss_data
            )
        else:
            logging.warning("loss (=%f) is not correct", loss_data)
        return self.loss

    
    def recognize(self, x, recog_args, char_list=None, rnnlm=None):
        """Recognize input speech.

        :param ndnarray x: input acoustic feature (B, T, D) or (T, D)
        :param Namespace recog_args: argment Namespace contraining options
        :param list char_list: list of characters
        :param torch.nn.Module rnnlm: language model module
        :return: decoding result
        :rtype: list
        """

        def num2str(char_list):
            def f(yl):
                cl = [char_list[y] for y in yl]
                return "".join(cl).replace("<space>", " ")

            return f

        n2s = num2str(char_list)

        self.eval()
        h = self.encode(x).unsqueeze(0)

        cs_pad, cs_mask, loss_qua = self.cif(
            h,
            None,
            None,
            tf=False,
            pad_value=self.eos,
            T_max=0,
        )

        # 2. forward decoder
        if self.decoder is not None:
            # non autoregressive decoder
            # jzmo: optinally use autoregressive to help training here
            ys_mask = None
            if not self.cif_nat_decoder:
                raise NotImplementedError
            else:
                pred_pad, pred_mask = self.decoder(
                    cs_pad, None, h, None
                )
                ys_hat_att = pred_pad.argmax(dim=-1)
                self.pred_pad = pred_pad
                pred_score_att, pred_id_att = torch.softmax(pred_pad[0], dim=-1).max(dim=-1)
                
        # greedy ctc outputs
        ctc_probs, ctc_ids = torch.exp(self.ctc.log_softmax(h)).max(dim=-1)
        ys_hat_ctc = torch.stack([x[0] for x in groupby(ctc_ids[0])])
        y_idx_ctc = torch.nonzero(ys_hat_ctc != 0).squeeze(-1)
        pred_id_ctc = ys_hat_ctc[y_idx_ctc]
        # calculate token-level ctc probabilities by taking
        # the maximum probability of consecutive frames with
        # the same ctc symbols
        pred_score_ctc = []
        cnt = 0
        for i, y in enumerate(ys_hat_ctc.tolist()):
            pred_score_ctc.append(-1)
            while cnt < ctc_ids.shape[1] and y == ctc_ids[0][cnt]:
                if pred_score_ctc[i] < ctc_probs[0][cnt]:
                    pred_score_ctc[i] = ctc_probs[0][cnt].item()
                cnt += 1
        pred_score_ctc = torch.from_numpy(numpy.array(pred_score_ctc))[y_idx_ctc]
        ctc_weight = recog_args.ctc_weight
        ys_hat, ys_prob = dynamic_matching(
            pred_id_ctc,
            pred_id_att,
            pred_score_ctc*ctc_weight,
            pred_score_att*(1-ctc_weight)
        )
        ret = [
            yi[0] if yp[0] > yp[1] else yi[1]
            for yi, yp in zip(ys_hat, ys_prob)
        ]
        hyp = {"score": 0.0, "yseq": [self.sos] + ret + [self.eos]}

        return [hyp]
