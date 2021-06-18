
import copy
from typing import Optional
from typing import Tuple
from typing import Union

import humanfriendly
import numpy as np
import torch
from torch_complex.tensor import ComplexTensor
from typeguard import check_argument_types
import torchaudio

from espnet.nets.pytorch_backend.frontends.frontend import Frontend
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.layers.log_mel import LogMel
from espnet2.layers.stft import Stft
from espnet2.utils.get_default_kwargs import get_default_kwargs


class MFCCFrontend(AbsFrontend):
    """Conventional frontend structure for ASR.

    Stft -> WPE -> MVDR-Beamformer -> Power-spec -> Mel-Fbank -> CMVN
    """

    def __init__(
        self,
        fs: Union[int, str] = 16000,
        num_ceps : int = 37,
        deltas: bool = True,
        ddeltas: bool = True,
        frontend_conf: Optional[dict] = get_default_kwargs(Frontend),
    ):
        assert check_argument_types()
        super().__init__()
        if isinstance(fs, str):
            fs = humanfriendly.parse_size(fs)

        self.fs = fs
        self.num_ceps = num_ceps 
        self.use_deltas = deltas
        self.use_ddeltas = ddeltas
    def output_size(self) -> int:

        #todo change to  ....
        input_size = self.num_ceps
        if self.use_deltas:
            input_size += self.num_ceps
        if self.use_ddeltas:
            input_size += self.num_ceps
        return input_size

    def forward(
        self, input: torch.Tensor, input_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 0. mfcc input(B, T)
        feats = []
        with torch.no_grad():
            for inp in input:
                x = inp.view(1, -1)
                mfccs = torchaudio.compliance.kaldi.mfcc(
                    waveform=x,
                    sample_frequency=self.fs,
                    num_ceps=self.num_ceps,
                    use_energy=False,
                )  # (time, freq)
                mfccs = mfccs.transpose(0, 1)  # (freq, time)
            
                if self.use_deltas:
                    deltas = torchaudio.functional.compute_deltas(mfccs)
                if self.use_ddeltas:
                    ddeltas = torchaudio.functional.compute_deltas(deltas)
                concat = torch.cat([mfccs, deltas, ddeltas], dim=0)
                concat = concat.transpose(0, 1).contiguous()  # (freq, time)
                feats.append(concat)
            feats = torch.stack(feats)

        return feats, input_lengths
