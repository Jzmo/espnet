# Copyright 2021 Tianzi Wang
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Encoder definition."""
import contextlib
import copy
from pathlib import Path
import yaml
from filelock import FileLock
import logging
import os
from typing import Optional
from typing import Tuple

import torch
from typeguard import check_argument_types
from argparse import Namespace
from omegaconf import DictConfig, OmegaConf, open_dict

from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet2.asr.encoder.abs_encoder import AbsEncoder

class FairseqHubertEncoder(AbsEncoder):
    """FairSeq Hubert encoder module.

    Args:
        input_size: input dim
        output_size: dimension of attention
        hubert_url: url to Wav2Vec2.0 pretrained model
        hubert_dir_path: directory to download the Wav2Vec2.0 pretrained model.
        normalize_before: whether to use layer_norm before the first block
        finetune_last_n_layers: last n layers to be finetuned in Wav2Vec2.0
                                0 means to finetune every layer if freeze_w2v=False.
    """

    def __init__(
        self,
        input_size: int, # doesn't use here
        hubert_url: str,
        hubert_dir_path: str = "./",
        output_size: int = 256,
        normalize_before: bool = False,
        freeze_finetune_updates: int = 0,
        dropout_rate: float = 0.0,
        activation_dropout: float = 0.1,
        attention_dropout: float = 0.0,
        mask_length: int = 10,
        mask_prob: float = 0.75,
        mask_selection: str = "static",
        mask_other: int = 0,
        apply_mask: bool = True,
        mask_channel_length: int = 64,
        mask_channel_prob: float = 0.5,
        mask_channel_other: int = 0,
        mask_channel_selection: str =  "static",
        layerdrop: float = 0.1,
        feature_grad_mult: float = 0.0,
    ):
        assert check_argument_types()
        super().__init__()
        self.apply_mask = apply_mask
        # https://github.com/pytorch/fairseq/blob/master/fairseq/models/hubert/hubert_asr.py#L241
        arg_overrides = {
            "dropout": dropout_rate,
            "activation_dropout": activation_dropout,
            "attention_dropout": attention_dropout,
            "mask_length": mask_length,
            "mask_prob": mask_prob,
            "mask_selection": mask_selection,
            "mask_other": mask_other,
            "mask_channel_length": mask_channel_length,
            "mask_channel_prob": mask_channel_prob,
            "mask_channel_selection": mask_channel_selection,
            "mask_channel_other": mask_channel_other,
            "encoder_layerdrop": layerdrop,
            "feature_grad_mult": feature_grad_mult,
            "data": hubert_dir_path,
        }        

        try:
            import fairseq
            from fairseq.models.hubert.hubert import HubertModel
        except Exception as e:
            print("Error: FairSeq is not properly installed.")
            print(
                "Please install FairSeq: cd ${MAIN_ROOT}/tools && make fairseq.done"
            )
            raise e

        if hubert_url == "espnet":
            self.hubert_model_path = hubert_dir_path
            s = torch.load(
                os.path.join(self.hubert_model_path, "valid.acc.best.pth"),
                map_location=torch.device('cpu'),
            )
            if all("encoder.encoder" in k for k in s):
                try:
                    state = {
                        k.replace("encoder.encoder.", ""):v
                        for k, v in s.items()
                    }
                except Exception as e:
                    raise e                    

            config_file = os.path.join(
                self.hubert_model_path, "config.yaml",
            )
            config_file = Path(config_file)

            with config_file.open("r", encoding="utf-8") as f:
                self.pretrained_cfg = yaml.safe_load(f)
                
            model = FairseqHubertPretrainEncoder(
                self.pretrained_cfg["input_size"],
                **self.pretrained_cfg["encoder_conf"]
            )
            model = model.encoder
            d = self.pretrained_cfg["encoder_conf"]["output_size"]
            self.pretrained_params = copy.deepcopy(state)
            
        else:
            
            self.hubert_model_path = download_hubert(hubert_url, hubert_dir_path)

            #state = fairseq.checkpoint_utils.load_checkpoint_to_cpu(
            #    self.hubert_model_path, arg_overrides=arg_overrides,
            #)
            #self.pretrained_cfg = state.get("cfg", None)
            models, self.pretrained_cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
                [self.hubert_model_path],
                arg_overrides=arg_overrides,
                strict=False,
            )
            model = models[0]
            d = self.pretrained_cfg.model.encoder_embed_dim
            self.pretrained_params = copy.deepcopy(model.state_dict())
            
        self._output_size = output_size

        if not isinstance(model, HubertModel):
            try:
                model = model.hubert_encoder.hubert_model
            except Exception as e:
                print(
                    "Error: pretrained models should be within: "
                    "'HubertModel, Hubertctc' classes, etc."
                )
                raise e
            
        self.encoders = model

        self.normalize_before = normalize_before
        if self.normalize_before:
            self.after_norm = LayerNorm(output_size)

        
        if output_size and output_size != d:
            self.output_layer = torch.nn.Sequential(
                torch.nn.Linear(d, output_size),
                )
        else:
            self.output_layer = None            

        self.freeze_finetune_updates = freeze_finetune_updates
        self.register_buffer("num_updates", torch.LongTensor([0]))

    def output_size(self) -> int:
        return self._output_size

    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        prev_states: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Forward FairHubert Encoder.

        Args:
            xs_pad: input tensor (B, L, D)
            ilens: input length (B)
            prev_states: Not to be used now.
        Returns:
            position embedded tensor and mask
        """
        #print("model params:", self.encoders.encoder.layers[11].final_layer_norm.bias.sum().item())
        masks = make_pad_mask(ilens).to(xs_pad.device)

        ft = self.freeze_finetune_updates <= self.num_updates

        if self.num_updates <= self.freeze_finetune_updates:
            self.num_updates += 1
        elif ft and self.num_updates == self.freeze_finetune_updates + 1:
            self.num_updates += 1
            logging.info("Start fine-tuning hubert parameters!")
        else:
            self.num_updates += 1
        with torch.no_grad() if not ft else contextlib.nullcontext():
            enc_outputs = self.encoders(
                xs_pad,
                padding_mask = masks,
                mask = self.apply_mask and self.training,
                features_only=True,
                output_layer=None,
            )

        xs_pad = enc_outputs["x"]  # (B,T,C),
        masks = enc_outputs["padding_mask"]  # (B, T)

        #save gpu memory
        del enc_outputs
        
        olens = (~masks).sum(dim=1)

        if self.output_layer is not None:
            xs_pad = self.output_layer(xs_pad)

        if self.normalize_before:
            xs_pad = self.after_norm(xs_pad)

        #if self.normalize_before:
        #    xs_pad = self.after_norm(xs_pad)

        return xs_pad, olens, None

    def reload_pretrained_parameters(self):
        self.encoders.load_state_dict(self.pretrained_params)
        logging.info("Pretrained Hubert model parameters reloaded!")

        
class FairseqHubertPretrainEncoder(AbsEncoder):
    """FairSeq Hubert encoder module.
    Ref: 
        https://github.com/pytorch/fairseq/blob/master/fairseq/models/hubert/hubert.py#L39
    Args:
        input_size: input dim
        output_size: dimension of attention

        normalize_before: whether to use layer_norm before the first block
    """

    def __init__(
        self,
        # encoder size
        input_size: int, #
        output_size: int = 1024,
        linear_units: int = 1024, #
        attention_heads: int = 12, #,
        num_blocks: int = 12, #
        # dropout
        dropout_rate: float = 0.0, #
        attention_dropout_rate: float = 0.0, #
        activation_dropout_rate: float = 0.0, #
        hubert_dict_dir = './',
        label_rate: int = 100,
        sample_rate: int = 16000,
        hubert_url: str = "./",
        hubert_dir_path: str = "./",

        **kwargs,
    ):
        """
        activation_fn: str = "gelu", #
        encoder_layerdrop: float = 0.0,
        dropout_input: float = 0.0,
        dropout_features: float = 0.0,
        # output
        final_dim: int = 0,
        untie_final_proj: bool = False,
        layer_norm_first: bool = False,
        conv_feature_layers: str = "[(512,10,5)] + [(512,3,2)] * 4 + [(512,2,2)] * 2",
        conv_bias: bool = False,
        logit_temp: float = 0.1,
        target_glu: bool = False,
        feature_grad_mult: float = 1.0,
        #masking
        mask_length: int = 10,
        mask_prob: float = 0.65,
        mask_selection: str = "static",
        mask_other: int = 0,
        no_mask_overlap: bool = False,
        mask_min_space: int = 1,
        # channel masking
        mask_channel_length: int = 64,
        mask_channel_prob: float = 0.0,
        mask_channel_selection: str =  "static",
        mask_channel_other: int = 0,
        no_mask_channel_overlap: bool = False,
        mask_channel_min_space: int = 1,
        # positional embeddings
        conv_pos: int = 128,
        conv_pos_groups: int = 16,
        # loss computation
        skip_masked: bool = False,
        skip_nomask: bool = False,
        # dictionary:
        hubert_dict_path: "./",
        ):
        """
        assert check_argument_types()
        super().__init__()
        self._output_size = output_size
        try:
            import fairseq
            from fairseq.models.hubert.hubert import (HubertModel,
                                                      HubertConfig,
                                                      HubertPretrainingConfig,
            )
            from fairseq.data.dictionary import Dictionary
        except Exception as e:
            print("Error: FairSeq is not properly installed.")
            print(
                "Please install FairSeq: cd ${MAIN_ROOT}/tools && make fairseq.done"
            )
            raise e
        if hubert_url.startswith("http"):
            self.hubert_model_path = download_hubert(hubert_url, hubert_dir_path)

            #state = fairseq.checkpoint_utils.load_checkpoint_to_cpu(
            #    self.hubert_model_path, arg_overrides=arg_overrides,
            #)
            #self.pretrained_cfg = state.get("cfg", None)
            models, self.pretrained_cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
                [self.hubert_model_path],
                strict=False,
            )
            model = models[0]
            d = self.pretrained_cfg.model.encoder_embed_dim
            self.pretrained_params = copy.deepcopy(model.state_dict())

        cfg_overides = {
            "encoder_embed_dim": input_size,
            "encoder_ffn_embed_dim": linear_units,
            "encoder_attention_heads": attention_heads,
            "encoder_layers": num_blocks,
            "final_dim": output_size,
            "dropout": dropout_rate,
            "attention_dropout": attention_dropout_rate,
            "label_rate": label_rate,
        }
        cfg_overides = {**cfg_overides, **kwargs}
        self.cfg = HubertConfig()
        
        for key, value in cfg_overides.items():
            if hasattr(self.cfg, key):
                setattr(self.cfg, key, value)

        hubert_task_cfg = HubertPretrainingConfig()
        hubert_task_cfg_overides = {
            "label_rate": label_rate,
            "sample_rate": sample_rate,
        }
        for key, value in hubert_task_cfg_overides.items():
            if hasattr(hubert_task_cfg, key):
                setattr(hubert_task_cfg, key, value)

        self.dictionaries = [Dictionary.load(f"{hubert_dict_dir}")
            if os.path.exists(f"{hubert_dict_dir}")
            else None
        ]
        self.encoder = HubertModel(self.cfg, hubert_task_cfg, self.dictionaries)        
    
    def output_size(self) -> int:
        return self._output_size
    
    def forward(
        self,
        xs_pad: torch.Tensor,
        ilens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_length: torch.Tensor,
        prev_states: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Forward FairHubert Encoder.

        Args:
            xs_pad: input tensor (B, L, D)
            ilens: input length (B)
            prev_states: Not to be used now.
        Returns:
            position embedded tensor and mask
        """
        masks = make_pad_mask(ilens).to(xs_pad.device)
        ys_pad = ys_pad[:, :min(ys_pad_length)]
        
        enc_outputs = self.encoder(
            xs_pad,
            padding_mask = masks,
            mask = True,
            target_list = [ys_pad],
            features_only = False,
        )

        return enc_outputs

    def reload_pretrained_parameters(self):
        self.encoder.load_state_dict(self.pretrained_params, strict=False)
        logging.info("Pretrained Hubert model parameters reloaded!")

    
def download_hubert(model_url, dir_path):
    os.makedirs(dir_path, exist_ok=True)

    model_name = model_url.split("/")[-1]
    model_path = os.path.join(dir_path, model_name)

    #dict_url = "https://dl.fbaipublicfiles.com/fairseq/wav2vec/dict.ltr.txt"
    #dict_path = os.path.join(dir_path, dict_url.split("/")[-1])

    with FileLock(model_path + ".lock"):
        if not os.path.exists(model_path):
            torch.hub.download_url_to_file(model_url, model_path)
            #torch.hub.download_url_to_file(dict_url, dict_path)
            logging.info(f"Hubert model downloaded {model_path}")
        else:
            logging.info(f"Hubert model {model_path} already exists.")

    return model_path

