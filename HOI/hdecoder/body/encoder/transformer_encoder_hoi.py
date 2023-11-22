# Copyright (c) Facebook, Inc. and its affiliates.
import logging
from typing import Callable, Dict, List, Optional, Tuple, Union

from torch import nn
from torch.nn import functional as F
import fvcore.nn.weight_init as weight_init
from detectron2.layers import Conv2d, DeformConv, ShapeSpec, get_norm

from .registry import register_encoder
from ..transformer_blocks import TransformerEncoder, TransformerEncoderLayer
from ...modules import PositionEmbeddingSine
from ...utils import configurable


class TransformerEncoderOnly(nn.Module):
    def __init__(
        self,
        d_model=512,
        nhead=8,
        num_encoder_layers=6,
        dim_feedforward=2048,
        dropout=0.1,
        activation="relu",
        normalize_before=False,
    ):
        super().__init__()

        encoder_layer = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, activation, normalize_before
        )
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        self._reset_parameters()

        self.d_model = d_model
        self.nhead = nhead

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, src, mask, pos_embed):
        # flatten NxCxHxW to HWxNxC
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = pos_embed.flatten(2).permute(2, 0, 1)
        if mask is not None:
            mask = mask.flatten(1)

        memory = self.encoder(src, src_key_padding_mask=mask, pos=pos_embed)
        return memory
        # return memory.permute(1, 2, 0).view(bs, c, h, w)


class BaseEncoder(nn.Module):
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        conv_dim: int,
        mask_dim: int,
        mask_on: bool,
        norm: Optional[Union[str, Callable]] = None,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            conv_dims: number of output channels for the intermediate conv layers.
            mask_dim: number of output channels for the final conv layer.
            norm (str or callable): normalization for all conv layers
        """
        super().__init__()

        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        self.in_features = [k for k, v in input_shape]  # starting from "res2" to "res5"
        feature_channels = [v.channels for k, v in input_shape]

        lateral_convs = []
        output_convs = []

        use_bias = norm == ""
        for idx, in_channels in enumerate(feature_channels):
            if idx == len(self.in_features) - 1:
                output_norm = get_norm(norm, conv_dim)
                output_conv = Conv2d(
                    in_channels,
                    conv_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=use_bias,
                    norm=output_norm,
                    activation=F.relu,
                )
                weight_init.c2_xavier_fill(output_conv)
                self.add_module("layer_{}".format(idx + 1), output_conv)

                lateral_convs.append(None)
                output_convs.append(output_conv)
            else:
                lateral_norm = get_norm(norm, conv_dim)
                output_norm = get_norm(norm, conv_dim)

                lateral_conv = Conv2d(
                    in_channels,
                    conv_dim,
                    kernel_size=1,
                    bias=use_bias,
                    norm=lateral_norm,
                )
                output_conv = Conv2d(
                    conv_dim,
                    conv_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=use_bias,
                    norm=output_norm,
                    activation=F.relu,
                )
                weight_init.c2_xavier_fill(lateral_conv)
                weight_init.c2_xavier_fill(output_conv)
                self.add_module("adapter_{}".format(idx + 1), lateral_conv)
                self.add_module("layer_{}".format(idx + 1), output_conv)

                lateral_convs.append(lateral_conv)
                output_convs.append(output_conv)
        # Place convs into top-down order (from low to high resolution)
        # to make the top-down computation in forward clearer.
        self.lateral_convs = lateral_convs[::-1]
        self.output_convs = output_convs[::-1]

        self.mask_on = mask_on
        if self.mask_on:
            self.mask_dim = mask_dim
            self.mask_features = Conv2d(
                conv_dim,
                mask_dim,
                kernel_size=3,
                stride=1,
                padding=1,
            )
            weight_init.c2_xavier_fill(self.mask_features)

        self.maskformer_num_feature_levels = 3  # always use 3 scales

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        enc_cfg = cfg["MODEL"]["ENCODER"]
        ret = {}
        ret["input_shape"] = {
            k: v for k, v in input_shape.items() if k in enc_cfg["IN_FEATURES"]
        }
        ret["conv_dim"] = enc_cfg["CONVS_DIM"]
        ret["mask_dim"] = enc_cfg["MASK_DIM"]
        ret["norm"] = enc_cfg["NORM"]
        return ret

    def forward_features(self, features):
        pass

    def forward(self, features, targets=None):
        # logger = logging.getLogger(__name__)
        # logger.warning(
        #     "Calling forward() may cause unpredicted behavior of TransformerEncoderHOI module."
        # )
        return self.forward_features(features)

class TransformerEncoderHOI(BaseEncoder):
    @configurable
    def __init__(
        self,
        input_shape: Dict[str, ShapeSpec],
        *,
        transformer_dropout: float,
        transformer_nheads: int,
        transformer_dim_feedforward: int,
        transformer_enc_layers: int,
        transformer_pre_norm: bool,
        conv_dim: int,
        mask_dim: int,
        mask_on: int,
        norm: Optional[Union[str, Callable]] = None,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            input_shape: shapes (channels and stride) of the input features
            transformer_dropout: dropout probability in transformer
            transformer_nheads: number of heads in transformer
            transformer_dim_feedforward: dimension of feedforward network
            transformer_enc_layers: number of transformer encoder layers
            transformer_pre_norm: whether to use pre-layernorm or not
            conv_dims: number of output channels for the intermediate conv layers.
            mask_dim: number of output channels for the final conv layer.
            norm (str or callable): normalization for all conv layers
        """
        super().__init__(
            input_shape,
            conv_dim=conv_dim,
            mask_dim=mask_dim,
            norm=norm,
            mask_on=mask_on,
        )

        input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        self.in_features = [k for k, v in input_shape]  # starting from "res2" to "res5"
        feature_strides = [v.stride for k, v in input_shape]
        feature_channels = [v.channels for k, v in input_shape]

        in_channels = feature_channels[len(self.in_features) - 1]
        self.input_proj = Conv2d(in_channels, conv_dim, kernel_size=1)
        
        # TODO 주석처리
        # weight_init.c2_xavier_fill(self.input_proj)
        
        self.transformer = TransformerEncoderOnly(
            d_model=conv_dim,
            dropout=transformer_dropout,
            nhead=transformer_nheads,
            dim_feedforward=transformer_dim_feedforward,
            num_encoder_layers=transformer_enc_layers,
            normalize_before=transformer_pre_norm,
        )
        N_steps = conv_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)
        delattr(self, "layer_{}".format(len(self.in_features)))

    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        enc_cfg = cfg["MODEL"]["ENCODER"]
        dec_cfg = cfg["MODEL"]["DECODER"]

        ret = super().from_config(cfg, input_shape)
        ret["transformer_dropout"] = dec_cfg["DROPOUT"]
        ret["transformer_nheads"] = dec_cfg["NHEADS"]
        ret["transformer_dim_feedforward"] = dec_cfg["DIM_FEEDFORWARD"]
        ret["transformer_enc_layers"] = enc_cfg[
            "TRANSFORMER_ENC_LAYERS"
        ]  # a separate config
        ret["transformer_pre_norm"] = dec_cfg["PRE_NORM"]

        ret["mask_on"] = cfg["MODEL"]["DECODER"]["MASK"]
        return ret
    
    def forward_features(self, features):
        # "res5"
        x = features[self.in_features[-1]]
        transformer = self.input_proj(x)
        pos = self.pe_layer(x)
        transformer_encoder_features = self.transformer(transformer, None, pos)
        return transformer_encoder_features, pos

    def forward(self, features, targets=None):
        return self.forward_features(features)

@register_encoder
def get_transformer_encoder_hoi(cfg, input_shape):
    """
    Build a pixel decoder from `cfg.MODEL.MASK_FORMER.PIXEL_DECODER_NAME`.
    """
    model = TransformerEncoderHOI(cfg, input_shape)    
    forward_features = getattr(model, "forward_features", None)
    if not callable(forward_features):
        raise ValueError(
            "Only SEM_SEG_HEADS with forward_features method can be used as pixel decoder. "
            f"Please implement forward_features for {name} to only return mask features."
        )
    return model