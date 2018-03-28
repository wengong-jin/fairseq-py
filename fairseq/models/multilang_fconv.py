# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import utils
from fairseq.data import LanguagePairDataset
from fairseq.modules import BeamableMM, GradMultiply, LearnedPositionalEmbedding, LinearizedConvolution

from . import register_model, register_model_architecture
from .multilang_model import MultiLangModel
from .fconv import FConvEncoder, FConvDecoder


@register_model('mlfconv')
class MLFConvModel(MultiLangModel):
    def __init__(self, encoder, decoder, src_decoder):
        super().__init__(encoder, decoder, src_decoder)
        self.encoder.num_attention_layers = sum(layer is not None for layer in decoder.attention)

    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser."""
        parser.add_argument('--dropout', default=0.1, type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension')
        parser.add_argument('--encoder-layers', type=str, metavar='EXPR',
                            help='encoder layers [(dim, kernel_size), ...]')
        parser.add_argument('--decoder-embed-dim', type=int, metavar='N',
                            help='decoder embedding dimension')
        parser.add_argument('--decoder-layers', type=str, metavar='EXPR',
                            help='decoder layers [(dim, kernel_size), ...]')
        parser.add_argument('--decoder-out-embed-dim', type=int, metavar='N',
                            help='decoder output embedding dimension')
        parser.add_argument('--decoder-attention', type=str, metavar='EXPR',
                            help='decoder attention [True, ...]')
        parser.add_argument('--share-input-output-embed', action='store_true',
                            help='share input and output embeddings (requires'
                                 ' --decoder-out-embed-dim and --decoder-embed-dim'
                                 ' to be equal)')

    @classmethod
    def build_model(cls, args, src_dict, dst_dict):
        """Build a new model instance."""
        encoder = FConvEncoder(
            src_dict,
            embed_dim=args.encoder_embed_dim,
            convolutions=eval(args.encoder_layers),
            dropout=args.dropout,
            max_positions=args.max_source_positions,
        )
        decoder = FConvDecoder(
            dst_dict,
            embed_dim=args.decoder_embed_dim,
            convolutions=eval(args.decoder_layers),
            out_embed_dim=args.decoder_out_embed_dim,
            attention=eval(args.decoder_attention),
            dropout=args.dropout,
            max_positions=args.max_target_positions,
            share_embed=args.share_input_output_embed
        )
        src_decoder = FConvDecoder(
            src_dict,
            embed_dim=args.decoder_embed_dim,
            convolutions=eval(args.decoder_layers),
            out_embed_dim=args.decoder_out_embed_dim,
            attention=eval(args.decoder_attention),
            dropout=args.dropout,
            max_positions=args.max_target_positions,
            share_embed=args.share_input_output_embed
        )
        return MLFConvModel(encoder, decoder, src_decoder)


@register_model_architecture('mlfconv', 'mlfconv')
def base_architecture(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_layers = getattr(args, 'encoder_layers', '[(512, 3)] * 20')
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 512)
    args.decoder_layers = getattr(args, 'decoder_layers', '[(512, 3)] * 20')
    args.decoder_out_embed_dim = getattr(args, 'decoder_out_embed_dim', 256)
    args.decoder_attention = getattr(args, 'decoder_attention', 'True')
    args.share_input_output_embed = getattr(args, 'share_input_output_embed', False)


@register_model_architecture('mlfconv', 'mlfconv_iwslt_de_en')
def fconv_iwslt_de_en(args):
    base_architecture(args)
    args.encoder_embed_dim = 256
    args.encoder_layers = '[(256, 3)] * 4'
    args.decoder_embed_dim = 256
    args.decoder_layers = '[(256, 3)] * 3'
    args.decoder_out_embed_dim = 256


@register_model_architecture('mlfconv', 'mlfconv_wmt_en_ro')
def fconv_wmt_en_ro(args):
    base_architecture(args)
    args.encoder_embed_dim = 512
    args.encoder_layers = '[(512, 3)] * 20'
    args.decoder_embed_dim = 512
    args.decoder_layers = '[(512, 3)] * 20'
    args.decoder_out_embed_dim = 512


@register_model_architecture('mlfconv', 'mlfconv_wmt_en_de')
def fconv_wmt_en_de(args):
    base_architecture(args)
    convs = '[(512, 3)] * 9'       # first 9 layers have 512 units
    convs += ' + [(1024, 3)] * 4'  # next 4 layers have 1024 units
    convs += ' + [(2048, 1)] * 2'  # final 2 layers use 1x1 convolutions
    args.encoder_embed_dim = 768
    args.encoder_layers = convs
    args.decoder_embed_dim = 768
    args.decoder_layers = convs
    args.decoder_out_embed_dim = 512


@register_model_architecture('mlfconv', 'mlfconv_wmt_en_fr')
def fconv_wmt_en_fr(args):
    base_architecture(args)
    convs = '[(512, 3)] * 6'       # first 6 layers have 512 units
    convs += ' + [(768, 3)] * 4'   # next 4 layers have 768 units
    convs += ' + [(1024, 3)] * 3'  # next 3 layers have 1024 units
    convs += ' + [(2048, 1)] * 1'  # next 1 layer uses 1x1 convolutions
    convs += ' + [(4096, 1)] * 1'  # final 1 layer uses 1x1 convolutions
    args.encoder_embed_dim = 768
    args.encoder_layers = convs
    args.decoder_embed_dim = 768
    args.decoder_layers = convs
    args.decoder_out_embed_dim = 512
