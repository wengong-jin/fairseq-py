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
from torch.autograd import Variable

from fairseq import utils
from fairseq.data import LanguagePairDataset
from fairseq.modules import BeamableMM, GradMultiply, LearnedPositionalEmbedding, LinearizedConvolution

from . import FairseqEncoder, FairseqDecoder, FairseqIncrementalDecoder, FairseqModel, register_model, register_model_architecture
from .fconv import Embedding, PositionalEmbedding, LinearizedConv1d, FConvEncoder, FConvDecoder


@register_model('fconv_align')
class FConvAlignModel(FairseqModel):
    def __init__(self, src_encoder, src_decoder, tgt_encoder, tgt_decoder):
        super().__init__(src_encoder, tgt_decoder) #For compatibility
        self.src_encoder = src_encoder
        self.src_decoder = src_decoder
        self.tgt_encoder = tgt_encoder
        self.tgt_decoder = tgt_decoder

        self.src_encoder.num_attention_layers = sum(layer is not None for layer in tgt_decoder.attention)
        self.tgt_encoder.num_attention_layers = sum(layer is not None for layer in src_decoder.attention)

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
    def build_model(cls, args, src_dict, tgt_dict):
        """Build a new model instance."""
        src_encoder = FConvEncoder(
            src_dict,
            embed_dim=args.encoder_embed_dim,
            convolutions=eval(args.encoder_layers),
            dropout=args.dropout,
            max_positions=args.max_source_positions,
        )

        tgt_encoder = FConvEncoder(
            tgt_dict,
            embed_dim=args.encoder_embed_dim,
            convolutions=eval(args.encoder_layers),
            dropout=args.dropout,
            max_positions=args.max_source_positions,
        )

        src_decoder = FConvDecoder(
            src_dict,
            embed_dim=args.decoder_embed_dim,
            convolutions=eval(args.decoder_layers),
            out_embed_dim=args.decoder_out_embed_dim,
            attention=eval(args.decoder_attention),
            dropout=args.dropout,
            max_positions=args.max_source_positions,
            share_embed=args.share_input_output_embed,
            left_pad=LanguagePairDataset.LEFT_PAD_SOURCE
        )

        tgt_decoder = FConvDecoder(
            tgt_dict,
            embed_dim=args.decoder_embed_dim,
            convolutions=eval(args.decoder_layers),
            out_embed_dim=args.decoder_out_embed_dim,
            attention=eval(args.decoder_attention),
            dropout=args.dropout,
            max_positions=args.max_target_positions,
            share_embed=args.share_input_output_embed,
            left_pad=LanguagePairDataset.LEFT_PAD_TARGET
        )

        return cls(src_encoder, src_decoder, tgt_encoder, tgt_decoder)

    def forward(self, src_tokens, tgt_tokens, src_lengths, prev_src_tokens, prev_output_tokens):
        bsz = src_tokens.size(0)
        N,M = src_tokens.size(1),tgt_tokens.size(1)
        T = max(N,M)

        src_tokens = F.pad(src_tokens, (0,T-N))
        tgt_tokens = F.pad(tgt_tokens, (0,T-M))
        src_encoder_out = self.src_encoder(src_tokens, src_lengths)
        tgt_encoder_out = self.tgt_encoder(tgt_tokens, src_lengths)
        src_decoder_out, _ = self.src_decoder(prev_src_tokens, tgt_encoder_out)
        tgt_decoder_out, _ = self.tgt_decoder(prev_output_tokens, src_encoder_out)

        attn_score = (src_encoder_out[0].unsqueeze(1) * tgt_encoder_out[0].unsqueeze(2)).sum(dim=3)
        target = Variable(torch.LongTensor(range(T))).cuda()
        target = torch.stack([target] * bsz, dim=0)
        if self.training:
            align_loss = F.cross_entropy(attn_score, target) + F.cross_entropy(attn_score.transpose(1, 2), target)
        else:
            align_loss = 0

        return src_decoder_out, tgt_decoder_out, align_loss * bsz


@register_model_architecture('fconv_align', 'fconv_align')
def base_architecture(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_layers = getattr(args, 'encoder_layers', '[(512, 3)] * 20')
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 512)
    args.decoder_layers = getattr(args, 'decoder_layers', '[(512, 3)] * 20')
    args.decoder_out_embed_dim = getattr(args, 'decoder_out_embed_dim', 256)
    args.decoder_attention = getattr(args, 'decoder_attention', 'True')
    args.share_input_output_embed = getattr(args, 'share_input_output_embed', False)


@register_model_architecture('fconv_align', 'fconv_align_iwslt_de_en')
def align_iwslt_de_en(args):
    base_architecture(args)
    args.encoder_embed_dim = 256
    args.encoder_layers = '[(256, 3)] * 4'
    args.decoder_embed_dim = 256
    args.decoder_layers = '[(256, 3)] * 3'
    args.decoder_out_embed_dim = 256

