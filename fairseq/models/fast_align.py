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


@register_model('fast_align')
class FastAlignModel(FairseqModel):
    def __init__(self, src_embedding, src_controller, src_decoder, tgt_embedding, tgt_controller, tgt_decoder, all_encoder):
        super().__init__(src_embedding, tgt_decoder) #For compatibility
        self.src_embedding = src_embedding
        self.src_controller = src_controller
        self.src_decoder = src_decoder
        self.tgt_embedding = tgt_embedding
        self.tgt_controller = tgt_controller
        self.tgt_decoder = tgt_decoder
        self.all_encoder = all_encoder

        #tgt_decoder and src_decoder must have the same depth
        self.all_encoder.num_attention_layers = sum(layer is not None for layer in tgt_decoder.attention)
        self.Ireg = False

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
        src_embedding = Embedder(src_dict, embed_dim=args.encoder_embed_dim)
        tgt_embedding = Embedder(tgt_dict, embed_dim=args.encoder_embed_dim)
        
        all_encoder = FConvEncoder(
            src_dict, #Just a dummy
            embed_dim=args.encoder_embed_dim,
            convolutions=eval(args.encoder_layers),
            dropout=args.dropout,
            max_positions=args.max_source_positions,
            embed=False
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

        tgt_decoder = FConvDecoder(
            tgt_dict,
            embed_dim=args.decoder_embed_dim,
            convolutions=eval(args.decoder_layers),
            out_embed_dim=args.decoder_out_embed_dim,
            attention=eval(args.decoder_attention),
            dropout=args.dropout,
            max_positions=args.max_target_positions,
            share_embed=args.share_input_output_embed
        )

        return cls(src_embedding, src_controller, src_decoder, tgt_embedding, tgt_controller, tgt_decoder, all_encoder)

    def forward(self, src_tokens, tgt_tokens, src_lengths, prev_src_tokens, prev_output_tokens):
        #src_lengths is actually not used 
        bsz = src_tokens.size(0)
        src_x = self.src_embedding(src_tokens, src_lengths)
        tgt_x = self.tgt_embedding(tgt_tokens, src_lengths)

        
        src_encoder_out = self.all_encoder(src_x, src_lengths)
        tgt_encoder_out = self.all_encoder(tgt_x, src_lengths)

        tgt_decoder_out,attn_cb = self.tgt_decoder(prev_output_tokens, src_encoder_out)
        src_decoder_out,attn_ca = self.src_decoder(prev_src_tokens, tgt_encoder_out)

        reg = F.mse_loss(src_x, tgt_x) * bsz #size_average takes mean over all dimensions
        #print("%.6f" % (reg.data.item() / bsz))

        if self.Ireg and self.training:
            attn_aa = torch.bmm(attn_ca, attn_ac)
            attn_bb = torch.bmm(attn_cb, attn_bc)
            I_a = Variable(torch.eye(src_tokens.size(1))).cuda()
            I_b = Variable(torch.eye(tgt_tokens.size(1))).cuda()
            I_a = torch.stack([I_a] * bsz, dim=0)
            I_b = torch.stack([I_b] * bsz, dim=0)
            Ireg = F.mse_loss(attn_aa, I_a) + F.mse_loss(attn_bb, I_b)
            reg += Ireg * bsz
        
        attn_aa = torch.bmm(attn_ca, attn_ac)
        attn_bb = torch.bmm(attn_cb, attn_bc)
        xx,yy = attn_ac[0].max(dim=1)
        xx1,yy1 = attn_bc[0].max(dim=1)
        #print(torch.stack([xx,yy.float(),xx1,yy1.float()],dim=1))
        print(attn_aa[0])

        return src_decoder_out, tgt_decoder_out, reg

class Embedder(FairseqEncoder):
    def __init__(self, dictionary, embed_dim, dropout=0.1, max_positions=1024):
        super().__init__(dictionary)
        self.embed_dim = embed_dim
        self.dropout = dropout

        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()
        self.embed_tokens = Embedding(num_embeddings, embed_dim, padding_idx)
        self.embed_positions = PositionalEmbedding(max_positions, embed_dim, padding_idx, left_pad=LanguagePairDataset.LEFT_PAD_SOURCE)

    def forward(self, src_tokens, src_lengths):
        x = self.embed_tokens(src_tokens) + self.embed_positions(src_tokens)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return self.embed_positions.max_positions()


def LSTM(input_size, hidden_size, **kwargs):
    m = nn.LSTM(input_size, hidden_size, **kwargs)
    for name, param in m.named_parameters():
        if 'weight' in name or 'bias' in name:
            param.data.normal_(mean=0, std=0.1)
    return m

def LSTMCell(input_size, hidden_size, **kwargs):
    m = nn.LSTMCell(input_size, hidden_size, **kwargs)
    for name, param in m.named_parameters():
        if 'weight' in name or 'bias' in name:
            param.data.normal_(mean=0, std=0.1)
    return m

def Linear(in_features, out_features, dropout=0, bias=True):
    """Weight-normalized Linear layer (input: N x T x C)"""
    m = nn.Linear(in_features, out_features, bias=bias)
    m.weight.data.normal_(mean=0, std=0.1)
    if bias:
        m.bias.data.zero_()
    return m

@register_model_architecture('fast_align', 'fast_align')
def base_architecture(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_layers = getattr(args, 'encoder_layers', '[(512, 3)] * 20')
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 512)
    args.decoder_layers = getattr(args, 'decoder_layers', '[(512, 3)] * 20')
    args.decoder_out_embed_dim = getattr(args, 'decoder_out_embed_dim', 256)
    args.decoder_attention = getattr(args, 'decoder_attention', 'True')
    args.share_input_output_embed = getattr(args, 'share_input_output_embed', False)


@register_model_architecture('fast_align', 'fast_align_iwslt_de_en')
def align_iwslt_de_en(args):
    base_architecture(args)
    args.encoder_embed_dim = 256
    args.encoder_layers = '[(256, 3)] * 4'
    args.decoder_embed_dim = 256
    args.decoder_layers = '[(256, 3)] * 3'
    args.decoder_out_embed_dim = 256

