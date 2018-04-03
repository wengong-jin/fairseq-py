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

from . import FairseqEncoder, FairseqIncrementalDecoder, FairseqModel, register_model, register_model_architecture
from .fconv import FConvEncoder, FConvDecoder
from .uni_model import UniSeqModel


@register_model('uni_zphrase')
class UniZPhraseModel(UniSeqModel):
    def __init__(self, encoder, decoder, src_decoder, z_encoder, z_decoder):
        super().__init__(encoder, decoder, src_decoder)
        self.z_encoder = z_encoder
        self.z_decoder = z_decoder
        self.encoder.num_attention_layers = sum(layer is not None for layer in decoder.attention)
        self.z_encoder.num_attention_layers = 1

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
        z_encoder = FConvEncoder(
            src_dict,
            embed_dim=args.encoder_embed_dim,
            convolutions=[(256, 3)] * 1,
            dropout=args.dropout,
            max_positions=args.max_source_positions,
            use_fc=False
        )

        z_decoder = LSTMController(
            encoder_hdim = 256,
            hdim = 256,
            zdim = 256,
            num_layers = 1,
            length = 0.8
        )

        encoder = FConvEncoder(
            src_dict, #Just a dummy
            embed_dim=args.encoder_embed_dim,
            convolutions=eval(args.encoder_layers),
            dropout=args.dropout,
            max_positions=args.max_source_positions,
            embed=False
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
        
        return MLZSeqModel(encoder, decoder, src_decoder, z_encoder, z_decoder)

    def forward(self, src_tokens, src_lengths, prev_src_tokens, prev_output_tokens):
        #src_lengths is actually not used in encoders
        z_encoder_out,_ = self.z_encoder(src_tokens, src_lengths)
        z_decoder_out = self.z_decoder(z_encoder_out)

        encoder_out = self.encoder(z_decoder_out, src_lengths)
        decoder_out, _ = self.decoder(prev_output_tokens, encoder_out)
        src_decoder_out,_ = self.src_decoder(prev_src_tokens, encoder_out )

        return src_decoder_out, decoder_out

class Controller(nn.Module):
    """LSTM Controller."""
    def __init__(self, encoder_hdim=512, hdim=512, zdim=128, num_layers=1, dropout=0, window_size=3, length=0.8):
        super().__init__()
        self.hdim = hdim
        self.zdim = zdim
        self.dropout = dropout
        self.length = length

        self.layers = nn.ModuleList([LSTMCell(zdim , hdim)] #input feeding
            + [LSTMCell(hdim, hdim) for layer in range(1, num_layers)]
        )
        self.attention = AttentionLayer(encoder_hdim, hdim)
        #self.attention = LocalAttentionLayer(encoder_hdim, hdim, window_size)
        self.mu_out = Linear(hdim, zdim, dropout=0, bias=False)
        self.sg_out = Linear(hdim, zdim, dropout=0, bias=False)
        self.init_z = nn.Parameter(torch.Tensor(zdim).normal_())
        #self.fc = Linear(self.zdim, encoder_hdim, dropout=0, bias=False)

    def forward(self, encoder_hiddens):
        num_layers = len(self.layers)
        bsz,seqlen = encoder_hiddens.size(0),encoder_hiddens.size(1)
        seqlen = int(seqlen * self.length) + 1

        zero = Variable(encoder_hiddens.data.new(bsz, self.hdim).zero_())
        prev_hiddens = [zero for i in range(num_layers)]
        prev_cells = [zero for i in range(num_layers)]

        init_z = torch.stack([self.init_z] * bsz, dim=0)
        zouts = [init_z]
        coverage = Variable(zero.data.new(bsz, encoder_hiddens.size(1)).zero_()).cuda()
        for j in range(seqlen):
            input = zouts[-1]

            for i, rnn in enumerate(self.layers):
                # recurrent cell
                hidden, cell = rnn(input, (prev_hiddens[i], prev_cells[i]))

                # hidden state becomes the input to the next layer
                input = F.dropout(hidden, p=self.dropout, training=self.training)

                # save state for next time step
                prev_hiddens[i] = hidden
                prev_cells[i] = cell

            # apply attention using the last layer's hidden state
            cxt, attn_score = self.attention(hidden, encoder_hiddens, coverage)
            #cxt, attn_score = self.attention(hidden, encoder_hiddens, coverage, j*self.cratio)
            coverage = coverage + attn_score

            # sample z from context vector cxt 
            cxt = F.dropout(cxt, p=self.dropout, training=self.training)
            mu = self.mu_out(cxt)
            log_sigma = -torch.abs(self.sg_out(cxt))
            if self.training:
                eps = Variable(mu.data.clone().normal_(std=0.0001))
            else:
                eps = 0
            zouts.append( mu + eps * torch.exp(log_sigma / 2) )

        # collect outputs across time steps
        zouts = torch.cat(zouts[1:], dim=0).view(seqlen, bsz, self.zdim)
        #zouts = self.fc(zouts)
        return zouts.transpose(1, 0) # T x B x C -> B x T x C

class PhCNNEncoder(nn.Module):
    def __init__(self, dictionary, embed_dim, dropout=0.1, max_positions=1024):
        super().__init__()
        self.dictionary = dictionary
        self.embed_dim = embed_dim
        self.dropout = dropout

        num_embeddings = len(dictionary)
        self.padding_idx = dictionary.pad()
        self.embed_tokens = Embedding(num_embeddings, embed_dim, self.padding_idx)
        self.l_gate = Linear(embed_dim, 1)
        self.r_gate = Linear(embed_dim, 1)
        self.W_conv = Linear(3 * embed_dim, embed_dim)

        self.embed_positions = PositionalEmbedding(
            max_positions,
            embed_dim,
            self.padding_idx,
            left_pad=LanguagePairDataset.LEFT_PAD_SOURCE,
        )
        
        self.lstm = LSTM(
            input_size=embed_dim,
            hidden_size=embed_dim,
            num_layers=1,
            dropout=0,
            bidirectional=True
        )

    def lstm_forward(self, src_tokens, src_lengths):
        if LanguagePairDataset.LEFT_PAD_SOURCE:
            # convert left-padding to right-padding
            src_tokens = utils.convert_padding_direction(
                src_tokens,
                src_lengths,
                self.padding_idx,
                left_to_right=True,
            )

        bsz, seqlen = src_tokens.size()

        # embed tokens
        x = self.embed_tokens(src_tokens)
        x = F.dropout(x, p=self.dropout, training=self.training)
        embed_dim = x.size(2)

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # pack embedded source tokens into a PackedSequence
        packed_x = nn.utils.rnn.pack_padded_sequence(x, src_lengths.data.tolist())

        # apply LSTM
        packed_outs, (final_hiddens, final_cells) = self.lstm(packed_x)

        # unpack outputs and apply dropout
        x, _ = nn.utils.rnn.pad_packed_sequence(packed_outs, padding_value=0.)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = x.transpose(0, 1)

        return torch.split(x, embed_dim, dim=2)

    def forward(self, src_tokens, src_lengths):
        f_hids, b_hids = self.lstm_forward(src_tokens, src_lengths)
        l_gate = F.sigmoid(self.l_gate(f_hids))
        r_gate = F.sigmoid(self.r_gate(b_hids))
        print(torch.cat((l_gate,r_gate), dim=2)[0].data)

        x = self.embed_tokens(src_tokens) + self.embed_positions(src_tokens)
        x = F.dropout(x, p=self.dropout, training=self.training)
        pad_x = F.pad(x, (0, 0, 1, 1))
        lx, rx = pad_x[:,:-2,:], pad_x[:,2:,:]
        lx = l_gate * lx
        rx = r_gate * rx
        x = torch.cat([lx,x,rx], dim=2)

        l_gate,r_gate = l_gate.squeeze(2), r_gate.squeeze(2)
        ll_gate = F.pad(l_gate[:,:-1], (1, 0))
        rr_gate = F.pad(r_gate[:,1:], (0, 1))

        return F.relu(self.W_conv(x)), (1 - ll_gate) * (1 - rr_gate)

class AttentionLayer(nn.Module):
    """T. Luong's global attention"""
    def __init__(self, input_embed_dim, output_embed_dim):
        super().__init__()
        self.input_proj = Linear(input_embed_dim, output_embed_dim, bias=False)

    def forward(self, input, source_hids):
        # input: bsz x input_embed_dim
        # source_hids: bsz x srclen x output_embed_dim
        # x: bsz x output_embed_dim
        x = self.input_proj(input)
        
        # compute attention
        attn_scores = (source_hids * x.unsqueeze(1)).sum(dim=2)
        attn_scores = F.softmax(attn_scores, dim=1)  # bsz x srclen
        #xx,yy = attn_scores.max(dim=1)
        #print(xx[0].item(), yy[0].item())

        # sum weighted sources
        x = (attn_scores.unsqueeze(2) * source_hids).sum(dim=1)
        return x, attn_scores

class MaxPoolAttentionLayer(nn.Module):
    def __init__(self, input_embed_dim, output_embed_dim):
        super().__init__()
        self.input_proj = Linear(input_embed_dim, output_embed_dim, bias=False)

    def forward(self, input, source_hids, coverage):
        # input: bsz x input_embed_dim
        # source_hids: bsz x srclen x output_embed_dim
        # x: bsz x output_embed_dim
        x = self.input_proj(input)
        
        # compute attention
        attn_scores = (source_hids * x.unsqueeze(1)).sum(dim=2)
        attn_scores = attn_scores + torch.log(1 - coverage + 1e-6)
        attn_scores = F.softmax(attn_scores) * (1 - coverage)
        score,idx = F.adaptive_max_pool1d(attn_scores.unsqueeze(1), 1, return_indices=True)
        #print(score[0][0].item(), idx[0][0].item())

        seqlen = source_hids.size(1)
        attn_scores = F.max_unpool1d(score, idx, kernel_size=seqlen, stride=seqlen)
        x = (source_hids * attn_scores.transpose(1,2)).sum(dim=1)

        return x, attn_scores.squeeze(1)

class LocalAttentionLayer(nn.Module):
    """T. Luong's local attention"""
    def __init__(self, input_embed_dim, output_embed_dim, window_size):
        super().__init__()
        self.input_proj = Linear(input_embed_dim, output_embed_dim, bias=False)
        self.window_size = window_size

    def forward(self, input, source_hids, coverage, pos):
        # input: bsz x input_embed_dim
        # source_hids: bsz x srclen x output_embed_dim
        # x: bsz x output_embed_dim
        x = self.input_proj(input)
        
        # compute attention
        bsz,seqlen = source_hids.size(0),source_hids.size(1)
        l,r = max(0, pos-self.window_size), min(seqlen, pos+self.window_size+1)
        source_hids = source_hids[:, l:r, :]
        coverage = coverage[:, l:r]
        attn_scores = (source_hids * x.unsqueeze(1)).sum(dim=2)
        attn_scores = attn_scores + torch.log(1 - coverage + 1e-6)
        attn_scores = F.softmax(attn_scores, dim=1)  # bsz x srclen
        attn_scores = attn_scores * (1 - coverage)

        # sum weighted sources
        x = (attn_scores.unsqueeze(2) * source_hids).sum(dim=1)
        if l > 0:
            zero = Variable(coverage.data.new(bsz,l).zero_())
            attn_scores = torch.cat((zero,attn_scores), dim=1)
        if r < seqlen:
            zero = Variable(coverage.data.new(bsz,seqlen-r).zero_())
            attn_scores = torch.cat((attn_scores,zero), dim=1)

        return x, attn_scores

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

@register_model_architecture('uni_zphrase', 'uni_zphrase')
def base_architecture(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_layers = getattr(args, 'encoder_layers', '[(512, 3)] * 20')
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 512)
    args.decoder_layers = getattr(args, 'decoder_layers', '[(512, 3)] * 20')
    args.decoder_out_embed_dim = getattr(args, 'decoder_out_embed_dim', 256)
    args.decoder_attention = getattr(args, 'decoder_attention', 'True')
    args.share_input_output_embed = getattr(args, 'share_input_output_embed', False)


@register_model_architecture('uni_zphrase', 'uni_zphrase_iwslt_de_en')
def fconv_iwslt_de_en(args):
    base_architecture(args)
    args.encoder_embed_dim = 256
    args.encoder_layers = '[(256, 3)] * 4'
    args.decoder_embed_dim = 256
    args.decoder_layers = '[(256, 3)] * 3'
    args.decoder_out_embed_dim = 256

