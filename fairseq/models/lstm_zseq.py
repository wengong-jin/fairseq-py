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
from .lstm import LSTMDecoder
from .zseq import LSTM, LSTMCell, Linear, AttentionLayer


@register_model('lstm_zseq')
class LSTMZSeqModel(FairseqModel):
    def __init__(self, encoder, decoder, z_encoder, z_decoder):
        super().__init__(encoder, decoder)
        self.z_encoder = z_encoder
        self.z_decoder = z_decoder
        self.z_encoder.num_attention_layers = 0.5

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

        z_decoder = LSTMSampler(
            encoder_hdim = 256,
            hdim = 256,
            zdim = 128,
            num_layers = 1
        )

        encoder = LSTMEncoder(
            src_dict, #Just a dummy
            embed_dim=args.encoder_embed_dim,
            num_layers=1,
            dropout_in=0,
            dropout_out=0
        )

        decoder = LSTMDecoder(
            dst_dict,
            encoder_embed_dim=args.encoder_embed_dim,
            embed_dim=args.decoder_embed_dim,
            out_embed_dim=args.decoder_out_embed_dim,
            num_layers=1,
            attention=True,
            dropout_in=0,
            dropout_out=0
        )

        return LSTMZSeqModel(encoder, decoder, z_encoder, z_decoder)

    def forward(self, src_tokens, src_lengths, prev_output_tokens):

        #src_lengths is actually not used in encoders
        z_encoder_out,_ = self.z_encoder(src_tokens, src_lengths)
        z_length = z_encoder_out.size()[1] 
        z_decoder_out = self.z_decoder(z_encoder_out, z_length)

        encoder_out = self.encoder(z_decoder_out, src_lengths)
        decoder_out, _ = self.decoder(prev_output_tokens, encoder_out)
        return decoder_out

class LSTMEncoder(FairseqEncoder):
    """LSTM encoder."""
    def __init__(self, dictionary, embed_dim=512, num_layers=1, dropout_in=0.1,
                 dropout_out=0.1):
        super().__init__(dictionary)
        self.num_layers = num_layers
        self.dropout_in = dropout_in
        self.dropout_out = dropout_out

        self.lstm = LSTM(
            input_size=embed_dim,
            hidden_size=embed_dim,
            num_layers=num_layers,
            dropout=self.dropout_out,
            bidirectional=False,
        )

    def forward(self, src_tokens, src_lengths):
        bsz, seqlen, embed_dim = src_tokens.size()

        # B x T x C -> T x B x C
        x = src_tokens.transpose(0, 1)

        # pack embedded source tokens into a PackedSequence
        packed_x = nn.utils.rnn.pack_padded_sequence(x, src_lengths.data.tolist())

        # apply LSTM
        h0 = Variable(x.data.new(self.num_layers , bsz, embed_dim).zero_())
        c0 = Variable(x.data.new(self.num_layers , bsz, embed_dim).zero_())
        packed_outs, (final_hiddens, final_cells) = self.lstm(
            packed_x,
            (h0, c0),
        )

        # unpack outputs and apply dropout
        x, _ = nn.utils.rnn.pad_packed_sequence(packed_outs, padding_value=0.)
        x = F.dropout(x, p=self.dropout_out, training=self.training)

        return x, final_hiddens, final_cells

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return int(1e5)  # an arbitrary large number

class LSTMSampler(nn.Module):
    """LSTM Z Sampler."""
    def __init__(self, encoder_hdim=512, hdim=512, zdim=128, num_layers=1, dropout=0):
        super().__init__()
        self.hdim = hdim
        self.zdim = zdim
        self.dropout = dropout

        self.layers = nn.ModuleList([LSTMCell(zdim , hdim)] #input feeding
            + [LSTMCell(hdim, hdim) for layer in range(1, num_layers)]
        )
        self.attention = AttentionLayer(encoder_hdim, hdim)
        self.mu_out = Linear(hdim, zdim, dropout=0)
        self.sg_out = Linear(hdim, zdim, dropout=0)
        #self.W_feed = Linear(2*hdim, hdim, bias=False)
        self.init_z = nn.Parameter(torch.Tensor(zdim).normal_())
        self.fc = Linear(self.zdim, encoder_hdim, dropout=0)

    def forward(self, encoder_hiddens, seqlen):
        num_layers = len(self.layers)
        bsz = encoder_hiddens.size()[0]

        zero = Variable(encoder_hiddens.data.new(bsz, self.hdim).zero_())
        prev_hiddens = [zero for i in range(num_layers)]
        prev_cells = [zero for i in range(num_layers)]

        #In the beginning, uniform attention
        #input_feed = encoder_hiddens.mean(dim=1) 
        #input_feed = F.tanh(self.W_feed(torch.cat((input_feed, zero), dim=1)))

        init_z = torch.stack([self.init_z] * bsz, dim=0)
        zouts = [init_z]
        for j in range(seqlen):
            # input feeding: concatenate context vector from previous time step
            #input = torch.cat((zouts[-1], input_feed), dim=1)
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
            cxt = self.attention(hidden, encoder_hiddens)
            #out = F.tanh(self.W_feed(torch.cat((cxt, hidden), dim=1)))

            cxt = F.dropout(cxt, p=self.dropout, training=self.training)
            #out = F.dropout(out, p=self.dropout, training=self.training)

            # input feeding
            #input_feed = out

            # sample z from context vector cxt 
            mu = self.mu_out(cxt)
            log_sigma = -torch.abs(self.sg_out(cxt))
            eps = Variable(mu.data.clone().normal_(std=0.0001))
            zouts.append( mu + eps * torch.exp(log_sigma / 2) )

        # collect outputs across time steps
        zouts = torch.cat(zouts[1:], dim=0).view(seqlen, bsz, self.zdim)
        zouts = self.fc(zouts)
        return zouts.transpose(1, 0) # T x B x C -> B x T x C

@register_model_architecture('lstm_zseq', 'lstm_zseq')
def base_architecture(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_layers = getattr(args, 'encoder_layers', '[(512, 3)] * 20')
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 512)
    args.decoder_layers = getattr(args, 'decoder_layers', '[(512, 3)] * 20')
    args.decoder_out_embed_dim = getattr(args, 'decoder_out_embed_dim', 256)
    args.decoder_attention = getattr(args, 'decoder_attention', 'True')
    args.share_input_output_embed = getattr(args, 'share_input_output_embed', False)


@register_model_architecture('lstm_zseq', 'lstm_zseq_iwslt_de_en')
def fconv_iwslt_de_en(args):
    base_architecture(args)
    args.encoder_embed_dim = 256
    args.encoder_layers = '[(256, 3)] * 4'
    args.decoder_embed_dim = 256
    args.decoder_layers = '[(256, 3)] * 3'
    args.decoder_out_embed_dim = 256

