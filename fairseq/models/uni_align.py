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


@register_model('align')
class AlignModel(FairseqModel):
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
        src_controller = Controller(embed_hdim=args.encoder_embed_dim, hdim=args.encoder_embed_dim)
        tgt_controller = Controller(embed_hdim=args.encoder_embed_dim, hdim=args.encoder_embed_dim)

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

        T = min(src_tokens.size(1), tgt_tokens.size(1))
        src_x, attn_ac = self.src_controller(src_x, T) #B * T * N
        tgt_x, attn_bc = self.tgt_controller(tgt_x, T) #B * T * M

        src_encoder_out = self.all_encoder(src_x, src_lengths)
        tgt_encoder_out = self.all_encoder(tgt_x, src_lengths)

        tgt_decoder_out,attn_cb = self.tgt_decoder(prev_output_tokens, src_encoder_out)
        src_decoder_out,attn_ca = self.src_decoder(prev_src_tokens, tgt_encoder_out)

        reg = F.mse_loss(src_x, tgt_x) * bsz #size_average takes mean over all dimensions
        #print(torch.norm(src_x[0] - tgt_x[0], p=1, dim=1) / 256)

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

        return src_decoder_out, tgt_decoder_out, reg

class Controller(nn.Module):
    def __init__(self, embed_hdim=512, hdim=512, zdim=256, num_layers=1, dropout=0):
        super().__init__()
        self.hdim = hdim
        self.zdim = zdim
        self.dropout = dropout

        self.layers = nn.ModuleList([LSTMCell(zdim , hdim)] #input feeding
            + [LSTMCell(hdim, hdim) for layer in range(1, num_layers)]
        )
        self.attention = AttentionLayer(embed_hdim, hdim)
        self.mu_out = Linear(hdim, zdim, dropout=0, bias=False)
        #self.sg_out = Linear(hdim, zdim, dropout=0, bias=False)
        self.init_z = nn.Parameter(torch.Tensor(hdim).normal_(0, 0.1))
        #self.fc = Linear(self.zdim, encoder_hdim, dropout=0, bias=False)

    def forward(self, encoder_hiddens, seqlen):
        num_layers = len(self.layers)
        bsz = encoder_hiddens.size(0)

        zero = Variable(torch.zeros(bsz, self.hdim)).cuda()
        prev_hiddens = [zero for i in range(num_layers)]
        prev_cells = [zero for i in range(num_layers)]

        init_z = torch.stack([self.init_z] * bsz, dim=0)
        zouts = [init_z]
        attns = []
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
            cxt, attn_score = self.attention(hidden, encoder_hiddens)
            attns.append(attn_score)

            # sample z from context vector cxt 
            cxt = F.dropout(cxt, p=self.dropout, training=self.training)
            mu = self.mu_out(cxt)
            zouts.append( mu )

        # collect outputs across time steps => B * T * N
        zouts = torch.stack(zouts[1:], dim=1)
        attns = torch.stack(attns, dim=1) 
        return zouts, attns

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

class SimpleDecoder(FairseqDecoder):
    def __init__(self, dictionary, embed_dim):
        super().__init__(dictionary)
        num_embeddings = len(dictionary)
        self.output = Linear(embed_dim, num_embeddings)

    def forward(self, cxt):
        return self.output(cxt), None
    
    def max_positions(self):
        """Maximum input length supported by the decoder."""
        return 1e5

class NoattConvDecoder(FairseqIncrementalDecoder):
    """Convolutional decoder"""
    def __init__(self, dictionary, embed_dim=512, in_embed_dim=256, out_embed_dim=256, max_positions=1024, convolutions=((256, 3),) * 2, dropout=0.1, share_embed=False, left_pad=LanguagePairDataset.LEFT_PAD_TARGET):
        super().__init__(dictionary)
        self.register_buffer('version', torch.Tensor([2]))
        self.dropout = dropout

        in_channels = convolutions[0][0]
        
        num_embeddings = len(dictionary)
        padding_idx = dictionary.pad()
        self.embed_tokens = Embedding(num_embeddings, embed_dim, padding_idx)
        self.embed_positions = PositionalEmbedding(
            max_positions,
            embed_dim,
            padding_idx,
            left_pad=left_pad
        )

        self.fc1 = Linear(embed_dim, in_channels, dropout=dropout)
        self.projections = nn.ModuleList()
        self.convolutions = nn.ModuleList()
        for i, (out_channels, kernel_size) in enumerate(convolutions):
            self.projections.append(Linear(in_channels, out_channels)
                                    if in_channels != out_channels else None)
            self.convolutions.append(
                LinearizedConv1d(in_channels, out_channels * 2, kernel_size,
                                 padding=(kernel_size - 1), dropout=dropout)
            )
            in_channels = out_channels

        self.fc2 = Linear(in_channels + in_embed_dim, out_embed_dim)
        if share_embed:
            assert out_embed_dim == embed_dim, \
                "Shared embed weights implies same dimensions " \
                " out_embed_dim={} vs embed_dim={}".format(out_embed_dim, embed_dim)
            self.fc3 = nn.Linear(out_embed_dim, num_embeddings)
            self.fc3.weight = self.embed_tokens.weight
        else:
            self.fc3 = Linear(out_embed_dim, num_embeddings, dropout=dropout)

    def forward(self, prev_output_tokens, encoder_out, incremental_state=None):
        # split and transpose encoder outputs
        encoder_a, encoder_b = self._split_encoder_out(encoder_out, incremental_state)

        # embed tokens and combine with positional embeddings
        x = self._embed_tokens(prev_output_tokens, incremental_state)
        x += self.embed_positions(prev_output_tokens, incremental_state)
        x = F.dropout(x, p=self.dropout, training=self.training)
        target_embedding = x

        # project to size of convolution
        x = self.fc1(x)

        # B x T x C -> T x B x C
        x = self._transpose_if_training(x, incremental_state)

        # temporal convolutions
        for proj, conv in zip(self.projections, self.convolutions):
            residual = x if proj is None else proj(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = conv(x, incremental_state)
            x = F.glu(x, dim=2)
            x = (x + residual) * math.sqrt(0.5)

        # T x B x C -> B x T x C
        x = self._transpose_if_training(x, incremental_state)
        x = torch.cat([x, encoder_out[0]], dim=2)

        # project back to size of vocabulary
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc3(x)

        return x, None

    def max_positions(self):
        """Maximum output length supported by the decoder."""
        return self.embed_positions.max_positions()

    def upgrade_state_dict(self, state_dict):
        if state_dict.get('decoder.version', torch.Tensor([1]))[0] < 2:
            # old models use incorrect weight norm dimension
            for i, conv in enumerate(self.convolutions):
                # reconfigure weight norm
                nn.utils.remove_weight_norm(conv)
                self.convolutions[i] = nn.utils.weight_norm(conv, dim=0)
            state_dict['decoder.version'] = torch.Tensor([1])
        return state_dict

    def _embed_tokens(self, tokens, incremental_state):
        if incremental_state is not None:
            # keep only the last token for incremental forward pass
            tokens = tokens[:, -1:]
        return self.embed_tokens(tokens)

    def _split_encoder_out(self, encoder_out, incremental_state):
        """Split and transpose encoder outputs.

        This is cached when doing incremental inference.
        """
        cached_result = utils.get_incremental_state(self, incremental_state, 'encoder_out')
        if cached_result is not None:
            return cached_result

        # transpose only once to speed up attention layers
        encoder_a, encoder_b = encoder_out
        encoder_a = encoder_a.transpose(1, 2).contiguous()
        result = (encoder_a, encoder_b)

        if incremental_state is not None:
            utils.set_incremental_state(self, incremental_state, 'encoder_out', result)
        return result

    def _transpose_if_training(self, x, incremental_state):
        if incremental_state is None:
            x = x.transpose(0, 1)
        return x

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

@register_model_architecture('align', 'align')
def base_architecture(args):
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_layers = getattr(args, 'encoder_layers', '[(512, 3)] * 20')
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 512)
    args.decoder_layers = getattr(args, 'decoder_layers', '[(512, 3)] * 20')
    args.decoder_out_embed_dim = getattr(args, 'decoder_out_embed_dim', 256)
    args.decoder_attention = getattr(args, 'decoder_attention', 'True')
    args.share_input_output_embed = getattr(args, 'share_input_output_embed', False)


@register_model_architecture('align', 'align_iwslt_de_en')
def align_iwslt_de_en(args):
    base_architecture(args)
    args.encoder_embed_dim = 256
    args.encoder_layers = '[(256, 3)] * 4'
    args.decoder_embed_dim = 256
    args.decoder_layers = '[(256, 3)] * 3'
    args.decoder_out_embed_dim = 256

