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

from sklearn.metrics import f1_score

from fairseq import utils
from fairseq.data import LanguagePairDataset
from fairseq.modules import BeamableMM, GradMultiply, LearnedPositionalEmbedding, LinearizedConvolution

from . import FairseqEncoder, FairseqIncrementalDecoder, FairseqModel, register_model, register_model_architecture
from .fconv import FConvEncoder, FConvDecoder, Embedding, PositionalEmbedding


@register_model('uni_zphrase')
class UniZPhraseModel(FairseqModel):
    def __init__(self, encoder, decoder, z_encoder, z_decoder, align_file):
        super().__init__(encoder, decoder)
        self.z_encoder = z_encoder
        self.z_decoder = z_decoder
        self.encoder.num_attention_layers = sum(layer is not None for layer in decoder.attention)
        self.z_encoder.num_attention_layers = 1
        self.all_align = []

        with open(align_file) as f:
            self.align = [line.strip("\r\n ").split() for line in f]

        for idx,align in enumerate(self.align):
            align = [x.split('-') for x in align]
            align = [(int(x[0]),int(x[1])) for x in align]
            srclen = max([x for x,y in align]) + 1
            amap = [set() for i in range(srclen)]
            for x,y in align:
                amap[x].add(y)

            label = torch.zeros(srclen) #there is an extra token </s> at the end
            for i in range(srclen - 1):
                #if len(amap[i]) == 0 or len(amap[i + 1]) == 0 or len(amap[i] & amap[i + 1]) > 0:
                if len(amap[i] & amap[i + 1]) > 0:
                    label[i] = 1

            self.all_align.append(Variable(label))

        self.train_align = self.all_align[:160215]
        self.valid_align = self.all_align[160215:]

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
        parser.add_argument('--align-file', type=str) 

    @classmethod
    def build_model(cls, args, src_dict, dst_dict):
        """Build a new model instance."""
        z_encoder = SegEncoder(
            src_dict,
            embed_dim=args.encoder_embed_dim,
            dropout=args.dropout,
            max_positions=args.max_source_positions
        )

        z_decoder = Controller(
            encoder_hdim = 256,
            hdim = 256,
            zdim = 256,
            num_layers = 1,
            length = 1
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

        return cls(encoder, decoder, z_encoder, z_decoder, args.align_file)

    def forward(self, src_tokens, tgt_tokens, src_lengths, prev_src_tokens, prev_output_tokens, sent_ids):
        if self.training:
            seg_scores = [self.train_align[i] for i in sent_ids]
        else:
            seg_scores = [self.valid_align[i] for i in sent_ids]
        seg_scores = sorted(seg_scores, key=lambda x:x.size(0), reverse=True)
        seg_scores = nn.utils.rnn.pad_sequence(seg_scores, batch_first=True).cuda()

        z_encoder_out, seg_scores = self.z_encoder(src_tokens, src_lengths, seg_scores)
        z_decoder_out = self.z_decoder(z_encoder_out, seg_scores)

        encoder_out = self.encoder(z_decoder_out, src_lengths)
        decoder_out, _ = self.decoder(prev_output_tokens, encoder_out)

        return decoder_out

class Controller(nn.Module):
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
        self.mu_out = Linear(hdim, zdim, dropout=0, bias=False)
        self.sg_out = Linear(hdim, zdim, dropout=0, bias=False)
        self.init_z = nn.Parameter(torch.Tensor(zdim).normal_(0, 0.1))
        #self.fc = Linear(self.zdim, encoder_hdim, dropout=0, bias=False)

    def forward(self, encoder_hiddens, seg_scores):
        num_layers = len(self.layers)
        bsz,seqlen = encoder_hiddens.size(0),encoder_hiddens.size(1)
        seqlen = max(int(seqlen * self.length), 1)

        zero = Variable(encoder_hiddens.data.new(bsz, self.hdim).zero_())
        prev_hiddens = [zero for i in range(num_layers)]
        prev_cells = [zero for i in range(num_layers)]

        init_z = torch.stack([self.init_z] * bsz, dim=0)
        zouts = [init_z]
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
            cxt, attn_score = self.attention(hidden, encoder_hiddens, seg_scores)

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

class SegEncoder(nn.Module):
    def __init__(self, dictionary, embed_dim, dropout=0.1, max_positions=1024, max_num_segs=3):
        super().__init__()
        self.dictionary = dictionary
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.max_num_segs = max_num_segs

        num_embeddings = len(dictionary)
        self.padding_idx = dictionary.pad()
        self.embed_tokens = Embedding(num_embeddings, embed_dim, self.padding_idx)
        #self.conv1 = Linear(embed_dim, embed_dim)
        self.conv2 = Conv1d(embed_dim, embed_dim, kernel_size=2)

        self.embed_positions = PositionalEmbedding(
            max_positions,
            embed_dim,
            self.padding_idx,
            left_pad=LanguagePairDataset.LEFT_PAD_SOURCE,
        )
        
    def forward(self, src_tokens, src_lengths, seg_scores):
        seg_scores = F.pad(seg_scores, (1, 1))
        l_seg = seg_scores[:, :-1]
        r_seg = seg_scores[:, 1:]
        pscore1 = (1 - l_seg) * (1 - r_seg)

        #2-word phrase
        l_seg = seg_scores[:, :-2]
        r_seg = seg_scores[:, 2:]
        m_seg = seg_scores[:, 1:-1]
        pscore2 = (1 - l_seg) * (1 - r_seg) * m_seg

        x = self.embed_tokens(src_tokens) + self.embed_positions(src_tokens)
        x = F.dropout(x, p=self.dropout, training=self.training)
        #x1 = F.relu(self.conv1(x))
        x1 = x * pscore1.unsqueeze(2)

        x2 = F.relu(self.conv2(x.transpose(1, 2))) #no padding
        x2 = F.dropout(x2, p=self.dropout, training=self.training)
        x2 = x2.transpose(1, 2)
        x2 = x2 * pscore2.unsqueeze(2)

        x = torch.cat((x1[:,:-1,:] + x2, x1[:,-1:,:]), dim=1)
        phrase_scores = torch.cat((pscore1[:,:-1] + pscore2, pscore1[:,-1:]), dim=1)

        return x, phrase_scores

class AttentionLayer(nn.Module):
    """T. Luong's global attention"""
    def __init__(self, input_embed_dim, output_embed_dim):
        super().__init__()
        self.input_proj = Linear(input_embed_dim, output_embed_dim, bias=False)

    def forward(self, input, source_hids, seg_scores):
        # input: bsz x input_embed_dim
        # source_hids: bsz x srclen x output_embed_dim
        # x: bsz x output_embed_dim
        x = self.input_proj(input)
        
        # compute attention
        attn_scores = (source_hids * x.unsqueeze(1)).sum(dim=2)
        attn_scores = attn_scores - 100.0 * (1 - seg_scores)
        attn_scores = F.softmax(attn_scores, dim=1)  # bsz x srclen
        
        xx,yy = attn_scores.max(dim=1)
        xx,yy = xx[0].item(), yy[0].item()
        #print(xx,yy)
        """
        srclen = (source_hids.size(1) + 1) // 2
        if yy < srclen:
            print(xx, 0, yy, seg_scores[0,yy].item())
        else:
            print(xx, 1, yy - srclen, seg_scores[0,yy].item())
        """

        # sum weighted sources
        x = (attn_scores.unsqueeze(2) * source_hids).sum(dim=1)
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
    m = nn.Linear(in_features, out_features, bias=bias)
    m.weight.data.normal_(mean=0, std=0.1)
    if bias:
        m.bias.data.zero_()
    return m

def Conv1d(in_features, out_features, kernel_size, dropout=0):
    m = nn.Conv1d(in_features, out_features, kernel_size)
    m.weight.data.normal_(mean=0, std=0.1)
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

