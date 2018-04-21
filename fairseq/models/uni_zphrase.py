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
    def __init__(self, src_encoder, tgt_encoder, src_decoder, tgt_decoder, uni_encoder, data_file):
        super().__init__(src_encoder, tgt_decoder)
        self.uni_encoder = uni_encoder
        self.uni_encoder.num_attention_layers = sum(layer is not None for layer in tgt_decoder.attention)

        self.src_encoder = src_encoder
        self.tgt_encoder = tgt_encoder

        self.src_decoder = src_decoder
        self.tgt_decoder = tgt_decoder

        self.update_step = 0
        self.warmup_step = args.warmup_step

        with open(data_file, 'rb') as f:
            self.train_src, self.train_tgt, self.valid_src, self.valid_tgt = torch.load(f)
        
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
        parser.add_argument('--data-file', type=str) 

    @classmethod
    def build_model(cls, args, src_dict, tgt_dict):
        """Build a new model instance."""

        uni_encoder = FConvEncoder(
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

        src_encoder = PhraseEncoder(
            src_dict,
            embedding=src_decoder.embed_tokens,
            hdim=args.encoder_embed_dim,
            dropout=args.dropout,
            project=True
        )

        tgt_encoder = PhraseEncoder(
            tgt_dict,
            embedding=tgt_decoder.embed_tokens,
            hdim=args.encoder_embed_dim,
            dropout=args.dropout,
            project=False
        )

        return cls(src_encoder, tgt_encoder, src_decoder, tgt_decoder, uni_encoder, args.data_file)

    def forward(self, src_tokens, tgt_tokens, src_lengths, prev_src_tokens, prev_output_tokens, sent_ids):
        if self.training:
            src_phrases = [self.train_src[i] for i in sent_ids]
            tgt_phrases = [self.train_tgt[i] for i in sent_ids]
        else:
            src_phrases = [self.valid_src[i] for i in sent_ids]
            tgt_phrases = [self.valid_tgt[i] for i in sent_ids]

        self.update_step += 1

        src_phrases = self.pad_sequence(src_phrases)
        tgt_phrases = self.pad_sequence(tgt_phrases)

        src_encoder_out = self.src_encoder(src_phrases)
        tgt_encoder_out = self.tgt_encoder(tgt_phrases)

        if self.update_step > self.warmup_step:
            src_encoder_out = F.normalize(src_encoder_out, dim=2)
            tgt_encoder_out = F.normalize(tgt_encoder_out, dim=2)
            uni_out = F.normalize(src_encoder_out + tgt_encoder_out, dim=2)
        else:
            uni_out = 0.5 * (src_encoder_out + tgt_encoder_out)

        uni_encoder_out = self.uni_encoder(uni_out, src_lengths)
        src_decoder_out, _ = self.src_decoder(prev_src_tokens, uni_encoder_out)
        tgt_decoder_out, _ = self.tgt_decoder(prev_output_tokens, uni_encoder_out)

        #Negative sampling
        src_encoder_out = src_encoder_out.view(-1, src_encoder_out.size(-1))
        tgt_encoder_out = tgt_encoder_out.view(-1, tgt_encoder_out.size(-1))
        shuf_idx = torch.randperm(len(tgt_encoder_out))
        shuf_encoder_out = tgt_encoder_out[shuf_idx, ...]
        #reg_loss = F.cosine_similarity(src_encoder_out, shuf_encoder_out, dim=1).sum() - F.cosine_similarity(src_encoder_out, tgt_encoder_out, dim=1).sum()
        if self.update_step > self.warmup_step:
            reg_loss = (src_encoder_out * shuf_encoder_out).sum() - (src_encoder_out * tgt_encoder_out).sum()
        else:
            reg_loss = Variable(uni_out.data.new(1).zero_())

        #print(F.cosine_similarity(src_encoder_out, tgt_encoder_out, dim=1).mean().item(), end=' ')
        #print(F.cosine_similarity(src_encoder_out, shuf_encoder_out, dim=1).mean().item())
        
        return src_decoder_out, tgt_decoder_out, reg_loss
    
    def pad_sequence(self, sequences):
        max_size = max([seq.size(0) for seq in sequences])
        out_dims = (len(sequences), max_size) + sequences[0].size()[1:]
        out_tensor = sequences[0].new(*out_dims).zero_()

        for i, tensor in enumerate(sequences):
            length = tensor.size(0)
            out_tensor[i, :length, ...] = tensor
        return Variable(out_tensor.long().cuda())

class PhraseEncoder(FairseqEncoder):
    """Phrase Conv Encoder"""
    def __init__(self, dictionary, embedding, hdim, ksize=5, dropout=0.2, project=False):
        super().__init__(dictionary)
        self.dropout = dropout
        self.hdim = hdim
        self.ksize = ksize
        self.padding_idx = 0 #not dictionary.pad(), which is 1
        self.embed_dim = embedding.embedding_dim
        self.project = project

        self.embedding = nn.Embedding(len(dictionary), self.embed_dim, self.padding_idx)
        self.embedding.weight = embedding.weight #weight is shared between encoder and decoder
        self.cnn = Linear(ksize * self.embed_dim, hdim * 2)
        if project:
            self.W_o = Linear(self.hdim, self.hdim)

    def forward(self, tokens):
        bsz,seqlen,ksize = tokens.size()
        x = self.embedding(tokens)
        mask = torch.ne(tokens, self.padding_idx).float()
        x = x * mask.unsqueeze(-1) #mask out padding
        x = x.view(-1, self.ksize * self.embed_dim)

        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.glu(self.cnn(x), dim=1)
        if self.project:
            x = self.W_o(x)

        return x.view(bsz, seqlen, self.hdim)

    def max_positions(self):
        return 1e5

class Controller(nn.Module):
    def __init__(self, embedding, hdim, num_layers=1, dropout=0.2):
        super().__init__()
        self.hdim = hdim
        self.zdim = zdim
        self.dropout = dropout

        self.layers = nn.ModuleList([LSTMCell(zdim , hdim)] #input feeding
            + [LSTMCell(hdim, hdim) for layer in range(1, num_layers)]
        )
        self.attention = AttentionLayer(encoder_hdim, hdim)
        self.mu_out = Linear(hdim, zdim, dropout=0, bias=False)
        #self.sg_out = Linear(hdim, zdim, dropout=0, bias=False)
        self.init_z = nn.Parameter(torch.Tensor(zdim).normal_(0, 0.1))
        #self.fc = Linear(self.zdim, encoder_hdim, dropout=0, bias=False)

    def forward(self, encoder_hiddens, seqlen):
        num_layers = len(self.layers)
        bsz,seqlen = encoder_hiddens.size(0),encoder_hiddens.size(1)

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
            cxt, attn_score = self.attention(hidden, encoder_hiddens)

            # sample z from context vector cxt 
            cxt = F.dropout(cxt, p=self.dropout, training=self.training)
            mu = self.mu_out(cxt)
            zouts.append( mu )

        # collect outputs across time steps
        #zouts = self.fc(zouts)
        return torch.stack(zouts[1:], dim=1) 

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
    args.warmup_step = 50


@register_model_architecture('uni_zphrase', 'uni_zphrase_iwslt_de_en')
def fconv_iwslt_de_en(args):
    base_architecture(args)
    args.encoder_embed_dim = 256
    args.encoder_layers = '[(256, 3)] * 4'
    args.decoder_embed_dim = 256
    args.decoder_layers = '[(256, 3)] * 3'
    args.decoder_out_embed_dim = 256

