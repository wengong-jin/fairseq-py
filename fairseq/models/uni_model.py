# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import torch.nn as nn

from . import FairseqModel, FairseqDecoder, FairseqEncoder


class UniSeqModel(FairseqModel):
    """Base class for multi lingual encoder-decoder models."""

    def __init__(self, encoder, decoder, src_decoder):
        super().__init__(encoder, decoder)
        self.src_decoder = src_decoder
        
    def forward(self, src_tokens, src_lengths, prev_src_tokens, prev_output_tokens):
        encoder_out = self.encoder(src_tokens, src_lengths)
        decoder_out, _ = self.decoder(prev_output_tokens, encoder_out)
        src_decoder_out, _ = self.src_decoder(prev_src_tokens, encoder_out)
        return src_decoder_out, decoder_out
