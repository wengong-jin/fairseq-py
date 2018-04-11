# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the LICENSE file in
# the root directory of this source tree. An additional grant of patent rights
# can be found in the PATENTS file in the same directory.

import math
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from . import FairseqCriterion, register_criterion
from fairseq import utils

@register_criterion('align_cross_entropy')
class AlignCrossEntropyCriterion(FairseqCriterion):

    def __init__(self, args, src_dict, dst_dict):
        super().__init__(args, src_dict, dst_dict)

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss, as a Variable
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        net_input = sample['net_input']
        src_tokens = net_input['src_tokens']
        start = Variable(src_tokens.data[:,0:1].clone().fill_(self.eos_idx))
        net_input['prev_src_tokens'] = torch.cat((start, src_tokens[:,:-1]), dim=1)
        net_input['tgt_tokens'] = Variable(sample['target'].data.clone())
        net_input['sent_ids'] = sample['id']
        tgt_output = model(**net_input)

        lprobs = model.get_normalized_probs(tgt_output, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = sample['target'].view(-1)
        loss = F.nll_loss(lprobs, target, size_average=False, ignore_index=self.padding_idx, reduce=reduce)

        sample_size = sample['target'].size(0) if self.args.sentence_avg else sample['ntokens'] 
        logging_output = {
            'loss': utils.item(tgt_loss.data) if reduce else tgt_loss.data,
            'ntokens': sample['ntokens'],
            'sample_size': sample_size,
        }
        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get('loss', 0) for log in logging_outputs)
        ntokens = sum(log.get('ntokens', 0) for log in logging_outputs)
        sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)
        agg_output = {
            'loss': loss_sum / sample_size / math.log(2),
        }
        if sample_size != ntokens:
            agg_output['nll_loss'] = loss_sum / ntokens / math.log(2)
        return agg_output
